import sys
import os.path
import math
import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import copy
import argparse
import time

cudnn.enabled = True
cudnn.benchmark = True

from ansemb.config import cfg, set_random_seed, update_train_configs
import ansemb.dataset.vqa as data
import ansemb.models.embedding as model
import ansemb.utils as utils

from ansemb.utils import cosine_sim
from ansemb.vector import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--max_negative_answer', default=12000, type=int)
parser.add_argument('--answer_batch_size', default=3000, type=int)
parser.add_argument('--loss_temperature', default=0.01, type=float)
parser.add_argument('--pretrained_model', default=None, type=str)
parser.add_argument('--context_embedding', default='SAN', choices=['SAN', 'BoW'])
parser.add_argument('--answer_embedding', default='BoW', choices=['BoW', 'RNN'])
parser.add_argument('--name', default=None, type=str)
args = parser.parse_args()

# fix random seed
set_random_seed(cfg.seed)

def test(context_net, answer_net, loader, tracker, args, prefix='', epoch=0):
  context_net.eval()
  answer_net.eval()
  tracker_class, tracker_params = tracker.MeanMonitor, {}
  ans_ids, que_ids = [], []
  accs, masks = [], []

  tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
  acc_tracker  = tracker.track('{}_acc'.format(prefix),  tracker_class(**tracker_params))

  var_params = { 'volatile': True, 'requires_grad': False }
  if args.answer_embedding == 'RNN':
    answer_var, answer_len = loader.dataset._get_answer_sequences(range(cfg.TEST.max_answer_index))
  else:
    answer_var, answer_len = loader.dataset._get_answer_vectors(range(cfg.TEST.max_answer_index))

  answer_var = Variable(answer_var.cuda(), **var_params)
  answer_embedding = answer_net.forward(answer_var, answer_len)

  cnt = 0
  for v, q, _, _, labels, idx, q_len in tq:
    v = Variable(v.cuda(), **var_params)
    q = Variable(q.cuda(), **var_params)
    q_len = Variable(q_len.cuda(), **var_params)

    context_embedding = context_net(v, q, q_len)

    predicts = cosine_sim(context_embedding, answer_embedding) / args.loss_temperature #temperature
    acc = utils.batch_accuracy(predicts.data, labels.cuda()).cpu()

    _, flag = labels.max(1)
    flag = ( flag >= 2 ).byte()
    masks.append(flag)

    accs.append(acc.view(-1))
    acc_tracker.append(acc.mean())

    # collect stats
    _, _ans_ids = predicts.data.cpu().max(dim=1)
    ans_ids.append(_ans_ids.view(-1))
    que_ids.append(idx.view(-1).clone())

    fmt = '{:.4f}'.format
    tq.set_postfix(acc=fmt(acc_tracker.mean.value))

  return accs, masks, ans_ids, que_ids

def train(context_net, answer_net, loader, optimizer, tracker, args, prefix='', epoch=0):
  """ Run an epoch over the given loader """
  context_net.train()
  answer_net.train()
  tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

  tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
  loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
  acc_tracker  = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
  lr_tracker   = tracker.track('{}_lr'.format(prefix), tracker_class(**tracker_params))

  var_params = { 'volatile': False, 'requires_grad': False, }
  log_softmax = nn.LogSoftmax().cuda()
  cnt = 0
  start_tm=time.time()
  for v, q, avocab, a, labels, idx, q_len in tq:
    data_tm = time.time() - start_tm
    start_tm=time.time()

    if args.answer_embedding == 'RNN':
      answer_var, answer_len = loader.dataset._get_answer_sequences(avocab)
    else:
      answer_var, answer_len = loader.dataset._get_answer_vectors(avocab)
    answer_var = Variable(answer_var.cuda(), **var_params)

    v = Variable(v.cuda(), **var_params)
    q = Variable(q.cuda(), **var_params)
    a = Variable(a.cuda(), **var_params)
    q_len = Variable(q_len.cuda(), **var_params)

    encode_tm = time.time() - start_tm
    start_tm=time.time()

    context_embedding = context_net(v, q, q_len)
    answer_embedding  = answer_net(answer_var, answer_len)

    predicts = cosine_sim(context_embedding, answer_embedding) / args.loss_temperature #temperature
    nll = -log_softmax(predicts)
    loss = (nll * a / a.sum(1, keepdim=True)).sum(dim=1).mean()

    acc = utils.batch_accuracy(predicts.data, a.data).cpu()

    global total_iterations
    lr = utils.update_learning_rate(optimizer, epoch)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    model_tm = time.time() - start_tm
    start_tm=time.time()

    loss_tracker.append(loss.data[0])
    acc_tracker.append(acc.mean())
    lr_tracker.append(lr)
    fmt = '{:.6f}'.format
    tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value), lr=fmt(lr_tracker.mean.value), t_data=data_tm, t_model=model_tm, t_encode=encode_tm)

def main(args):
  if args.name is None:
    from datetime import datetime
    name = args.context_embedding+"_"+args.answer_embedding+"_vqa_batch_softmax_embedding_"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = ( name + '_finetune' ) if args.finetune else name
  else:
    name = args.context_embedding+"_"+args.answer_embedding+"_vqa_batch_softmax_embedding_"+args.name

  output_filepath = os.path.join(cfg.output_path, '{}.pth'.format(name))
  print('Output data would be saved to {}'.format(output_filepath))

  word2vec = Vector()
  train_loader = data.get_loader(word2vec, train=True)
  val_loader = data.get_loader(word2vec, val=True)

  question_word2vec = word2vec._prepare(train_loader.dataset.token_to_index)

  if args.context_embedding == 'SAN':
    context_net = model.StackedAttentionEmbedding(
                      train_loader.dataset.num_tokens,
                      question_word2vec).cuda()
  elif args.context_embedding == 'BoW':
    context_net = model.VisualSemanticEmbedding(
                      train_loader.dataset.num_tokens,
                      question_word2vec).cuda()
  else:
    raise TypeError('Unsupported Context Model')

  if args.answer_embedding == 'BoW':
    answer_net = model.MLPEmbedding(train_loader.dataset.vector.dim).cuda()
  elif args.answer_embedding == 'RNN':
    answer_net = model.RNNEmbedding(train_loader.dataset.vector.dim).cuda()
  else:
    raise TypeError('Unsupported Answer Model')

  print('Context Model:')
  print(context_net)

  print('Answer Model:')
  print(answer_net)

  if args.pretrained_model is not None:
    states = torch.load(args.pretrained_model)
    answer_state, context_state = states['answer_net'], states['context_net']

    answer_net.load_state_dict(answer_state)
    context_net.load_state_dict(context_state)

  params_for_optimization = list(context_net.parameters()) + list(answer_net.parameters())
  optimizer = optim.Adam([p for p in params_for_optimization if p.requires_grad])

  tracker = utils.Tracker()
  if args.pretrained_model is not None:
    accs, masks, ans_ids, que_ids  = test(context_net, answer_net, val_loader, tracker, args, prefix='val', epoch=-1)

    total_accs  = torch.cat(accs)
    total_masks = torch.cat(masks)
    print('* VQA2 Val Accuracy: {}'.format(total_accs.mean()))
    accs_ = total_accs[total_masks].sum()
    print('* VQA2- Val Accuracy: {}'.format(accs_ / total_masks.sum()))

    results = { 'name': name,
                'eval': [{'answer_ids': ans_ids, 'question_ids': que_ids}],
                'vocab': { 'answer_to_index': train_loader.dataset.answer_to_index,
                           'index_to_answer': train_loader.dataset.index_to_answer  } }
    print('* Dumpping output to: {}'.format(output_filepath))
    torch.save(results, output_filepath)

    if not args.finetune:
      raise ValueError('Testing Finished')

  best_val_acc = 0
  best_context_net, best_answser_net = None, None
  _eval = []
  for i in range(cfg.TRAIN.epochs):
    train(context_net, answer_net, train_loader, optimizer, tracker, args, prefix='train', epoch=i)
    accs, _, ans_ids, que_ids = test(context_net, answer_net, val_loader, tracker, args, prefix='val', epoch=i)

    _eval.append({ 'accuracies': accs, 'answer_ids': ans_ids, 'question_ids': que_ids })
    val_acc = torch.mean( torch.cat(accs, dim=0) )
    if best_val_acc < val_acc:
      best_val_acc = val_acc
      best_context_net = copy.deepcopy( context_net.state_dict() )
      best_answer_net  = copy.deepcopy( answer_net.state_dict() )

    results = {
      'name': name,
      'tracker': tracker.to_dict(),
      'config': cfg,
      'context_net': best_context_net,
      'answer_net':  best_answer_net,
      'eval': _eval,
      'vocab': { 'answer_to_index': train_loader.dataset.answer_to_index,
                 'index_to_answer': train_loader.dataset.index_to_answer }
    }
    torch.save(results, output_filepath)

if __name__ == '__main__':
  torch.cuda.set_device(args.gpu_id)
  print(args.__dict__)
  print(cfg)
  update_train_configs(args)
  main(args)
