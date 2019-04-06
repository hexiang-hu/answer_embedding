import json
import os
import os.path as osp
import nltk

from collections import Counter, OrderedDict
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate

from ansemb.config import cfg
from ansemb.dataset.preprocess import process_punctuation
from ansemb.dataset.base import VisualQA

import ansemb.utils as utils
import ansemb.dataset.data_utils as data_utils
from IPython import embed

def path_for(train=False, val=False, test=False):
  assert train + val + test == 1
  if train:  split = cfg.Visual7W.train_qa
  elif val:  split = cfg.Visual7W.val_qa
  elif test: split = cfg.Visual7W.test_qa
  else:    raise ValueError('Unsupported split.')

  return os.path.join(cfg.Visual7W.qa_path, split)

def path_for_decoys(train=False, val=False, test=False):
  assert train + val + test == 1
  if train:  split = cfg.Visual7W.train_v7w_decoys
  elif val:  split = cfg.Visual7W.val_v7w_decoys
  elif test: split = cfg.Visual7W.test_v7w_decoys
  else:    raise ValueError('Unsupported split.')

  return os.path.join(cfg.Visual7W.qa_path, split)

def get_loader(vector, train=False, val=False, test=False, vocab_path=None):
  assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
  split = Visual7W(
    path_for(train=train, val=val, test=test),
    path_for_decoys(train=train, val=val, test=test),
    cfg.Visual7W.feature_path,
    vector,
    vocab_path=vocab_path,
    answerable_only=train,
  )
  loader = torch.utils.data.DataLoader(
    split,
    batch_size=cfg.TRAIN.batch_size,
    shuffle=train,  # only shuffle the data in training
    pin_memory=True,
    num_workers=cfg.TRAIN.data_workers,
    collate_fn=data_utils.collate_fn,
  )
  return loader

class Visual7W(VisualQA):
  def __init__(self, questions_path, decoys_path, image_features_path,
                     vector, vocab_path=None, answerable_only=False):
    answer_vocab_path = cfg.Visual7W.answer_vocab_path if vocab_path is None else vocab_path
    super(Visual7W, self).__init__(vector,
                              image_features_path,
                              answer_vocab_path=answer_vocab_path)

    # load annotation
    with open(questions_path, 'r') as fd: questions_json = json.load(fd)
    with open(decoys_path, 'r') as fd:    decoys_json = json.load(fd)

    # q and a
    cache_filepath = osp.join(cfg.cache_path, "visual7w.{}.pt".format(questions_path.split('/')[-1]))

    if not os.path.exists( cache_filepath ):
      print('extracting answers...')
      self.answers = [ ans for ans in prepare_answers(questions_json) ]
      self.v7w_answers = [ ans for ans in prepare_v7w_answers(questions_json, decoys_json) ]

      print('encoding questions...')
      self.questions = list(prepare_questions(questions_json))
      self.questions  = [self._encode_question(q) for q in self.questions]

      self.answer_indices     = [ [ self.answer_to_index.get(_a, -1) for _a in a ] for a in self.answers ]
      self.v7w_answer_indices = [ [ self.answer_to_index.get(_a, -1) for _a in a ] for a in self.v7w_answers ]
      print('saving cache to: {}'.format(cache_filepath))
      torch.save({'questions':          self.questions,
                  'answer_indices':     self.answer_indices,
                  'v7w_answer_indices': self.v7w_answer_indices
                  }, cache_filepath)
    else:
      print('loading cache from: {}'.format(cache_filepath))
      _cache = torch.load(cache_filepath)
      self.questions = _cache['questions']
      self.answer_indices = _cache['answer_indices']
      self.v7w_answer_indices = _cache['v7w_answer_indices']

    # process images
    self.image_features_path = image_features_path
    self.image_id_to_index = self._create_image_id_to_index()
    self.image_ids = [q['image_index'] for q in questions_json]

    # swtich for determing the data serving [default is False]
    self.serving_v7w = False

  def set_v7w_server(self, value):
    self.serving_v7w = value
    if self.serving_v7w:
      print('Now serving V7W.')
    else:
      print('Now serving Visual7W.')

  def __getitem__(self, item):
    question, question_length = self.questions[item]

    # sample answers
    answer_indices = self.answer_indices[item]

    #TODO: beautify the hack that hard code decoys to be zero
    if self.serving_v7w == True:
      choices  = self.v7w_answer_indices[item]
      counts   = [0, 0, 0, 0, 0, 0, 1]
    else:
      choices  = self.answer_indices[item]
      counts   = [0, 0, 0, 1]

    image_id = self.image_ids[item]
    image = self._load_image(image_id)
    return image, question, answer_indices, counts, choices, None, item, question_length

def prepare_questions(questions_json):
  questions = [q['question'] for q in questions_json]
  for question in questions:
    question = question.lower()[:-1]
    yield nltk.word_tokenize(process_punctuation(question))

#################################################################################
# Note that we processed the data so that correct choice is always the last one,
# this is a hack just for convience. So models that take all answer choices as input
# should be cautious about cheating by learning the bias in order
#################################################################################

def prepare_answers(answers_json):
  answers = [ [ _a.lower().strip('.') for _a in a['multiple_choices'] ] for a in answers_json]
  for answer in answers:
    yield [ process_punctuation(a) for a in answer ]

def prepare_v7w_answers(answers_json, decoys_json):
  answers= []
  for ans, decoy in zip(answers_json,   decoys_json):
    assert ans['qa_id'] == decoy['qa_id'], 'inconsistent qa_id: {}, decoy_id: {}'.format(ans['qa_id'], decoy['qa_id'])
    answers.append(decoy['IoU_decoys'] + decoy['QoU_decoys'] + [ ans['answer'] ])

  answers = [ [ _a.lower().strip('.') for _a in a ] for a in answers]
  for answer in answers:
    yield [ process_punctuation(a) for a in answer ]
