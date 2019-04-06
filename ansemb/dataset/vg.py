import json
import os
import os.path as osp
import nltk

from collections import Counter
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate

from ansemb.config import cfg
from ansemb.dataset.base import VisualQA
from ansemb.dataset.preprocess import process_punctuation, invert_dict

import ansemb.utils as utils
import ansemb.dataset.data_utils as data_utils

from IPython import embed

def path_for(train=False, val=False, test=False):
  assert train + val + test == 1
  if train: split = cfg.VG.train_qa
  elif val: split = cfg.VG.val_qa
  else:     split = cfg.VG.test_qa

  return os.path.join(cfg.VG.qa_path, split)

def create_trainval_loader(vector):
  datasets = [
    VG(path_for(train=True), cfg.VG.feature_path, vector),
    VG(path_for(val=True),   cfg.VG.feature_path, vector)
  ]

  data_loader = torch.utils.data.DataLoader(
    data_utils.Composite(*datasets),
    batch_size=cfg.TRAIN.batch_size,
    shuffle=True,  # only shuffle the data in training
    pin_memory=True,
    num_workers=cfg.TRAIN.data_workers,
    collate_fn=data_utils.collate_fn,
  )
  return data_loader

def get_loader(vector, train=False, val=False, test=False, vocab_path=None):
  assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
  split = VG(
    path_for(train=train, val=val, test=test),
    cfg.VG.feature_path,
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

def invert_dict(d):
  return {v: k for k, v in d.items()}

class VG(VisualQA):
  def __init__(self, questions_path, image_features_path,
               vector, vocab_path=None, answerable_only=False):
    answer_vocab_path = cfg.VG.answer_vocab_path if vocab_path is None else vocab_path
    super(VG, self).__init__(vector,
                             image_features_path,
                             answer_vocab_path=answer_vocab_path)

    # load annotation
    with open(questions_path, 'r') as fd: questions_json = json.load(fd)

    # q and a
    cache_filepath = osp.join(cfg.cache_path, "{}.pt".format(questions_path.split('/')[-1]))

    if not os.path.exists( cache_filepath ):
      print('extracting answers...')
      self.answers  = list(prepare_answers(questions_json))
      self.choices  = list(prepare_choices(questions_json))

      print('encoding questions...')
      self.questions = list(prepare_questions(questions_json))
      self.questions  = [self._encode_question(q) for q in self.questions]

      self.answer_indices = [ [ self.answer_to_index.get(_a, -1) for _a in a ] for a in self.answers ]
      self.choice_indices = [ [ self.answer_to_index.get(_a, -1) for _a in a ] for a in self.choices ]
      self.answer_vectors = torch.cat( [self._encode_answer_vector(answer)
                  for answer, index in  self.answer_to_index.items() ], dim=0).float()
      print('saving cache to: {}'.format(cache_filepath))
      torch.save({'questions': self.questions, 'answer_indices': self.answer_indices,
                  'choice_indices': self.choice_indices, 'answer_vectors': self.answer_vectors}, cache_filepath)
    else:
      print('loading cache from: {}'.format(cache_filepath))
      _cache = torch.load(cache_filepath)
      self.questions  = _cache['questions']
      self.answer_indices = _cache['answer_indices']
      self.choice_indices = _cache['choice_indices']
      self.answer_vectors = _cache['answer_vectors']

    # process images
    self.image_features_path = image_features_path
    self.image_id_to_index = self._create_image_id_to_index()
    self.image_ids = [q['image_id'] for q in questions_json]

  def __getitem__(self, item):
    question, question_length = self.questions[item]

    # sample answers
    answer_indices = self.answer_indices[item]
    counts         = [1]

    choices = self.choice_indices[item]
    image_id = self.image_ids[item]
    image = self._load_image(image_id)
    return image, question, answer_indices, counts, choices, None, item, question_length

  def evaluate(self, predictions):
    raise NotImplementedError

def prepare_questions(questions_json):
  questions = [q['question'] for q in questions_json]
  for question in questions:
    question = question.lower()[:-1]
    yield nltk.word_tokenize(process_punctuation(question))

def prepare_answers(answers_json):
  answers = [a['answer'] for a in answers_json]
  for answer in answers:
    yield [ process_punctuation(answer.lower().strip('.')) ]

#################################################################################
# Note that we processed the data so that correct choice is always the last one,
# this is a hack just for convience. So models that take all answer choices as input
# should be cautious about cheating by learning the bias in order
#################################################################################

def prepare_choices(answers_json):
  answers = [ a['IoU_decoys'] + a['QoU_decoys'] + [ a['answer'] ] for a in answers_json]
  for answer in answers:
    yield [ process_punctuation(a.lower().strip('.')) for a in answer ]
