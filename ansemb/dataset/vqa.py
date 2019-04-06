import json
import os
import os.path as osp
import nltk
import random

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

def path_for(train=False, val=False, test=False, question=False, answer=False):
  assert train + val + test == 1
  assert question + answer == 1
  assert not (test and answer), 'loading answers from test split not supported'
  if train: split = cfg.VQA2.train_qa
  elif val: split = cfg.VQA2.val_qa
  else:     split = cfg.VQA2.test_qa
  if question: fmt = 'v2_{0}_{1}_{2}_questions.json'
  else: fmt = 'v2_{1}_{2}_annotations.json'
  s = fmt.format(cfg.VQA2.task, cfg.VQA2.dataset, split)
  return os.path.join(cfg.VQA2.qa_path, s)


def get_loader(vector, train=False, val=False, test=False, vocab_path=None, _batch_size=None):
  """ Returns a data loader for the desired split """
  assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
  batch_size = _batch_size if _batch_size is not None else cfg.TRAIN.batch_size
  if test == True:
    split = VQAEval( path_for(test=test, question=True),
                     cfg.VQA2.feature_path,
                     vector,
                     vocab_path=vocab_path)
    loader = torch.utils.data.DataLoader(
      split,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=cfg.TRAIN.data_workers,
      collate_fn=data_utils.collate_fn,
    )
  else:
    split = VQA(
      path_for(train=train, val=val, question=True),
      path_for(train=train, val=val, answer=True),
      cfg.VQA2.feature_path,
      vector,
      vocab_path=vocab_path,
    )
    loader = torch.utils.data.DataLoader(
      split,
      batch_size=batch_size,
      shuffle=train,  # only shuffle the data in training
      pin_memory=True,
      num_workers=cfg.TRAIN.data_workers,
      collate_fn=data_utils.collate_fn,
    )
  return loader

def create_trainval_loader(vector):
  datasets = [
    VQA(path_for(train=True, question=True), path_for(train=True, answer=True), cfg.VQA2.feature_path, vector),
    VQA(path_for(val=True, question=True),   path_for(val=True, answer=True),   cfg.VQA2.feature_path, vector)
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

class VQA(VisualQA):
  """ VQA dataset, open-ended """
  def __init__(self, questions_path, answers_path, image_features_path, vector,
                     vocab_path=None):
    answer_vocab_path = cfg.VQA2.answer_vocab_path if vocab_path is None else vocab_path
    super(VQA, self).__init__(vector, image_features_path,
                              answer_vocab_path=answer_vocab_path)

    # load annotation
    with open(questions_path, 'r') as fd: questions_json = json.load(fd)
    with open(answers_path,   'r') as fd:   answers_json = json.load(fd)

    # q and a
    cache_filepath = osp.join(cfg.cache_path, "{}.{}.pt".format(questions_path.split('/')[-1],
                  answers_path.split('/')[-1]))

    print('extracting answers...')
    self.answers  = list(prepare_answers(answers_json['annotations']))

    if not os.path.exists( cache_filepath ):
      print('encoding questions...')
      self.questions = list(prepare_questions(questions_json['questions']))
      self.questions = [self._encode_question(q) for q in self.questions]

      self.answer_indices = [ [ self.answer_to_index.get(_a, -1) for _a in a ] for a in self.answers ]
      self.answer_vectors = torch.cat( [self._encode_answer_vector(answer)
                                for answer, index in  self.answer_to_index.items() ], dim=0).float()
      print('saving cache to: {}'.format(cache_filepath))
      torch.save({'questions': self.questions, 'answer_indices': self.answer_indices,
                  'answer_vectors': self.answer_vectors}, cache_filepath)
    else:
      print('loading cache from: {}'.format(cache_filepath))
      _cache = torch.load(cache_filepath)
      self.questions  = _cache['questions']
      self.answer_indices = _cache['answer_indices']
      self.answer_vectors = _cache['answer_vectors']

    # process images
    self.image_features_path = image_features_path
    self.image_id_to_index = self._create_image_id_to_index()
    self.image_ids = [q['image_id'] for q in questions_json['questions']]

    self.vqa_minus  = torch.Tensor([ (anno['answer_type'] != 'yes/no') for anno in answers_json['annotations'] ]).byte()

  def __getitem__(self, item):
    question, question_length = self.questions[item]

    # sample answers
    answer_cands = Counter(self.answer_indices[item])
    answer_indices = list(answer_cands.keys())
    counts = list(answer_cands.values())

    label = self._encode_multihot_labels(self.answers[item])
    image_id = self.image_ids[item]
    image = self._load_image(image_id)
    return image, question, answer_indices, counts, None, label, item, question_length

def prepare_questions(questions_json):
  """ Tokenize and normalize questions from a given question json in the usual VQA format. """
  questions = [q['question'] for q in questions_json]
  for question in questions:
    question = question.lower()[:-1]
    yield nltk.word_tokenize(process_punctuation(question))

def prepare_answers(answers_json):
  """ Normalize answers from a given answer json in the usual VQA format. """
  answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json]
  for answer_list in answers:
    ret = list(map(process_punctuation, answer_list))
    yield ret

def prepare_multiple_choice_answer(answers_json):
  """ Normalize answers from a given answer json in the usual VQA format. """
  multiple_choice_answers = [ ans_dict['multiple_choice_answer'] for ans_dict in answers_json]
  for answer in multiple_choice_answers:
    yield [ process_punctuation(answer) ]

class VQAEval(VisualQA):
  """ VQA dataset, open-ended """
  def __init__(self, questions_path, image_features_path, vector, vocab_path=None):
    answer_vocab_path = cfg.VQA2.answer_vocab_path if vocab_path is None else vocab_path
    super(VQAEval, self).__init__(vector,
                                  image_features_path,
                                  answer_vocab_path=answer_vocab_path)

    with open(questions_path, 'r') as fd: questions_json = json.load(fd)

    # q and a
    cache_filepath = osp.join(cfg.cache_path, "{}.pt".format(questions_path.split('/')[-1]))

    if not os.path.exists( cache_filepath ):
      print('encoding questions...')
      self.questions = list(prepare_questions(questions_json['questions']))
      self.questions  = [self._encode_question(q) for q in self.questions]

      print('saving cache to: {}'.format(cache_filepath))
      torch.save({'questions': self.questions}, cache_filepath)
    else:
      print('loading cache from: {}'.format(cache_filepath))
      _cache = torch.load(cache_filepath)
      self.questions  = _cache['questions']

    # v
    self.image_features_path = image_features_path
    self.image_id_to_index = self._create_image_id_to_index()
    self.image_ids = [q['image_id'] for q in questions_json['questions']]

  def __getitem__(self, item):
    question, question_length = self.questions[item]

    image_id = self.image_ids[item]
    image = self._load_image(image_id)
    return image, question, None, None, None, None, item, question_length

