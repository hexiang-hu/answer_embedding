import json
import os
import os.path as osp
import re
import nltk
import random

from collections import Counter
from ansemb.config import cfg
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

class Composite(data.Dataset):
  """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
  def __init__(self, *datasets):
    self.datasets = datasets

  def __getitem__(self, item):
    current = self.datasets[0]
    for d in self.datasets:
      if item < len(d):
        return d[item]
      item -= len(d)
    else:
      raise IndexError('Index too large for composite dataset')

  def __len__(self):
    return sum(map(len, self.datasets))

  def _get_answer_vectors(self, answer_indices):
    return self.datasets[0]._get_answer_vectors(answer_indices)

  def _get_answer_sequences(self, answer_indices):
    return self.datasets[0]._get_answer_sequences(answer_indices)

  @property
  def vector(self):
    return self.datasets[0].vector

  @property
  def token_to_index(self):
    return self.datasets[0].token_to_index

  @property
  def answer_to_index(self):
    return self.datasets[0].answer_to_index

  @property
  def index_to_answer(self):
    return self.datasets[0].index_to_answer

  @property
  def num_tokens(self):
    return self.datasets[0].num_tokens

  @property
  def num_answer_tokens(self):
    return self.datasets[0].num_answer_tokens

  @property
  def vocab(self):
    return self.datasets[0].vocab

def generate_batch_answer(indices, counts):
  unique_answers = set( aid for aids in indices for aid in aids )
  negative_answers = random.sample( set(range(cfg.TRAIN.max_negative_answer)) - unique_answers,
                                    max(cfg.TRAIN.answer_batch_size - len(unique_answers), 0))
  unique_answers = list(unique_answers) + negative_answers
  # unique_answers = list(set( aid for aids in indices for aid in aids ))
  answer_dict   = { k: i for i, k in enumerate(unique_answers)}
  answer_vector = torch.zeros(len(indices), len(unique_answers))
  for i in range(len(counts)):
    for j, c in zip(indices[i], counts[i]):answer_vector[i, answer_dict[j]] = c

  return unique_answers, answer_vector

def collate_fn(batch):
  # put question lengths in descending order so that we can use packed sequences later
  _images, _questions, _answer_indices, _answer_counts, _choices, _labels, _indices, _question_lengths = zip(*batch)

  # universal contents
  images    = default_collate(_images)
  questions = default_collate(_questions)
  indices        = default_collate(_indices)
  question_lengths = default_collate(_question_lengths)

  if ( _answer_indices[0] == None ) and ( _answer_counts[0] == None ):
    return images, questions, indices, question_lengths

  # flatten nested list
  _unique_answers, _answer_vectors = generate_batch_answer(_answer_indices, _answer_counts)

  unique_answers = default_collate(_unique_answers)
  answer_vectors = default_collate(_answer_vectors)

  if _choices[0] is not None:
    return images, questions, unique_answers, answer_vectors, _choices, indices, question_lengths
  elif _labels[0] is not None:
    answer_labels  = default_collate(_labels)
    return images, questions, unique_answers, answer_vectors, answer_labels, indices, question_lengths
  else:
    raise NotImplementedError('Something is wrong with dataloader')

def eval_collate_fn(batch):
  # put question lengths in descending order so that we can use packed sequences later
  batch.sort(key=lambda x: x[-1], reverse=True)
  return data.dataloader.default_collate(batch)
