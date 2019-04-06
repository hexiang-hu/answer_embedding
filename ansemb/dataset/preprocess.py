import json
import os
import os.path as osp
import re
import nltk
import random
import itertools
from collections import Counter

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import ansemb.utils as utils

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

def process_punctuation(s):
  # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
  # this version should be faster since we use re instead of repeated operations on str's
  original_s = s
  if _punctuation.search(s) is None:
      return s
  s = _punctuation_with_a_space.sub('', s)
  if re.search(_comma_strip, s) is not None:
      s = s.replace(',', '')
  s = _punctuation.sub(' ', s)
  s = _period_strip.sub('', s)
  if s.strip() == '': return original_s.strip()
  else:               return s.strip()

def extract_vocab(iterable, top_k=None, start=0, input_vocab=None):
  """ Turns an iterable of list of tokens into a vocabulary.
      These tokens could be single answers or word tokens in questions.
  """
  all_tokens = itertools.chain.from_iterable(iterable)
  counter = Counter(all_tokens)
  if top_k:
    most_common = counter.most_common(top_k)
    most_common = (t for t, c in most_common)
  else:
    most_common = counter.keys()
  # descending in count, then lexicographical order
  tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)

  vocab = {t: i for i, t in enumerate(tokens, start=start)}
  return vocab

class CocoImages(data.Dataset):
  def __init__(self, path, transform=None):
    super(CocoImages, self).__init__()
    self.path = path
    self.id_to_filename = self._find_images()
    self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
    print('found {} images in {}'.format(len(self), self.path))
    self.transform = transform

  def _find_images(self):
    id_to_filename = {}
    for filename in os.listdir(self.path):
      if not filename.endswith('.jpg'): continue
      id_and_extension = filename.split('_')[-1]
      id = int(id_and_extension.split('.')[0])
      id_to_filename[id] = filename
    return id_to_filename

  def __getitem__(self, item):
    id = self.sorted_ids[item]
    path = os.path.join(self.path, self.id_to_filename[id])
    img = Image.open(path).convert('RGB')

    if self.transform is not None: img = self.transform(img)
    return id, img

  def __len__(self):
    return len(self.sorted_ids)

class VGImages(data.Dataset):
  def __init__(self, path, transform=None):
    super(VGImages, self).__init__()
    self.path = path
    self.id_to_filename = self._find_images()
    self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
    print('found {} images in {}'.format(len(self), self.path))
    self.transform = transform

  def _find_images(self):
    id_to_filename = {}
    for filename in os.listdir(self.path):
      if not filename.endswith('.jpg'):
        continue
      id = int(filename.split('.')[0])
      id_to_filename[id] = filename
    return id_to_filename

  def __getitem__(self, item):
    id = self.sorted_ids[item]
    path = os.path.join(self.path, self.id_to_filename[id])
    img = Image.open(path).convert('RGB')

    if self.transform is not None:
      img = self.transform(img)
    return id, img

  def __len__(self):
    return len(self.sorted_ids)

def invert_dict(d): return {v: k for k, v in d.items()}

class VizwizImages(data.Dataset):
  def __init__(self, path, transform=None):
    super(VizwizImages, self).__init__()
    self.path = path
    self.id_to_filename = self._find_images()
    self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
    print('found {} images in {}'.format(len(self), self.path))
    self.transform = transform

  def _find_images(self):
    id_to_filename = {}
    for filename in os.listdir(self.path):
      if not filename.endswith('.jpg'): continue
      id_and_extension = filename.split('_')[-1]
      id = int(id_and_extension.split('.')[0])
      id_to_filename[id] = filename
    return id_to_filename

  def __getitem__(self, item):
    id = self.sorted_ids[item]
    path = os.path.join(self.path, self.id_to_filename[id])
    img = Image.open(path).convert('RGB')

    if self.transform is not None: img = self.transform(img)
    return id, img

  def __len__(self):
    return len(self.sorted_ids)
