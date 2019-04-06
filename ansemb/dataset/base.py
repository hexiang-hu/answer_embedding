import json
import os
import os.path as osp
import random
import nltk
import h5py

from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from ansemb.config import cfg
from ansemb.dataset.preprocess import invert_dict

class VisualQA(data.Dataset):
  def __init__(self,
               vector,
               image_features_path,
               answer_vocab_path=None):
    super(VisualQA, self).__init__()

    # vocab
    self.vector = vector

    # process question
    with open(cfg.question_vocab_path, 'r') as fd: question_vocab = json.load(fd)
    self.token_to_index = question_vocab['question']
    self._max_question_length = question_vocab['max_question_length']

    self.index_to_token  = invert_dict(self.token_to_index)

    if answer_vocab_path is not None:
      print('import answer vocabulary from: {}'.format(answer_vocab_path))
      # process answer
      with open(answer_vocab_path, 'r') as fd: answer_vocab = json.load(fd)
      self.answer_to_index = answer_vocab['answer']
      self.index_to_answer = invert_dict(self.answer_to_index)

    self.cached_answers = {}
    self.unk_vector = self.vector['UNK']

  @property
  def max_question_length(self):
    return self._max_question_length

  @property
  def max_answer_length(self):
    assert hasattr(self, answers), 'Dataloader must have access to answers'
    if not hasattr(self, '_max_answer_length'):
      self._max_answer_length = max(map(len, self.answers))
    return self._max_answer_length

  @property
  def num_tokens(self):
    return len(self.token_to_index)

  @property
  def num_answers(self):
    return len(self.answer_to_index)

  def __len__(self):
    return len(self.questions)

  ### Internal data utility---------------------------------------
  def _load_image(self, image_id):
    """ Load an image """
    if not hasattr(self, 'features_file'):
      # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
      # forks for multiple works, every child would use the same file object and fail
      # Having multiple readers using different file objects is fine though, so we just init in here.
      self.features_file = h5py.File(self.image_features_path, 'r')
    index = self.image_id_to_index[image_id]
    dataset = self.features_file['features']
    img = dataset[index].astype('float32')
    return torch.from_numpy(img)

  def _get_answer_vectors(self, answer_indices):
    if isinstance(answer_indices[0], list):
      N, C = len(answer_indices), len(answer_indices[0])
      vector = torch.zeros(N, C, self.vector.dim)
      for i, answer_ids in enumerate(answer_indices):
        for j, answer_id in enumerate(answer_ids):
          if answer_id != -1:
            vector[i, j, :] = self._encode_answer_vector(self.index_to_answer[answer_id])
          else:
            vector[i, j, :] = self.unk_vector
    else:
      vector = torch.zeros(len(answer_indices), self.vector.dim)
      for idx, answer_id in enumerate(answer_indices):
        if answer_id != -1:
          vector[idx, :] = self._encode_answer_vector(self.index_to_answer[answer_id])
        else:
          vector[idx, :] = self.unk_vector
    return vector, []

  def _get_answer_sequences(self, answer_indices):
    seqs, lengths = [], []
    max_seq_length = 0
    if isinstance(answer_indices[0], list):
      N, C = len(answer_indices), len(answer_indices[0])
      for i, answer_ids in enumerate(answer_indices):
        _seqs = []
        for j, answer_id in enumerate(answer_ids):
          if answer_id != -1:
            _seqs.append( self._encode_answer_sequence(self.index_to_answer[answer_id]) )
          else:
            _seqs.append([ self.unk_vector ])
          if max_seq_length < len(_seqs[-1]): max_seq_length = len(_seqs[-1]) # determing max length
        seqs.append(_seqs)

      vector = torch.zeros(N, C, max_seq_length, self.vector.dim)
      for i, _seqs in enumerate(seqs):
        for j, seq in enumerate(_seqs):
          if len(seq) != 0: vector[i, j, :len(seq), :] = torch.cat(seq, dim=0)
          lengths.append(len(seq))
      assert len(lengths) == N*C, 'Wrong lengths - length: {} vs N: {}, C: {} vs seqs: {}'.format(len(lengths), N, C, len(seqs))
    else:
      for idx, answer_id in enumerate(answer_indices):
        if answer_id != -1:
          seqs.append( self._encode_answer_sequence(self.index_to_answer[answer_id]) )
        else:
          seqs.append([ self.unk_vector ])

        if max_seq_length < len(seqs[-1]): max_seq_length = len(seqs[-1]) # determing max length

      vector = torch.zeros(len(answer_indices), max_seq_length, self.vector.dim)
      for idx, seq in enumerate(seqs):
        if len(seq) != 0: vector[idx, :len(seq), :] = torch.cat(seq, dim=0)
        lengths.append(len(seq))

    return vector, lengths

  def _create_image_id_to_index(self):
    """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
    with h5py.File(self.image_features_path, 'r') as features_file:
      image_ids = features_file['ids'][()]
    image_id_to_index = {id: i for i, id in enumerate(image_ids)}
    return image_id_to_index

  def _encode_question(self, question):
    """ Turn a question into a vector of indices and a question length """
    vec = torch.zeros(self.max_question_length).long()
    for i, token in enumerate(question):
      index = self.token_to_index.get(token, 0)
      vec[i] = index
    return vec, len(question)

  def _encode_answer_vector(self, answer):
    if isinstance(self.cached_answers.get(answer, -1), int):
      tokens = nltk.word_tokenize(answer)
      answer_vec = torch.zeros(1, self.vector.dim)
      cnt = 0
      for i, token in enumerate(tokens):
        if self.vector.check(token):
          answer_vec += self.vector[token]
          cnt += 1
      self.cached_answers[answer] = answer_vec / (cnt + 1e-12)

    return self.cached_answers[answer]

  def _encode_answer_sequence(self, answer):
    if isinstance(self.cached_answers.get(answer, -1), int):
      tokens = nltk.word_tokenize(answer)
      answer_seq = []
      for i, token in enumerate(tokens):
        if self.vector.check(token):
          answer_seq.append(self.vector[token].view(1, self.vector.dim))
        else:
          answer_seq.append(self.vector['<unk>'].view(1, self.vector.dim))
      self.cached_answers[answer] = answer_seq

    return self.cached_answers[answer]

  def _encode_multihot_labels(self, answers, max_answer_index=cfg.TEST.max_answer_index):
    """ Turn an answer into a vector """
    tail_index = max_answer_index
    answer_vec = torch.zeros(max_answer_index)
    for answer in answers:
      index = self.answer_to_index.get(answer)
      if index is not None:
        if index < max_answer_index:
          answer_vec[index] += 1
    return answer_vec

  def evaluate(self, predictions):
    raise NotImplementedError
