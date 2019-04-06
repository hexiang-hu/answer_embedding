import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

import time
from ansemb.models.layers import *
from IPython import embed

class StackedAttentionEmbedding(nn.Module):
  def __init__(self, embedding_tokens, embedding_weights=None,
               output_features=2048, embedding_size=1024,
               rnn_bidirectional=True, embedding_requires_grad=True):
    super(StackedAttentionEmbedding, self).__init__()
    question_features = 1024
    rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)
    vision_features = output_features
    glimpses = 2

    vocab_size = embedding_weights.size(0)
    vector_dim = embedding_weights.size(1)
    self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)

    self.drop = nn.Dropout(0.5)
    self.text = Seq2SeqRNN(
      input_features=vector_dim,
      rnn_features=int(rnn_features),
      rnn_type='LSTM',
      rnn_bidirectional=rnn_bidirectional,
    )
    self.attention = Attention(
      v_features=vision_features,
      q_features=question_features,
      mid_features=512,
      glimpses=2,
      drop=0.5,
    )
    self.mlp = GroupMLP(
      in_features=glimpses * vision_features + question_features,
      mid_features=4096,
      out_features=embedding_size,
      drop=0.5,
      groups=64,
    )

    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        if m.bias is not None:
          m.bias.data.zero_()

    self.embedding.weight.data = embedding_weights
    self.embedding.weight.requires_grad = embedding_requires_grad

  def forward(self, v, q, q_len):
    q = self.text(self.drop(self.embedding(q)), list(q_len.data))
    # q = self.text(self.embedding(q), list(q_len.data))

    v = F.normalize(v, p=2, dim=1)
    a = self.attention(v, q)
    v = apply_attention(v, a)

    combined = torch.cat([v, q], dim=1)
    embedding = self.mlp(combined)
    return embedding

class VisualSemanticEmbedding(nn.Module):
  def __init__(self, embedding_tokens, embedding_weights=None,
                     output_features=2048, embedding_size=1024,
                     rnn_bidirectional=True, embedding_requires_grad=True):
    super(VisualSemanticEmbedding, self).__init__()
    question_features = 300
    rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)
    vision_features = output_features
    glimpses = 2

    # self.text = BagOfWordsMLPProcessor(
    self.text = BagOfWordsProcessor(
      embedding_tokens=embedding_weights.size(0) if embedding_weights is not None else embedding_tokens,
      embedding_weights=embedding_weights,
      embedding_features=300,
      embedding_requires_grad=True,
      rnn_features=rnn_features,
      drop=0.5,
    )
    self.mlp = GroupMLP(
      in_features= vision_features + question_features,
      mid_features=4096,
      out_features=embedding_size,
      drop=0.5,
      groups=64,
    )

    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        if m.bias is not None:
          m.bias.data.zero_()

  def forward(self, v, q, q_len):
    q = F.normalize(self.text(q, list(q_len.data)), p=2, dim=1)
    v = F.normalize(F.avg_pool2d(v, (v.size(2), v.size(3))).squeeze(), p=2, dim=1)

    combined = torch.cat([v, q], dim=1)
    embedding = self.mlp(combined)
    return embedding

class MLPEmbedding(nn.Module):
  def __init__(self, embedding_features,
               embedding_weights=None,
               embedding_size=1024):
    super(MLPEmbedding, self).__init__()

    self.mlp = GroupMLP(
      in_features=embedding_features,
      mid_features=4096,
      out_features=embedding_size,
      drop=0.5,
      groups=64,
    )

    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        if m.bias is not None:
          m.bias.data.zero_()

  def forward(self, a, a_len=None):
    return self.mlp(F.normalize(a, p=2))

class RNNEmbedding(nn.Module):
  def __init__(self, embedding_features,
               embedding_weights=None,
               rnn_bidirectional=True,
               embedding_size=1024):
    super(RNNEmbedding, self).__init__()

    rnn_features = int(embedding_size // 2) if rnn_bidirectional else int(embedding_size)
    self.text = MaxoutRNN(
      input_features=embedding_features,
      rnn_features=int(rnn_features),
      rnn_type='GRU',
      num_layers=2,
      rnn_bidirectional=rnn_bidirectional,
      drop=0.5,
    )

  def forward(self, a, a_len):
    return self.text(a, a_len)
    # return self.text(F.normalize(a, p=2), a_len)
