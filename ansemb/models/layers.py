import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from IPython import embed

class MLP(nn.Sequential):
  def __init__(self, in_features, mid_features, out_features, drop=0.0, groups=1):
    super(MLP, self).__init__()
    self.add_module('drop1', nn.Dropout(drop))
    self.add_module('lin1', nn.Linear(in_features, mid_features))
    self.add_module('relu', nn.LeakyReLU())
    self.add_module('drop2', nn.Dropout(drop))
    self.add_module('lin2', nn.Linear(mid_features, out_features))

class MaxoutRNN(nn.Module):
  def __init__(self, input_features, rnn_features, num_layers=1, drop=0.0,
               rnn_type='LSTM', rnn_bidirectional=False):
    super(MaxoutRNN, self).__init__()
    self.bidirectional = rnn_bidirectional

    if rnn_type == 'LSTM':
      self.rnn = nn.LSTM(input_size=input_features,
              hidden_size=rnn_features, dropout=drop,
              num_layers=num_layers, batch_first=True,
              bidirectional=rnn_bidirectional)
    elif rnn_type == 'GRU':
      self.rnn = nn.GRU(input_size=input_features,
              hidden_size=rnn_features, dropout=drop,
              num_layers=num_layers, batch_first=True,
              bidirectional=rnn_bidirectional)
    else:
      raise ValueError('Unsupported RNN type')

    self.features = rnn_features

    self._init_rnn(self.rnn.weight_ih_l0)
    self._init_rnn(self.rnn.weight_hh_l0)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

  def _init_rnn(self, weight):
    for w in weight.chunk(3, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len, hidden=None):
    lengths = torch.LongTensor(q_len)
    lens, indices = torch.sort(lengths, 0, True)

    packed_batch = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if hidden is not None:
      N_, H_ = hidden.size()
      hs, _ = self.rnn(packed_batch, hidden[indices.cuda()].view(1, N_, H_))
    else:
      hs, _ = self.rnn(packed_batch)
    outputs, _ = pad_packed_sequence(hs, batch_first=True, padding_value=np.float('-inf'))

    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]
    N, L, H = outputs.size()
    return F.max_pool1d(outputs.transpose(1, 2), L).squeeze().view(N, H)

class Seq2SeqRNN(nn.Module):
  def __init__(self, input_features, rnn_features, num_layers=1, drop=0.0,
               rnn_type='LSTM', rnn_bidirectional=False):
    super(Seq2SeqRNN, self).__init__()
    self.bidirectional = rnn_bidirectional

    if rnn_type == 'LSTM':
      self.rnn = nn.LSTM(input_size=input_features,
                hidden_size=rnn_features, dropout=drop,
                num_layers=num_layers, batch_first=True,
                bidirectional=rnn_bidirectional)
    elif rnn_type == 'GRU':
      self.rnn = nn.GRU(input_size=input_features,
                hidden_size=rnn_features, dropout=drop,
                num_layers=num_layers, batch_first=True,
                bidirectional=rnn_bidirectional)
    else:
      raise ValueError('Unsupported Type')

    self.init_weight(rnn_bidirectional, rnn_type)

  def init_weight(self, bidirectional, rnn_type):
    self._init_rnn(self.rnn.weight_ih_l0, rnn_type)
    self._init_rnn(self.rnn.weight_hh_l0, rnn_type)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

    if bidirectional:
      self._init_rnn(self.rnn.weight_ih_l0_reverse, rnn_type)
      self._init_rnn(self.rnn.weight_hh_l0_reverse, rnn_type)
      self.rnn.bias_ih_l0_reverse.data.zero_()
      self.rnn.bias_hh_l0_reverse.data.zero_()

  def _init_rnn(self, weight, rnn_type):
    chunk_size = 4 if rnn_type == 'LSTM' else 3
    for w in weight.chunk(chunk_size, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len):
    lengths = torch.LongTensor(q_len)
    lens, indices = torch.sort(lengths, 0, True)

    packed = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if isinstance(self.rnn, nn.LSTM):
      _, ( outputs, _ ) = self.rnn(packed)
    elif isinstance(self.rnn, nn.GRU):
      _, outputs = self.rnn(packed)

    if self.bidirectional:
      outputs = torch.cat([ outputs[0, :, :], outputs[1, :, :] ], dim=1)
    else:
      outputs = outputs.squeeze(0)

    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]

    return outputs

class SelfAttention(nn.Module):
  def __init__(self, v_features, mid_features, glimpses, drop=0.0):
    super(SelfAttention, self).__init__()
    self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
    self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

    self.drop = nn.Dropout(drop)
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, v):
    v = self.v_conv(self.drop(v))
    x = self.relu(v)
    x = self.x_conv(self.drop(x))
    return x

class Attention(nn.Module):
  def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
    super(Attention, self).__init__()
    self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
    self.q_lin = nn.Linear(q_features, mid_features)
    self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

    self.drop = nn.Dropout(drop)
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, v, q):
    v = self.v_conv(self.drop(v))
    q = self.q_lin(self.drop(q))
    q = tile_2d_over_nd(q, v)
    x = self.relu(v + q)
    x = self.x_conv(self.drop(x))
    return x

def apply_attention(input, attention):
  """ Apply any number of attention maps over the input.
    The attention map has to have the same size in all dimensions except dim=1.
  """
  n, c = input.size()[:2]
  glimpses = attention.size(1)

  # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
  input = input.view(n, c, -1)
  attention = attention.view(n, glimpses, -1)
  s = input.size(2)

  # apply a softmax to each attention map separately
  # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
  # so that each glimpse is normalized separately
  attention = attention.view(n * glimpses, -1)
  attention = F.softmax(attention)

  # apply the weighting by creating a new dim to tile both tensors over
  target_size = [n, glimpses, c, s]
  input = input.view(n, 1, c, s).expand(*target_size)
  attention = attention.view(n, glimpses, 1, s).expand(*target_size)
  weighted = input * attention
  # sum over only the spatial dimension
  weighted_mean = weighted.sum(dim=3)
  # the shape at this point is (n, glimpses, c, 1)
  return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
  """ Repeat the same feature vector over all spatial positions of a given feature map.
    The feature vector should have the same batch size and number of features as the feature map.
  """
  n, c = feature_vector.size()
  spatial_size = feature_map.dim() - 2
  tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
  return tiled

class BagOfWordsProcessor(nn.Module):
  def __init__(self, embedding_tokens, embedding_features, rnn_features,
             embedding_weights, embedding_requires_grad, drop=0.0):
    super(BagOfWordsProcessor, self).__init__()
    self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)

    self.embedding.weight.data = embedding_weights
    self.embedding.weight.requires_grad = embedding_requires_grad

  def forward(self, q, q_len):
    embedded = self.embedding(q)
    q_len = Variable(torch.Tensor(q_len).view(-1, 1) + 1e-12, requires_grad=False).cuda(async=True)

    return torch.div( torch.sum(embedded, 1), q_len )

class GroupMLP(nn.Module):
  def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
    super(GroupMLP, self).__init__()

    self.conv1 = nn.Conv1d(in_features, mid_features, 1)
    self.drop  = nn.Dropout(p=drop)
    self.relu  = nn.LeakyReLU()
    self.conv2 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

  def forward(self, a):
    N, C = a.size()
    h = self.relu(self.conv1(a.view(N, C, 1)))
    return self.conv2(self.drop(h)).view(N, -1)

