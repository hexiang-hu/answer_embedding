import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ansemb.utils import cosine_sim
from IPython import embed

class ContrastiveLoss(nn.Module):
  def __init__(self, margin=0.2, measure=False, dual=False, beta=10):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    if measure == 'cosine':
      self.sim = cosine_sim
    else:
      raise ValueError('Unknown similarity.[{}]'.format(measure))

    self.beta = beta

  def forward(self, scores, match, weight=None):
    N, M = scores.size()

    match_byte = ( match > 0 ).byte()
    pos_iq = torch.zeros(N, M)
    pos_iq = scores[match_byte].view(N, 1).expand(N, M)

    margin_weight = 1
    if weight is not None:
      _, _match = torch.max(match, 1)
      _weight = F.normalize(weight, p=2, dim=1)
      margin_weight = (1 - ((1 + cosine_sim(_weight, _weight)[_match]) / 2)**self.beta).clamp(min=0, max=1)

    cost_iq = (self.margin*margin_weight + scores - pos_iq).clamp(min=0)
    cost_iq = cost_iq.masked_fill_(match_byte, 0)

    return cost_iq.sum()
