from enum import Enum
import torch.nn as nn
import torch

bce_loss = nn.BCELoss(reduction='mean')
bce_sum_loss = nn.BCELoss(reduction='sum')
l1_loss = nn.L1Loss(reduction='mean')
l2_loss = nn.MSELoss(reduction='mean')
kl_loss = nn.KLDivLoss(reduction='batchmean')
l2_sum_loss = nn.MSELoss(reduction='sum')
l2_non_loss = nn.MSELoss(reduction='none')

def kl_loss_function(p, q):
    """
    :param p: model output parameters of N binary R.V.
    :param q: True distribution
    :return:
    """
    p1 = p + 1e-6
    p2 = 1 - p + 1e-6
    p1 = torch.log(p1)
    p2 = torch.log(p2)
    q2 = 1.0 - q
    return kl_loss(p1, q) + kl_loss(p2, q2)


def js_loss_function(p, q):
    return 0.5 * kl_loss_function(p, q) + 0.5 * kl_loss_function(q, p)

class DataType(Enum):
    uniform = 1
    biased = 2
    both = 3

class DatasetName(Enum):
    coat = 1
    yahooR3 = 2


