import os
import pdb
import math
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Pass(nn.Module):
    def __init__(self, opt):
        super(Pass, self).__init__()
        self.opt = opt

    def forward(self, feat):

        return feat


class FC(nn.Module):
    def __init__(self, opt):
        super(FC, self).__init__()
        self.opt = opt
        self.fc = nn.Sequential(
        	nn.Dropout(p=opt.dropout),
            nn.Linear(2048, 2*21)
        )

    def forward(self, feat):
        feat = self.fc(feat)
        feat = feat.view(feat.size(0), 21, 2)

        return feat
