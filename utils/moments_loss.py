import numpy as np
import scipy
import torch
import sys
import utils.freq_fourier_loss
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("..")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class CMD_loss(nn.Module):
    def __init__(self, target, axis=(0, 2, 3), k=5, weights=(1, 1, 1, 1, 1)):
        super(CMD_loss, self).__init__()
        self.target = target
        self.k = k
        self.axis = axis
        self.weights = weights
        self.rmse = RMSELoss()
        self.c1_y, self.m_ys = self._init_targets()

        assert k == len(self.weights)

    def _init_targets(self):
        c1_y = self.target.mean(dim=self.axis).view(1, -1, 1, 1)
        m_ys = list()
        for i in range(2, self.k + 1):
            # watch out: zeroth element is pow 2, first is pow 3...
            m_ys.append((self.target - c1_y).pow(i).mean(dim=self.axis))
        return c1_y, m_ys

    def __call__(self, x):
        c1_x = x.mean(dim=self.axis).view(1, -1, 1, 1)
        loss = self.weights[0] * self.rmse(c1_x, self.c1_y)
        for i in range(2, self.k + 1):
            m_x = (x - c1_x).pow(i).mean(dim=self.axis)
            loss = loss + self.weights[i - 1] * self.rmse(m_x, self.m_ys[i - 2])
        return loss


