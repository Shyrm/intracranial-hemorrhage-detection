import torch
from torch import nn


class LogLoss:

    def __init__(self, weights):

        self.loss = nn.BCELoss(reduction='none')
        self.weights = weights

    def __call__(self, input, target, **kwargs):

        nrl = self.loss(input, target)
        nrl = (nrl * self.weights.unsqueeze(0)).sum(1) / self.weights.sum()

        return nrl.mean()


def logloss_glob(weights):

    loss = nn.BCELoss(reduction='none')

    def logloss(pred, target):
        nrl = loss(pred, target)
        nrl = (nrl * weights.unsqueeze(0)).sum(1) / weights.sum()
        return nrl.mean()

    return logloss


class BCELossPost:

    def __init__(self):

        self.loss = nn.BCELoss(reduction='mean')

    def __call__(self, input, target, **kwargs):

        yh, yh_len = input['yh'], input['yh_len']
        losses = torch.zeros(yh.shape[0], dtype=torch.float32, device=yh.device)
        for i in range(yh.shape[0]):
            losses[i] = self.loss(yh[i, :yh_len[i], ...], target[i, :yh_len[i], ...])

        return losses.mean()


def logloss_post(weights):

    loss = nn.BCELoss(reduction='none')

    def logloss(pred, target):

        yh, yh_len = pred['yh'], pred['yh_len']
        losses = torch.zeros(yh.shape[0], dtype=torch.float32, device=yh.device)
        for i in range(yh.shape[0]):
            nrl = loss(yh[i, :yh_len[i], ...], target[i, :yh_len[i], ...])
            nrl = (nrl * weights.unsqueeze(0)).sum(1) / weights.sum()
            losses[i] = nrl.mean()

        return losses.mean()

    return logloss
