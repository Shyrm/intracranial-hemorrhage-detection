import random
import os
import torch
import numpy as np
from torch import nn
from fastai.callbacks import LearnerCallback
from fastai.vision import Learner
from dataclasses import dataclass
from tqdm import tqdm


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_BN_momentum(model, n_acc):

    momentum = 0.1 * n_acc
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            layer.momentum = momentum


@dataclass
class AccumulateStep(LearnerCallback):
    """
    Does accumlated step every nth step by accumulating gradients
    """

    def __init__(self, learn: Learner, n_step: int = 1):
        super().__init__(learn)
        self.n_step = n_step

    def on_epoch_begin(self, **kwargs):
        "init samples and batches, change optimizer"
        self.acc_batches = 0

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "accumulate samples and batches"
        self.acc_batches += 1

    def on_backward_end(self, **kwargs):
        "step if number of desired batches accumulated, reset samples"
        if (self.acc_batches % self.n_step) == self.n_step - 1:
            for p in (self.learn.model.parameters()):
                if p.requires_grad: p.grad.div_(self.acc_batches)

            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0

    def on_epoch_end(self, **kwargs):
        "step the rest of the accumulated grads"
        if self.acc_batches > 0:
            for p in (self.learn.model.parameters()):
                if p.requires_grad: p.grad.div_(self.acc_batches)
            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0


def collect_bn_modules(module, bn_modules):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        bn_modules.append(module)


def fix_batchnorm(swa_model, train_dl):
    bn_modules = []
    swa_model.apply(lambda module: collect_bn_modules(module, bn_modules))

    if not bn_modules: return

    swa_model.train()

    for module in bn_modules:
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

    momenta = [m.momentum for m in bn_modules]

    inputs_seen = 0

    for (*x, y) in tqdm(iter(train_dl), total=len(train_dl)):
        batch_size = x[0].size(0)

        momentum = batch_size / (inputs_seen + batch_size)
        for module in bn_modules:
            module.momentum = momentum

        res = swa_model(*x)
        del res
        torch.cuda.empty_cache()

        inputs_seen += batch_size

    for module, momentum in zip(bn_modules, momenta):
        module.momentum = momentum

