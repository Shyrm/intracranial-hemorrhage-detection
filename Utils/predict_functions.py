import torch
import gc
from fastai.basic_data import DatasetType
from fastai.vision import flip_lr, dihedral
import numpy as np


def pred_with_flip(learn, ds_type=DatasetType.Valid, use_tta=False, flips=[]):

    # get prediction
    if use_tta:
        preds, acts = learn.TTA(ds_type)
        return preds, acts

    preds, acts = learn.get_preds(ds_type)

    if len(flips) == 0:
        return preds, acts

    # add flip to dataset and get prediction
    for f in flips:
        learn.data.dl(ds_type).dl.dataset.tfms.append(dihedral(k=f, is_random=False))
        predsf, _ = learn.get_preds(ds_type)

        del learn.data.dl(ds_type).dl.dataset.tfms[-1]

        preds += predsf

        del predsf
        gc.collect()
        torch.cuda.empty_cache()

    preds /= (len(flips) + 1)

    return preds, acts
