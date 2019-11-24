import pandas as pd
import os
import torch
from torch import nn
from fastai.basic_train import Learner
from sync_batchnorm import convert_model

from Architectures import SEResNeXt50, EfficientNetB4
from Utils.generators_fastai import get_data_prob
from Utils.utils import fix_batchnorm

from modelling import (DEVICE, DATA_FOLDER, SZ, STATISTICS, N_CLASSES, DROPOUT, DATA_OPTIONS,
                       CRITERION, OPTIMIZER, WD)


MODELS_FOLDER = './BloodModels'
DATA_MODE = 'blood_wo_contrast'

MODEL = SEResNeXt50

BATCH_SIZE = 76
NUM_WORKERS = 32


if __name__ == '__main__':

    # read and filter labels
    labels = pd.read_csv(f'{DATA_FOLDER}/Target.csv', sep=';', header=0)
    labels['ImageId'] = labels['ImageId'].apply(lambda x: f'{x}.png')
    labels = labels[labels['ImageId'].isin(set(os.listdir(f'{DATA_FOLDER}/{DATA_OPTIONS[DATA_MODE]["train"]}')))]

    # read and format validation index
    valid_idx = pd.read_csv(f'{DATA_FOLDER}/ValidIdx.csv', sep=';', header=0)['Idx'].values
    valid_idx = [f'{idx}.png' for idx in valid_idx]

    # init dataloader
    data = get_data_prob(
        train_folder=f'{DATA_FOLDER}/{DATA_OPTIONS[DATA_MODE]["train"]}',
        labels=labels,
        valid_idx=valid_idx,
        test_path=f'{DATA_FOLDER}/{DATA_OPTIONS[DATA_MODE]["test"]}',
        img_size=(SZ, SZ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization=STATISTICS,
    )

    # init model
    swa_model = MODEL(num_classes=N_CLASSES, dropout_p=DROPOUT)
    model = MODEL(num_classes=N_CLASSES, dropout_p=DROPOUT)

    # nullify all swa model parameters
    swa_params = swa_model.parameters()
    for swa_param in swa_params:
        swa_param.data = torch.zeros_like(swa_param.data)

    # average model
    n_swa = len(os.listdir(MODELS_FOLDER))
    print(f"Averaging {n_swa} models")
    for file in os.listdir(MODELS_FOLDER):
        model.load_state_dict(torch.load(f'{MODELS_FOLDER}/{file}')['model'])
        model_params = model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data += model_param.data / n_swa

    # fix batch norm
    print("Fixing batch norm")
    swa_model.to(DEVICE)
    learn = Learner(data, model, model_dir=MODELS_FOLDER, loss_func=CRITERION, opt_func=OPTIMIZER, wd=WD)
    learn.model = convert_model(learn.model)
    learn.model = nn.DataParallel(learn.model).to(DEVICE)
    fix_batchnorm(learn.model, learn.data.train_dl)
    learn.save('swa_model')
