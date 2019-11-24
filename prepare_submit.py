import pandas as pd
import os
from Utils.predict_functions import pred_with_flip
from Utils.generators_fastai import get_data_prob
from Utils.acc_grad_learner import AccGradLearner
from fastai.basic_data import DatasetType
from Architectures import SEResNeXt50, EfficientNetB4
import torch
from torch import nn
from sync_batchnorm import convert_model
import numpy as np
from modelling import (DEVICE, DATA_FOLDER, SZ, STATISTICS, N_CLASSES, DROPOUT, DATA_OPTIONS)


SUBMISSIONS_FOLDER = './Submissions'
SUBMISSION_FILE = 'BloodS1_SE_cp4.csv'

MODELS_FOLDER = './PredictModels'  # all models from this folder will be used and predictions averaged
DATA_MODE = 'blood_wo_contrast'

MODEL = SEResNeXt50

BATCH_SIZE = 256
NUM_WORKERS = 32

# FLIPS = []  # no flips
FLIPS = [2]  # flip_lr
# FLIPS = [1]  # flip_tb
# FLIPS = [1, 2, 3]  # flip_lr + flip_tb
# FLIPS = [1, 2, 3, 4, 5, 6, 7]  # flip_lr + flip_tb + rotate90


def prepare_submit(learn, flips=FLIPS):

    # get test predictions
    preds, _ = pred_with_flip(learn, ds_type=DatasetType.Test, flips=flips)
    ids = [o.stem for o in data.test_ds.items]

    sub = pd.DataFrame(preds.detach().cpu().numpy(), index=ids,
                       columns=['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'])

    sub.index.name = 'ImageId'
    sub = pd.melt(sub.reset_index(), id_vars='ImageId', var_name='Type', value_name='Label')
    sub['ID'] = sub['ImageId'] + '_' + sub['Type']
    return sub[['ID', 'Label']].sort_values('ID')


def save_submit(submit, folder=SUBMISSIONS_FOLDER, file_name=SUBMISSION_FILE):

    submit.to_csv(f'{folder}/{file_name}', sep=',', header=True, index=False)


def merge_submissions(subs):

    res = subs[0]
    for dt in subs[1:]:
        res = pd.merge(
            left=res,
            right=dt,
            on='ID',
            how='inner'
        )

    res['Label'] = np.mean(res[[col for col in res.columns if col != 'ID']].values, axis=1)

    return res[['ID', 'Label']].sort_values('ID')


if __name__ == '__main__':

    if not os.path.exists(SUBMISSIONS_FOLDER):
        os.makedirs(SUBMISSIONS_FOLDER)

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
    model = MODEL(num_classes=N_CLASSES, dropout_p=DROPOUT)

    # iterate though state dicts in specified folder and obtain predictions for each
    submissions = []
    for file in os.listdir(MODELS_FOLDER):

        model.load_state_dict(torch.load(f'{MODELS_FOLDER}/{file}')['model'])
        # if 'swa' in file:
        #     model.load_state_dict(torch.load(f'{MODELS_FOLDER}/{file}'))
        # else:
        #     model.load_state_dict(torch.load(f'{MODELS_FOLDER}/{file}')['model'])

        model.to(DEVICE)
        learner = AccGradLearner(data, model)
        learner.loss_func = None
        learner.model = convert_model(learner.model)
        learner.model = nn.DataParallel(learner.model).to(DEVICE)
        learner.model.eval()

        print(f'Inferring from model {file}')
        submissions.append(prepare_submit(learner))

    # merge submissions and store into file
    if len(submissions) == 1:
        save_submit(submissions[0])
    else:
        save_submit(merge_submissions(submissions))

