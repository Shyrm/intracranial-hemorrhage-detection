import argparse
import pandas as pd
import numpy as np
import random
import os

import torch
from torch import nn

from Architectures import SEResNeXt50, EfficientNetB4
from Utils.generators_fastai import get_data_prob
from Utils.acc_grad_learner import AccGradLearner
from Utils.utils import set_BN_momentum, AccumulateStep
from Utils.losses import logloss_glob
from sync_batchnorm import convert_model
from fastai.vision import AdamW

from fastai.callbacks import CSVLogger, EarlyStoppingCallback, SaveModelCallback
from fastai.callback import annealing_poly

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', '-se', help="Start epoch of learner", type=int, default=0)
print(parser.format_help())
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_FOLDER = './Data'

MODELS_FOLDER = './FittedModels'
LOGGING_FOLDER = f'{MODELS_FOLDER}/ProbModel_v01'


START_EPOCH = args.start_epoch  # 0 = first epoch!!!
# CONTINUE_FROM = f'{MODELS_FOLDER}/ProbModel_v01/inter_model_9.pth'
CONTINUE_FROM = None


SZ = 512
N_CLASSES = 6

DATA_MODE = 'blood_wo_contrast'

DATA_OPTIONS = {
    'original_wo_contrast': {
        'train': 'train_orig',
        'test': 'test_orig',
        'statistics': ([0.1743, 0.1743, 0.1743], [0.3238, 0.3238, 0.3238])
    },
    'bone_wo_contrast': {
        'train': 'train_bone',
        'test': 'test_bone',
        'statistics': ([0.11655, 0.11655, 0.11655], [0.1780, 0.1780, 0.1780])
    },
    'blood_wo_contrast': {
        'train': 'train_blood',
        'test': 'test_blood',
        'statistics': ([0.13295, 0.13295, 0.13295], [0.2815, 0.2815, 0.2815])
    },
    'brain_wo_contrast': {
        'train': 'train_brain',
        'test': 'test_brain',
        'statistics': ([0.16177, 0.16177, 0.16177], [0.3142, 0.3142, 0.3142])
    },
    'stroke_wo_contrast': {
        'train': 'train_stroke',
        'test': 'test_stroke',
        'statistics': ([0.14945, 0.14945, 0.14945], [0.31915, 0.31915, 0.31915])
    },
    'soft_tissue_wo_contrast': {
        'train': 'train_soft_tissue',
        'test': 'test_soft_tissue',
        'statistics': ([0.18298, 0.18298, 0.18298], [0.3063, 0.3063, 0.3063])
    }
}

STATISTICS = DATA_OPTIONS[DATA_MODE]['statistics']

DROPOUT = None

MODEL = EfficientNetB4

N_ACC = 1
BATCH_SIZE = 96
NUM_WORKERS = 64
LEARNING_RATE = 0.001
PATIENCE = 8
NUM_EPOCHS = 10
ANNEALING = annealing_poly(degree=0.9)

CRITERION = nn.BCELoss()
METRIC = logloss_glob(weights=torch.tensor([2, 1, 1, 1, 1, 1], dtype=torch.float32, device=DEVICE))
OPTIMIZER = AdamW
WD = 1e-2
DIV_FACTOR = 10

SEED = 42


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    seed_everything(SEED)

    # create folders if not exist
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if not os.path.exists(LOGGING_FOLDER):
        os.makedirs(LOGGING_FOLDER)

    # read and filter labels
    labels = pd.read_csv(f'{DATA_FOLDER}/Target.csv', sep=';', header=0)
    # labels_cq = pd.read_csv(f'{DATA_FOLDER}/TargetCQ500.csv', sep=';', header=0)
    # cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    # labels_cq[cols] = np.clip(labels_cq[cols], 0, 1)
    # labels = pd.concat([labels, labels_cq])
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
    if CONTINUE_FROM is not None:
        model.load_state_dict(torch.load(CONTINUE_FROM)['model'])

    # init learner
    learn = AccGradLearner(data, model, model_dir=LOGGING_FOLDER, loss_func=CRITERION,
                           metrics=[METRIC], opt_func=OPTIMIZER, wd=WD)
    set_BN_momentum(learn.model, n_acc=N_ACC)
    learn.clip_grad(1.)

    learn.model = convert_model(learn.model)
    learn.model = nn.DataParallel(learn.model).to(DEVICE)

    # init callbacks
    csv_logger = CSVLogger(learn=learn, filename=f'{LOGGING_FOLDER}/fit_trace', append=True)
    early_stopping = EarlyStoppingCallback(learn=learn, monitor='valid_loss', patience=PATIENCE)
    save_model = SaveModelCallback(learn=learn, monitor='valid_loss', name='inter_model', every='epoch')
    acc_grad = AccumulateStep(learn, N_ACC)

    # fit one cycle
    learn.fit_one_cycle(
        cyc_len=NUM_EPOCHS,
        max_lr=LEARNING_RATE,
        div_factor=DIV_FACTOR,
        final_div=DIV_FACTOR,
        annealing_func=ANNEALING,
        start_epoch=START_EPOCH,
        callbacks=[
            acc_grad,
            csv_logger,
            early_stopping,
            save_model
        ]
    )
