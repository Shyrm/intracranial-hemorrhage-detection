import pandas as pd
import numpy as np
from fastai.vision import *
from ast import literal_eval
from torch.utils.data import Dataset


def get_data_prob(
        train_folder,
        labels,
        valid_idx,
        test_path,
        img_size=(256, 256),
        batch_size=16,
        num_workers=3,
        normalization=([0.540, 0.540, 0.540], [0.264, 0.264, 0.264]),
):

    data = (ImageList.from_df(df=labels, path=train_folder)
            .split_by_files(valid_idx)
            .label_from_df(cols=[1, 2, 3, 4, 5, 6])
            .add_test(Path(test_path).ls(), label=None)
            .transform(get_transforms(), size=img_size, tfm_y=False)
            .databunch(path=Path('.'), bs=batch_size, num_workers=num_workers)
            .normalize(normalization))

    return data


def prepare_windows(train_meta_fn, test_meta_fn, train_prob_fn, test_prob_fn, train_target_fn):

    # get meta
    meta_train = pd.read_csv(train_meta_fn, sep=';', header=0)[['ImageId', 'StudyInstanceUID', 'ImagePositionPatient']]
    meta_test = pd.read_csv(test_meta_fn, sep=';', header=0)[['ImageId', 'StudyInstanceUID', 'ImagePositionPatient']]
    meta = pd.concat([meta_train, meta_test])
    meta['ImagePositionPatient'] = meta['ImagePositionPatient'].apply(literal_eval)
    meta[['ImagePositionPatient0', 'ImagePositionPatient1', 'ImagePositionPatient2']] = pd.DataFrame(meta.ImagePositionPatient.values.tolist(), index=meta.index)
    meta['ImagePositionPatient2'] = meta['ImagePositionPatient2'].astype(np.float)
    meta = meta[['ImageId', 'StudyInstanceUID', 'ImagePositionPatient2']]

    # get probs
    pred_train = pd.read_csv(train_prob_fn, sep=',', header=0)
    pred_test = pd.read_csv(test_prob_fn, sep=',', header=0)
    pred = pd.concat([pred_train, pred_test])
    pred['ImageId'], pred['ClassId'] = pred['ID'].str.rsplit('_', 1).str
    pred = pd.pivot_table(pred[['ImageId', 'ClassId', 'Label']], values='Label', index='ImageId', columns='ClassId', aggfunc=np.sum).reset_index()
    df = pd.merge(meta, pred, how='inner', on='ImageId')
    df.sort_values(by=['StudyInstanceUID', 'ImagePositionPatient2', 'ImageId'], inplace=True, ascending=True)

    # train X
    df_train = df[~df['StudyInstanceUID'].isin(meta_test['StudyInstanceUID'])]

    # test X
    df_test = df[df['StudyInstanceUID'].isin(meta_test['StudyInstanceUID'])]

    # train Y
    y_train = pd.read_csv(train_target_fn, sep=';', header=0)
    dfy = pd.merge(df_train[['ImageId', 'StudyInstanceUID', 'ImagePositionPatient2']], y_train, how='inner', on='ImageId')
    dfy.sort_values(by=['StudyInstanceUID', 'ImagePositionPatient2', 'ImageId'], inplace=True, ascending=True)

    return df_train, dfy, df_test


class StudyDataset(Dataset):

    def __init__(self, df, idxs, target=None, is_test=False):
        super().__init__()

        self.df = df
        self.idxs = idxs
        self.target = target
        self.is_test = is_test

        self.image_ids = []
        for study_idx in self.idxs:
            self.image_ids.extend(self.df.get_group(study_idx)['ImageId'].values)

        self.cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        self.max_len = 60

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        study_idx = self.idxs[idx]

        x = np.zeros(shape=(self.max_len, len(self.cols)))
        xt = self.df.get_group(study_idx)[self.cols].values
        x[:xt.shape[0], :] = xt
        x = torch.tensor(x, dtype=torch.float32)

        if self.is_test:
            return {'x': x, 'x_len': xt.shape[0]}
        else:
            y = np.zeros(shape=(self.max_len, len(self.cols)))
            yt = self.target.get_group(study_idx)[self.cols].values
            y[:yt.shape[0], :] = yt
            y = torch.tensor(y, dtype=torch.float32)
            return {'x': x, 'x_len': xt.shape[0]}, y


def get_data_post(
        train_meta_fn, test_meta_fn, train_prob_fn, test_prob_fn, train_target_fn,
        valid_size=0.2, seed=42,
        batch_size=64, num_workers=4
):

    df_train, dfy, df_test = prepare_windows(train_meta_fn, test_meta_fn, train_prob_fn, test_prob_fn, train_target_fn)

    dfg_train = df_train.groupby('StudyInstanceUID')
    gk_train = list(dfg_train.groups.keys())

    random.seed(seed)
    gk_valid = [gk_train[i] for i in sorted(random.sample(range(len(gk_train)), int(len(gk_train) * valid_size)))]
    gk_train = [idx for idx in gk_train if idx not in gk_valid]

    dfy = dfy.groupby('StudyInstanceUID')

    dfg_test = df_test.groupby('StudyInstanceUID')
    gk_test = list(dfg_test.groups.keys())

    ds_train = StudyDataset(df=dfg_train, idxs=gk_train, target=dfy, is_test=False)
    ds_valid = StudyDataset(df=dfg_train, idxs=gk_valid, target=dfy, is_test=False)
    ds_test = StudyDataset(df=dfg_test, idxs=gk_test, target=None, is_test=True)

    return DataBunch.create(train_ds=ds_train, valid_ds=ds_valid, test_ds=ds_test, bs=batch_size, num_workers=num_workers)

