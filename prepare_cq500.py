import os
import cv2
import glob2
import pydicom
import pandas as pd
import numpy as np
from skimage import exposure
from tqdm import tqdm

DATA_INPUT = './QureaiCQ500/DICOM'
DATA_INPUT_TARGET = os.path.realpath('./QureaiCQ500/qureai-cq500-boxes.csv')


DATA_OUTPUT = './QureaiCQ500/Data'
VIEWS = {
    'ORIG': {
        'DATA_OUTPUT_TRAIN': f'{DATA_OUTPUT}/train_orig'
    },
    'BRAIN': {
        'WC': 40,
        'WW': 80,
        'DATA_OUTPUT_TRAIN': f'{DATA_OUTPUT}/train_brain'
    },
    'BLOOD': {
        'WC': 80,
        'WW': 200,
        'DATA_OUTPUT_TRAIN': f'{DATA_OUTPUT}/train_blood'
    },
    'BONE': {
        'WC': 600,
        'WW': 2800,
        'DATA_OUTPUT_TRAIN': f'{DATA_OUTPUT}/train_bone'
    },
    'STROKE': {
        'WC': 40,
        'WW': 40,
        'DATA_OUTPUT_TRAIN': f'{DATA_OUTPUT}/train_stroke'
    },
    'SOFT_TISSUE': {
        'WC': 40,
        'WW': 375,
        'DATA_OUTPUT_TRAIN': f'{DATA_OUTPUT}/train_soft_tissue'
    }
}

DATA_INPUT_STATS_M1 = os.path.realpath('./Data/StatsViewsM1.csv')
DATA_INPUT_STATS_M2 = os.path.realpath('./Data/StatsViewsM2.csv')

DATA_OUTPUT_TARGET_FULL = os.path.realpath(f'{DATA_OUTPUT}/TargetCQ500Full.csv')
DATA_OUTPUT_TARGET = os.path.realpath(f'{DATA_OUTPUT}/TargetCQ500.csv')
DATA_OUTPUT_STATS_M1 = os.path.realpath(f'{DATA_OUTPUT}/StatsViewsM1.csv')
DATA_OUTPUT_STATS_M2 = os.path.realpath(f'{DATA_OUTPUT}/StatsViewsM2.csv')

target_sz = 512
sz0 = 512


def window_image(img, window_center, window_width, intercept, slope):

    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    img = (img - img_min) / (img_max - img_min)

    return (img * 255.0).clip(0, 255).astype(np.uint8)


def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):

    dicom_fields = [
        data[('0028', '1050')].value,  # window center
        data[('0028', '1051')].value,  # window width
        data[('0028', '1052')].value,  # intercept
        data[('0028', '1053')].value  # slope
    ]

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def convert_image(filename, idx, views, sz, is_test=False, sz0=sz0, add_contrast=True):

    md = {'ImageId': idx}

    ds = pydicom.read_file(str(filename))

    for k in ds.dir():
        if k != 'PixelData':
            md[k] = ds.data_element(k).value

    img = ds.pixel_array
    window_center, window_width, intercept, slope = get_windowing(ds)

    img_out = {}
    for vk, vp in views.items():
        if vk == 'ORIG':
            img_out[vk] = window_image(img, window_center, window_width, intercept, slope)
        else:
            img_out[vk] = window_image(img, vp['WC'], vp['WW'], intercept, slope)

    if sz != sz0:
        for vk, img in img_out.items():
            img_out[vk] = cv2.resize(img, (sz, sz))

    if add_contrast:
        for vk, img in img_out.items():
            img_out[vk] = exposure.equalize_adapthist(img)  # contrast correction

    # image statistics
    if add_contrast:
        x_tot = {vk: img.mean() for vk, img in img_out.items()}
        x2_tot = {vk: (img**2).mean() for vk, img in img_out.items()}
    else:
        x_tot = {vk: (img / 255.).mean() for vk, img in img_out.items()}
        x2_tot = {vk: ((img / 255.) ** 2).mean() for vk, img in img_out.items()}

    if add_contrast:
        for vk, img in img_out.items():
            img_out[vk] = (img * 255).clip(0, 255).astype(np.uint8)

    name = idx + '.png'

    for vk, img in img_out.items():
        if not is_test:
            output_folder = views[vk]['DATA_OUTPUT_TRAIN']
        else:
            output_folder = views[vk]['DATA_OUTPUT_TEST']
        cv2.imwrite(os.path.realpath(f'{output_folder}/{name}'), img)

    return x_tot, x2_tot, md


if __name__ == '__main__':

    # ============================ create all necessary folders =====================================

    for vt, vp in VIEWS.items():
        DATA_OUTPUT_TEST = vp['DATA_OUTPUT_TRAIN']

        if not os.path.exists(os.path.realpath(DATA_OUTPUT_TEST)):
            os.makedirs(os.path.realpath(DATA_OUTPUT_TEST))

    # ============================ create dataset for learning ======================================

    print('Preparing new train data...')

    stats_m1, stats_m2 = [], []
    meta = []
    train_dcm_list = glob2.glob(os.path.join(DATA_INPUT, '**/*.dcm'))

    for i, file in tqdm(enumerate(train_dcm_list), total=len(train_dcm_list)):

        idx = 'cq500_image_' + str(i)

        img_mean, img2_mean, md = convert_image(
            filename=file,
            idx=idx,
            views=VIEWS,
            sz=target_sz,
            is_test=False,
            add_contrast=False
        )

        meta.append(md)
        img_mean['ImageId'], img2_mean['ImageId'] = idx, idx
        img_mean['IsTest'], img2_mean['IsTest'] = 0, 0
        stats_m1.append(img_mean)
        stats_m2.append(img2_mean)

    print('Preparing new target...')

    meta = pd.DataFrame(meta)[['ImageId', 'SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']]
    target_data = pd.read_csv(DATA_INPUT_TARGET, sep=',', header=0)[['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID', 'labelName']]
    target_data = pd.merge(meta, target_data, how='left', on=['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']).fillna('any')
    target_data['labelName'] = target_data['labelName'].str.lower()
    target_data = pd.pivot_table(target_data[['ImageId', 'labelName']], index='ImageId', columns='labelName', aggfunc='size', fill_value=0).reset_index()
    target_data['any'] = 1 - target_data['any']
    target_data.to_csv(DATA_OUTPUT_TARGET_FULL, sep=';', header=True, index=False)
    target_data.drop(columns='chronic', inplace=True)
    target_data['any'] = target_data[['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']].max(axis=1)
    target_data.to_csv(DATA_OUTPUT_TARGET, sep=';', header=True, index=False)

    print('Store dataset stats...')

    stats_m1 = pd.DataFrame(stats_m1)
    stats_m1.to_csv(DATA_OUTPUT_STATS_M1, sep=';', header=True, index=False)
    stats_m2 = pd.DataFrame(stats_m2)
    stats_m2.to_csv(DATA_OUTPUT_STATS_M2, sep=';', header=True, index=False)

    for vk, vp in VIEWS.items():
        img_mean = stats_m1[vk].mean()
        img_std = np.sqrt(stats_m2[vk].mean() - img_mean ** 2)
        print('CQ500 {0} mean = {1}, std = {2}'.format(vk, img_mean, img_std))

    # recalc full statistics
    stats_s1_m1 = pd.read_csv(DATA_INPUT_STATS_M1, sep=';', header=0)
    stats_m1 = pd.concat([stats_s1_m1, stats_m1])
    stats_s1_m2 = pd.read_csv(DATA_INPUT_STATS_M2, sep=';', header=0)
    stats_m2 = pd.concat([stats_s1_m2, stats_m2])

    for vk, vp in VIEWS.items():
        img_mean = stats_m1[vk].mean()
        img_std = np.sqrt(stats_m2[vk].mean() - img_mean ** 2)
        print('Full {0} mean = {1}, std = {2}'.format(vk, img_mean, img_std))

# CQ500 ORIG mean = 0.16694029327851254, std = 0.29330657059063564
# CQ500 BRAIN mean = 0.162406161144795, std = 0.31413759759428744
# CQ500 BLOOD mean = 0.12696418513976326, std = 0.2680984646057333
# CQ500 BONE mean = 0.11566735308935254, std = 0.18136913323034656
# CQ500 STROKE mean = 0.15685024051724106, std = 0.3276258558787897
# CQ500 SOFT_TISSUE mean = 0.17803718972332064, std = 0.29681395074418576

# Full ORIG mean = 0.17295798740380083, std = 0.31838217074833247
# Full BRAIN mean = 0.16188517931917512, std = 0.3141988326310683
# Full BLOOD mean = 0.13183924248876017, std = 0.27908733696929094
# Full BONE mean = 0.11638688345451915, std = 0.17865636238215613
# Full STROKE mean = 0.15082186721794164, std = 0.32075074649981894
# Full SOFT_TISSUE mean = 0.18206785437209608, std = 0.3045920150492521
