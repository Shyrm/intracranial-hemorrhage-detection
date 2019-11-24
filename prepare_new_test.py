import os
import cv2
import glob2
import pydicom
import pandas as pd
import numpy as np
from skimage import exposure
from tqdm import tqdm

DATA_INPUT = './RawData'
DATA_INPUT_TARGET = os.path.realpath(f'{DATA_INPUT}/stage_2_train.csv')
DATA_INPUT_TEST = os.path.realpath(f'{DATA_INPUT}/stage_2_test_images')


DATA_OUTPUT = './Data'
VIEWS = {
    'ORIG': {
        'DATA_OUTPUT_TEST': f'{DATA_OUTPUT}/test_orig'
    },
    'BRAIN': {
        'WC': 40,
        'WW': 80,
        'DATA_OUTPUT_TEST': f'{DATA_OUTPUT}/test_brain'
    },
    'BLOOD': {
        'WC': 80,
        'WW': 200,
        'DATA_OUTPUT_TEST': f'{DATA_OUTPUT}/test_blood'
    },
    'BONE': {
        'WC': 600,
        'WW': 2800,
        'DATA_OUTPUT_TEST': f'{DATA_OUTPUT}/test_bone'
    },
    'STROKE': {
        'WC': 40,
        'WW': 40,
        'DATA_OUTPUT_TEST': f'{DATA_OUTPUT}/test_stroke'
    },
    'SOFT_TISSUE': {
        'WC': 40,
        'WW': 375,
        'DATA_OUTPUT_TEST': f'{DATA_OUTPUT}/test_soft_tissue'
    }
}

DATA_INPUT_STATS_M1 = os.path.realpath(f'{DATA_OUTPUT}/StatsViewsM1.csv')
DATA_INPUT_STATS_M2 = os.path.realpath(f'{DATA_OUTPUT}/StatsViewsM2.csv')

DATA_OUTPUT_TARGET = os.path.realpath(f'{DATA_OUTPUT}/TargetS2.csv')
DATA_OUTPUT_STATS_M1 = os.path.realpath(f'{DATA_OUTPUT}/StatsViewsS2M1.csv')
DATA_OUTPUT_STATS_M2 = os.path.realpath(f'{DATA_OUTPUT}/StatsViewsS2M2.csv')

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


def convert_image(filename, views, sz, is_test=False, sz0=sz0, add_contrast=True):

    ds = pydicom.read_file(str(filename))
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

    name = filename.split('/')[-1][:-4] + '.png'

    for vk, img in img_out.items():
        if not is_test:
            output_folder = views[vk]['DATA_OUTPUT_TRAIN']
        else:
            output_folder = views[vk]['DATA_OUTPUT_TEST']
        cv2.imwrite(os.path.realpath(f'{output_folder}/{name}'), img)

    return x_tot, x2_tot


if __name__ == '__main__':

    # ============================ create all necessary folders =====================================

    for vt, vp in VIEWS.items():
        DATA_OUTPUT_TEST = vp['DATA_OUTPUT_TEST']

        if not os.path.exists(os.path.realpath(DATA_OUTPUT_TEST)):
            os.makedirs(os.path.realpath(DATA_OUTPUT_TEST))

    # ============================ create dataset for learning ======================================

    print('Preparing new target...')

    target_data = pd.read_csv(DATA_INPUT_TARGET, sep=',', header=0)
    target_data['ImageId'], target_data['ClassId'] = target_data['ID'].str.rsplit('_', 1).str
    target_data = pd.pivot_table(target_data[['ImageId', 'ClassId', 'Label']], values='Label', index='ImageId', columns='ClassId', aggfunc=np.sum).reset_index()
    target_data.to_csv(DATA_OUTPUT_TARGET, sep=';', header=True, index=False)

    print('Preparing new test data...')

    stats_m1, stats_m2 = [], []
    test_dcm_list = glob2.glob(os.path.join(DATA_INPUT_TEST, '**/*.dcm'))

    for file in tqdm(test_dcm_list):

        idx = file.split('/')[-1][:-4]  # get index of the image

        # ignore corrupted images
        if idx == 'ID_6431af929':
            continue

        img_mean, img2_mean = convert_image(
            filename=file,
            views=VIEWS,
            sz=target_sz,
            is_test=True,
            add_contrast=False
        )

        img_mean['ImageId'], img2_mean['ImageId'] = idx, idx
        img_mean['IsTest'], img2_mean['IsTest'] = 1, 1
        stats_m1.append(img_mean)
        stats_m2.append(img2_mean)

    # store statistics
    stats_m1 = pd.DataFrame(stats_m1)
    stats_s1_m1 = pd.read_csv(DATA_INPUT_STATS_M1, sep=';', header=0)
    stats_m1 = pd.concat([stats_s1_m1, stats_m1])
    stats_m1.to_csv(DATA_OUTPUT_STATS_M1, sep=';', header=True, index=False)
    stats_m2 = pd.DataFrame(stats_m2)
    stats_s1_m2 = pd.read_csv(DATA_INPUT_STATS_M2, sep=';', header=0)
    stats_m2 = pd.concat([stats_s1_m2, stats_m2])
    stats_m2.to_csv(DATA_OUTPUT_STATS_M2, sep=';', header=True, index=False)

    for vk, vp in VIEWS.items():
        img_mean = stats_m1[vk].mean()
        img_std = np.sqrt(stats_m2[vk].mean() - img_mean ** 2)
        print('{0} mean = {1}, std = {2}'.format(vk, img_mean, img_std))

