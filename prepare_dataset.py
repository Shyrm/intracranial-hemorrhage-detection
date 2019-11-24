import os
import cv2
import glob2
import pydicom
import pandas as pd
import numpy as np
from skimage import exposure
from tqdm import tqdm
from shutil import copyfile

DATA_INPUT = './RawData'
DATA_INPUT_TRAIN = os.path.realpath(f'{DATA_INPUT}/stage_1_train_images')
DATA_INPUT_TEST = os.path.realpath(f'{DATA_INPUT}/stage_1_test_images')
DATA_INPUT_TARGET = os.path.realpath(f'{DATA_INPUT}/stage_1_train.csv')


DATA_OUTPUT = './Data'

SEED = 2019
TRAIN_SMALL_SIZE = 80000
VALIDATION_SIZE = 15000

DATA_OUTPUT_TRAIN_BIG = f'{DATA_OUTPUT}/train_big'
DATA_OUTPUT_TRAIN_SMALL = f'{DATA_OUTPUT}/train_small'
DATA_OUTPUT_TEST = f'{DATA_OUTPUT}/test'

DATA_OUTPUT_TRAIN_META = os.path.realpath(f'{DATA_OUTPUT}/TrainMeta.csv')
DATA_OUTPUT_TEST_META = os.path.realpath(f'{DATA_OUTPUT}/TestMeta.csv')

DATA_OUTPUT_TARGET = os.path.realpath(f'{DATA_OUTPUT}/Target.csv')
DATA_OUTPUT_STATS = os.path.realpath(f'{DATA_OUTPUT}/Stats.csv')
DATA_OUTPUT_VALID_IDX = os.path.realpath(f'{DATA_OUTPUT}/ValidIdx.csv')


target_sz = 512
sz0 = 512


def window_image(img, window_center, window_width, intercept, slope):

    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    img = (img - img_min) / (img_max - img_min)

    return np.uint8(img * 255.0)


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


def convert_image(filename, output_folders, sz, is_dicom=True, sz0=sz0, add_contrast=True):

    imid = str(filename).split('/')[-1][:-4]
    md = {'ImageId': imid}

    if is_dicom:
        ds = pydicom.read_file(str(filename))

        for k in ds.dir():
            if k != 'PixelData':
                md[k] = ds.data_element(k).value

        img = ds.pixel_array
        window_center, window_width, intercept, slope = get_windowing(ds)
        img = window_image(img, window_center, window_width, intercept, slope)
    else:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if sz != sz0:
        img = cv2.resize(img, (sz, sz))

    if add_contrast:
        img = exposure.equalize_adapthist(img)  # contrast correction

    # image statistics
    x_tot = img.mean()
    x2_tot = (img**2).mean()

    if add_contrast:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    name = filename.split('/')[-1][:-4] + '.png'

    for output_folder in output_folders:
        cv2.imwrite(os.path.realpath(f'{output_folder}/{name}'), img)

    return x_tot, x2_tot, md


if __name__ == '__main__':

    # ============================ create all necessary folders =====================================

    if not os.path.exists(os.path.realpath(DATA_OUTPUT_TRAIN_BIG)):
        os.makedirs(os.path.realpath(DATA_OUTPUT_TRAIN_BIG))

    if not os.path.exists(os.path.realpath(DATA_OUTPUT_TRAIN_SMALL)):
        os.makedirs(os.path.realpath(DATA_OUTPUT_TRAIN_SMALL))

    if not os.path.exists(os.path.realpath(DATA_OUTPUT_TEST)):
        os.makedirs(os.path.realpath(DATA_OUTPUT_TEST))

    # ============================ create dataset for learning ======================================

    print('Preparing train data...')

    target_data = pd.read_csv(DATA_INPUT_TARGET, sep=',', header=0)
    target_data['ImageId'], target_data['ClassId'] = target_data['ID'].str.rsplit('_', 1).str
    target_data = pd.pivot_table(target_data[['ImageId', 'ClassId', 'Label']], values='Label', index='ImageId', columns='ClassId', aggfunc=np.sum).reset_index()
    target_data.to_csv(DATA_OUTPUT_TARGET, sep=';', header=True, index=False)

    idxs = set(target_data.ImageId)

    np.random.seed(SEED)
    small_train_idx = np.random.choice(list(idxs), size=(TRAIN_SMALL_SIZE + VALIDATION_SIZE), replace=False)

    statistics = []
    meta = []
    train_dcm_list = glob2.glob(os.path.join(DATA_INPUT_TRAIN, '**/*.dcm'))

    for file in tqdm(train_dcm_list):

        idx = file.split('/')[-1][:-4]  # get index of the image

        # ignore images without target or corrupted
        if idx not in idxs or idx == 'ID_6431af929':
            continue

        output_folders = [DATA_OUTPUT_TRAIN_BIG]
        is_small = 1 if idx in small_train_idx else 0
        if is_small == 1:
            output_folders.append(DATA_OUTPUT_TRAIN_SMALL)

        img_mean, img2_mean, md = convert_image(
            filename=file,
            sz=target_sz,
            output_folders=output_folders
        )

        statistics.append([idx, 0, is_small, img_mean, img2_mean])
        md['IsSmall'] = is_small
        meta.append(md)

    pd.DataFrame(meta).to_csv(DATA_OUTPUT_TRAIN_META, sep=';', header=True, index=False)

    print('Preparing test data...')

    meta = []
    test_dcm_list = glob2.glob(os.path.join(DATA_INPUT_TEST, '**/*.dcm'))

    for file in tqdm(test_dcm_list):

        idx = file.split('/')[-1][:-4]  # get index of the image

        # ignore corrupted images
        if idx == 'ID_6431af929':
            continue

        img_mean, img2_mean, md = convert_image(
            filename=file,
            sz=target_sz,
            output_folders=[DATA_OUTPUT_TEST]
        )

        statistics.append([idx, 1, 0, img_mean, img2_mean])
        meta.append(md)

    pd.DataFrame(meta).to_csv(DATA_OUTPUT_TEST_META, sep=';', header=True, index=False)

    # store statistics for mask data separately
    stats = pd.DataFrame(statistics, columns=['ImageId', 'IsTest', 'IsSmall', 'mu', '2mu'])
    stats.to_csv(DATA_OUTPUT_STATS, sep=';', header=True, index=False)

    img_mean = stats['mu'].mean()
    img_std = np.sqrt(stats['2mu'].mean() - img_mean ** 2)
    print('Big mean = {0}, std = {1}'.format(img_mean, img_std))

    img_mean = stats[(stats['IsSmall'] == 1) | (stats['IsTest'] == 1)]['mu'].mean()
    img_std = np.sqrt(stats[(stats['IsSmall'] == 1) | (stats['IsTest'] == 1)]['2mu'].mean() - img_mean ** 2)
    print('Small mean = {0}, std = {1}'.format(img_mean, img_std))

    # ================================ create and store validation index =========================================
    np.random.seed(SEED)
    validation_idx = np.random.choice(small_train_idx, size=VALIDATION_SIZE, replace=False)

    pd.DataFrame(validation_idx, columns=['Idx']).to_csv(DATA_OUTPUT_VALID_IDX, sep=';', header=True, index=False)
