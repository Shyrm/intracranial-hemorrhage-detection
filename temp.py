import numpy as np
import pandas as pd
# import cv2
import os
from tqdm import tqdm

# FOLDERS = {
#     'train': {
#         'source_folders': [
#             './Data/train_bone',
#             './Data/train_blood',
#             './Data/train_brain',
#         ],
#         'output_folder': './Data/train_bbb'
#     },
#     'test': {
#         'source_folders': [
#             './Data/test_bone',
#             './Data/test_blood',
#             './Data/test_brain'
#         ],
#         'output_folder': './Data/test_bbb'
#     }
# }
#
# for data_part in FOLDERS.values():
#
#     output_folder = data_part['output_folder']
#     source_folders = data_part['source_folders']
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for img_id in tqdm(os.listdir(source_folders[0])):
#
#         channel1 = cv2.imread(f'{source_folders[0]}/{img_id}', cv2.IMREAD_GRAYSCALE)
#         channel2 = cv2.imread(f'{source_folders[1]}/{img_id}', cv2.IMREAD_GRAYSCALE)
#         channel3 = cv2.imread(f'{source_folders[2]}/{img_id}', cv2.IMREAD_GRAYSCALE)
#
#         full_image = np.stack([channel1, channel2, channel3], axis=2)
#
#         cv2.imwrite(f'{output_folder}/{img_id}', full_image)


# ============================== snippet to merge sevaral submissions by averaging ==============================
sub1 = pd.read_csv('./Submissions/Blood_cp1_6.csv', sep=',', header=0)
sub1['Label'] = sub1['Label'] * 6 / 7
sub2 = pd.read_csv('./Submissions/Blood_swa.csv', sep=',', header=0)
sub2['Label'] = sub2['Label'] * 1 / 7

def merge_submissions(subs):

    res = subs[0]
    for dt in subs[1:]:
        res = pd.merge(
            left=res,
            right=dt,
            on='ID',
            how='inner'
        )

    res['Label'] = np.sum(res[[col for col in res.columns if col != 'ID']].values, axis=1)

    return res[['ID', 'Label']].sort_values('ID')


merged_sub = merge_submissions([sub1, sub2])
merged_sub.to_csv('./Submissions/Blood_cp1_6_swa.csv', sep=',', header=True, index=False)