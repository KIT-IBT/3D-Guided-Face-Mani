import pickle
import argparse
import glob
import os
import cv2
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from morphable_model.local_shape_loader import LocalShapeLoader
from morphable_model.utils import render_diff, render_pncc

############################################
# Configs
############################################
IMAGE_SIZE = (450, 450, 3)
MULTIPLIERS = ['n', 'p']
SHAPE_MODS = [['chin_1', 'chin_2'],
              ['nose_1', 'nose_2']]
IMG_FORMAT = '.jpg'
#IMG_FORMAT = '.png'
CONTINUOUS_MULTIPLIER = False
dataset_path = 'data/datasets'
LOCAL_SHAPE_MOD_CSV = 'data/configs/shape_mods/BFM200_local_deformation_vectors.csv'
shape_mod_loader = LocalShapeLoader(LOCAL_SHAPE_MOD_CSV)

parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
parser.add_argument('dataset')
args = parser.parse_args()

if args.dataset == 'AFLW2000':
    dataset_list = ['AFLW2000']
elif args.dataset == '300W-LP':
    dataset_list = ['AFW', 'HELEN',
                    'IBUG', 'LFPW',
                    'AFW_Flip', 'HELEN_Flip',
                    'IBUG_Flip', 'LFPW_Flip',
                    'validset', 'validset_Flip']
    dataset_path = os.path.join(dataset_path,
                                '300W_LP')
else:
    raise Exception('Only datasets "AFLW2000" or "300W-LP" accepted!')


############################################
# Functions
############################################

def get_shape_mod(alpha, multiplier, shape_mod):

    mod = shape_mod_loader.load_vector(shape_mod)
    f_min, f_max = shape_mod_loader.get_factor_range(shape_mod)
    if multiplier == 'n':
        factor = f_min
    elif multiplier == 'p':
        factor = f_max
    else:
        raise Exception(f'Given param MULTIPLIER not valid: {multiplier}')

    if CONTINUOUS_MULTIPLIER:
        factor = np.random.uniform(0, factor) if factor > 0 else np.random.uniform(factor, 0)

    alpha_mod = alpha.copy()
    alpha_mod[:] += mod[:, None] * factor

    return alpha_mod, mod, factor


############################################
# Dataset rendering 
############################################
for name in dataset_list:
    dataset = os.path.join(dataset_path, name)
    sample_list = [s for s in os.listdir(dataset) if IMG_FORMAT in s]
    n_samples = len(sample_list)

    # Delete previous PNCCs
    for f in glob.glob(dataset + "/*pncc*"):
        os.remove(f)
    for f in glob.glob(dataset + "/*diff*"):
        os.remove(f)

    print(f'Found {n_samples} images in {dataset}! Now rendering corresponding PNCCs ...')
    for s in tqdm(sample_list):

        mat_path = os.path.join(dataset, s[:-4] + '.mat')

        mat = sio.loadmat(mat_path)
        alpha = mat['Shape_Para']
        alpha_exp = mat['Exp_Para']

        # phi, gamma, theta, tx, ty, tz, f
        pose = mat['Pose_Para']
        angles = pose[0, :3]
        t3d = pose[0, 3:6]
        t3d = t3d[:, None]
        f = pose[0, 6]

        pncc = render_pncc(alpha, alpha_exp, angles, t3d, f)
        cv2.imwrite(mat_path[:-4] + '_pncc.png',
                    pncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR

#        multiplier = np.random.choice(MULTIPLIERS)
        for shape_mods in SHAPE_MODS:
            shape_mod = np.random.choice(shape_mods)
            for multiplier in MULTIPLIERS:
                alpha_mod, mod, factor = get_shape_mod(alpha, multiplier, shape_mod)
                pncc = render_pncc(alpha_mod, alpha_exp, angles, t3d, f)
                cv2.imwrite(mat_path[:-4] + f'_pncc_mod_{shape_mod}_{multiplier}.png',
                            pncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
                mod = mod[:, None] * factor
                diff = render_diff(mod, alpha_mod, alpha_exp, angles, t3d, f)
                cv2.imwrite(mat_path[:-4] + f'_diff_mod_{shape_mod}_{multiplier}.png',
                            diff[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
