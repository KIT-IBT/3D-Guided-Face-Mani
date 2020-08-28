import pickle
import glob
import os
import cv2
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from morphable_model.face3d.face3d import mesh
from morphable_model.utils.ddfa import reconstruct_vertex
from morphable_model.utils.render import cpncc_v3, cncc_v3, crender_colors
from morphable_model.morphable_model_300W import MorphabelModel300W
from morphable_model.local_deformations.local_shape_loader import LocalShapeLoader


IMAGE_SIZE = (450, 450, 3)
MULTIPLIERS = ['n', 'p']
#SHAPE_MODS = [['chin_1', 'chin_2'],
#              ['nose_1', 'nose_2']]
SHAPE_MODS = [['chin_1',],['chin_2',]]

#dataset_list = ['AFW', 'HELEN',
#                'IBUG', 'LFPW',]
                #'validset']
#dataset_list = ['IBUG', 'LFPW',
#                'validset']
dataset_list = ['AFLW2000_chin_only']
#dataset_list = ['AFW_Flip', 'HELEN_Flip',
#                'IBUG_Flip', 'LFPW_Flip']
#                'validset_Flip']
#dataset_list = ['LFPW',
#                'validset']
#dataset_list = ['testset']

IMG_FORMAT = '.jpg'
#IMG_FORMAT = '.png'

PNCC = True
CONTINUOUS_MULTIPLIER = False


dataset_path = 'data/datasets'
LOCAL_SHAPE_MOD_CSV = 'data/configs/shape_mods/BFM200_local_deformation_vectors.csv'

BFM_model = MorphabelModel300W(SYSTEM)
shape_mod_loader = LocalShapeLoader(LOCAL_SHAPE_MOD_CSV)

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


def render_diff(mod, alpha, alpha_exp, angles, t3d, f):
    vertices_mod = BFM_model.w @ mod
    vertices_mod = vertices_mod.reshape(3, -1, order='F')
    dist = np.linalg.norm(vertices_mod, axis=0)
    dist = dist / np.max(dist)
    colors = np.zeros((3, dist.shape[0]))
    colors[0, :] = dist
    colors[1, :] = dist
    colors[2, :] = dist
    colors = np.minimum(np.maximum(colors, 0), 1) * 255

    vertices = BFM_model.reconstruct_vertex(alpha=alpha_mod,
                                            alpha_exp=alpha_exp,
                                            angles=angles,
                                            t3d=t3d,
                                            f=f,
                                            img_size=IMAGE_SIZE
                                           )
#    image_vertices = mesh.transform.to_image(vertices,
#                                             450,
#                                             450)
    img = crender_colors(vertices.T, BFM_model.tri.T, colors.T,
                         450,
                         450, c=3)
    img = img[:, :, 0, None]
    return img



def render_pncc(alpha, alpha_exp, angles, t3d, f, render_ncc=False):

    if not render_ncc:
        vertices = BFM_model.reconstruct_vertex(alpha=alpha,
                                                alpha_exp=alpha_exp,
                                                angles=angles,
                                                t3d=t3d,
                                                f=f)
        pncc_feature = cpncc_v3([vertices,],
                                BFM_model.tri,
                                img_size=IMAGE_SIZE)  # cython version
    else:
        ncc_vertices = BFM_model.reconstruct_vertex(alpha=alpha,
                                                    alpha_exp=alpha_exp,
                                                    angles=None,
                                                    t3d=None,
                                                    f=None,
                                                    transform=False)
        vertices = BFM_model.transform_face(ncc_vertices.copy(),
                                            angles=angles,
                                            t3d=t3d,
                                            f=f)

        pncc_feature = cncc_v3(vertices,
                               ncc_vertices,
                               BFM_model.tri,
                               img_size=IMAGE_SIZE)  # cython version

    ### Remove all pixels that are not connected to the biggest blob
    # First prepare binary mask
    mask = pncc_feature.astype(np.float32)
    grey_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (grey_img > 0).astype(np.uint8)
    # Find connected pixels in mask
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask,
                                                                               connectivity=8)
    # Remove background from list
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # Set not connected pixels to zero e.g. back part of the ear
    min_size = max(sizes)
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            pncc_feature[output == i + 1] = 0

    return pncc_feature


for name in dataset_list:
    dataset = os.path.join(dataset_path, name)
    sample_list = [s for s in os.listdir(dataset) if IMG_FORMAT in s]
    n_samples = len(sample_list)

#    for f in glob.glob(dataset + "/*.png"):
#        os.remove(f)

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

        if PNCC:
            pncc = render_pncc(alpha, alpha_exp, angles, t3d, f)
            cv2.imwrite(mat_path[:-4] + '_pncc.png',
                        pncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
        else:
            ncc = render_pncc(alpha, alpha_exp, angles, t3d, f,
                              render_ncc=True)
            cv2.imwrite(mat_path[:-4] + '_ncc.png',
                        ncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR

#        multiplier = np.random.choice(MULTIPLIERS)
        for shape_mods in SHAPE_MODS:
            shape_mod = np.random.choice(shape_mods)
            for multiplier in MULTIPLIERS:
                alpha_mod, mod, factor = get_shape_mod(alpha, multiplier, shape_mod)
                if PNCC:
                    pncc = render_pncc(alpha_mod, alpha_exp, angles, t3d, f)
                    cv2.imwrite(mat_path[:-4] + f'_pncc_mod_{shape_mod}_{multiplier}.png',
                                pncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
                else:
                    ncc = render_pncc(alpha_mod, alpha_exp, angles, t3d, f,
                                      render_ncc=True)
                    cv2.imwrite(mat_path[:-4] + f'_ncc_mod_{shape_mod}_{multiplier}.png',
                                ncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
                mod = mod[:, None] * factor
                diff = render_diff(mod, alpha_mod, alpha_exp, angles, t3d, f)
                cv2.imwrite(mat_path[:-4] + f'_diff_mod_{shape_mod}_{multiplier}.png',
                            diff[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
