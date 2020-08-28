import os
import cv2
import scipy.io as sio
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm
from dl_utils.nets.pix2pix.networks import UnetGenerator
import dl_utils.nets.pytorch_tiramisu.tiramisu as tiramisu
from morphable_model.utils.render import cpncc_v3

from dl_utils.dataset.subset_300W import read_pncc, read_img, denorm, TF2torch, TFresize_l, TFnormalize, TFinv_normalize
from morphable_model.visualizations.morphable_model_300W import MorphabelModel300W
from morphable_model.local_deformations.local_shape_loader import LocalShapeLoader


#SINGLE_MODE = False
SINGLE_MODE = True
#CREATE_GRID = True
CREATE_GRID = False
#SYSTEM = 'IBT128'
SYSTEM = 'NOTEBOOK'

if SYSTEM == 'NOTEBOOK':
    LOCAL_SHAPE_MOD_CSV = '/home/host/projects/models/shape_vectors/BFM200_local_deformation_vectors/BFM200_local_deformation_vectors.csv'
    MODEL_PATH = '/home/host/projects/pilot_heika/thesis/tools/gi_results/run48_460k/ckpt_G_000044_0000460000.tar'
else:
    LOCAL_SHAPE_MOD_CSV = r'C:\Users\ra321wn\repos\pilot_heika\morphable_model\local_deformations\BFM200_local_deformation_vectors\BFM200_local_deformation_vectors.csv'
    MODEL_PATH = None
shape_mod_loader = LocalShapeLoader(LOCAL_SHAPE_MOD_CSV)
m = MorphabelModel300W(SYSTEM)

class Render3DMM:

    def __init__(self,
                 model,
                 image_size=(450, 450, 3)):
        self.model = model
        self.image_size = image_size

    def load_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        self.alpha = mat['Shape_Para']
        self.exp = mat['Exp_Para']

        # phi, gamma, theta, tx, ty, tz, f
        pose = mat['Pose_Para']
        self.angles = pose[0, :3]
        self.t3d = pose[0, 3:6]
        self.t3d = self.t3d[:, None]
        self.f = pose[0, 6]

    def get_alpha(self):
        return np.copy(self.alpha)

    def render_pncc(self, alpha):
        vertices = self.model.reconstruct_vertex(alpha=alpha,
                                                alpha_exp=self.exp,
                                                angles=self.angles,
                                                t3d=self.t3d,
                                                f=self.f)
        pncc_feature = cpncc_v3([vertices,],
                                self.model.tri,
                                img_size=self.image_size)  # cython version

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

    def apply_shape_mod(self,
                        alpha,
                        shape_mod,
                        multiplier,
                       ):
        mod = shape_mod_loader.load_vector(shape_mod)
        f_min, f_max = shape_mod_loader.get_factor_range(shape_mod)
        if multiplier == 'n':
            factor = f_min
        elif multiplier == 'p':
            factor = f_max
        elif multiplier is None:
            factor = 0
        else:
            raise Exception(f'Given param MULTIPLIER not valid: {multiplier}')
        alpha = alpha + mod[:, None] * factor

        return alpha

def convert2tensor(img_path, pncc_path):
    img = read_img(img_path)
    img = TFresize_l(img)
    img = TF2torch(img)
    img = TFnormalize(img)

    pncc, mask = read_pncc(pncc_path)

    return img[None, :], pncc[None, :]

def load_model(model_path, model_name):
    class G_Wrapper(nn.Module):
        def __init__(self, G):
            super().__init__()
            self.G = G
        def forward(self, x, p):
            p, mask = p
            x = torch.cat((x, p), dim=1)
            return self.G(x)
    if model_name  == 'DenseUnet':
        net = UnetGenerator(6, 3, 4, 64) # with BN, no DO
        net = G_Wrapper(net)
    elif model_name == 'DenseNet57':
        net = tiramisu.FCDenseNet57(n_classes=5)
    else:
        raise NotImplementedError(f'Given gi_net not implemented {self.gi_net}')

    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['model_state_dict'])

    return net.cuda()

def predct_inmod(model, mm, alpha, img_path):
    pncc = mm.render_pncc(alpha)
    cv2.imwrite('tmp.png',
                pncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
    img, pncc = convert2tensor(img_path, 'tmp.png')
    # I am too lazy to rewrite the preprocessing 
    os.remove('tmp.png')
    img = img.cuda()
    pncc = pncc.cuda()
    out = G(img, [pncc, ...])
    img = denorm(img[0])
    out = denorm(out[0])
    return out, img

def create_mod_grid(img_path, mat_path, model):
    multipliers = ['n', None, 'p']
    mm = Render3DMM(m)
    mm.load_mat(mat_path)

    grid_list = list()
    for m1 in multipliers:
        alpha = mm.get_alpha()
        alpha = mm.apply_shape_mod(alpha, 'chin_1', m1)
        for m2 in multipliers:
            alpha_mod = np.copy(alpha)
            alpha_mod = mm.apply_shape_mod(alpha_mod, 'nose_1', m2)

            inmod, ori = predct_inmod(model, mm, alpha_mod, img_path)
            grid_list.append(inmod)

    return grid_list, ori


if SINGLE_MODE:
    ### parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('img_path')
    parser.add_argument('--model', default=MODEL_PATH, type=str)
    parser.add_argument('-o','--output_dir', type=str)
    args = parser.parse_args()

    img_path = os.path.abspath(args.img_path)
    mat_path = os.path.abspath(args.img_path[:-4] + '.mat')
    model_path = os.path.abspath(args.model)
else:
    ### parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path')
    parser.add_argument('--model', default=MODEL_PATH, type=str)
    parser.add_argument('-o','--output_dir', type=str)
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)
    model_path = os.path.abspath(args.model)

if args.output_dir:
    dir_path = os.path.abspath(args.output_dir)
else:
    dir_path = os.getcwd()
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)


G = load_model(model_path, 'DenseUnet')
# eval is very important if network has bn layers
G = G.eval()

if SINGLE_MODE:
    grid_list, ori = create_mod_grid(img_path, mat_path, model=G)
    if CREATE_GRID:
        grid_list = vutils.make_grid(grid_list, nrow=3, normalize=True, range=(0,1))
        out_name = os.path.join(dir_path,
                                os.path.basename(img_path)[:-4] + '_grid.png')
        vutils.save_image(grid_list, out_name, nrow=3, padding=0)
    else:
        out_name = os.path.join(dir_path,
                            os.path.basename(img_path)[:-4] + f'_ori.png')
        vutils.save_image(ori, out_name, padding=0)
        for l, img in enumerate(grid_list):
            out_name = os.path.join(dir_path,
                                os.path.basename(img_path)[:-4] + f'_combinations_{l}.png')
            vutils.save_image(img, out_name, padding=0, normalize=True)


else:
    samples = [(os.path.join(dataset_path, f),
                os.path.join(dataset_path, f[:-4] + '.mat')) for f in os.listdir(dataset_path)
               if '.jpg' in f]
    for img_path, mat_path in tqdm(samples):
        grid_list = create_mod_grid(img_path, mat_path, model=G)
        out_name = os.path.join(dir_path,
                                os.path.basename(img_path)[:-4] + '_grid.png')
        vutils.save_image(grid_list, out_name, nrow=3, padding=0)



