import os
import cv2
import scipy.io as sio
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from morphable_model.utils import render_diff, render_pncc
from morphable_model.local_shape_loader import LocalShapeLoader
from morphable_model.morphable_model_300W import MorphabelModel300W
from subset_300W import read_pncc, read_img, denorm, TF2torch, TFresize_l, TFnormalize, TFinv_normalize
from model import UnetGenerator


LOCAL_SHAPE_MOD_CSV = 'data/configs/shape_mods/BFM200_local_deformation_vectors.csv'
MODEL_PATH = 'data/configs/ckpt_G_000042_0000900000.tar'
shape_mod_loader = LocalShapeLoader(LOCAL_SHAPE_MOD_CSV)
m = MorphabelModel300W()

class Render3DMM:
    """Class to modify the BFM and render the PNCC"""
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
        return render_pncc(alpha, self.exp,
                           self.angles, self.t3d, self.f)

    def apply_shape_mod(self,
                        alpha,
                        shape_mod,
                        multiplier,
                       ):
        mod = shape_mod_loader.load_vector(shape_mod)
        f_min, f_max = shape_mod_loader.get_factor_range(shape_mod)
        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        if multiplier == 'n':
            factor = f_min
        elif multiplier == 'p':
            factor = f_max
        elif multiplier is None:
            factor = 0
        elif isfloat(multiplier):
            factor = float(multiplier)
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
    if model_name  == 'DenseUnet':
        class G_Wrapper(nn.Module):
            def __init__(self, G):
                super().__init__()
                self.G = G
            def forward(self, x, p):
                p, mask = p
                x = torch.cat((x, p), dim=1)
                return self.G(x)
        net = UnetGenerator(6, 3, 4, 64)
        net = G_Wrapper(net)
    else:
        raise NotImplementedError(f'Given gi_net not implemented {self.gi_net}')

    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['model_state_dict'])

    return net.cuda()

def predct_inmod(model, mm, alpha, img_path, return_pncc=False):
    # Write PNCC
    pncc_img = mm.render_pncc(alpha)
    cv2.imwrite('tmp.png',
                pncc_img[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
    img, pncc = convert2tensor(img_path, 'tmp.png')
    # I am too lazy to rewrite the preprocessing 
    os.remove('tmp.png')
    img = img.cuda()
    pncc = pncc.cuda()
    out = G(img, [pncc, ...])
    img = denorm(img[0])
    out = denorm(out[0])
    if return_pncc:
        return out, img, pncc_img
    else:
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

############################################
# Parsing 
############################################
parser = argparse.ArgumentParser(description='')
parser.add_argument('path', help="Either path to image.jpg or a directory containing both *.jpg and *.mat")
parser.add_argument('--model', default=MODEL_PATH, type=str, help="alternative path to model weights")
parser.add_argument('-o','--output_dir', type=str)
parser.add_argument('--no_grid', action="store_true", help="If the output should be singe images instead of an image grid")
parser.add_argument('-mod', '--modifier', type=str, help="Choose between 'chin_1' and 'nose_1'. If not set, a grid is created with all combinations of the shape modifications")
parser.add_argument('-mult', '--multiplier', type=str, help="Choose between 'p', 'n', or an integer value e.g. '400000'. If not set, a grid with all modifications (single and combined is created)" )
args = parser.parse_args()

# I did not consider cases here
if os.path.isdir(args.path):
    single_mode = False
    dataset_path = args.path
else:
    single_mode = True
    img_path = args.path
create_grid = not args.no_grid

single_modifier = False
if args.modifier and args.multiplier:
    create_grid = False
    single_modifier = True

if single_mode:
    img_path = os.path.abspath(img_path)
    mat_path = os.path.abspath(img_path[:-4] + '.mat')
    model_path = os.path.abspath(args.model)
else:
    dataset_path = os.path.abspath(dataset_path)
    model_path = os.path.abspath(args.model)

if args.output_dir:
    dir_path = os.path.abspath(args.output_dir)
else:
    dir_path = os.getcwd()
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

############################################
# Run inference 
############################################
G = load_model(model_path, 'DenseUnet')
G = G.eval()

if single_modifier:
    mm = Render3DMM(m)
    mm.load_mat(mat_path)
    alpha = mm.get_alpha()
    alpha_mod = mm.apply_shape_mod(alpha, args.modifier, args.multiplier)
    inmod, ori, pncc = predct_inmod(G, mm, alpha_mod, img_path, return_pncc=True)
    out_name = os.path.join(dir_path,
                            os.path.basename(img_path)[:-4] + f'{args.modifier}_{args.multiplier}')
    vutils.save_image(inmod, out_name + '_prediction.png')
    vutils.save_image(ori, out_name + '_input.png')
    cv2.imwrite(out_name + '_pncc.png',
                pncc[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
elif single_mode:
    grid_list, ori = create_mod_grid(img_path, mat_path, model=G)
    if create_grid:
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
        grid_list, _ = create_mod_grid(img_path, mat_path, model=G)
        out_name = os.path.join(dir_path,
                                os.path.basename(img_path)[:-4] + '_grid.png')
        vutils.save_image(grid_list, out_name, nrow=3, padding=0)
