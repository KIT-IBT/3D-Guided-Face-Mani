import torch
import numpy as np
import scipy.io as sio
from submodules.face3d.face3d import mesh

class MorphabelModel300W():

    def __init__(self, use_torch=False):

        self.use_torch = use_torch

        bfm_model = sio.loadmat('data/configs/Model_Shape.mat')
        exp_model = sio.loadmat('data/configs/Model_Expression.mat')
        self.tri = sio.loadmat('data/configs/tri.mat')['tri']

        self.tri = self.tri - 1

        self.mu = bfm_model['mu_shape'] + exp_model['mu_exp']
        self.w = bfm_model['w']
        self.sigma = bfm_model['sigma']
        self.w_exp = exp_model['w_exp']
        self.sigma_exp = exp_model['sigma_exp']

        self.rot_matrix = mesh.transform.angle2matrix_3ddfa


    def reconstruct_vertex(self, alpha, alpha_exp, angles, t3d, f,
                           transform=True,
                           img_size=(450, 450, 3)):

        if len(alpha.shape) == 1:
            alpha = alpha[:, None]
        if len(alpha_exp.shape) == 1:
            alpha_exp = alpha_exp[:, None]

        # Apply PCA model
        vertices = self.mu + self.w @ alpha + self.w_exp @ alpha_exp
        vertices = vertices.reshape(3, -1, order='F')

        if transform:
            vertices = self.transform_face(vertices, angles, t3d, f,
                                           img_size)

        if self.use_torch:
            vertices = torch.from_numpy(vertices)

        return vertices


    def transform_face(self, vertices,
                       angles, t3d, f,
                       img_size=(450, 450, 3)):
        if len(t3d.shape) == 1:
            t3d = t3d[:, None]
        R = self.rot_matrix(angles)
        h, w, c = img_size

        # Transform vertices (Scale, Rotation and Translation)
        vertices = f * R @ vertices  + t3d

        # Convert to image space
        vertices[1, :] = h + 1 - vertices[1, :]

        return vertices


    def coef2mesh(self, alpha, alpha_exp=None, model=None):


        if type(alpha) is not np.ndarray:
            alpha = alpha.detach().numpy()
        if type(alpha_exp) is not np.ndarray and alpha_exp is not None:
            alpha_exp = alpha_exp.detach().numpy()

        if len(alpha.shape) == 1:
            alpha = alpha[:, None]
        if len(alpha_exp.shape) == 1:
            alpha_exp = alpha_exp[:, None]

        if alpha_exp is None:
            vertices = self.mu + self.w @ alpha
        else:
            vertices = self.mu + self.w @ alpha + self.w_exp @ alpha_exp

        vertices = vertices.reshape(3, -1, order='F')

        if self.use_torch:
            vertices = torch.from_numpy(vertices)

        return (np.swapaxes(vertices, 0, 1),
                np.swapaxes(self.tri, 0, 1))
