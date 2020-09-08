"""
crender_colors, cpncc_v3 modified from:
- https://raw.githubusercontent.com/YadiraF/PRNet/master/utils/render.py
- https://github.com/cleardusk/3DDFA/blob/master/utils/render.py
"""

import cv2
import numpy as np
from submodules.PyTorch3DDFA.utils.cython import mesh_core_cython

from submodules.PyTorch3DDFA.utils.ddfa import reconstruct_vertex
from morphable_model.morphable_model_300W import MorphabelModel300W
from submodules.face3d.face3d import mesh


BFM_model = MorphabelModel300W()
pncc_code = np.load('submodules/PyTorch3DDFA/train.configs/pncc_code.npy')


def crender_colors(vertices, triangles, colors, h, w, c=3,
                   BG=None, ignore_backside=True, invert_normals=False):
    """ render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3]
        h: height
        w: width
        c: channel
        BG: background image
        ignore_border: render only vertices with normals pointing towards the camera
    Returns:
        image: [h, w, c]. rendered image./rendering.
    """

    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG.astype(np.float32).copy(order='C')
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # to C order
    vertices = vertices.astype(np.float32).copy(order='C')
    triangles = triangles.astype(np.int32).copy(order='C')
    colors = colors.astype(np.float32).copy(order='C')
    normals  = np.zeros_like(vertices).astype(np.float32).copy(order='C')
    if ignore_backside:
        """
             Calculate scalar product of vertex normal vector and the vector
             <p: vertex - q: distant point behind the camera>. If negative, keep
             color of vertex. If positive, set color to 0
        """
        mesh_core_cython.get_normal(normals, vertices, triangles,
                                    vertices.shape[0], triangles.shape[0])
        q = [0, 0, 10000000]
        for l in range(normals.shape[0]):
            p = vertices[l]
            dot = np.dot(p - q, normals[l])
            if invert_normals:
                dot *= -1
            colors[l] = colors[l] if dot <= 0 else [0.0, 0.0, 0.0]


    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, c
    )
    return image

def cpncc_v3(vertices_lst, tri, img_size,
             ignore_backside=True,
             invert_normals=False):
    """cython version for PNCC render: original paper"""
    h, w, c = img_size

    pnccs_img = np.zeros((h, w, c))
    for i in range(len(vertices_lst)):
        vertices = vertices_lst[i]

        pncc_img = crender_colors(vertices.T, tri.T, pncc_code.T, h, w, c,
                                  ignore_backside=ignore_backside,
                                  invert_normals=invert_normals)
        pnccs_img[pncc_img > 0] = pncc_img[pncc_img > 0]

    pnccs_img = pnccs_img.squeeze() * 255
    return pnccs_img

def render_diff(alpha_mod, alpha, alpha_exp, angles, t3d, f, image_size=(450, 450, 3)):
    vertices_mod = BFM_model.w @ alpha_mod
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
                                            img_size=image_size
                                           )
    img = crender_colors(vertices.T, BFM_model.tri.T, colors.T,
                         450,
                         450, c=3)
    img = img[:, :, 0, None]
    return img



def render_pncc(alpha, alpha_exp, angles, t3d, f, image_size=(450, 450, 3)):

    vertices = BFM_model.reconstruct_vertex(alpha=alpha,
                                            alpha_exp=alpha_exp,
                                            angles=angles,
                                            t3d=t3d,
                                            f=f)
    pncc_feature = cpncc_v3([vertices,],
                            BFM_model.tri,
                            img_size=image_size)  # cython version

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
