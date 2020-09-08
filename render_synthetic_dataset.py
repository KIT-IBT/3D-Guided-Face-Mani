import cv2
from PIL import Image
import os
import numpy as np
import torchvision
from tqdm import tqdm

from morphable_model.local_shape_loader import LocalShapeLoader
from morphable_model.random_bfm_sampler import BFMSampler
from morphable_model.morphable_model_300W import MorphabelModel300W
from morphable_model.utils import crender_colors, cpncc_v3, render_diff
from submodules.face3d.face3d import mesh
from submodules.face3d.face3d.morphable_model import MorphabelModel
from submodules.PyTorch3DDFA.utils.ddfa import reconstruct_vertex

LOCAL_SHAPE_MOD_CSV = 'data/configs/shape_mods/BFM200_local_deformation_vectors.csv'
DATASET_PATH = 'data/datasets/synthetic'
BG_IMAGE_DIR = 'data/datasets/Images'
TEXTURE_BFM_PATH = 'data/configs/BFM.mat'

class DatasetRenderer():

    def __init__(self,
                 model='BFM200',
                 width=450,
                 height=450,
                ):

        self.uniform_sampler = BFMSampler()


        assert model in {'BFM40', 'BFM200'}, f'Given param MODEL not supported: {model}'
        self.model = model

        self.bfm = MorphabelModel300W()
        self.tri = self.bfm.tri.astype(dtype=np.int32)

        self.tex_model = MorphabelModel(TEXTURE_BFM_PATH)

        self.shape_mod_loader = LocalShapeLoader(LOCAL_SHAPE_MOD_CSV)

        self.width = width
        self.height = height

        self.bg_images = list()
        for r, d, f in os.walk(BG_IMAGE_DIR):
            for file in f:
                if '.jpg' in file or 'png' in file:
                    self.bg_images.append(os.path.join(r,
                                                       file))
        self.crop = torchvision.transforms.CenterCrop((self.width,
                                                       self.height))
        self.resize = torchvision.transforms.Resize((self.width,
                                                     self.height))


    def load_random_bg(self):
        img_path = np.random.choice(self.bg_images)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
            w, h = img.size
            if not (w >= self.width and h >= self.height):
                img = self.resize(img)
            img = self.crop(img)
            img = np.array(img)
            # Sample again if greyvalue img or rgb with alpha channel
            if len(img.shape) == 3 and img.shape[2] == 3:
                return img
            else:
                return self.load_random_bg()


    def create_dataset(self,
                       dataset_path,
                       num_samples,
                       shape_mod=None,
                       add_light=True,
                       white_bg=False,
                       image_format='.png'
                      ):

        assert image_format in {'.png', '.jpg'}, f'Given param IMAGE_FORMAT is not supported {image_format}!'

        step = 0
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        else:
            step_list = [f for f in os.listdir(dataset_path)
                         if f'sample' in f]
            step = max(int(x[7:13]) for x in step_list)
            print(f'Resuming at step {step}!')

        self.dir = os.path.abspath(dataset_path)

        if shape_mod is not None:
            assert shape_mod in self.shape_mod_loader.get_vector_names(), 'Given param SHAPE_MOD not found: {shape_mod}!'

        print(f'Creating {num_samples} samples in {self.dir}')


        for s in tqdm(range(step, num_samples)):
            sample_name = 'sample_' + str(s).zfill(6)
            sample_name = os.path.join(self.dir, sample_name)

            alpha, tex = self.uniform_sampler.get_random_face()
            exp = self.uniform_sampler.get_random_expression()
            s, angles, t = self.uniform_sampler.get_random_transform()

            alpha = alpha[:, None]
            tex = tex[:, None]
            exp = exp[:, None]
    #            angles = angles[:, None]
            t = t[:, None]

            light_intensity = np.random.uniform(0.9, 1.1)

            bg = self.load_random_bg()
            if white_bg:
                bg = bg*0 + 255

            self._render_and_save_image(alpha,
                                        exp,
                                        tex,
                                        s,
                                        angles,
                                        t,
                                        sample_name+image_format,
                                        light_intensity=light_intensity,
                                        add_light=add_light,
                                        background=bg.copy(),
                                       )

            if shape_mod is not None:
                multiplier = np.random.choice(['n', 'p'])
                alpha_mod, mod_vector, factor = self._get_shape_mod(alpha,
                                                                    multiplier,
                                                                    shape_mod)
                sample_name= f'{sample_name}_pncc_mod_{shape_mod}_{multiplier}'
                self._render_and_save_image(alpha_mod,
                                            exp,
                                            tex,
                                            s,
                                            angles,
                                            t,
                                            sample_name+image_format,
                                            light_intensity=light_intensity,
                                            add_light=add_light,
                                            background=bg,
                                            alpha_ori=alpha,
                                            render_diff_map=True
                                           )


    def _render_and_save_image(self,
                               alpha, exp, tex,
                               s, angles, t,
                               image_path,
                               light_intensity,
                               add_light=True,
                               background=None,
                               alpha_ori=None,
                               render_diff_map=False
                              ):
        vertices = self.bfm.reconstruct_vertex(alpha=alpha,
                                               alpha_exp=exp,
                                               angles=angles,
                                               t3d=t,
                                               f=s,
                                               img_size=(self.width,
                                                         self.height, 3)
                                              )

        colors = self.tex_model.generate_colors(tex)
        colors_max = np.max(colors)
        colors_min = np.min(colors)
        if colors_max <= 1:
            colors_max = 1
        if colors_min > 0:
            colors_min = 0
        colors = (colors - colors_min) / (colors_max - colors_min)

        if add_light:
            v = vertices.copy().T
            v[:, 1] = self.height + 1 - v[:, 1]
            a = light_intensity
            light_positions = np.array([[100,0,1e6]])
            light_intensities = np.array([[a, a, a]])
            colors = mesh.light.add_light(v,
                                          self.tri.T,
                                          colors,
                                          light_positions,
                                          light_intensities)
            colors = colors.T
        colors = colors * 255

        image = crender_colors(vertices.T, self.tri.T,
                               colors.T,
                               self.width,
                               self.height, c=3)
        pncc_feature = cpncc_v3([vertices.copy(), ],
                                self.tri,
                                img_size=(self.width, self.height, 3))

        ### Remove all pixels that are not connected to the biggest blob
        # First prepare binary mask
        mask = pncc_feature.astype(np.float32)
        grey_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (grey_img > 0).astype(np.uint8)
        # Find connected pixels in mask
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Remove background from list
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        # Set not connected pixels to zero e.g. back part of the ear
        min_size = max(sizes)
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                pncc_feature[output == i + 1] = 0
        ###

        cv2.imwrite(image_path[:-4] + '_pncc.png', pncc_feature[:, :, ::-1])

        if background is not None:
            background[np.max(pncc_feature, axis=2) > 0, :] = 0.0
            image = image + background
        cv2.imwrite(image_path, image[:, :, ::-1])

        if render_diff_map:
            assert alpha_ori is not None, 'Must provide original alpha without modification to render diff_map!'
            diff_img = render_diff(alpha-alpha_ori, alpha_ori,
                                   exp, angles, t, s)
            cv2.imwrite(image_path[:-4] + '_diff.png',
                        diff_img[:, :, ::-1])



    def _get_shape_mod(self, alpha, multiplier, shape_mod):

        mod = self.shape_mod_loader.load_vector(shape_mod)
        f_min, f_max = self.shape_mod_loader.get_factor_range(shape_mod)
        if multiplier == 'n':
            factor = f_min
        elif multiplier == 'p':
            factor = f_max
        else:
            raise Exception(f'Given param MULTIPLIER not valid: {multiplier}')


        alpha_mod = alpha.copy()
        alpha_mod[:, 0] += mod * factor

        return alpha_mod, mod, factor


r = DatasetRenderer(model='BFM200')

r.create_dataset(DATASET_PATH + os.sep + 'synthetic_chin',
                 50000, 'chin_1',
                 add_light=True,
                 image_format='.png')
r.create_dataset(DATASET_PATH + os.sep + 'synthetic_nose',
                 50000, 'nose_1',
                 add_light=True,
                 image_format='.png')
r.create_dataset(DATASET_PATH + os.sep + 'synthetic_chin_white_bg',
                 10000, 'chin_1',
                 add_light=True,
                 white_bg=True,
                 image_format='.png')
r.create_dataset(DATASET_PATH + os.sep + 'synthetic_nose_white_bg',
                 10000, 'nose_1',
                 add_light=True,
                 white_bg=True,
                 image_format='.png')
