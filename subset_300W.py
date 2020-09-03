import os
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F

TF2torch = torchvision.transforms.ToTensor()
TFRotate = F.rotate
TFcrop = F.crop
TFresize_b = torchvision.transforms.Resize((128, 128))
TFresize_n = torchvision.transforms.Resize((128, 128), Image.NEAREST)
TFresize_l = torchvision.transforms.Resize((128, 128), Image.LANCZOS)
TF2pil = torchvision.transforms.ToPILImage()
TFnormalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5])
TFinv_normalize = torchvision.transforms.Normalize(
                                                mean=[-1, -1, -1],
                                                std=[2, 2, 2])

def denorm(x):
    x = (x + 1) / 2
    return x.clamp_(0, 1)


def read_img(img_path, rgb=True):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            img.convert('RGB')
        # top, left, height, width
        img = TFcrop(img, 120, 75, 310, 300)

        return img

def read_grey(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        # top, left, height, width
        img = TFcrop(img, 120, 75, 310, 300)

        return img

def read_pncc(pncc_path, rotation=None):

    target = read_img(pncc_path)

    seg_mask = target.copy()
    if rotation is not None:
        target = TFRotate(target, rotation, resample=Image.BICUBIC)

    target = TF2torch(TFresize_l(target))

    seg_mask = np.max(np.array(seg_mask), axis=2)

    seg_mask = (seg_mask > 0).astype(np.float32)
    seg_mask = (seg_mask * 255).astype(np.uint8)
    seg_mask = Image.fromarray(seg_mask)

    if rotation is not None:
        seg_mask = TFRotate(seg_mask,
                            rotation,
                            resample=Image.BICUBIC)

    seg_mask = TFresize_l(seg_mask)
    mask = ((np.array(seg_mask) / 255) > 0.9).astype("long")

    mask = torch.from_numpy(mask).long()

    return target, mask


class Subset300W(torch.utils.data.Dataset):

    def __init__(self,
                 subset_path,
                 mod_names=['chin', 'nose'],
                 multipliers=['n', 'p'],
                 rotate_images=True
                ):

        self.rotate_images = rotate_images

        assert os.path.isdir(subset_path), f'{subset_path} is not a directory!'

        self.subset_path = os.path.abspath(subset_path)
        print('Load dataset: {}'.format(self.subset_path))

        sample_list = [s for s in os.listdir(self.subset_path)
                            if '.jpg' in s]
        sample_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        pncc_mod_list = list()
        diff_mod_list = list()
        for mod in mod_names:
            pncc_mod_list.append([s for s in os.listdir(self.subset_path)
                                  if f'_pncc_mod_{mod}' in s])
            diff_mod_list.append([s for s in os.listdir(self.subset_path)
                                  if f'_diff_mod_{mod}' in s])


        def find_match(string_list, substring, m):
            for n in string_list:
                if substring in n:
                    if f'_{m}.' in n:
                        return n
            else:
                raise Exception(f'Couldnt find matching file in the dataset for substring: {substring}')

        self.sample_list = list()
        for s in sample_list:
            for l, n in enumerate(mod_names):
                for m in multipliers:
                    shape = s[:-4] + '_pncc.png'
                    mod_list = pncc_mod_list[l]
                    mod = find_match(mod_list, shape[:-4], m)
                    diff_list = diff_mod_list[l]
                    diff = find_match(diff_list, shape[:-8], m)
                    self.sample_list.append([s, shape, mod, diff])


    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):

        paths = self.sample_list[idx]
        paths = [os.path.join(self.subset_path, s) for s in paths]
        img_path, shape_path, mod_path, diff_path = paths

        if self.rotate_images:
            rotation = np.random.uniform(-90, 90)
        else:
            rotation = 0.0

        img = read_img(img_path)
        img = F.rotate(img, rotation,
                       resample=Image.BICUBIC)
        img = TFresize_l(img)
        img = TF2torch(img)
        img = TFnormalize(img)

        pncc, mask = read_pncc(shape_path,
                               rotation=rotation)
        pncc_mod, mask_mod = read_pncc(mod_path,
                                       rotation=rotation)

        diff = read_img(diff_path)
        diff = F.rotate(diff, rotation,
                        resample=Image.BICUBIC)
        diff = TFresize_l(diff)
        diff = TF2torch(diff)

        return img, [pncc, mask], [pncc_mod, mask_mod], diff
