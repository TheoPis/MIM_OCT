import json
import os
from torch import Tensor
from typing import Union
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image, ImageFile
from utils import DATASETS_INFO, remap_mask, printlog, create_new_directory
import numpy as np
import pathlib
from glob import glob
import random
from typing import List, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_files_with_suffix(directory, suffix='.png'):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                png_files.append(os.path.join(root, file))
    return png_files


class OCT5K(Dataset):
    valid_splits = ['train', 'val', 'all']

    splits = {'train': ['AMD (48).E2E', 'AMD (35).E2E', 'DME (6).E2E', 'Normal (9).E2E', 'DME (4).E2E',
                        'Normal (38).E2E', 'Normal (13).E2E', 'Normal (26).E2E', 'AMD (40).E2E', 'Normal (47).E2E',
                        'AMD (9).E2E', 'DME (17).E2E', 'Normal (25).E2E', 'DME (9).E2E', 'AMD (22).E2E', 'DME (31).E2E',
                        'DME (30).E2E', 'Normal (35).E2E', 'DME (28).E2E', 'Normal (40).E2E', 'DME (22).E2E',
                        'AMD (31).E2E', 'Normal (39).E2E', 'Normal (34).E2E', 'DME (15).E2E', 'AMD (2).E2E',
                        'DME (11).E2E', 'AMD (21).E2E', 'DME (33).E2E', 'AMD (10).E2E', 'AMD (26).E2E', 'AMD (24).E2E',
                        'DME (19).E2E', 'DME (26).E2E', 'AMD (11).E2E', 'Normal (7).E2E', 'AMD (3).E2E',
                        'Normal (36).E2E', 'AMD (28).E2E', 'AMD (23).E2E', 'DME (14).E2E', 'Normal (15).E2E',
                        'Normal (14).E2E', 'AMD (27).E2E', 'Normal (6).E2E'],

              'val': ['Normal (12).E2E', 'DME (29).E2E', 'DME (7).E2E', 'Normal (27).E2E', 'DME (21).E2E',
                      'DME (37).E2E', 'AMD (34).E2E', 'Normal (49).E2E', 'Normal (37).E2E', 'Normal (10).E2E',
                      'AMD (29).E2E', 'AMD (1).E2E', 'AMD (13).E2E', 'AMD (8).E2E', 'DME (35).E2E']
              }

    def __init__(self, root,
                 transforms_dict: dict,
                 img_channels: int = 1,
                 split: Union[str, list] = 'train',
                 return_metadata=True,
                 debug=False):

        super(OCT5K, self).__init__()
        self.root = root
        self.experiment = 1  # hardcoded
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        assert (split in self.valid_splits), f'split {split} is not in valid_modes {self.valid_splits}'
        self.split = split  # "train", "test", "val"
        self.debug = debug
        self.img_channels = img_channels
        self.dataset = 'OCT5K'
        self.img_suffix = '.png'
        self.target_suffix = '.png'
        self.imgs_dir = os.path.join(self.root, 'Images', 'Images_Manual')
        self.masks_dir = os.path.join(self.root, 'Masks', 'Masks_Manual', 'Grading_1')
        self.pathologies = ['AMD', 'DME', 'Normal']

        paths_to_pathology_dirs = [os.path.join(self.imgs_dir, f) for f in os.listdir(self.imgs_dir)]
        paths_to_pathology_dirs_masks = [os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir)]
        assert len(paths_to_pathology_dirs) == len(paths_to_pathology_dirs_masks),\
            f'different number of pathology dirs for images and masks {len(paths_to_pathology_dirs)} ' \
            f'{len(paths_to_pathology_dirs_masks)}'

        #################################### obtain dataset metadata ###################################################
        # the dict below is useful for stratified splitting of the dataset
        # pathology_to_subjects_to_files[pathology][subject_folder] = [(path_to_img1, path_to_mask1), ...]
        self.pathology_to_subjects_to_files = {p: [] for p in self.pathologies}
        self.subjects = []

        # data is stored in the following way:
        # $root/Images/Images_Manual/$pathology_dir/$subject_folder/*.png
        # $root/Masks/Masks_Manual/Grading_1/$pathology_dir_mask/$subject_folder/*.png
        # note: there are annotations by 3 graders but we focus on Grading_1 for now.

        for p in self.pathologies:
            self.pathology_to_subjects_to_files[p] = {}
            for pathology_dir, pathology_dir_mask in zip(paths_to_pathology_dirs, paths_to_pathology_dirs_masks):
                if p in pathology_dir:  # e.g AMD in "AMD (X).E2E" or DME in "DME (X).E2E"
                    assert p in pathology_dir_mask, f'pathology {p} not found in mask dir {pathology_dir_mask}'
                    for subject_folder in os.listdir(pathology_dir):
                        self.subjects.append(subject_folder)
                        if subject_folder not in self.pathology_to_subjects_to_files[p]:
                            # use glob to get any file with the suffix inside subect_folder
                            paired_img_label = list(zip(find_files_with_suffix(
                                os.path.join(pathology_dir, subject_folder),
                                self.img_suffix),
                                find_files_with_suffix(os.path.join(pathology_dir_mask, subject_folder),
                                                       self.target_suffix)))

                            self.pathology_to_subjects_to_files[p][subject_folder] = paired_img_label

        #################################### apply pre-specified split #################################################
        self.images = []
        self.targets = []

        if self.split != 'all':
            # get only the subjects in the split
            for pathology in self.pathologies:
                for subject_folder in self.pathology_to_subjects_to_files[pathology].keys():
                    if subject_folder in self.splits[self.split]:
                        for img, mask in self.pathology_to_subjects_to_files[pathology][subject_folder]:
                            self.images.append(img)
                            self.targets.append(mask)
        else:
            # get everything
            for pathology in self.pathologies:
                for subject_folder in self.pathology_to_subjects_to_files[pathology].keys():
                    for img, mask in self.pathology_to_subjects_to_files[pathology][subject_folder]:
                        self.images.append(img)
                        self.targets.append(mask)

        printlog(f'OCT5K data all found split = {self.split}, images {len(self.images)}, targets {len(self.targets)}'
                 f' image_channels: {self.img_channels}')

        self.return_metadata = return_metadata
        self.return_filename = False
        self.store = False

    def __getitem__(self, index) -> Union[Tuple[Tensor, Tensor, dict], Tuple[Tensor, Tensor]]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        target = remap_mask(np.array(target), DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0],
                            to_network=True).astype('int32')
        target = Image.fromarray(target)
        # print(np.unique(target).tolist()) # to make sure remap_mask works see this

        if self.return_metadata:
            metadata = {'index': index,
                        'img_filename': self.images[index],
                        'target_filename': self.targets[index],
                        'subject_name': pathlib.Path(self.targets[index]).parent.parent.name}
            if 'AMD' in metadata['img_filename']:
                metadata['pathology'] = 'AMD'
            elif 'DME' in metadata['img_filename']:
                metadata['pathology'] = 'DME'
            elif 'Normal' in metadata['img_filename']:
                metadata['pathology'] = 'Normal'

        image, target, metadata = self.common_transforms((image, target, metadata))
        img_tensor = self.img_transforms(image)
        lbl_tensor = self.lbl_transforms(target).squeeze()

        if self.debug:
            self.visualize_image_mask_colormap(img_tensor, lbl_tensor, index)

        if self.img_channels == 1:
            img_tensor = img_tensor[:, 0].unsqueeze(1)
        if self.return_metadata:
            return img_tensor, lbl_tensor, metadata
        else:
            return img_tensor, lbl_tensor

    def __len__(self):
        return len(self.images)

    def visualize_image_mask_colormap(self, img_tensor, lbl_tensor, index):
        """ for debugging purposes"""
        from utils import mask_to_colormap, get_remapped_colormap
        ToPILImage()(img_tensor).show()
        color_lbl = mask_to_colormap(to_numpy(lbl_tensor),
                                     get_remapped_colormap(
                                         DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0],
                                         self.dataset),
                                     from_network=True, experiment=self.experiment,
                                     dataset=self.dataset)
        Image.fromarray(color_lbl).show()
        print(f'\nafter aug index: {np.unique(lbl_tensor)} '
              f'lbl {lbl_tensor.shape} image {img_tensor.shape} fname:{self.images[index]},'
              f'lbl unique values: {np.unique(lbl_tensor)}\n')


if __name__ == '__main__':
    import pathlib
    import torch
    from utils import parse_transform_lists
    import json
    import cv2
    from torch.nn import functional as F
    from utils import Pad, RandomResize, RandomCropImgLbl, Resize, FlipNP, to_numpy, pil_plot_tensor, to_comb_image
    from torchvision.transforms import ToTensor
    import PIL.Image as Image
    from utils import printlog, to_numpy, to_comb_image, un_normalise

    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\OCT5K\\'
    d = {"dataset": 'OCT5K', "experiment": 1}
    path_to_config = '../configs/OCT5K/vitd.json'

    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']

    transforms_dict_ = parse_transform_lists(transforms_list, transforms_values, **d)
    transforms_list_val = config['data']['transforms_val']
    transforms_values_val = config['data']['transform_values_val']

    if 'torchvision_normalise' in transforms_list_val:
        del transforms_list_val[-1]
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, **d)

    train_set = OCT5K(root=data_path,
                      debug=True,
                      split='train',
                      transforms_dict=transforms_dict_)

    val_set = OCT5K(root=data_path,
                    debug=True,
                    split='val',
                    transforms_dict=transforms_dict_)

    for i in val_set:
        print(i)