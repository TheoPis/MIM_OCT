import json
import os
from typing import Union
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image, ImageFile
from utils import DATASETS_INFO, remap_mask, printlog, create_new_directory
import numpy as np
import pathlib
from utils import get_remapped_colormap

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RAVIR(Dataset):
    hardcoded_splits = {'hardcoded_train': ['11', '17', '19', '20', '21', '22', '23', '26', '34', '35', '36', '38',
                                            '40', '41', '43', '44', '49', '51', '52', ],
                        'hardcoded_val': ['53', '55', '56', '58']
                        }

    def __init__(self, root,
                 transforms_dict: dict,
                 img_channels: int = 1,
                 split: Union[str, list] = 'train', return_metadata=True, debug=False):

        super(RAVIR, self).__init__()
        self.root = root
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        # assert(mode in ("fine", "coarse"))
        valid_splits = ["train", "val", ['train', 'val'], 'hardcoded_train', 'hardcoded_val', 'all']
        assert (split in valid_splits), f'split {split} is not in valid_modes {valid_splits}'
        # self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.split = split  # "train", "test", "val"
        self.debug = debug
        # self.target_type = target_type
        self.images = []
        self.targets = []
        # this can only take the following values so hardcoded
        self.img_channels = img_channels
        self.dataset = 'RAVIR'
        self.experiment = 1
        self.img_suffix = '.png'
        self.target_suffix = '.png'
        if self.split == ['test']:
            raise NotImplementedError('test split not implemented yet')
        elif self.split == 'all':
            self.images_dir = os.path.join(self.root, 'train', 'training_images')
            self.targets_dir = os.path.join(self.root, 'train', 'training_masks')
            for image_filename in os.listdir(self.images_dir):
                img_path = os.path.join(self.images_dir, image_filename)
                # print(image_filename)
                target_path = os.path.join(self.targets_dir,
                                           image_filename.split(self.img_suffix)[-2] + self.target_suffix) #todo
                self.images.append(img_path)
                self.targets.append(target_path)
                assert (pathlib.Path(img_path).exists() and pathlib.Path(target_path).exists())
                assert (pathlib.Path(img_path).stem == pathlib.Path(target_path).stem)

        elif 'hardcoded' in self.split:
            self.images_dir = os.path.join(self.root, 'train', 'training_images')
            self.targets_dir = os.path.join(self.root, 'train', 'training_masks')
            split_ids = self.hardcoded_splits[self.split]
            for image_filename in os.listdir(self.images_dir):
                img_path = os.path.join(self.images_dir, image_filename)
                target_path = os.path.join(self.targets_dir,
                                           image_filename.split(self.img_suffix)[-2] + self.target_suffix)
                id_ = str(pathlib.Path(img_path).stem).split('_')[-1][1:]
                if id_ in split_ids:
                    self.images.append(img_path)
                    self.targets.append(target_path)
                    assert (pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists()) #todo
                    assert (pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem) #todo

        # label remap_array (workaroudn to using utils.remap_mask
        remap_array = np.full(256, 0, dtype=np.uint8)
        remap_array[128] = 1
        remap_array[255] = 2
        self.remap_mask = remap_array

        printlog(f'RAVIR data all found split = {self.split}, images {len(self.images)}, targets {len(self.targets)}'
                 f' image_channels: {self.img_channels}')

        self.return_metadata = return_metadata
        self.return_filename = False
        self.store = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        metadata = {'index': index}

        # workaround note for the remap_mask function  todo : update remap_mask
        # target = remap_mask(np.array(target),
        #                     DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], ignore_label=259,
        #                     to_network=True).astype('int32')
        if self.experiment == 1:
            target = np.array(target)
            if self.debug:
                print(f'before remap index: {np.unique(target)} lbl {target.shape} image {np.array(image).shape} fname:{self.images[index]},'
                      f'lbl unique values: {np.unique(target)}\n')
            target = self.remap_mask[target].astype('int32')
        else:
            raise NotImplementedError(f'experiment {self.experiment} not implemented')
        target = Image.fromarray(target)

        metadata.update({'img_filename': self.images[index],
                         'target_filename': self.targets[index]})

        image, target, metadata = self.common_transforms((image, target, metadata))
        img_tensor = self.img_transforms(image)
        lbl_tensor = self.lbl_transforms(target).squeeze()

        if self.debug:
            from utils import mask_to_colormap
            ToPILImage()(img_tensor).show()
            # ToPILImage()(lbl_tensor).show()
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

        if self.img_channels == 1:
            img_tensor = img_tensor[:, 0].unsqueeze(1)
        if self.return_metadata:
            return img_tensor, lbl_tensor, metadata
        else:
            return img_tensor, lbl_tensor

    def __len__(self):
        return len(self.images)


def find_roi_and_crop(x, show=False, verbose=False):
    # 1) thresholds oct image
    # 2) finds bounding box around foreground
    # 3) padds the bounding box to make it equal to a multiple of the input_stride of the network
    # input_stride = 2 ** self.network_settings['layers']
    ret, thresh1 = cv2.threshold(x, 90, 255, cv2.THRESH_BINARY)
    thresh1 = np.asarray(thresh1, dtype=np.uint8)
    c, r, _, h = cv2.boundingRect(thresh1)
    # _ = print(c, r, 512, h) if verbose else ''
    roi = x[r:r + h, :]
    _ = print('ROI: {}'.format(roi.shape)) if verbose else ''
    w = roi.shape[-1]
    return roi, r, h,


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

    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\RAVIR\\RAVIR Dataset\\'
    d = {"dataset": 'RAVIR', "experiment": 1}
    path_to_config = '../configs/AROI/vitd.json'

    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']

    tdict = parse_transform_lists(transforms_list, transforms_values, **d)
    transforms_list_val = config['data']['transforms_val']
    transforms_values_val = config['data']['transform_values_val']

    if 'torchvision_normalise' in transforms_list_val:
        del transforms_list_val[-1]
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, **d)

    train_set = RAVIR(root=data_path,
                      debug=True,
                      split='all',
                      transforms_dict=tdict)

    valid_set = RAVIR(root=data_path,
                      debug=True,
                      split='hardcoded_val',
                      transforms_dict=transforms_dict_val)

    # train_set.store = True
    issues = []
    train_set.return_filename = True
    hs = []
    ws = []
    for i, ret in enumerate(train_set):
        hs.append(ret[0].shape[1])
        ws.append(ret[0].shape[2])
        present_classes = torch.unique(ret[1])
        print(ret[-1])
        im = ret[0]
        pred = ret[1]
