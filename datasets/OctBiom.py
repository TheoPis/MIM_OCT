import h5py
import pathlib
import torch

import numpy as np
from typing import Union

import torchvision.transforms
from PIL import ImageFile, Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from utils import DATASETS_INFO, printlog

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OctBiom(Dataset):
    def __init__(self,
                 root,
                 hdf5_file,
                 transforms_dict: dict,
                 split: Union[str, list] = 'train',
                 img_channels=3,
                 debug=False,
                 using_minival=True):
        """
        - train:
            21,723 slices
            labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
            10 biomarkers labelled by BScan (1 if present, 0 otherwise). No position information.
            [5956 1028 4721 5199 4687 1829 5585 1031 2120 4218]
            All types of conditions

        -val (minival): (subset of train)
            1000 slices
            labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
            10 biomarkers labelled by BScan (1 if present, 0 otherwise). No position information.
            [252 110 202 256 147  91 288  48  85 246] positives in each class

        - test
            1,002 OCT BScans (496x512)
            labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
            10 biomarkers labelled by BScan (1 if present, 0 otherwise). No position information.
            AMD patients

        :param root: path to
        :param transforms_dict: see dataset_from_df.py
        :param split: any of "train", "test", "val"
        :param img_channels: number of channels in the image
        :param debug: if True, visualizes every image in the dataset using PIL
        :param using_minival: if True, uses a subset of 'train' split (fixed to be 1000 last images) as 'val'
                              when split='test' this is ignored and has not effect
        """
        super(OctBiom, self).__init__()
        self.root = root
        self.hdf5_file = hdf5_file

        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])

        valid_splits = ['train', 'val', 'test']

        assert (split in valid_splits), f'split {split} is not in valid_modes {valid_splits}'
        self.split = split  # "train", "test", "val"
        self.img_channels = img_channels

        # the following can only take the following values so hardcoded
        self.dataset = 'OctBiom'
        self.experiment = 0
        self.num_classes = len(DATASETS_INFO['OctBiom']['CLASS_INFO'][self.experiment][1])
        self.modality = 'OCT'
        self.mode = 'slices'

        self.image_set_name = "data/slices"
        self.slice_set_name = None
        self.label_set_name = 'data/markers'

        if self.split == 'train':
            self.hdf5_file = 'train.hdf5'
        elif self.split == 'test':
            self.hdf5_file = 'test.hdf5'
        elif self.split == 'val':
            assert using_minival, f'using_minival must be True (got {using_minival} for split {self.split}'
            self.hdf5_file = 'train.hdf5'
        else:
            raise ValueError(f'invalid split {self.split} not {valid_splits}')

        # specifying minival
        # i.e whether to use a subset of 'train' split (fixed to be 1000 last images) as 'val'
        self.using_minival = using_minival and not self.split == 'test'

        # load all data to memory
        self._reset_hdf5()

        self.dataset_len = self.image_set.shape[0]
        self.lbl_len = self.label_set.shape[0]
        assert self.dataset_len == self.lbl_len, \
            f' number of images [{self.dataset_len}] and labels [{self.lbl_len}] do not match'

        self.weights = self._get_weights_loss()

        printlog(
            f'{self.dataset} data found '
            f'split = {self.split}, images {self.__len__()}, targets {len(self.label_set)} '
            f'using minival {self.using_minival}')

        self.debug = debug
        self.return_filename = False

        self._normalize_min1_1 = torchvision.transforms.Normalize(mean=[0.5]*self.img_channels, std=[0.5]*self.img_channels)

    def _get_weights_loss(self):
        labels_sum = np.sum(self.label_set, axis=0)
        largest_class = max(labels_sum)
        weights = largest_class / labels_sum
        weights = torch.from_numpy(weights)
        return weights

    def _reset_hdf5(self):
        root = pathlib.Path(self.root)
        if self.using_minival:
            if self.split == 'val':
                self.image_set = h5py.File(root / self.hdf5_file, 'r')[self.image_set_name][-1000:]
                self.label_set = h5py.File(root / self.hdf5_file, 'r')[self.label_set_name][-1000:]

            elif self.split == 'train':
                self.image_set = h5py.File(root / self.hdf5_file, 'r')[self.image_set_name][:-1000]
                self.label_set = h5py.File(root / self.hdf5_file, 'r')[self.label_set_name][:-1000]
        else:
            self.image_set = h5py.File(root / self.hdf5_file, 'r')[self.image_set_name]
            self.label_set = h5py.File(root / self.hdf5_file, 'r')[self.label_set_name]

    def __getitem__(self, index):
        """
        :param index: integer dataset index
        :return: oct slices (H,W,3), labels (10,)
        """
        if self.image_set is None or self.label_set is None:
            self._reset_hdf5()

        metadata = {'index': index}
        image = np.array(self.image_set[index])  # (H,W,1)
        labels = self.label_set[index]  # .astype(np.float32) # (10,)
        image = Image.fromarray(image[..., 0]).convert('RGB')  # replicate (H,W) to (H,W,3) (needed for transforms)
        image, target, metadata = self.common_transforms((image, image, metadata))
        img_tensor, metadata = self.img_transforms((image, metadata))
        # img_tensor is (3,H,W) now
        if self.return_filename:
            metadata.update({'modality': self.modality})

        if self.img_channels == 1:
            img_tensor = img_tensor[0].unsqueeze(0)
            # img_tensor = (1,H,W)

        lbl_tensor = torch.from_numpy(labels).float()
        if self.debug:
            ToPILImage()(img_tensor[0]).show()
            labels_list = lbl_tensor.numpy()
            print([f'{DATASETS_INFO[self.dataset]["CLASS_INFO"][self.experiment][1][i]}:'
                   f' {labels_list[i]}' for i in range(len(labels_list))])

        # map img_tensor to [-1,1] if no imagenet normalization is used
        # if 'torchvision_normalize' not in self.img_transforms.transforms:
        #     img_tensor = self._normalize_min1_1(img_tensor)
        return img_tensor, lbl_tensor, metadata

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    # import pathlib
    # import torch
    from utils import parse_transform_lists
    import json
    # import PIL.Image as Image
    # C:\Users\thopis\Documents\PycharmProjects\DICOM\output_image_10perc_subset
    # data_path = 'C:\\Users\\thopis\\Documents\\PycharmProjects\\DICOM\\output_image_10perc_subset'
    # hdf5_filename = 'oct_test_all.hdf5'
    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\OctBiom\\'

    d = {"dataset": 'OctBiom', "experiment": 0}
    path_to_config = '../configs/OctBiom/vit.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    if 'torchvision_normalise' in transforms_list:
        del transforms_list[-1]

    transforms_diction = parse_transform_lists(transforms_list, transforms_values, **d)
    train_set = OctBiom(root=data_path,
                        hdf5_file=None,
                        debug=True,
                        split='test',
                        transforms_dict=transforms_diction, img_channels=1)

    issues = []
    # valid_set.return_filename = True
    train_set.return_filename = True
    hs = []
    ws = []
    for ret in train_set:
        hs.append(ret[0].shape[1])
        ws.append(ret[0].shape[2])
        present_classes = torch.unique(ret[1])
        print(ret[-1])
