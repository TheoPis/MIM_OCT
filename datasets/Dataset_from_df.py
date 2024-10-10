import os
import pathlib
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from utils import DATASETS_INFO, remap_mask
import numpy as np
from .omnivision import Sample


class DatasetFromDF(Dataset):
    def __init__(self, dataframe, experiment, transforms_dict, config: dict, data_path=None, labels_remaped=False,
                 return_pseudo_property=False, dataset='CADIS', debug=False, return_metadata=True):
        self.config = config.get('data', {})
        self.img_channels = self.config.get('img_channels', 1)
        self.df = dataframe
        self.experiment = experiment
        self.dataset = dataset
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        self.labels_are_remapped = labels_remaped  # used when reading pseudo labeled data
        self.return_pseudo_property = return_pseudo_property  # used to return whether the datapoint is pseudo labelled
        self.preloaded = False if data_path is not None else True
        if self.preloaded:  # Data preloaded, need to assert that 'image' and 'label' exist in the dataframe
            assert 'image' in self.df and 'label' in self.df, "For preloaded data, the dataframe passed to the " \
                                                              "PyTorch dataset needs to contain the columns 'image' " \
                                                              "and 'label'"
        else:  # Standard case: data not preloaded, needs base path to get images / labels from
            assert 'img_path' in self.df and 'lbl_path' in self.df, "The dataframe passed to the PyTorch dataset needs"\
                                                                    " to contain the columns 'img_path' and 'lbl_path'"
            self.data_path = data_path
        self.debug = debug

        self.return_metadata = return_metadata
        self.use_ominvision_api = self.config.get('use_ominvision_api', False)

    def __getitem__(self, item):
        if self.preloaded:
            img = self.df.iloc[item].loc['image']
            lbl = self.df.iloc[item].loc['label']
        else:
            # img = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['img_path']))[..., ::-1]
            img = cv2.imread(
                os.path.join(
                    self.data_path,
                    os.path.join(*self.df.iloc[item].loc['img_path'].split('\\'))))[..., ::-1]
            img = img - np.zeros_like(img)  # deals with negative stride error
            # lbl = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['lbl_path']), 0)
            lbl = cv2.imread(
                os.path.join(
                    self.data_path,
                    os.path.join(*self.df.iloc[item].loc['lbl_path'].split('\\'))), 0)
            lbl = lbl - np.zeros_like(lbl)

        if self.labels_are_remapped:
            # if labels are pseudo they are already remapped to experiment label set
            lbl = lbl.astype('int32')
        else:
            lbl = remap_mask(lbl, DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], to_network=True).astype('int32')

        # Note: .astype('i') is VERY important. If left in uint8, ToTensor() will normalise the segmentation classes!

        # Here (and before Compose(lbl_transforms) we'd need to set the random seed and pray, following this idea:
        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        # Big yikes. Big potential problem source, see here: https://github.com/pytorch/pytorch/issues/7068
        # If that doesn't work, the whole transforms structure needs to be changed into all-custom functions that will
        # transform both img and lbl at the same time, with one random shift / flip / whatever being applied to both
        metadata = {'index': item, 'filename': self.df.iloc[item].loc['img_path'],
                    'target_filename': str(pathlib.Path(self.df.iloc[item].loc['img_path']).stem)}

        # dataset specific metadata information
        if self.dataset == 'RETOUCH':
            subject_id = pathlib.Path(metadata['filename']).parent.stem
            slice_id = pathlib.Path(self.df.iloc[item].loc['lbl_path']).stem
            metadata['subject_id'] = subject_id
            metadata['target_filename'] = f"{subject_id}_{slice_id}"

        img, lbl, metadata = self.common_transforms((img, lbl, metadata))

        img_tensor = self.img_transforms(img)
        lbl_tensor = self.lbl_transforms(lbl).squeeze()
        if self.return_pseudo_property:
            # pseudo_tensor = torch.from_numpy(np.asarray(self.df.iloc[item].loc['pseudo']))
            metadata.update({'pseudo': self.df.iloc[item].loc['pseudo']})

        if self.debug:
            # ToPILImage()(img_tensor).show()
            # ToPILImage()(lbl_tensor).show()
            print(f'\nafter aug index : {np.unique(lbl_tensor)}  lbl {lbl_tensor.shape} image {img_tensor.shape}')

        if (self.img_channels == 1) and len(img_tensor.shape) == 3:
            img_tensor = img_tensor[0].unsqueeze(0)

        if self.use_ominvision_api:
            return self.create_sample(item, img_tensor, lbl_tensor)

        if self.return_metadata:
            return img_tensor, lbl_tensor, metadata
        else:
            return img_tensor, lbl_tensor

    @staticmethod
    def create_sample(idx, img, lbl):
        return Sample(
            data=img, label=lbl, data_idx=idx, data_valid=True
        )

    def __len__(self):
        return len(self.df)


class DatasetFromDFSub(Dataset):
    def __init__(self, dataframe, experiment, transforms_dict, config: dict, data_path=None,
                 return_pseudo_property=False, dataset='CADIS', debug=False, return_metadata=True):
        self.config = config.get('data', {})
        self.img_channels = self.config.get('img_channels', 1)
        self.df = dataframe
        self.experiment = experiment
        self.dataset = dataset
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        assert 'img_path' in self.df, "The dataframe passed to the PyTorch dataset needs"\
                                      " to contain the columns 'img_path' and 'lbl_path'"
        self.data_path = data_path
        self.debug = debug

        self.return_metadata = return_metadata
        self.use_ominvision_api = self.config.get('use_ominvision_api', False)

    def __getitem__(self, item):

        # img = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['img_path']))[..., ::-1]
        img = cv2.imread(
            os.path.join(
                self.data_path,
                os.path.join(*self.df.iloc[item].loc['img_path'].split('\\'))))[..., ::-1]
        img = img - np.zeros_like(img)  # deals with negative stride error

        metadata = {'index': item, 'filename': self.df.iloc[item].loc['img_path'],
                    'target_filename': str(pathlib.Path(self.df.iloc[item].loc['img_path']).stem)}

        # dataset specific metadata information
        if self.dataset == 'RETOUCH':
            subject_id = pathlib.Path(metadata['filename']).parent.stem
            slice_id = pathlib.Path(self.df.iloc[item].loc['img_path']).stem
            metadata['subject_id'] = subject_id
            metadata['target_filename'] = f"{subject_id}_{slice_id}"

        img, lbl, metadata = self.common_transforms((img, img, metadata))
        img_tensor = self.img_transforms(img)
        if self.debug:
            # ToPILImage()(img_tensor).show()
            # ToPILImage()(lbl_tensor).show()
            print(f'\nafter aug index : image {img_tensor.shape}')

        if (self.img_channels == 1) and len(img_tensor.shape) == 3:
            img_tensor = img_tensor[0].unsqueeze(0)

        if self.return_metadata:
            return img_tensor, img_tensor, metadata  # for compatibility with the rest of the code return image x2
        else:
            return img_tensor, img_tensor

    def __len__(self):
        return len(self.df)
