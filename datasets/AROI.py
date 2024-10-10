import json
import os
from typing import Union
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image, ImageFile
from utils import DATASETS_INFO, remap_mask, printlog, create_new_directory
import numpy as np
import pathlib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AROI(Dataset):
    hardcoded_splits = {'hardcoded_train': [2, 3, 5, 6, 8, 9, 11, 13, 16, 18, 19, 20, 22, 23],
                        'hardcoded_val': [1, 4, 7, 10, 12, 14, 15, 17, 21, 24]
                        }



    def __init__(self, root,
                 transforms_dict: dict,
                 img_channels: int = 1,
                 split: Union[str, list] = 'train', return_metadata=True, debug=False):
        """

        :param root: path to cityscapes dir (i.e where directories "leftImg8bit" and "gtFine" are located)
        :param transforms_dict: see dataset_from_df.py
        :param split: any of "train", "test", "val"
        :param mode: if "fine" then loads finely annotated images else Coarsely uses coarsely annotated
        :param target_type: currently only expects the default: 'semantic' (todo: test other target_types if needed)
        """

        super(AROI, self).__init__()
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
        self.dataset = 'AROI'
        self.experiment = 1
        self.img_suffix = '.png'
        self.target_suffix = '.png'
        if self.split == ['train', 'val']:
            # for training on train + val
            printlog('train set is train+val splits')
            # todo not checked because no train or val splits are defined
            for i, s in enumerate(self.split):
                self.images_dir = os.path.join(self.root, 'images', s)
                self.targets_dir = os.path.join(self.root, 'labels', s)
                for image_filename in os.listdir(self.images_dir):
                    img_path = os.path.join(self.images_dir, image_filename)
                    target_path = os.path.join(self.targets_dir, image_filename.split(self.img_suffix)[-2] + self.target_suffix)
                    self.images.append(img_path)
                    self.targets.append(target_path)
                    assert (pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists())
                    assert(pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem)
        elif self.split in ['train', 'val']:
            # todo not checked because no train or val splits are defined
            self.images_dir = os.path.join(self.root, 'images', split)
            self.targets_dir = os.path.join(self.root, 'labels', split)
            for image_filename in os.listdir(self.images_dir):
                img_path = os.path.join(self.images_dir, image_filename)
                target_path = os.path.join(self.targets_dir, image_filename.split(self.img_suffix)[-2] + self.target_suffix) #todo
                self.images.append(img_path)
                self.targets.append(target_path)
                assert (pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists()) #todo
                assert(pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem) #todo

        elif self.split == 'all':
            self.images_dir = os.path.join(self.root, 'images')
            self.targets_dir = os.path.join(self.root, 'labels')
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
            self.images_dir = os.path.join(self.root, 'images')
            self.targets_dir = os.path.join(self.root, 'labels')
            split_ids = self.hardcoded_splits[self.split]
            for image_filename in os.listdir(self.images_dir):
                img_path = os.path.join(self.images_dir, image_filename)
                target_path = os.path.join(self.targets_dir,
                                           image_filename.split(self.img_suffix)[-2] + self.target_suffix)
                id_ = int(str(pathlib.Path(img_path).stem).split('_')[0][7:]) #todo
                if id_ in split_ids:
                    self.images.append(img_path)
                    self.targets.append(target_path)
                    assert (pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists()) #todo
                    assert (pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem) #todo

        printlog(f'AROI data all found split = {self.split}, images {len(self.images)}, targets {len(self.targets)}'
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
        target = remap_mask(np.array(target),
                            DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], to_network=True).astype('int32')
        # print(index, ': ', np.unique(target))
        target = Image.fromarray(target)
        # print(np.unique(target).tolist())

        # if 14 in np.unique(target).tolist():  # tracl
        #     target.show()
        #     target.close()
        # return 0 , 0 , 0

        metadata.update({'img_filename': self.images[index],
                         'target_filename': self.targets[index]})

        if self.store:
            image, r, h = find_roi_and_crop(np.array(image)[..., 0], verbose=True)
            target = np.array(target)[r:r+h, :]

            dir_image = pathlib.Path(self.root) / 'roi' / 'images'
            dir_target = pathlib.Path(self.root) / 'roi' / 'labels'
            create_new_directory(dir_target)
            create_new_directory(dir_image)
            ToPILImage()(image).save(os.path.join(dir_image, pathlib.Path(metadata['img_filename']).name))
            ToPILImage()(target).save(os.path.join(dir_target, pathlib.Path(metadata['target_filename']).name))

        image, target, metadata = self.common_transforms((image, target, metadata))
        img_tensor = self.img_transforms(image)
        lbl_tensor = self.lbl_transforms(target).squeeze()

        if self.debug:
            ToPILImage()(img_tensor).show()
            ToPILImage()(lbl_tensor).show()
            print(f'\nafter aug index: {np.unique(lbl_tensor)} '
                  f'lbl {lbl_tensor.shape} image {img_tensor.shape} fname:{self.images[index]}')

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

    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\AROI\\roi'
    d = {"dataset": 'AROI', "experiment": 1}
    path_to_config = '../configs/AROI/vitd_init_both.json'

    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']

    transforms_dict = parse_transform_lists(transforms_list, transforms_values, **d)
    transforms_list_val = config['data']['transforms_val']
    transforms_values_val = config['data']['transform_values_val']

    if 'torchvision_normalise' in transforms_list_val:
        del transforms_list_val[-1]
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, **d)

    train_set = AROI(root=data_path,
                     debug=True,
                     split='all',
                     transforms_dict=transforms_dict)

    # valid_set = AROI(root=data_path,
    #                  debug=True,
    #                  split='hardcoded_val',
    #                  transforms_dict=transforms_dict_val)

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



        # # 3,H,W - H,W,3
        # image = image.permute(1, 2, 0)
        #
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(to_numpy(image.float()))
        # axarr[1].imshow(to_numpy(pred.float()), cmap=plt.cm.jet, vmax=8)
        # plt.show()



        #to_comb_image(image, pred, None, 1, 'AROI', save=f'../debug/lbl_{i}.png')

        # elif 15 in present_classes:
        #     issues.append([ret[-1], present_classes])
        #     print('bus found !!!! ')
        #     print(present_classes)
        #     pil_plot_tensor(ret[0], is_rgb=True)
        #     pil_plot_tensor(ret[1], is_rgb=False)

        # a = 1
    # print(max(hs))
    # print(max(ws))