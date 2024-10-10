import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from utils import DATASETS_INFO
from typing import Dict, List, Tuple, Union
from collections import namedtuple
from enum import Enum
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image
from utils import printlog, FlipNP, create_new_directory

ImageItem = namedtuple("ImageItem", ["id", "path"])
DataSplit = Enum("DataSplit", ["TRAIN", "VAL", "TEST"])
ModalityModeData = Dict[str, Dict[str, torch.Tensor]]


class OLIVES(Dataset):
    VALID_SPLITS = ["train", "val", "test", 'all']
    VALID_GROUND_TRUTHS = ["biomarkers", "measurements"]
    # classes used for multilabel classification task in the OLIVES paper
    BIOMARKER_CLASSES = ['Fluid (IRF)', 'DRT/ME', 'IR HRF', "Atrophy / thinning of retinal layers",
                         'Fully attached vitreous face', 'Partially attached vitreous face']
    VALID_MODALITIES = ["OCT", "IR"]
    VALID_MODALITY_MODES = {"OCT": ["slices"], "IR": ["en_face"]}
    SPLITS = {
        'train': ['/TREX DME/GILA/0241GOD', '/TREX DME/TREX/0201TOS', '/TREX DME/TREX/0242TOS', '/TREX DME/Monthly/0218MOD',
                  '/Prime_FULL/01-027', '/Prime_FULL/02-008', '/TREX DME/Monthly/0245MOD', '/TREX DME/Monthly/0248MOD',
                  '/TREX DME/GILA/0243GOD', '/TREX DME/TREX/0228TOS', '/TREX DME/Monthly/0249MOD', '/Prime_FULL/02-032',
                  '/Prime_FULL/02-031', '/Prime_FULL/01-012', '/TREX DME/Monthly/0206MOD', '/TREX DME/TREX/0232TOS',
                  '/TREX DME/TREX/0222TOD', '/Prime_FULL/01-013', '/TREX DME/TREX/0230TOS', '/TREX DME/GILA/0201GOD',
                  '/Prime_FULL/01-047', '/TREX DME/Monthly/0212MOS', '/Prime_FULL/02-036', '/TREX DME/TREX/0213TOS',
                  '/Prime_FULL/02-005', '/Prime_FULL/01-028', '/TREX DME/TREX/0209TOD', '/TREX DME/TREX/0211TOD',
                  '/Prime_FULL/02-017', '/TREX DME/TREX/0226TOS', '/TREX DME/Monthly/0222MOS', '/TREX DME/TREX/0224TOD',
                  '/Prime_FULL/02-041', '/Prime_FULL/02-018', '/Prime_FULL/02-029', '/Prime_FULL/02-016',
                  '/Prime_FULL/02-042', '/Prime_FULL/02-039', '/Prime_FULL/02-004', '/TREX DME/GILA/0221GOD',
                  '/Prime_FULL/02-030', '/Prime_FULL/01-038', '/Prime_FULL/02-015', '/Prime_FULL/01-035',
                  '/Prime_FULL/01-025', '/TREX DME/GILA/0204GOS', '/Prime_FULL/01-020', '/Prime_FULL/01-023',
                  '/TREX DME/GILA/0236GOS', '/TREX DME/TREX/0237TOS'],
        'val': ['/TREX DME/GILA/0232GOD', '/TREX DME/TREX/0220TOD', '/TREX DME/GILA/0215GOS',
                '/TREX DME/Monthly/0204MOD', '/TREX DME/GILA/0239GOD', '/TREX DME/TREX/0255TOD',
                '/Prime_FULL/02-045', '/TREX DME/GILA/0229GOD', '/TREX DME/Monthly/0210MOD',
                '/TREX DME/GILA/0251GOD', '/Prime_FULL/01-002', '/TREX DME/Monthly/0216MOD',
                '/Prime_FULL/01-040', '/TREX DME/TREX/0234TOS', '/TREX DME/GILA/0256GOD',
                '/TREX DME/TREX/0208TOD', '/TREX DME/GILA/0252GOD', '/Prime_FULL/02-044',
                '/Prime_FULL/02-034', '/TREX DME/GILA/0225GOS'],
        'test': ['/TREX DME/GILA/0203GOS', '/TREX DME/TREX/0231TOD', '/TREX DME/GILA/0213GOD', '/TREX DME/GILA/0238GOD',
                 '/TREX DME/TREX/0207TOS', '/TREX DME/GILA/0247GOD', '/Prime_FULL/02-024', '/Prime_FULL/02-043',
                 '/TREX DME/TREX/0254TOD', '/TREX DME/Monthly/0219MOS', '/TREX DME/GILA/0234GOD', '/Prime_FULL/01-026',
                 '/Prime_FULL/02-019', '/Prime_FULL/01-037', '/Prime_FULL/01-048', '/Prime_FULL/01-014',
                 '/Prime_FULL/02-046', '/TREX DME/GILA/0217GOS', '/TREX DME/GILA/0249GOS', '/TREX DME/GILA/0226GOD',
                 '/Prime_FULL/01-001', '/TREX DME/TREX/0235TOS', '/TREX DME/TREX/0243TOS', '/TREX DME/TREX/0240TOS',
                 '/Prime_FULL/02-010', '/TREX DME/Monthly/0253MOS']}

    def __init__(
            self,
            root: str,
            split: str,
            transforms_dict: Dict,
            img_channels: int = 3,
            return_metadata: bool = True,
            debug: bool = False,
            ground_truth_type: str = "biomarkers",
            modalities=('OCT', 'IR'),
            modes: Union[List, Tuple] = ('slices', 'en_face')

    ):
        assert os.path.exists(root), f"Root folder not found at {root}"
        assert split in self.VALID_SPLITS, f"Invalid split name. Choose from {self.VALID_SPLITS}"
        assert ground_truth_type in self.VALID_GROUND_TRUTHS, f"ground_truth_type {ground_truth_type} not in {self.VALID_GROUND_TRUTHS}"
        assert all([m in self.VALID_MODALITIES for m in modalities]), f"modalities {modalities} not in {self.VALID_MODALITIES}"
        assert all([m in self.VALID_MODALITY_MODES.keys() for m in modalities]), f"modalities {modalities} not in {self.VALID_MODALITY_MODES.keys()}"

        self.root = root
        self.split = split
        self.img_channels = img_channels
        self.return_metadata = return_metadata
        self.debug = debug
        self.modalities = modalities
        self.modes = {m: modes[i] for i, m in enumerate(modalities)}   # todo for now unused
        # self.common_transforms = Compose(transforms_dict["common"])
        # self.img_transforms = Compose(transforms_dict["img"])

        # get transforms
        self.common_transforms = {}
        self.img_transforms = {}
        self.do_flip = False
        if len(self.modalities) > 1:
            self.do_flip = all(any([isinstance(item, FlipNP) for item in transforms_dict[m]['common']])
                               for m in self.modalities)
        for m in self.modalities:
            if self.do_flip:  # remove flip from common transforms
                for i, t in enumerate(transforms_dict[m]['common']):
                    if isinstance(t, FlipNP):
                        transforms_dict[m]['common'].pop(i)
                        break
            if len(self.modalities) == 1:
                # transforms_dict is a Dict['common': List[Transform], 'img': List[Transform]]
                self.common_transforms[m] = Compose(transforms_dict['common'])
                self.img_transforms[m] = Compose(transforms_dict['img'])
            else:
                # transforms_dict is a Dict['OCT': {'common': List[Transform], 'img': List[Transform]},
                #                           'IR': {'common': List[Transform], 'img': List[Transform]}]
                self.common_transforms[m] = Compose(transforms_dict[m]['common'])
                self.img_transforms[m] = Compose(transforms_dict[m]['img'])

        # load metadata file
        if split == 'all':
            self.subjects = self.SPLITS['train'] + self.SPLITS['val'] + self.SPLITS['test']
        else:
            self.subjects = self.SPLITS[self.split]
        self.ground_truth_type = ground_truth_type
        self.metadata = self._load_ground_truth_data(ground_truth_type)
        printlog(f"Loaded OLIVES split {self.split} with {len(self)} images, ground truth: {ground_truth_type}")

    @staticmethod
    def get_all_subjects(metadata: pd.DataFrame):
        meta_subjects = metadata['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].tolist()
        s = ['/'.join(v.split('/')[0:4]) if 'Prime' not in v else '/'.join(v.split('/')[0:3]) for v in meta_subjects]
        assert len(set(s)) == 96, f"Error in the metadata file, expected 96 unique subjects got {len(set(s))}:{set(s)}"
        return list(set(s))

    @staticmethod
    def init_from_config(root: str, split: str, transforms_dict: Dict, config: Dict):
        """Initialize the dataset from a configuration dictionary."""
        assert (
                config["dataset"] == "OLIVES"
        ), f"Invalid dataset name. Expected OLIVES, got {config['dataset']}"

        # make sure the default values correspond to the default ones of the class
        return OLIVES(
            root=root,
            split=split,
            transforms_dict=transforms_dict,
            img_channels=config.get("img_channels", 3),
            return_metadata=config.get("return_metadata", False),
            debug=config.get("debug", False),
            ground_truth_type=config.get("ground_truth_type", "biomarkers"),
            modalities=config.get("modality", ('OCT', 'IR')),
            modes=config.get("modes", ('slices', 'en_face'))
            )

    def __len__(self):
        return len(self.metadata)

    def preprocess(self, idx, p, metadata):
        target_root = 'E:\\datasets\\OLIVES\\processed\\' # where the preprocessed data will be saved
        # note: we just create a copy of the dataset without any resolution change or cropping to isolate the images for
        # which there are biomarker annotations

        filename = Path(self.metadata.iloc[idx, 0][1:]).name
        new_dir = Path(self.metadata.iloc[idx, 0][1:]).parent
        p_new = Path(target_root) / new_dir  / filename
        create_new_directory(os.path.join(target_root, new_dir))
        if not os.path.exists(p_new):
            img = Image.open(p).convert("RGB")
            img.save(p_new)
        # metadata['week'] / metadata['laterality'] /
        p = p.parent / f"fundus_{metadata['laterality']}_{metadata['week']}.tif"
        p_new = target_root / new_dir / f"fundus_{metadata['laterality']}_{metadata['week']}.tif"
        if not os.path.exists(p_new):
            img = Image.open(p).convert("RGB")
            img.save(p_new)
        return

    def __getitem__(self, idx) -> Union[Tuple[ModalityModeData, Dict],
                                        Tuple[torch.Tensor, torch.Tensor, Dict],
                                        ModalityModeData]:
        data = {}
        metadata = {'index': idx}
        # path to image
        p = Path(self.root) / Path(self.metadata.iloc[idx, 0][1:])  # remove leading '/'
        # labels
        labels = self.metadata.iloc[idx, 1:].values.astype(np.float32)
        # metadata
        laterality = Path(self.metadata.iloc[idx, 0]).parts[-2]
        week = Path(self.metadata.iloc[idx, 0]).parts[-3]
        metadata.update({'laterality': laterality, 'week': week})

        # self.preprocess(idx, p, metadata)  # uncomment to preprocess
        # return # uncomment to preprocess

        h_flip = False
        if self.do_flip:
            # do flipping outside of transforms because we want it to be applied to all images
            h_flip = torch.rand(1) < 0.5

        for modality in self.modalities:
            data[modality] = {}
            # for mode in self.modes[modality]:  # todo for now unused remove
            if modality == 'IR':  # IR
                # metadata['week'] / metadata['laterality'] /
                p = p.parent / f"fundus_{metadata['laterality']}_{metadata['week']}.tif"
            image = Image.open(p).convert("RGB")
            image, _, metadata = self.common_transforms[modality]((image, image, metadata))
            img_tensor, metadata = self.img_transforms[modality]((image, metadata))
            if h_flip:
                img_tensor = img_tensor.flip(2)
            data[modality][self.modes[modality]] = img_tensor

        data['label'] = torch.from_numpy(labels).float()

        # debug only
        if self.debug:
            for modality in self.modalities:
                mode = self.modes[modality]
                ToPILImage()(data[modality][mode]).show()

        if self.return_metadata:
            metadata.update({'subject_id': '_'.join(self.metadata.iloc[idx, 0].split('/')[2:6])})

        # if single modality return Tensor, Tensor, Dict
        if len(self.modalities) == 1:
            labels = data['label']
            data = data[self.modalities[0]][self.modes[self.modalities[0]]]
            if self.return_metadata:
                return data, labels, metadata
            else:
                return data, labels

        # if multimodal return Dict[Dict[str, Tensor]], Dict
        if self.return_metadata:
            return data, metadata
        else:
            return data

    def _load_ground_truth_data(self, ground_truth_type: str):
        """Load visual field ground truth data for the selected task."""
        if ground_truth_type == "biomarkers":
            metadata = pd.read_csv(os.path.join(self.root, 'OLIVES_Dataset_Labels/full_labels/Biomarker_Clinical_Data_Images.csv'))
            # keep only split's subjects
            metadata = metadata[metadata['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'].str.contains('|'.join(self.subjects))]
            # keep only the path to image and the biomarker annotations that we train the detection model on
            metadata = metadata[['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'] + self.BIOMARKER_CLASSES]
            # replace any NaN values with 0
            metadata.fillna(0, inplace=True)
            # add new column in Metadata called 'Healthy'
            is_healthy = metadata[self.BIOMARKER_CLASSES].sum(axis=1) == 0

            metadata.insert(metadata.shape[-1], 'Healthy', 1.0*is_healthy)
            printlog(f"-------Biomarker distribution for split: {self.split}---------")
            printlog(metadata[self.BIOMARKER_CLASSES].sum())  # print the sum of each biomarker class for this split
            return metadata
        elif ground_truth_type == "measurements":
            # TODO: implement loading of measurements ground truth (BCVA, CST)
            raise NotImplementedError("Measurements ground truth not implemented yet")

    def __eq__(self, other) -> bool:
        if not isinstance(other, OLIVES):
            return False
        return (
                self.root == other.root
                and self.split == other.split
                and self.img_channels == other.img_channels
                and self.return_metadata == other.return_metadata
                and self.debug == other.debug
                and self.modalities == other.modalities
                and self.modes == other.modes
                and self.common_transforms == other.common_transforms
                and self.img_transforms == other.img_transforms
                and self.metadata.equals(other.metadata)
        )


if __name__ == "__main__":

    from torchvision.transforms import ToTensor
    import torch
    import time
    import matplotlib
    from utils import parse_transform_lists
    import json

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    test_root = "E:\\datasets\\OLIVES\\"
    # test_root = 'C:\\Users\\thopis\\Documents\\datasets\\OLIVES'
    d = {"dataset": 'OLIVES', "experiment": 1}
    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\DR\\data\\'
    path_to_config = '../configs/DR/mvitseq_ft.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    modalities_ = config['data']['modality']
    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    transforms_dictionary = {}
    for ind, moda in enumerate(modalities_):
        t_dict = parse_transform_lists(transforms_list[ind], transforms_values[ind], **d)
        transforms_dictionary[moda] = t_dict

    train_set = OLIVES(root=test_root, split="train", debug=True, transforms_dict=transforms_dictionary, ground_truth_type='biomarkers')
    val_set = OLIVES(root=test_root, split="val", debug=True, transforms_dict=transforms_dictionary, ground_truth_type='biomarkers')

    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")

    # Visualize a slice
    im, grt = train_set[0]
    print(im.shape, grt.shape)
    plt.imshow(im[0][16], cmap='gray')
    plt.show()

    # Test initialization time
    tic = time.perf_counter()
    dataset = OLIVES(root=test_root, split="train", debug=True, transforms_dict=transforms_dictionary, ground_truth_type='biomarkers')
    toc = time.perf_counter()
    print(f"Loading the dataset took {toc - tic:0.4f} seconds")

    #  Test loading time
    n_loadings = 1
    tic = time.perf_counter()
    for i in range(n_loadings):
        im, grt = dataset[i]
    toc = time.perf_counter()
    print(f"Loading {n_loadings} images took {toc - tic:0.4f} seconds")


    def print_split_ids():
        train, val = train_test_split(dataset.metadata, test_size=0.2, random_state=42,
                                      stratify=dataset.metadata["Glaucoma_stage"])
        print_split_overview(train, "Training")
        print_split_overview(val, "Validation")


    def print_split_overview(df: pd.DataFrame, split_name: str):
        print(f'----- {split_name} split -----')
        print(f'Overview: {df["Glaucoma_stage"].value_counts()}')
        print(f'IDs: {list(df.index)}')
        print('\n')
    # print_split_ids()