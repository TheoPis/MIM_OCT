import os
from os.path import join as pjoin
import ast
import json
from utils import printlog
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Union, Dict, Tuple, List
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToPILImage
from utils import FlipNP
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DRDataset(Dataset):
    valid_modalities = ["OCT", "IR"]
    valid_modality_modes = {"OCT": ["horizontal", "vertical"], "IR": ["fundus"]}
    valid_splits = ["train", "train_refined", "val", "test", "test_refined",
                    "minival_refined_30", "train_minival_172",
                    "train_refined_minival_149", "train_1", "train_2", "train_3", "train_4", "train_5",
                    "val_1", "val_2", "val_3", "val_4", "val_5"]

    def __init__(self,
                 root: str,
                 split: str,
                 transforms_dict: Dict[str, Dict[str, List]],
                 modalities: List[str] = ('OCT', 'IR'),
                 modes: Union[List[List], Tuple[Tuple]] = (('horizontal', 'vertical'), ('en_face', )),
                 return_metadata: bool = False,
                 use_condor: bool = False,
                 max_condor_dates: int = 3,
                 debug=False):
        """ Dataset for the DR dataset: expected folder structure is
        root
        ├── images : where all the images are (1980 in total comprising horizontal/vertical OCT and en_face IR )
        ├── sets : where the csv files defining the splitting are stored

        :param root: a path to the root of the dataset (e.g. /home/user/datasets/DR)
        :param split: any of valid_splits
        :param transforms_dict: a nested dict[modality]['common'/'img'/'lbl'] -> list of transforms
                                or dict['common'/'img'/'lbl'] -> list of transforms
        :param modalities: any of valid_modalities (e.g. ('OCT', 'IR'))
        :param modes: a list/tuple of lists/tuples of any of the modes
        :param return_metadata: whether to return metadata (e.g. image name, label, etc.)
        :param use_condor: whether to use condor images (default False)
                if True, the expected location of Condor relative to DR is the following:
                DR: root/data/images
                Condor: root/../Condor/data

        :param max_condor_dates: maximum number of dates per condor patient to use (default 3)
        :param debug:
        """
        self.debug = debug
        self.return_metadata = return_metadata
        # sanity checks
        assert split in self.valid_splits, f'split {split} is not in valid_modes {self.valid_splits}'
        assert os.path.exists(root), f'data path {root} does not exist'
        for modes_m, modality in zip(modes, modalities):
            assert modality in self.valid_modalities, f"{modality} not in {self.valid_modalities}"
            assert any([m in self.valid_modality_modes[modality] for m in modes_m]), \
                f"modes {modes_m} for modality {modality} not in {self.valid_modality_modes[modality]}"

        self.root = root
        self.split = split
        self.modalities = modalities
        self.modes = dict(zip(modalities, modes))
        self.img_path = os.path.join(root, 'images')
        csv_file = split + '.csv'
        csv_path = os.path.join(root, 'sets', csv_file)
        self.annotations = pd.read_csv(csv_path, converters={'frame_of_reference_UID': ast.literal_eval,
                                                             'image_hash': ast.literal_eval,
                                                             'image_uuid': ast.literal_eval})
        assert all(item in self.annotations.columns for item in
                   ['patient_hash', 'image_uuid']), 'CSV does not provide enough information'

        self.label_type = 'proliferation'
        self.label_dict = {'NPDR': 0, 'PDR': 1}

        self.common_transforms = {}
        self.img_transforms = {}

        self.do_flip = False
        flip_object = FlipNP
        if len(self.modalities) > 1:
            self.do_flip = all(any([isinstance(item, flip_object) for item in transforms_dict[m]['common']])
                               for m in self.modalities)
        for m in self.modalities:
            if self.do_flip:  # remove flip from common transforms
                for i, t in enumerate(transforms_dict[m]['common']):
                    if isinstance(t, flip_object):
                        transforms_dict[m]['common'].pop(i)
                        break
            if len(self.modalities) == 1:
                self.common_transforms[m] = Compose(transforms_dict['common'])
                self.img_transforms[m] = Compose(transforms_dict['img'])
            else:
                self.common_transforms[m] = Compose(transforms_dict[m]['common'])
                self.img_transforms[m] = Compose(transforms_dict[m]['img'])


        # paths to triplets of images from original DR dataset
        self.triplet_paths = [{k: os.path.join(self.img_path, triplet[k] + '.png') for k in triplet.keys()}
                              for triplet in self.annotations['image_uuid'].tolist()]


        self.img_labels = [self.label_dict[label] for label in self.annotations[self.label_type]]

        # add condor
        img_names_condor = []
        if use_condor:
            img_names_condor, img_labels_condor = self.get_condor_paths(max_dates=max_condor_dates)
            self.triplet_paths.extend(img_names_condor)
            self.img_labels.extend(img_labels_condor)

        self.weights = self._get_weights_loss()
        printlog(
            f'DR data found \n'
            f'  using condor: {use_condor},  condor_images: {len(img_names_condor)} \n'
            f'  split = {self.split}, images {self.__len__()} \n '
            f'  num NPDR = {self.img_labels.count(0)}, num PDR = {self.img_labels.count(1)} \n '
        )

    def get_condor_paths(self, max_dates):
        condor_root = Path(self.root).parent.parent / 'Condor'
        if 'train' in self.split:
            condor_json = condor_root / f'triplets_train_patients_100_MaxDates_{max_dates}.json'
        elif 'test' in self.split:
            condor_json = condor_root / f'triplets_test_patients_51_MaxDates_{max_dates}.json'
        elif 'minival_refined_30' in self.split:
            return [], []
        else:
            raise ValueError(f'Unknown split {self.split} for Condor expected one that contains either train/test')

        # open json
        with open(condor_json, 'r') as f:
            condor_dict = json.load(f)
        # get paths
        condor_triplets = condor_dict['triplets']
        img_names_condor = []
        for p in condor_triplets:
            p_dict = {}
            for m in self.modalities:
                if 'horizontal' in self.modes[m]:
                    p_dict['horizontal'] = pjoin(condor_root, p[0] + '.png')
                elif 'vertical' in self.modes[m]:
                    p_dict['vertical'] = pjoin(condor_root, p[1] + '.png')
                elif 'fundus' in self.modes[m]:
                    p_dict['fundus'] = pjoin(condor_root, p[2] + '.png')
                else:
                    raise ValueError(f'Unknown mode {self.modes[m]} for modality {m}')
            img_names_condor.append(p_dict)
        img_labels = [1]*len(img_names_condor)  # all are PDR in condor
        return img_names_condor, img_labels

    def __len__(self):
        return len(self.img_labels)

    @property
    def do_transforms(self):
        return len(self.common_transforms) > 0

    def _get_weights_loss(self):
        # if 'minival_refined' in self.split:
        #     return torch.tensor([1.0, 1.0])
        lbls = torch.tensor(self.img_labels)
        one_hot_lbls = torch.nn.functional.one_hot(lbls.long(), num_classes=len(self.label_dict))
        labels_sum = torch.sum(one_hot_lbls, dim=0)
        largest_class = max(labels_sum)
        weights = largest_class / labels_sum
        return weights

    def __getitem__(self, index) -> Union[Dict[str, Dict[str, torch.Tensor]], Tuple[torch.Tensor, int]]:
        metadata = {'index': index}
        data = {}

        h_flip = False
        if self.do_flip:
            # do flipping outside of transforms because we want it to be applied to all images
            h_flip = torch.rand(1) < 0.5

        for modality in self.modalities:
            data[modality] = {}
            for mode in self.modes[modality]:
                p = self.triplet_paths[index][mode]
                image = Image.open(p).convert('RGB')
                image, _, metadata = self.common_transforms[modality]((image, image, metadata))
                img_tensor, metadata = self.img_transforms[modality]((image, metadata))
                if h_flip:
                    img_tensor = img_tensor.flip(2)
                data[modality][mode] = img_tensor
                # print(f'FOUND reading image {p}')

        data['label'] = self.img_labels[index]
        # debug only
        if self.debug:
            for modality in self.modalities:
                for mode in self.modes[modality]:
                    ToPILImage()(data[modality][mode]).show()

        # if only a single modality/mode is used, return only tensor without dict
        # this is for compatibility with ViT_Manager for finetuning on single modality
        if len(self.modalities) == 1 and len(self.modes[self.modalities[0]]) == 1:
            data = data[self.modalities[0]][self.modes[self.modalities[0]][0]]
            return data, self.img_labels[index]
        else:
            return data




if __name__ == '__main__':
    # script to generate train_minival.csv and minival.csv
    import json
    from utils import parse_transform_lists
    d = {"dataset": 'DR', "experiment": 1}
    data_path = 'C:\\Users\\thopis\\Documents\\datasets\\DR\\data\\'
    path_to_config = '../configs/DR/mvitseq_ft_IN.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    modalities_ = config['data']['modality']
    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    transforms_dictionary = {}
    for ind, moda in enumerate(modalities_):
        t_dict = parse_transform_lists(transforms_list[ind], transforms_values[ind], **d)
        transforms_dictionary[moda] = t_dict

    dataset = DRDataset(root=data_path,
                        split='train_refined_minival_149',
                        modes=config['data']['mode'],
                        transforms_dict=transforms_dictionary,
                        use_condor=True,
                        debug=True)

    patients_pathology_triplets = list(
        zip(dataset.annotations.patient_hash, dataset.annotations.proliferation, dataset.annotations.image_uuid))
    patients = list(set(dataset.annotations.patient_hash))
    pathologies = ['NPDR', 'PDR']
    # get lists of patients with each of the pathologies
    patients_per_pathology = {
        pathology: [patient for patient, pathology_, _ in patients_pathology_triplets if pathology_ == pathology] for
        pathology in pathologies}
    # remove duplicate patients per pathology
    patients_per_pathology = {pathology: list(set(patients)) for pathology, patients in patients_per_pathology.items()}

    # keep last 10 patients for validation per pathology
    N_patients_for_minival = 15  # per pathology

    patients_for_minival = {pathology: patients_per_pathology[pathology][-N_patients_for_minival:]
                            for pathology in pathologies}

    # create patients_pathology_triplets for minival
    patients_pathology_triplets_minival = [triplet for triplet in patients_pathology_triplets if
                                           triplet[0] in patients_for_minival[triplet[1]]]

    # create patients_pathology_triplets for train by removing patients_for_minival from patients_pathology_triplets
    patients_pathology_triplets_train_minival = [triplet for triplet in patients_pathology_triplets if
                                                 triplet[0] not in patients_for_minival[triplet[1]]]

    # get dataframe from patients_pathology_triplets_minival
    df = pd.DataFrame(patients_pathology_triplets_minival,
                      columns=['patient_hash', 'proliferation', 'image_uuid'])
    # save dataframe as csv
    df.to_csv(os.path.join(data_path, 'sets', f'minival_refined_{N_patients_for_minival*2}.csv'), index=False)

    # get dataframe from patients_pathology_triplets_train_minival
    N_train = len(patients) - N_patients_for_minival * 2
    df = pd.DataFrame(patients_pathology_triplets_train_minival,
                      columns=['patient_hash', 'proliferation', 'image_uuid'])

    # save dataframe as csv
    df.to_csv(os.path.join(data_path, 'sets', f'train_refined_minival_{N_train}.csv'), index=False)
