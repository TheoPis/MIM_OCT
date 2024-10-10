import json
import os
from os.path import join as pjoin
import random
import datetime
from typing import Union, Dict, List, Tuple
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import ImageFile, Image
from pathlib import Path
from utils import printlog, FlipNP
import itertools


def find_jpeg_files(directory):
    jpeg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg'):
                jpeg_files.append(os.path.join(root, file))
    return jpeg_files


try:
    from .omnivision import Sample
except:
    from omnivision import Sample

ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoids an error with PIL

# meta file
patient_hash = str
date_hash = str
modality_name = str
laterality_name = str
path_to_image = str

meta_type = Dict[patient_hash,
                 Dict[date_hash,
                      Dict[modality_name,
                           Tuple[laterality_name,
                                 List[path_to_image]]]]]

ood_meta_type = Dict[patient_hash,
                     Dict[date_hash,
                          Dict[modality_name,
                               Dict[path_to_image, float]]]]


class OCTIRSingle(Dataset):
    valid_splits = ['all', 'train', 'val', 'train_small']
    valid_modalities = ["OCT", "FA", "IR"]
    valid_modality_modes = {"OCT": ["slices", "slices_v", "slices_h"], "FA": ["single"], "IR": ["en_face"]}
    val_set_size = 1000  # number of patients in the validation set
    # modality_dir_name = {"OCT": "OCT_volume", "FA": "SLO_FA_single", "IR":  "SLO_IR_en_face"}
    valid_get_paths_methods = ['naive', 'stratified', 'deduplication']
    valid_stratification_methods = ['cap_per_patient']
    min_per_modality_per_date_per_patient = {"OCT": 6, "FA": 2, "IR": 4}

    def __init__(self,
                 root,
                 metafile: str,
                 transforms_dict: Dict[str, List],
                 split: str = 'train',
                 modality: str = 'OCT',
                 mode: str = 'slices',
                 debug=False,
                 img_channels=1,
                 return_metadata=True,
                 use_omnivision_api=False,
                 get_paths_method='naive',
                 stratification_method=None,
                 max_per_modality_per_patient: int = 10000,
                 keep_every_nth_patient: int = 1,
                 ood_modality_to_patient_to_filename_to_score: str = None,
                 ood_remove_percent=0.5,
                 use_kermany=False,
                 log_dir=None):

        """ OCTIR dataset of 2D. Loads the data from paths

        :param root: data path root (e.x home/data/KEKI/) were all the data for a single modality is stored
        :param metafile: path to the json file containing the metadata (nested dict id_to_modality_to_dcm)
        :param transforms_dict: dictionary of transforms to apply to the images of each modality
               transforms_dict[modality]['common']
               transforms_dict[modality]['img']

        :param split: split of the dataset to use, can be 'all', 'train', 'val' or a list of indices
        :param debug: if true shows images with PIL
        # :param keep_every_nth_slice: if > 1 keeps only every nth slice (default: 1)
        :param img_channels: number of channels in images (1 or 3)
        :param return_metadata: if True returns additional information in a dictionary (default: True)
        :param use_omnivision_api: if True returns a Sample object for the omnivision dataloader (default: False)
        :param get_paths_method:
        :param stratification_method:
        :param max_per_modality_per_patient:
        :param keep_every_nth_patient:
        :param ood_modality_to_patient_to_filename_to_score: a file with a list of ood img
        :param ood_remove_percent: top-% of images per subject wrt to their ood scores that are ignored when sampling
        :param log_dir: path to the log directory
        """
        super().__init__()
        # sanity checks
        assert split in self.valid_splits, f'split {split} is not in valid_modes {self.valid_splits}'
        assert os.path.exists(root), f'data path {root} does not exist'
        assert modality in self.valid_modalities, f"{modality} not in {self.valid_modalities}"
        assert mode in self.valid_modality_modes[modality], f"mode '{mode}' for modality '{modality}' " \
                                                            f"not in {self.valid_modality_modes[modality]}"

        # metafile must exist
        assert os.path.isfile(metafile), f'metafile {metafile} does not exist'

        self.debug = debug
        self.root = root
        self.log_dir = log_dir if log_dir is not None else str(Path(log_dir).parent)
        self.metafile = metafile

        self.experiment = 1  # hardcoded no effect here
        self.modality = modality  # e.g OCT, IR etc
        self.modality_tag = modality  # fixme not necessary for OCTIR
        self.mode = mode

        # modality specific augmentation pipelines
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])

        self.split = split  # "train", "test", "val"
        assert self.split in self.valid_splits, f"split {self.split} not in {self.valid_splits}"

        self.img_channels = img_channels
        self.dataset_view = f"OCTIR_{self.modality}_{self.mode}"
        self.suffix = '.png'

        self.use_kermany = use_kermany
        extra_image_paths = []
        # path to kermany data
        if self.use_kermany:
            assert self.split == 'train', f"split {self.split} must be train if using kermany data"
            assert self.modality == 'OCT', f"modality {self.modality} must be OCT if using kermany data"
            root_extra = os.path.join(str(Path(self.root).parent), 'OCT2017resized')
            extra_image_paths = find_jpeg_files(root_extra)
            printlog(f"Adding {len(extra_image_paths)} images from Kermany")


        # names of unique dicom files
        self.unique_dicoms = []
        self.dicoms_to_num_slices = {}
        self.get_paths_method = get_paths_method

        assert get_paths_method in self.valid_get_paths_methods, \
            f"get_paths_method {get_paths_method} not in {self.valid_get_paths_methods}"
        self.stratification_method = stratification_method
        assert stratification_method in self.valid_stratification_methods, \
            f"stratification_method {stratification_method} not in {self.valid_stratification_methods}"

        self.max_per_modality_per_patient = max_per_modality_per_patient  # max images per modality per patient
        self.keep_every_nth_patient = keep_every_nth_patient  # keep every nth patient
        #
        # self.keep_every_nth_slice = {modality: keep_n for modality, keep_n in zip(self.modality_tags,
        #                                                                           keep_every_nth_slice)}
        #
        # self.keep_every_nth_dcm = {modality: keep_n for modality, keep_n in zip(self.modality_tags,
        #                                                                         keep_every_nth_dcm)}

        self.ood_remove_percent = ood_remove_percent
        if ood_modality_to_patient_to_filename_to_score is not None:
            with open(os.path.join(self.root, ood_modality_to_patient_to_filename_to_score), 'r') as fn:
                ood_modality_to_patient_to_filename_to_score = json.load(fn)

        # self.cotrain_with_volumes = cotrain_with_volumes
        self.image_paths, self.dates, self.patients = self.get_all_img_files(get_paths_method,
                                                                             ood_modality_to_patient_to_filename_to_score)

        self.image_paths += extra_image_paths
        printlog(f'{self.dataset_view} data found split = {self.split},'
                 f' all_images: {self.__len__()} (including {len(extra_image_paths)} from Kermany),'
                 f' all_patients: {len(set(self.patients))}'
                 f' all_dates: {len(set(self.dates))}')

        self.return_filename = False
        self.return_metadata = return_metadata
        self.use_omnivision_api = use_omnivision_api
        printlog(f"OCTIRSingle dataloader: using omnivision api for batching: {self.use_omnivision_api}")

    def get_all_img_files(self,
                          get_paths_method: str,
                          ood_modality_to_patient_to_filename_to_score=None,
                          ood_filtering: str = "remove_top_then_random",
                          pair_generation_method='same_laterality_all_combs'):
        """
        :param get_paths_method: string specifying how to get the paths to the images
        :param ood_modality_to_patient_to_filename_to_score: (Optional) Dict[modality, patient_id, filename] = ood_score
        :param ood_filtering: (Optional) string specifying how to filter the ood data
                                1) "remove_top_then_random" - remove n% (defaults to 50%) of data with top ood score,
                                 then randomly sample max_per_modality_per_patient from the rest
                                2) "keep_top" - keep at most max_per_modality_per_patient of data sorted descending
                                    with lowest ood scores
        :param pair_generation_method:
        :return: all_pairs (a list of pairs of paths to imgs)
        """
        all_dates = []
        all_patients = []

        random.seed(0)  # to ensure we always "randomly" sample the same files per patient
        if ood_modality_to_patient_to_filename_to_score is not None:
            assert ood_filtering in ["remove_top_then_random", "keep_top"], \
                f"ood_filtering {ood_filtering} not in ['remove_top_then_random', 'keep_top']"

        with open(os.path.join(self.root, self.metafile), 'r') as fn:
            meta: meta_type = json.load(fn)

        all_paths = []
        if get_paths_method == 'naive':
            for ph in meta.keys():
                for date in meta[ph].keys():
                    for modality in meta[ph][date].keys():
                        new_paths = [pjoin(self.root, 'data', p[0] + self.suffix) for p in meta[ph][date][modality]]
                        all_paths += new_paths
            a = 1
        elif get_paths_method == 'deduplication':
            ph_to_date_lat_to_paths = {}
            m = self.modality
            for ph in meta.keys():
                ph_to_date_lat_to_paths[ph] = {}
                for date in meta[ph].keys():
                    ph_to_date_lat_to_paths[ph][date] = {}
                    try:
                        ph_to_date_lat_to_paths[ph][date]['R'] = [pjoin(self.root, 'data', p[0] + self.suffix) for p in
                                                                  meta[ph][date][m] if p[1] == 'R']
                        ph_to_date_lat_to_paths[ph][date]['L'] = [pjoin(self.root, 'data', p[0] + self.suffix) for p in
                                                                  meta[ph][date][m] if p[1] == 'L']
                    except:
                        print(f"NO data from modality {m} for ph: {ph} date {date} : omitted from self.image_paths")
                        # remove this date for this patient from ph_to_date_lat_to_paths
                        ph_to_date_lat_to_paths[ph].pop(date)

            return ph_to_date_lat_to_paths

        elif get_paths_method == 'stratified':
            # in this case data are organized as:
            # root
            #   - data
            #     - data_XXX
            #       - {uuid}.png
            # we apply a stratification heuristic to keep only a subset of the data per patient
            # data from date and each laterality are stratified separately
            # for each patient, for each modality keep at most n_max_dcm_per_modality
            if self.stratification_method == 'cap_per_patient':
                random.seed(0)  # to ensure each time we randomly sample some dcms per patient, we get the same ones

                # note: we assume that all patients have the all the required modalities
                patients_to_keep = list(meta.keys())
                if self.split == 'train':
                    # we keep all patients except the last N (hardcoded)
                    patients_to_keep = patients_to_keep[:-self.val_set_size]
                    n_train_patients = len(patients_to_keep)
                    if self.keep_every_nth_patient > 1:
                        patients_to_keep = patients_to_keep[::self.keep_every_nth_patient]
                        printlog(f"keeping every {self.keep_every_nth_patient}th patient "
                                 f"of the starting {n_train_patients}"
                                 f" patients resulting in {len(patients_to_keep)} patients")

                elif self.split == 'val':
                    # we keep only the last N patients (hardcoded)
                    patients_to_keep = patients_to_keep[-self.val_set_size:]
                elif self.split == 'train_small':
                    # we keep only the first N patients (hardcoded)
                    patients_to_keep = patients_to_keep[:]
                else:
                    raise ValueError(f"split {self.split} not in {self.valid_splits}")

                num_patients = len(patients_to_keep)
                printlog(f"num_patients = {num_patients} -- {self.split}")
                stats_lat = {'R': 0, 'L': 0}
                stats_patient = dict()
                all_dates = []
                for patient_id in meta.keys():
                    if patient_id in patients_to_keep:  # only collect paths for patients in the train or val set
                        stats_patient[patient_id] = {self.modality: 0}
                        num_patient_dates = len(meta[patient_id].keys())  # number of dates for this patient
                        max_per_modality_per_patient = self.max_per_modality_per_patient

                        # note: we assume all dates have all modalities
                        max_per_modality_per_date_per_patient = max(max_per_modality_per_patient // num_patient_dates,
                                                                    self.min_per_modality_per_date_per_patient[
                                                                        self.modality])

                        # print(f"Patient {patient_id} "
                        #       f"max_per_modality_per_patient = {max_per_modality_per_patient}"
                        #       f" -- patient_dates = {num_patient_dates} "
                        #       f"--> max_per_modality_per_date_per_patient = {max_per_modality_per_date_per_patient}")

                        for i, date in enumerate(meta[patient_id].keys()):
                            date_to_right_left_lists_temp = {}  # reset for each patient_id
                            # meta[patient_id][date].keys():
                            m = self.modality
                            # date_to_right_left_lists_temp[m] = {'R': [], 'L': []}
                            uuids_lats = meta[patient_id][date][m]  # tuples (data_XXX/uuid, laterality)
                            right_paths = [pjoin(self.root, 'data', pjoin(*leaf[0].split("\\"))) for leaf in uuids_lats
                                           if leaf[1] == 'R']
                            left_paths = [pjoin(self.root, 'data', pjoin(*leaf[0].split("\\"))) for leaf in uuids_lats
                                          if leaf[1] == 'L']

                            if 'small' in self.split:
                                right_paths = [p + '.png' for p in right_paths]
                                left_paths = [p + '.png' for p in left_paths]

                            # optionally apply ood filtering
                            # fixme ood
                            if isinstance(ood_modality_to_patient_to_filename_to_score, dict):
                                raise NotImplementedError("ood filtering not implemented yet")
                            else:
                                # from each list, keep at most n_max_dcm_per_modality uniformly at random
                                if len(right_paths) > max_per_modality_per_date_per_patient // 2:
                                    right_paths = random.sample(right_paths, max_per_modality_per_date_per_patient // 2)
                                elif len(right_paths) <= max_per_modality_per_date_per_patient // 2:
                                    right_paths = right_paths
                                if len(left_paths) > max_per_modality_per_date_per_patient // 2:
                                    left_paths = random.sample(left_paths, max_per_modality_per_date_per_patient // 2)
                                elif len(left_paths) <= max_per_modality_per_date_per_patient // 2:
                                    left_paths = left_paths

                            # stats for laterality
                            stats_lat['R'] += len(right_paths)
                            stats_lat['L'] += len(left_paths)
                            stats_patient[patient_id][m] += len(right_paths + left_paths)

                            date_to_right_left_lists_temp['R'] = right_paths
                            date_to_right_left_lists_temp['L'] = left_paths
                            all_paths += right_paths + left_paths
                            all_dates += [date] * (len(right_paths) + len(left_paths))
                            all_patients += [patient_id] * (len(right_paths) + len(left_paths))

                            # print(f"    date:{i+1}/{num_patient_dates} "
                            #       f"new: {m} images: {len(right_paths + left_paths)}"
                            #       f" (L:{len(left_paths)} R:{len(right_paths)})")

                            # show images in right_paths and left_paths with PIL
                            # if num_patient_dates > 10:
                            #     for p in right_paths:
                            #         img = Image.open(p).convert('RGB')
                            #         img.show()

                stats = {'patient': stats_patient, 'laterality': stats_lat}
                # save stats using datetime in the filename
                date_now = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
                stas_fname = f'stats_{self.split}_{self.modality}_{self.max_per_modality_per_patient}_{date_now}.json'
                with open(os.path.join(self.log_dir, stas_fname), 'w') as fn:
                    json.dump(stats, fn, indent=4)

            else:
                raise ValueError(
                    f"stratification_method {self.stratification_method} not in {self.valid_stratification_methods}")
        else:
            raise ValueError(f"get_paths_method {get_paths_method} not in {self.valid_get_paths_methods}")

        return all_paths, all_dates, all_patients

    def __getitem__(self, index):
        metadata = {'index': index}
        p = self.image_paths[index]  # a tuple of paths to images of different modalities -- same subject/laterality
        image = Image.open(p).convert('RGB')
        image, _, metadata = self.common_transforms((image, image, metadata))
        img_tensor, metadata = self.img_transforms((image, metadata))
        if self.img_channels == 1:  # keep only first channel
            img_tensor = img_tensor[0].unsqueeze(0)
        if self.debug:
            metadata.update({'patient': None})
            printlog(f"metadata: {metadata}")
            ToPILImage()(img_tensor).show()
        if self.use_omnivision_api:
            return self.create_sample(index, img_tensor)
        else:
            if self.return_metadata:
                return img_tensor, metadata
            else:
                return img_tensor

    @staticmethod
    def create_sample(idx, img):
        return Sample(data=img, data_idx=idx)

    def __len__(self):
        return len(self.image_paths)


class OCTIRPaired(Dataset):
    valid_splits = ['all', 'train', 'val', 'train_small', 'val_small']
    valid_modalities = ["OCT", "FA", "IR"]
    valid_modality_modes = {"OCT": ["slices", "slices_v", "slices_h"], "FA": ["single"], "IR": ["en_face"]}
    val_set_size = 1000  # number of patients in the validation set
    # modality_dir_name = {"OCT": "OCT_volume", "FA": "SLO_FA_single", "IR":  "SLO_IR_en_face"}
    valid_get_paths_methods = ['naive', 'stratified']
    valid_stratification_methods = ['cap_per_patient']
    min_per_modality_per_date_per_patient = {"OCT": 6, "FA": 2, "IR": 4}

    def __init__(self,
                 root,
                 metafile: str,
                 transforms_dict: Dict,
                 split: Union[str, List, Tuple] = ('train', 'train'),
                 modalities: Union[Tuple, List] = ('OCT', 'FA'),
                 modes: Union[Tuple, List] = ('slices', 'single'),
                 debug: bool = False,
                 img_channels: Tuple = (1, 1),
                 return_metadata=True,
                 use_omnivision_api=False,
                 get_paths_method='naive',
                 stratification_method=None,
                 max_per_modality_per_patient: Union[Tuple, List] = (8, 6),
                 cap_dates_per_patient: Tuple = (None, None),
                 ood_modality_to_patient_to_filename_to_score: str = None,
                 ood_remove_percent=0.5,
                 log_dir=None):

        """ OCTIR dataset for 2D data that loads the data from paths

        :param root: data path root (e.x home/data/KEKI/) were all the data for a single modality is stored
        :param metafile: path to the json file containing the metadata (nested dict id_to_modality_to_dcm)
        :param transforms_dict: dictionary of transforms to apply to the images of each modality
               transforms_dict[modality]['common']
               transforms_dict[modality]['img']

        :param split: split of the dataset to use, can be 'all', 'train', 'val' or a list of indices
        :param debug: if true shows images with PIL
        :param img_channels: number of channels in images (1 or 3)
        :param return_metadata: if True returns additional information in a dictionary (default: True)
        :param use_omnivision_api: if True returns a Sample object for the omnivision dataloader (default: False)
        :param get_paths_method:
        :param stratification_method:
        :param max_per_modality_per_patient:
        :param ood_modality_to_patient_to_filename_to_score: a file with a list of ood img
        :param ood_remove_percent: top-% of images per subject wrt to their ood scores that are ignored when sampling
        :param log_dir: path to the log directory
        """
        super().__init__()
        self.debug = debug  # set to true will print a lot of info and also show images in self.__getitem__()

        ################################# Sanity Checks ##########################################
        assert split in self.valid_splits, f'split {split} is not in valid_modes {self.valid_splits}'
        assert os.path.exists(root), f'data path {root} does not exist'
        assert os.path.exists(metafile), f'metafile {metafile} does not exist'
        for mode, modality in zip(modes, modalities):
            assert modality in self.valid_modalities, f"{modality} not in {self.valid_modalities}"
            assert mode in self.valid_modality_modes[modality], f"mode '{mode}' for modality '{modality}' " \
                                                                f"not in {self.valid_modality_modes[modality]}"
        assert get_paths_method in self.valid_get_paths_methods, \
            f"get_paths_method {get_paths_method} not in {self.valid_get_paths_methods}"
        assert stratification_method in self.valid_stratification_methods, \
            f"stratification_method {stratification_method} not in {self.valid_stratification_methods}"

        #########################################################################################

        self.root = root
        self.log_dir = log_dir if log_dir is not None else str(Path(log_dir).parent)
        self.metafile = metafile

        self.experiment = 1  # hardcoded no effect here
        self.split = split  # "train", "test", "val"
        self.modalities = modalities  # e.g OCT, FA etc
        self.modality_tags = modalities  # different from KEKI, OCTIR has the same modalities and modalities_tags
        self.modes = modes
        self.img_channels = img_channels

        self.suffix = '.png'
        # self.return_filename = False
        self.return_metadata = return_metadata
        self.use_omnivision_api = use_omnivision_api
        self.dataset_view = f"OCTIR_{self.modalities}_{self.modes}"

        ################################# Augmentations ########################################
        # modality specific augmentation pipelines
        self.common_transforms = {}  # Dict[m, Compose(transforms_dict[m]['common'])]
        self.img_transforms = {}  # Dict[m, Compose(transforms_dict[m]['img'])]

        # workaround utils.transforms to ensure the same flipping is applied on all modalities
        # if FlipNP is used in common transforms then remove it and apply it self.__getitem__()
        self.do_flip = all(any([isinstance(item, FlipNP) for item in transforms_dict[m]['common']])
                           for m in self.modalities)

        for m in self.modalities:
            if self.do_flip:  # remove flip from common transforms
                for i, t in enumerate(transforms_dict[m]['common']):
                    if isinstance(t, FlipNP):
                        transforms_dict[m]['common'].pop(i)
                        break
            self.common_transforms[m] = Compose(transforms_dict[m]['common'])
            self.img_transforms[m] = Compose(transforms_dict[m]['img'])

        ################################# Paths to data / Stratification ########################################
        # method to collect paths to images
        self.get_paths_method = get_paths_method
        self.stratification_method = stratification_method
        # per modality cap of dates per patient
        if all([i is None for i in cap_dates_per_patient]):
            self.cap_dates_per_modality = None
        else:
            self.cap_dates_per_modality = dict()
            for modality, cap in zip(self.modality_tags, cap_dates_per_patient):
                if cap is None:
                    cap = 10000  # set to a very large number
                self.cap_dates_per_modality[modality] = cap

        # max number of images per modality per patient
        self.max_per_modality_per_patient = dict()
        for modality, num in zip(self.modality_tags, max_per_modality_per_patient):
            self.max_per_modality_per_patient[modality] = num  # max number of images per modality per patient

        # ood not applied on the fly anymore : todo safe remove
        # self.ood_remove_percent = ood_remove_percent
        # if ood_modality_to_patient_to_filename_to_score is not None:
        #     with open(os.path.join(self.root, ood_modality_to_patient_to_filename_to_score), 'r') as fn:
        #         ood_modality_to_patient_to_filename_to_score = json.load(fn)

        self.image_pairs = self.get_all_img_files_paired(get_paths_method, None)

        ################################# Summarize ##############################################
        printlog(
            f'{self.dataset_view} data found '
            f' split = {self.split}, all_pairs: {self.__len__()} -- all_images (not unique): '
            f'{len(self.modalities) * self.__len__()} -')

        printlog("stratification: {}".format(self.stratification_method))
        printlog(f"max_per_modality_per_patient: {self.max_per_modality_per_patient}")
        printlog(f"cap_dates_per_modality: {self.cap_dates_per_modality}")
        printlog(f"OCTIRPaired dataloader: using omnivision api for batching: {self.use_omnivision_api}")

    def get_all_img_files_paired(self,
                                 get_paths_method: str,
                                 ood_modality_to_patient_to_filename_to_score=None,
                                 ood_filtering: str = "remove_top_then_random",
                                 pair_generation_method='same_laterality_all_combs'):
        """
        :param get_paths_method: string specifying how to get the paths to the images
        :param ood_modality_to_patient_to_filename_to_score: (Optional) Dict[modality, patient_id, filename] = ood_score
        :param ood_filtering: (Optional) string specifying how to filter the ood data
                                1) "remove_top_then_random" - remove n% (defaults to 50%) of data with top ood score,
                                 then randomly sample max_per_modality_per_patient from the rest
                                2) "keep_top" - keep at most max_per_modality_per_patient of data sorted descending
                                    with lowest ood scores
        :param pair_generation_method:
        :return: all_pairs (a list of pairs of paths to imgs)
        """
        random.seed(0)  # to ensure we always "randomly" sample the same files per patient
        if ood_modality_to_patient_to_filename_to_score is not None:
            assert ood_filtering in ["remove_top_then_random", "keep_top"], \
                f"ood_filtering {ood_filtering} not in ['remove_top_then_random', 'keep_top']"

        with open(os.path.join(self.root, self.metafile), 'r') as fn:
            # with open(os.path.join(self.metafile), 'r') as fn:
            meta: meta_type = json.load(fn)

        all_pairs = []
        if get_paths_method == 'naive':
            raise NotImplementedError(f"get_paths_method {get_paths_method} not supported for OCTIRPaired")
        elif get_paths_method == 'stratified':
            # in this case data are organized as:
            # root
            #   - data
            #     - data_XXX
            #       - {uuid}.png
            # we apply a stratification heuristic to keep only a subset of the data per patient
            # data from date and each laterality are stratified separately

            # for each patient, for each modality keep at most n_max_dcm_per_modality
            if self.stratification_method == 'cap_per_patient':
                random.seed(0)  # to ensure we always "randomly" sample the same files per patient

                patients_to_keep = list(meta.keys())
                if self.split == 'train':
                    # we keep all patients except the last N (hardcoded)
                    patients_to_keep = patients_to_keep[:-self.val_set_size]
                elif self.split == 'val':
                    # we keep only the last N patients (hardcoded)
                    patients_to_keep = patients_to_keep[-self.val_set_size:]
                elif self.split == 'train_small': # 600 patients
                    assert self.debug, "train_small only works in debug mode"
                    # we keep only the first N patients (hardcoded)
                    patients_to_keep = patients_to_keep[:-500]
                elif self.split == 'val_small': # 600 patients
                    assert self.debug, "val_small only works in debug mode"
                    # we keep only the first N patients (hardcoded)
                    patients_to_keep = patients_to_keep[-500:]
                else:
                    raise NotImplementedError(f"split {self.split} not supported for OCTIRPaired")

                num_patients = len(patients_to_keep)
                num_patients_with_no_dates = 0
                printlog(f"num_patients = {num_patients} -- {self.split}")
                stats_lat = {'R': 0, 'L': 0}
                stats_patient = dict()
                for patient_id in meta.keys():
                    if patient_id in patients_to_keep:  # only collect paths for patients in the train or val set

                        dates = list(meta[patient_id].keys())
                        if self.debug:
                            printlog(f"Patient {patient_id} has {len(dates)} dates")
                        if isinstance(self.cap_dates_per_modality, dict):
                            # random uniform sampling of dates
                            # todo: for now we assume all modalities have the same cap and just use OCT
                            cap_dates_per_patient = self.cap_dates_per_modality['OCT']
                            dates = random.sample(dates, min(len(dates), cap_dates_per_patient))
                            if self.debug:
                                printlog(f"Patient {patient_id} has {len(dates)} dates after sampling with "
                                         f"cap {cap_dates_per_patient}")

                        num_patient_dates = len(dates)  # number of dates for this patient
                        if num_patient_dates == 0:
                            printlog(f"Patient {patient_id} has no dates")
                            num_patients -= 1
                            num_patients_with_no_dates += 1
                            continue

                        stats_patient[patient_id] = {m: 0 for m in self.modalities}
                        stats_patient.update({'num_dates': num_patient_dates})

                        # note: we assume all dates have all modalities
                        # depends on the number of dates for this patient hence we need to compute it for each patient
                        max_per_modality_per_date_per_patient = {
                            m: max(self.max_per_modality_per_patient[m] // num_patient_dates,
                                   self.min_per_modality_per_date_per_patient[m])
                            for m in self.modalities}

                        if self.debug:
                            print(f"Patient {patient_id} "
                                  f"max_per_modality_per_patient = "
                                  f"{self.max_per_modality_per_patient}"
                                  f" -- patient_dates = {num_patient_dates} "
                                  f"--> max_per_modality_per_date_per_patient = {max_per_modality_per_date_per_patient}")

                        patient_pairs = {'R': [], 'L': []}  # fixme: clean as this is only for debug to show images
                        for i, date in enumerate(dates):
                            modality_to_right_left_lists_temp = {}  # reset for each patient_
                            modality_to_right_left_lists_temp: Dict[modality_name, Dict[laterality_name, List[str]]]
                            for modality in meta[patient_id][date].keys():
                                if modality in self.modalities:
                                    modality_to_right_left_lists_temp[modality] = {'R': [], 'L': []}
                                    # get two lists from meta: one for right and one for left eye dcms
                                    uuids_lats = meta[patient_id][date][modality]  # tuples (data_XXX/uuid, laterality)
                                    right_paths = [pjoin(self.root, 'data', pjoin(*leaf[0].split("\\")))
                                                   for leaf in uuids_lats if leaf[1] == 'R']
                                    left_paths = [pjoin(self.root, 'data', pjoin(*leaf[0].split("\\")))
                                                  for leaf in uuids_lats if leaf[1] == 'L']

                                    # todo ammend this at some point to use the new naming convention
                                    if 'small' in self.split:
                                        right_paths = [p + '.png' for p in right_paths]
                                        left_paths = [p + '.png' for p in left_paths]
                                    # optionally apply ood filtering
                                    # fixme ood
                                    if isinstance(ood_modality_to_patient_to_filename_to_score, dict):
                                        raise NotImplementedError("ood done externally for OCTIR ")
                                    else:
                                        # from each list, keep at most n_max_dcm_per_modality uniformly at random
                                        if len(right_paths) > max_per_modality_per_date_per_patient[modality] // 2:
                                            right_paths = random.sample(right_paths,
                                                                        max_per_modality_per_date_per_patient[
                                                                            modality] // 2)
                                        elif len(right_paths) <= max_per_modality_per_date_per_patient[modality] // 2:
                                            right_paths = right_paths
                                        if len(left_paths) > max_per_modality_per_date_per_patient[modality] // 2:
                                            left_paths = random.sample(left_paths,
                                                                       max_per_modality_per_date_per_patient[
                                                                           modality] // 2)
                                        elif len(left_paths) <= max_per_modality_per_date_per_patient[modality] // 2:
                                            left_paths = left_paths

                                    # stats for laterality
                                    stats_lat['R'] += len(right_paths)
                                    stats_lat['L'] += len(left_paths)
                                    stats_patient[patient_id][modality] += len(right_paths + left_paths)

                                    # collect left right paths
                                    modality_to_right_left_lists_temp[modality]['R'] = right_paths
                                    modality_to_right_left_lists_temp[modality]['L'] = left_paths
                                    if self.debug:
                                        print(f"    date:{i + 1}/{num_patient_dates} "
                                              f"new: {modality} images: {len(right_paths + left_paths)}"
                                              f" (L:{len(left_paths)} R:{len(right_paths)})")

                            # For given patient_id and a given date
                            # try:
                            if pair_generation_method == 'same_laterality_all_combs':
                                # for each laterality seperately (i.e for each of the patient's eyes on that date)
                                for lat in ['R', 'L']:
                                    # get all modality pairs (OCT, IR)
                                    for path_OCT in modality_to_right_left_lists_temp['OCT'][lat]:
                                        for path_IR in modality_to_right_left_lists_temp['IR'][lat]:
                                            all_pairs.append((path_OCT, path_IR))
                                            patient_pairs[lat].append((path_OCT, path_IR))
                            else:
                                raise NotImplementedError(f"pair_generation_method={pair_generation_method}")
                            # except:
                            #     print(f'patient_id: {patient_id} date: {date}')
                            #     a =1

                        # # show images in right_paths and left_paths with PIL
                        # if num_patient_dates > 10:
                        #     for pair in patient_pairs['R']:
                        #         img = Image.open(pair[0]).convert('RGB')
                        #         img.show()
                        #     for pair in patient_pairs['R']:
                        #         img = Image.open(pair[1]).convert('RGB')
                        #         img.show()
                stats = {'patient': stats_patient, 'laterality': stats_lat}
                # save stats using datetime in the filename
                date = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
                stas_fname = f'stats_{self.split}_'
                suffix = 'cap'
                for mt in self.modality_tags:
                    stas_fname += f'{mt}_'
                    suffix += f'{self.max_per_modality_per_patient[mt]}_'
                stas_fname += f'{suffix}{date}.json'
                with open(os.path.join(self.log_dir, stas_fname), 'w') as fn:
                    json.dump(stats, fn, indent=4)

                printlog(f"After parsing meta: updated num_patients = {num_patients} "
                         f"(empty: {num_patients_with_no_dates}) -- {self.split}")

            else:
                raise ValueError(
                    f"stratification_method {self.stratification_method} not in {self.valid_stratification_methods}")
        else:
            raise ValueError(f"get_paths_method {get_paths_method} not in {self.valid_get_paths_methods}")

        return all_pairs

    def __getitem__(self, index):
        """
        Single hdf5 file for all splits
        different dicom name to index dictionaries used
        Args:
            index (int): Index that is 0-N for each set where N is the size of the set (ex train or val)
            Note that when indexing the val set and because the underlying hdf5 is the same for all sets in the split,
             the index is offset by the size of the train split
        Returns:

        """
        metadata = {'index': index}
        pair = self.image_pairs[index]  # a tuple of paths to images of different modalities -- same subject/laterality
        data = {}
        meta = {}

        h_flip = False
        if self.do_flip:
            # do flip outside utils.transforms because we want it to be applied to all images
            h_flip = torch.rand(1) < 0.5

        for ind, (path_to_img, m) in enumerate(zip(pair, self.modalities)):
            image = Image.open(path_to_img).convert('RGB')
            image, _, metadata = self.common_transforms[m]((image, image, metadata))
            img_tensor, metadata = self.img_transforms[m]((image, metadata))
            if self.img_channels[ind] == 1:
                # keep only first channel as they are duplicates
                # img tensor is H,W,C
                img_tensor = img_tensor[0].unsqueeze(0)
            if h_flip:
                img_tensor = torch.flip(img_tensor, dims=[2])
            data.update({m: img_tensor})
            meta.update({m: metadata})

        if self.debug:
            printlog(f"metadata: {metadata}")
            for m in self.modalities:
                ToPILImage()(data[m]).show()

        if self.use_omnivision_api:
            return self.create_sample(index, data)
        else:
            return data

    @staticmethod
    def create_sample(idx, img):
        return Sample(
            data=img, data_idx=idx
        )

    def __len__(self):
        return len(self.image_pairs)
