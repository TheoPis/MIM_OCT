import os
import pathlib
import json
from utils.defaults import DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_NESTED_DICT, DATASETS_INFO
from utils.logger import printlog
from torchvision.transforms import ToPILImage, ColorJitter, ToTensor, Normalize, RandomApply
from utils.transforms import BlurPIL, RandomCropImgLbl, RandomResize, Resize, RandomCropImg
from utils.np_transforms import AffineNP, PadNP, FlipNP


def parse_config(file_path, user, device, dataset, parallel):
    printlog(f"*** Parsing config file '{file_path}")
    # Load config
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found at given path '{}'".format(file_path))
        exit(1)
    # Fill in correct paths
    config_path = pathlib.Path('configs')
    # config_path = pathlib.Path(file_path).parent.parent
    with open(config_path / 'path_info.json', 'r') as f:
        path_info = json.load(f)  # Dict: keys are user codes, values are a list of 'data_path', 'log_path' (absolute)
    if dataset != -1:  # if dataset provided as an argument for main
        assert(dataset in ['KEKI', 'MKEKI', 'OctBiom', 'RETOUCH', 'DR', 'OCTIR', 'OCTID', 'OCTDL', 'AROI'])
        config_dict['data']['dataset'] = dataset
        # print('dataset set to {} '.format(dataset))
    else:
        dataset = config_dict['data']['dataset']

    dataset_user_suffix = ''
    if dataset == 'KEKI':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_KEKI'

    elif dataset == 'MKEKI':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_MKEKI'

    elif dataset == 'OctBiom':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_OctBiom'

    elif dataset == 'RETOUCH':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_RETOUCH'

    elif dataset == 'AROI':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_AROI'
    elif dataset == 'DR':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_DR'

    elif dataset == 'OCTIR':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_OCTIR'

    elif dataset == 'OCTID':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_OCTID'

    elif dataset == 'OCTDL':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_OCTDL'

    elif dataset == 'RAVIR':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_RAVIR'

    elif dataset == 'VA':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_VA'

    elif dataset == 'OCT5K':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_OCT5K'

    elif dataset == 'OLIVES':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_OLIVES'
    else:
        ValueError(f"Dataset '{dataset}' not found in configs/path_info.json")

    # ddp
    printlog(f"using ddp: {parallel}")
    config_dict['parallel'] = parallel

    if user+dataset_user_suffix in path_info:
        config_dict.update({
            'data_path': pathlib.Path(path_info[user+dataset_user_suffix][0]),
            'log_path': pathlib.Path(path_info[user+dataset_user_suffix][1]),
            # 'ss_pretrained_path': path_info['ss_pretrained_{}'.format(user)][0] # todo safe remove
        })

        # todo - this optional for now
        if len(path_info[user+dataset_user_suffix]) == 3:
            config_dict.update({
                'internal_pretrained_path': pathlib.Path(path_info[user+dataset_user_suffix][2])
            })

    else:
        ValueError("User '{}' not found in configs/path_info.json".format(user))

    if f'pytorch_checkpoints_{user}' in path_info:
        printlog(f"external checkpoints stored at {path_info[f'pytorch_checkpoints_{user}']}")
        config_dict['external_checkpoints'] = pathlib.Path(path_info[f'pytorch_checkpoints_{user}'][0])
        assert config_dict['external_checkpoints'].exists(), \
            'external_checkpoints  {} does not exist'.format(config_dict['external_checkpoints'])
    else:
        printlog(f"Warning: pytorch_checkpoints_{user} not found in configs/path_info.json so setting to None")
        config_dict['external_checkpoints'] = None

    config_dict['user'] = user
    if 'ubelix' in config_dict['user']:
        config_dict['data_path'] = pathlib.Path(os.path.expandvars(str(config_dict['data_path'])))
    config_dict['dataset_suffix'] = dataset_user_suffix

    assert config_dict['data_path'].exists(), 'data_path {} does not exist'.format(config_dict['data_path'])
    assert config_dict['log_path'].exists(),  'log_path  {} does not exist'.format(config_dict['log_path'])
    assert config_dict['external_checkpoints'].exists(), \
        'external_checkpoints  {} does not exist'.format(config_dict['external_checkpoints'])
    if 'internal_pretrained_path' in config_dict:
        assert config_dict['internal_pretrained_path'].exists(), 'internal_pretrained_path {} does not exist'.format(
            config_dict['internal_pretrained_path'])

    # Fill in GPU device if applicable
    if isinstance(device, list):
        config_dict['gpu_device'] = device
    elif device >= 0:  # Only update config if user entered a device (default otherwise -1)
        config_dict['gpu_device'] = device

    # Make sure all ne0000cessary default values exist
    default_dict = DEFAULT_CONFIG_DICT.copy()
    default_dict.update(config_dict)  # Keeps all default values not overwritten by the passed config
    nested_default_dicts = DEFAULT_CONFIG_NESTED_DICT.copy()
    for k, v in nested_default_dicts.items():  # Go through the nested dicts, set as default first, then update
        default_dict[k] = v  # reset to default values
        default_dict[k].update(config_dict[k])  # Overwrite defaults with the passed config values
    # printlog(pprint.pformat(default_dict))
    # Extra config bits needed
    if type(default_dict['data']['transform_values']) == list:
        for i in range(len(default_dict['data']['transform_values'])):
            default_dict['data']['transform_values'][i]['experiment'] = 1
        # default_dict['data']['transform_values']['experiment'] = default_dict['data']['experiment'][0]
    else:
        default_dict['data']['transform_values']['experiment'] = default_dict['data']['experiment']
    printlog(f"user+dataset_suffix {user + dataset_user_suffix} {config_dict['data_path']}")
    printlog(f"'data_path':  {config_dict['data_path']}")
    printlog(f"'log_path':  {config_dict['log_path']}")
    printlog(f"'internal_pretrained_path':  {config_dict.get('internal_pretrained_path', 'None')}")
    printlog(f"'external_checkpoints':  {config_dict['external_checkpoints']}")
    printlog(f"'DDP':  {config_dict['parallel']}")
    printlog(f"*** Done parsing config file ***")
    return default_dict


def parse_transform_lists(transform_list, transform_values, dataset, experiment=1):
    """Helper function to parse given dataset transform list. Order of things:
    - first the 'common' transforms are applied. At this point, the input is expected to be a numpy array.
    - then the img and lbl transforms are each applied as necessary. The input is expected to be a numpy array, the
        output will be a tensor, as required by PyTorch"""
    d = {"dataset": dataset, "experiment": experiment}
    printlog(f"------parsing transform list {transform_list}------")
    transforms_dict = \
        {
            'common': [],
            'img': [],
            'lbl': []
        }

    # Step 1: Go through all transforms that need to go into the 'commom' section, i.e. which rely on using the same
    # random parameters on both the image and the label: generally actual augmentation transforms.
    #   Input: np.ndarray/PIL; Output: np.ndarray
    # first = True
    for t in transform_list:
        if t == 'flip':
            transforms_dict['common'].append(FlipNP())
            # if first:
            #     transforms_dict = determine_affine(transform_list, transforms_dict, num_classes)
            #     first = False
        elif t == 'pad':
            assert dataset == 'CADIS'
            # this is for CADIS only
            # Needs to be added to img and lbl, train and valid
            if 'crop' not in transform_list:  # Padding only necessary if no cropping has happened
                for obj in ['img', 'lbl']:
                    transforms_dict[obj].append(PadNP(ver=(2, 2), hor=(0, 0), padding_mode='reflect'))

        elif t == 'resize':
            fit_stride = transform_values['fit_stride'] if 'fit_stride' in transform_values else None
            target_size = transform_values['target_size'] if 'target_size' in transform_values else None
            min_side_length = transform_values['min_side_length'] if 'min_size_length' in transform_values else None
            transforms_dict['common'].append(Resize(**d,
                                                    target_size=target_size,
                                                    min_side_length=min_side_length,
                                                    fit_stride=fit_stride))
        elif t == 'resize_val':
            # e.x Pascal Context: resizes with min_side, aspect ratio preserved,
            # pad to fit_Stride and returns original labels for validation
            transforms_dict['common'].append(Resize(**d,
                                                    min_side_length=transform_values['min_side_length'],
                                                    fit_stride=transform_values['fit_stride_val'],
                                                    return_original_labels=True))
        elif t == 'random_scale':
            aspect_range = transform_values['aspect_range'] if 'aspect_range' in transform_values else [0.9, 1.1]
            p_random_scale = transform_values['p_random_scale'] if 'p_random_scale' in transform_values else 1.0
            transforms_dict['common'].append(RandomResize(**d,
                                                          scale_range=transform_values['scale_range'],
                                                          target_size=transform_values['crop_shape'],
                                                          aspect_range=aspect_range,
                                                          probability=p_random_scale))

        elif t == 'RandomCropImgLbl':
            max_ratio = transform_values['crop_class_max_ratio'] if 'crop_class_max_ratio' in transform_values else None
            transforms_dict['common'].append(RandomCropImgLbl(**d,
                                                              shape=transform_values['crop_shape'],
                                                              crop_class_max_ratio=max_ratio))

        elif t == 'RandomCropImg':
            transforms_dict['img'].append(RandomCropImg(**d, shape=transform_values['crop_shape']))

        elif t == 'blur':
            transforms_dict['img'].append(BlurPIL(**d,
                                                  probability=.05,
                                                  kernel_limits=(3, 7)))

        elif t == 'colorjitter':
            raise NotImplementedError("colorjitter not implemented, use colorjitter_oct instead")

        elif t == 'colorjitter_oct':
            transforms_dict['img'].append(ToPILMeta())
            # transforms_dict['lbl'].append(ToPILImage())

            p = transform_values.get('colorjitter_p', 1.0)  # fixme: this has no effect, it is always going to be 1.0

            brightness = transform_values.get('brightness', (0.5, 1.5))
            contrast = transform_values.get('contrast', (0.5, 1.5))
            saturation = transform_values.get('saturation', (0.5, 1.5))

            colorjitter_func = ColorJitter(brightness=brightness,
                                           contrast=contrast,
                                           saturation=saturation)

            # ColorjitterMeta: wrapper for forwarding metadata
            # rnd_color_jitter =  RandomApply([colorjitter_func], p=p)
            jit = ColorjitterMeta(colorjitter_func)
            transforms_dict['img'].append(jit)
            printlog(f'{jit} with p {p}')

        elif t in ['torchvision_normalise']:
            continue
        else:
            raise ValueError(f' transform {t} not found')

    for obj in ['img', 'lbl']:
        transforms_dict[obj].append(ToTensorMeta())
    if 'torchvision_normalise' in transform_list:
        transforms_dict['img'].append(NormalizeMeta(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    printlog("----------------------------------")
    return transforms_dict


class TransformMeta:

    def __init__(self, transform):
        """Wrapper for transforms applied to some input (ex image) and also forward the metadata dict"""
        self.t = transform

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            # assume inputs[0] = img, inputs[1] = dict with metadata from previous transforms
            return self.t(inputs[0]), inputs[1]
        else:
            return self.t(inputs)


class NormalizeMeta(TransformMeta):
    def __init__(self, mean, std):
        transform = Normalize(mean, std)
        super().__init__(transform)


class ToPILMeta(TransformMeta):
    def __init__(self):
        transform = ToPILImage()
        super().__init__(transform)


class ColorjitterMeta(TransformMeta):
    def __init__(self, cjit: ColorJitter):
        super().__init__(cjit)

    def __repr__(self):
        return self.t.__repr__()


class ToTensorMeta(TransformMeta):
    def __init__(self):
        super().__init__(ToTensor())


def determine_affine(transform_list, transforms_dict, num_classes):
    rotation = 0
    rot_centre_offset = (.2, .2)
    shift = 0
    shear = (0, 0)
    shear_centre_offset = (.2, .2)
    set_affine = False
    if 'rot' in transform_list:
        rotation = 15
        set_affine = True
    if 'shift' in transform_list:
        shift = .1
        set_affine = True
    if 'shear' in transform_list:
        shear = (.1, .1)
        set_affine = True
    if 'affine' in transform_list:
        rotation = 10
        shear = (.1, .1)
        rot_centre_offset = (.1, .1)
        set_affine = True

    if set_affine:
        transforms_dict['train']['common'].append(AffineNP(num_classes=num_classes,
                                                           crop_to_fit=False,
                                                           rotation=rotation,
                                                           rot_centre_offset=rot_centre_offset,
                                                           shift=shift,
                                                           shear=shear,
                                                           shear_centre_offset=shear_centre_offset))

    return transforms_dict
