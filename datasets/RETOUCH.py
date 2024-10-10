import pandas as pd
from utils import DATASETS_INFO, printlog


def get_retouch_dataframes(config: dict):
    # Make dataframes for the training and the validation set
    assert 'data' in config
    dataset = config['data']['dataset']
    assert dataset == 'RETOUCH', f'dataset must be RETOUCH instead got {dataset}'
    df = pd.read_csv(f"{config['data_path']}/retouch_data.csv")
    if 'random_split' in config['data']:
        print("***Legacy mode: random split of all data used, instead of split of videos!***")
        train = df.sample(frac=config['data']['random_split'][0]).copy()
        valid = df.drop(train.index).copy()
        split_of_rest = config['data']['random_split'][1] / (1 - config['data']['random_split'][0])
        valid = valid.sample(frac=split_of_rest)
    else:
        if config['mode'] == 'submission_inference' and config['data']['split'] == 2:
            df = pd.read_csv(f"{config['data_path']}/retouch_data_test.csv")

            _, valid_videos = DATASETS_INFO[dataset].DATA_SPLITS[int(config['data']['split'])]
            valid = df.loc[(df['vid_num'].isin(valid_videos))].copy()  # No prop lbl in valid
            info_string = "Dataframes created. Number of records external test: {:06d}\n".format(len(valid.index))
            printlog(f" dataset {dataset}")
            printlog(info_string)
            return valid, valid  # return valid twice to avoid complicating the code in BaseMAnager
        else:
            train_videos, valid_videos = DATASETS_INFO[dataset].DATA_SPLITS[int(config['data']['split'])]
            train = df.loc[df['vid_num'].isin(train_videos)].copy()
            valid = df.loc[(df['vid_num'].isin(valid_videos))].copy()  # No prop lbl in valid
            info_string = "Dataframes created. Number of records training / validation: {:06d} / {:06d}\n" \
                          "                    Actual data split training / validation: {:.3f}  / {:.3f}" \
                .format(len(train.index), len(valid.index), len(train.index) / len(df), len(valid.index) / len(df))
            printlog(f" dataset {dataset}")
            printlog(info_string)
            return train, valid


if __name__ == '__main__':
    # import pathlib
    import torch
    from utils import parse_transform_lists
    import json
    # import cv2
    # from torch.nn import functional as F
    # from utils import Pad, RandomResize, RandomCropImgLbl, Resize, FlipNP, to_numpy, pil_plot_tensor, to_comb_image
    from datasets.Dataset_from_df import DatasetFromDF
    # import PIL.Image as Image
    from utils import printlog, to_numpy, to_comb_image, un_normalise

    # data_path = "D:\\datasets\\RETOUCH\\labelled_preprocessed"
    data_path = "C:\\Users\\thopis\\Documents\\datasets\\RETOUCH\\labelled_preprocessed"
    d = {"dataset":'RETOUCH', "experiment":1}
    path_to_config = '../configs/RETOUCH/vitd_mae_keki.json'

    with open(path_to_config, 'r') as f:
        config = json.load(f)
    config['data_path'] = data_path
    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    if 'torchvision_normalise' in transforms_list:
        del transforms_list[-1]

    transforms_dict = parse_transform_lists(transforms_list, transforms_values, **d)
    transforms_list_val = config['data']['transforms_val']
    transforms_values_val = config['data']['transform_values_val']

    if 'torchvision_normalise' in transforms_list_val:
        del transforms_list_val[-1]

    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, **d)

    train_df, valid_df = get_retouch_dataframes(config)
    train_set = DatasetFromDF(train_df, 1, transforms_dict, dataset='RETOUCH', data_path=data_path, debug=False)
    valid_set = DatasetFromDF(valid_df, 1, transforms_dict_val, dataset='RETOUCH', data_path=data_path, debug=True)

    issues = []
    train_set.return_filename = True
    hs=[]
    ws = []
    for i, ret in enumerate(valid_set):
        hs.append(ret[0].shape[1])
        ws.append(ret[0].shape[2])
        present_classes = torch.unique(ret[1])
        print(ret[-1])
        image = ret[0]
        pred = ret[1]
        # to_comb_image(image, pred, None, 1, 'RETOUCH', save=f'datasets/debug/RT_lbl_{i}.png')

        # elif 15 in present_classes:
        #     issues.append([ret[-1], present_classes])
        #     print('bus found !!!! ')
        #     print(present_classes)
        #     pil_plot_tensor(ret[0], is_rgb=True)
        #     pil_plot_tensor(ret[1], is_rgb=False)

        # a = 1
    # print(max(hs))
    # print(max(ws))