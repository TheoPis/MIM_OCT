import cv2
import os
import json
import torch
from .defaults import DATASETS_INFO, get_cityscapes_colormap, get_cadis_colormap, get_pascalc_colormap,\
    get_ade20k_colormap, get_iacl_colormap, get_retouch_colormap, get_aroi_colormap, get_ravir_colormap,\
    get_OCT5K_colormap
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from typing import Any, Union, List
from collections import OrderedDict
# from PIL import Image
import imageio
import random
import datetime


def time_funcall(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        end = datetime.datetime.now()
        print('                 {} {:.1f}'.format(func.__name__, (end - start).total_seconds()*1000))
        return result
    return wrapper

# def show_tensor2pil(x: torch.Tensor):
#     assert len(x.shape) in [3, 4]
#     if len(x.shape) == 4:
#         # assume BCHW
#         a = 1
#     raise NotImplementedError('')

def find_smallest_even_number(number):
    nearest_smaller_even = number - (number % 2)
    nearest_larger_even = number + (2 - (number % 2))
    return min(nearest_smaller_even, nearest_larger_even)


def save_tensors_as_gif(img_tensors: Union[torch.Tensor, List[torch.Tensor]], filename, duration=0.3):
    """Saves a spatio-temporal tensor (CTBH) or a list of spatial tensors [(CBH)] as a gif """
    if isinstance(img_tensors, list):
        ims = [np.array(to_numpy(x) * 255, dtype=np.uint8).transpose(1, 2, 0) for x in img_tensors]
    elif isinstance(img_tensors, torch.Tensor):
        img_tensors = torch.permute(img_tensors, (1, 2, 3, 0))  # (CTBH) -> (TBHC)
        T = img_tensors.shape[0]
        ims = [np.array(to_numpy(img_tensors[i]) * 255, dtype=np.uint8) for i in range(T)]
    else:
        raise ValueError(f'img_tensors must be a list of tensors or a tensor, got {type(img_tensors)}')
    imageio.mimsave(filename, ims, duration=duration)
    return


def recursive_search_submodule(module, submodule_name):
    """Recursively search for head in model"""
    if hasattr(module, submodule_name):
        submodule = getattr(module, submodule_name)
        if submodule is not None:
            return submodule
    else:
        for child in module.children():
            submodule = recursive_search_submodule(child, submodule_name)
            if submodule is not None:
                return submodule


def get_log_name(config):
    """utility to automate those damn log filenames"""
    run_str = config['name']
    if 'data' in config:
        s = config['data']['dataset']
        run_str += f"_{s}"

    if 'graph' in config:
        s = config['graph']['backbone']
        if s == 'resnet101':
            s = 'r101'
        elif s == 'resnet50':
            s = 'r50'
        elif s == 'hrnet48':
            s = 'hr48'
        run_str += f"_{s}"
        if config['graph']['sync_bn'] and config['parallel']:
            run_str += f"_{'sbn'}"

        phase = config['graph'].get('phase', 'pretraining')
        run_str += f"_{phase}"
        if phase in ['finetuning', 'linear_probing', 'fpn_probing']:
            if 'pretrained' in config['graph']:
                if config['graph']['pretrained']:
                    if config['graph'].get('internal_checkpoint_name', None):
                        if 'OCTIR' in config['graph']['internal_checkpoint_name']:
                            if 'mae' in config['graph']['internal_checkpoint_name']:
                                run_str += f"_OCTIRmae"
                            elif 'CLIP' in config['graph']['internal_checkpoint_name']:
                                run_str += f"_OCTIRclip"
                            # run_str += f"_OCTIRmae"
                        elif 'KEKI' in config['graph']['internal_checkpoint_name']:
                            run_str += f"_KEKImae"
                    elif config['graph'].get('external_checkpoint_name', None):
                        if 'RETFound' in config['graph']['external_checkpoint_name']:
                            run_str += f"_RETFoundmae"
                        elif 'mae_pretrain_vit' in config['graph']['external_checkpoint_name']:
                            run_str += f"_INmae"
                    elif 'dino' in config['graph']["backbone"]:
                        run_str += f"DINOv2"

    if 'loss' in config:
        c = config['loss']
        s = ''

        if config['loss']['name'] == 'manager':
            s = 'L2mae'
            if c.get('norm_pix_per_channel', False):
                s += '_pix_norm_per_channel'
            if c.get('norm_per_patch', False):
                s += '_norm_per_patch'
            run_str += f"_{s}"
        elif config['loss']['name'] == 'BCE':
            s = 'BCE'
            run_str += f"_{s}"

        if c.get('losses', None):
            for s in c['losses']:
                if s == 'CrossEntropyLoss':
                    s = 'CE'
                elif s == 'DenseContrastiveLoss':
                    s = 'DCv0'
                elif s == 'DenseContrastiveCenters':
                    s = 'DCC'
                elif s == 'DenseContrastiveLossV2':
                    s = 'DC'
                elif s == 'DenseContrastiveLossV2_ms':
                    s = 'DCms'
                    if 'cross_scale_contrast' in c:
                        s += '_cs'
                elif s == 'DenseContrastiveLossV3':
                    s = f'DC_mem-L{c["memory"]["L"]}-V{c["memory"]["V"]}'
                    if 'sampling' in c["memory"]:
                        if c["memory"]['sampling'] == 'random':
                            s += f'-Srand-Neg{c["memory"]["n_neg"]}'

            run_str += f"_{s}"

    if 'train' in config:
        run_str += f"_epochs{config['train']['epochs']}"
    if 'data' in config:
        run_str += f"_bs{config['data']['batch_size']}"
    return run_str


def shorten_key(key: str, **kwargs):
    is_cs = False
    if 'is_cs' in kwargs:
        is_cs = kwargs['is_cs']

    if 'CrossEntropyLoss' in key:
        return 'ce'
    elif 'DenseContrastiveLoss' == key:
        return 'DCv0'
    elif 'DenseContrastiveCenters' == key:
        return 'DCC'
    elif 'DenseContrastiveLossV2' == key:
        return 'DCv2'
    elif 'DenseContrastiveLossV2_ms' == key:
        skey = 'DCv2_ms'
        if is_cs:
            skey = f'{skey}_cs'
        return skey
    elif 'DenseContrastiveLossV2_ms' in key:
        suffix = key.split('DenseContrastiveLossV2_ms')[-1]
        return f'DCV2{suffix}'
    else:
        return key


def set_seeds(seed: int):

    """Function that sets all relevant seeds (by Claudio)
    :param seed: Seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % 2**32)  # To coincide with behaviour of worker_init_fn, and make sure the seed works
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remap_experiment(mask, experiment, dataset='CADIS'):
    """Remap mask for Experiment 'experiment' (needs to be int)"""
    colormap = get_remapped_colormap(DATASETS_INFO[dataset].CLASS_INFO[experiment][0], dataset)
    remapped_mask = remap_mask(mask, class_remapping=DATASETS_INFO[dataset].CLASS_INFO[experiment][0])
    return remapped_mask, DATASETS_INFO[dataset].CLASS_INFO[experiment][1], colormap


def remap_mask(mask, class_remapping, ignore_label=255, to_network=None):
    """
    Remaps mask to class_ids,
    Important note: if to_network is True, the ignore value (e.g. 255) is remapped to max_class_id+1
    :param mask: ndarray of input segmentation mask
    :param class_remapping: dictionary that indicates class remapping
    :param ignore_label: class id of ignore class
    :param to_network: default False. If true, ignore value (e.g. 255) is remapped to max_class_id+1
    :return: 2D/3D ndarray of remapped segmentation mask
    """
    to_network = False if to_network is None else to_network
    classes = []
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    n = max(len(classes), mask.max() + 1)
    remap_array = np.full(n, ignore_label, dtype=np.uint8)
    for key, val in class_remapping.items():
        for v in val:
            remap_array[v] = key
    mask_remapped = remap_array[mask]
    if to_network:
        mask_remapped[mask_remapped == ignore_label] = len(class_remapping) - 1
    return mask_remapped


def get_remapped_colormap(class_remapping, dataset):
    """
    Generated colormap of remapped classes
    Classes that are not remapped are indicated by the same color across all experiments
    :param class_remapping: dictionary that indicates class remapping
    :param dataset: CADIS or CITYSCAPES
    :return: 2D ndarray of rgb colors for remapped colormap
    """
    if dataset == 'CADIS':
        colormap = get_cadis_colormap()
    elif dataset == 'CITYSCAPES':
        colormap = get_cityscapes_colormap()
    elif dataset == 'PASCALC':
        colormap = get_pascalc_colormap()
    elif dataset == 'ADE20K':
        colormap = get_ade20k_colormap()
    elif dataset == 'IACL':
        colormap = get_iacl_colormap()
    elif dataset == 'RETOUCH':
        colormap = get_retouch_colormap()
    elif dataset == 'AROI':
        colormap = get_aroi_colormap()
    elif dataset == 'RAVIR':
        colormap = get_ravir_colormap()
    elif dataset == 'OCT5K':
        colormap = get_OCT5K_colormap()
    else:
        raise ValueError('dataset {} is not recognized'.format(dataset))
    remapped_colormap = {}
    for key, val in class_remapping.items():
        if key == 255:
            remapped_colormap.update({key: [0, 0, 0]})
        else:
            remapped_colormap.update({key: colormap[val[0]]})
    return remapped_colormap


def mask_from_network(mask, experiment, dataset):
    """
    Converts the segmentation masks as used in the network to using the IDs as used by the CaDISv2 paper
    :param mask: Input mask with classes numbered strictly from 0 to num_classes-1
    :param experiment: Experiment number
    :param dataset: dataset name in capital letters
    :return: Mask with classes numbered as required by CaDISv2 for the specific experiment (includes '255')
    """
    if experiment:
        if 255 in DATASETS_INFO[dataset].CLASS_INFO[experiment][0]:
            mask[mask == len(DATASETS_INFO[dataset].CLASS_INFO[experiment][1]) - 1] = 255
    return mask


def mask_to_colormap(mask, colormap, from_network=None, experiment=None, dataset='CADIS'):
    """
    Genarates RGB mask colormap from mask with class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param colormap: dictionary that indicates color corresponding to each class
    :param from_network: Default False. If True, class IDs as used in the network are first corrected to CaDISv2 usage
    :param experiment: Needed if from_network = True to determine which IDs need to be corrected
    :param dataset
    :return: 3D ndarray Generated RGB mask
    """
    from_network = False if from_network is None else from_network
    if from_network:
        mask = mask_from_network(mask, experiment, dataset)
    rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    # TODO: I feel this can be vectorised for speed
    for label, color in colormap.items():
        rgb[mask == label] = color
    return rgb


def mask_to_colormap_batched_torch(mask, colormap, from_network=None, experiment=None, dataset='CADIS'):
    """ Batched torch version of mask_to_colormap
    Genarates RGB mask colormap from mask with class ids
    :param mask: B,
    :param colormap: dictionary that indicates color corresponding to each class
    :param from_network: Default False. If True, class IDs as used in the network are first corrected to CaDISv2 usage
    :param experiment: Needed if from_network = True to determine which IDs need to be corrected
    :param dataset
    :return: B,3,H,W rgb version of mask (either predicted or label)
    """
    from_network = False if from_network is None else from_network
    if from_network:
        # replaces pixels with value ingore_class id (assumed to be num_classes-1) to 255
        mask = mask_from_network(mask, experiment, dataset)
    rgb = torch.zeros(size=mask.shape + (3,))
    for class_id, color in colormap.items():
        rgb[mask == class_id] = torch.tensor(color).float()
    return rgb


def plot_images(img, remapped_mask, remapped_colormap, classes_exp):
    """
    Generates plot of Image and RGB mask with class colorbar
    :param img: 3D ndarray of input image
    :param remapped_mask: 2D/3D ndarray of input segmentation mask with class ids
    :param remapped_colormap: dictionary that indicates color corresponding to each class
    :param classes_exp: dictionary of classes names and corresponding class ids
    :return: plot of image and rgb mask with class colorbar
    """
    mask_rgb = mask_to_colormap(remapped_mask, colormap=remapped_colormap, dataset='CADIS')

    fig, axs = plt.subplots(1, 2, figsize=(26, 7))
    plt.subplots_adjust(left=1 / 16.0, right=1 - 1 / 16.0, bottom=1 / 8.0, top=1 - 1 / 8.0)
    axs[0].imshow(img)
    axs[0].axis("off")

    img_u_labels = np.unique(remapped_mask)
    c_map = []
    cl = []
    for i_label in img_u_labels:
        for i_key, i_color in remapped_colormap.items():
            if i_label == i_key:
                c_map.append(i_color)
        for i_key, i_class in classes_exp.items():
            if i_label == i_key:
                cl.append(i_class)
    cl = np.asarray(cl)
    cmp = np.asarray(c_map) / 255
    cmap_mask = LinearSegmentedColormap.from_list("seg_mask_colormap", cmp, N=len(cmp))
    im = axs[1].imshow(mask_rgb, cmap=cmap_mask)
    intervals = np.linspace(0, 255, num=len(cl) + 1)
    ticks = intervals + int((intervals[1] - intervals[0]) / 2)
    divider = make_axes_locatable(axs[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(mappable=im, cax=cax1, ticks=ticks, orientation="vertical")
    cbar1.ax.set_yticklabels(cl)
    axs[1].axis("off")
    fig.tight_layout()

    return fig


def plot_experiment(img_path, mask_path, experiment=1):
    """
    Generates plot of image and rgb mask with colorbar for specified experiment
    :param img_path: Path to input image
    :param mask_path: Path to input segmentation mask
    :param experiment: int Experimental setup (1,2 or 3)
    :return: plot of image and rgb mask with class colorbar
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    remapped_mask, classes_exp, colormap = remap_experiment(mask, experiment)
    return plot_images(img, remapped_mask, colormap, classes_exp)


def pil_plot_tensor(tensor: torch.tensor, is_rgb=False):
    if is_rgb:
        # 3,H,W - H,W,3
        tensor = tensor.permute(1, 2, 0)
        # Image.fromarray(to_numpy(tensor.float().squeeze())).show(
        cv2.imshow('rgb', to_numpy(tensor.float()))
    else:
        cv2.imshow('lbl', to_numpy(tensor))
        return


def to_comb_image(img, lbl, lbl_pred, experiment, dataset, save=None):
    with torch.no_grad():
        img, lbl = to_numpy(img), to_numpy(lbl)
        img = np.round(np.moveaxis(img, 0, -1) * 255).astype('uint8')
        lbl = mask_to_colormap(lbl,
                               get_remapped_colormap(DATASETS_INFO[dataset].CLASS_INFO[experiment][0], dataset),
                               from_network=True, experiment=experiment, dataset=dataset)
        if lbl_pred is not None:
            lbl_pred = to_numpy(lbl_pred)
            lbl_pred = mask_to_colormap(lbl_pred,
                                        get_remapped_colormap(DATASETS_INFO[dataset].CLASS_INFO[experiment][0],
                                                              dataset),
                                        from_network=True, experiment=experiment, dataset=dataset)

            comb_img = np.concatenate((img, lbl, lbl_pred), axis=1)
        else:
            comb_img = np.concatenate((img, lbl), axis=1)

    if save:
        assert(type(save) == str)
        cv2.imwrite(save, comb_img)

    return comb_img


def get_matrix_fig(matrix, exp_num, dataset):
    labels = [item[1] for item in DATASETS_INFO[dataset].CLASS_INFO[exp_num][1].items()]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(.7*n, .7*n))
    im, cbar = heatmap(matrix, labels, labels, ax=ax, cbar_kw={}, cmap="YlGn", cbarlabel="Percentage probability")
    annotate_heatmap(im, valfmt='{x:.2f}', threshold=.6)
    fig.tight_layout()
    return fig


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """Create a heatmap from a numpy array and two lists of labels. {COPIED FROM MATPLOTLIB DOCS}

    :param data: A 2D numpy array of shape (N, M).
    :param row_labels: A list or array of length N with the labels for the rows.
    :param col_labels: A list or array of length M with the labels for the columns.
    :param ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted. If not provided, use current axes or
        create a new one. Optional.
    :param cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`. Optional.
    :param cbarlabel: The label for the colorbar. Optional.
    :param kwargs: All other arguments are forwarded to `imshow`.
    :return: im, cbar
    """
    cbar_kw = {} if cbar_kw is None else cbar_kw

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    im.set_clim(vmin=0, vmax=1)

    # Create colorbar
    # Code adapted from: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    aspect = 20
    pad_fraction = 0.5
    divider = make_axes_locatable(im.axes)
    width = axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels[:data.shape[0]])
    ax.set_yticklabels(row_labels[:data.shape[0]])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=None, threshold=None, **textkw):
    """A function to annotate a heatmap. {COPIED FROM MATPLOTLIB DOCS}

    :param im: The AxesImage to be labeled.
    :param data: Data used to annotate.  If None, the image's data is used.  Optional.
    :param valfmt: The format of the annotations inside the heatmap. This should either use the string format method,
        e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`. Optional.
    :param textcolors: A list or array of two color specifications. The first is used for values below a threshold,
        the second for those above. Optional.
    :param threshold: Value in data units according to which the colors from textcolors are applied. If None (the
        default) uses the middle of the colormap as separation. Optional.
    :param textkw: All other arguments are forwarded to each call to `text` used to create the text labels.
    :return: texts
    """
    textcolors = ["black", "white"] if textcolors is None else textcolors
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.max(data)) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        # noinspection PyUnresolvedReferences
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            # noinspection PyCallingNonCallable
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def un_normalise(arr: torch.Tensor, mean: list = None, std: list = None):
    """Reverts the action of torchvision.transforms.Normalize (on numpy). Assumes NCHW shape"""
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    mean = torch.as_tensor(mean, device=arr.device).view(-1, 1, 1)
    std = torch.as_tensor(std, device=arr.device).view(-1, 1, 1)
    unnorm_arr = arr * std + mean
    return unnorm_arr


def do_nothing(x):
    return x

def str2bool(s:str):
    assert type(s), f'input argument must be str instead {s}'
    if s in ['True', 'true']:
        return True
    elif s in ['False', 'false']:
        return False
    else:
        raise ValueError(f'string {s} ')


def to_numpy(tensor):
    """Tensor to numpy, calls .cpu() if necessary"""
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        return tensor.numpy()


def softmax(x, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.
    From: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html

    :param x: ND-Array. Probably should be floats.
    :param theta: (optional) float parameter, used as a multiplier prior to exponentiation. Default = 1.0
    :param axis: (optional) axis to compute values along. Default is the first non-singleton axis.
    :return: an array the same size as X. The result will sum to 1 along the specified axis.
    """
    # make x at least 2d
    y = np.atleast_2d(x)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if x was 1D
    if len(x.shape) == 1:
        p = p.flatten()

    return p


def fig_from_dist(elements: np.ndarray, counts: np.ndarray, desired_num_bins: int,
                  xlabel: str = '', ylabel: str = ''):
    """Returns bar chart figure with frequency count as y axis, bins along x axis"""
    els_per_bin = np.maximum(len(elements) // desired_num_bins, 1)
    num_bins = len(elements) // els_per_bin
    el_ind_lists = np.arange(els_per_bin * num_bins)\
        .reshape((num_bins, els_per_bin)).tolist()
    if len(elements) > els_per_bin * num_bins:
        el_ind_lists.append(np.arange(els_per_bin * num_bins, len(elements)).tolist())
        num_bins += 1  # correction due to extra bin for left-over elements
    chart_count = np.zeros(num_bins, 'i')
    for i, el_ind_list in enumerate(el_ind_lists):
        chart_count[i] = np.sum(counts[el_ind_list])
    fig, ax = plt.subplots(figsize=(.2 * num_bins, 8))
    tick_labels = []
    for i in range(num_bins):
        if len(el_ind_lists[i]) > 1:
            lbl = '{} - {}'.format(el_ind_lists[i][0], el_ind_lists[i][-1])
        else:
            lbl = '{}'.format(el_ind_lists[i][0])
        tick_labels.append(lbl)
    ax.bar(tick_labels, chart_count)
    plt.xticks(rotation=45)
    plt.axis('tight')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def get_class_info(training_df: pd.DataFrame, experiment: int, with_name=False, dataset='CADIS'):
    # with_name=True will add per frame class information with columns named with their "names" instead of numbrer id
    classes = [c for c in DATASETS_INFO[dataset].CLASS_INFO[experiment][0].keys() if c != 255]
    if with_name:
        classes = [DATASETS_INFO[dataset].CLASS_INFO[experiment][1][c] for c in classes]
    for c, c_name in enumerate(classes):  # Leave out the 'ignore' class, if it exists
        col_sum = training_df[[DATASETS_INFO[dataset].CLASS_INFO[0][1][i]
                               for i in DATASETS_INFO[dataset].CLASS_INFO[experiment][0][c]]].sum(1)
        if with_name:
            training_df[c_name] = col_sum
        else:
            training_df[c] = col_sum
    return training_df


def reverse_one_to_many_mapping(mapping: dict):
    """ inverts class experiment mappings or id to name dicts """
    rev_mapping = dict()
    for key in mapping.keys():
        vals = mapping[key]
        if isinstance(vals, list):
            for key_new in vals:
                rev_mapping[key_new] = key
        elif type(vals) == str:
            rev_mapping[vals] = key
    return rev_mapping


def reverse_mapping(mapping: dict):
    """ inverts class experiment mappings or id to name dicts """
    rev_mapping = dict()
    for key in mapping.keys():
        if key == 255:
            continue
        vals = mapping[key]
        if isinstance(vals, list):
            for key_new in vals:
                rev_mapping[key_new] = [key]
        elif type(vals) == str:
            rev_mapping[vals] = key
    return rev_mapping


def dict2json(d: Union[dict, OrderedDict], path_to_filename: str, verbose=True, sort_keys=True):
    d_ = json.dumps(d, indent=4, sort_keys=sort_keys)
    with open(path_to_filename, 'w') as json_file:
        json_file.write(d_)
    if verbose:
        print(f'dict was saved as {path_to_filename}')
    return 0


def create_new_directory(d):
    """create if it does not exist else do nothing and return -1"""
    _ = os.makedirs(d) if not(os.path.isdir(d)) else -1
    return _


def colourise_data(data: np.ndarray,  # NHW expected
                   low: float = 0, high: float = 1,
                   repeat: list = None,
                   perf_colour: tuple = (255, 0, 0)) -> np.ndarray:
    # perf_colour in RGB
    if high == -1:  # Scale by maximum present
        high = np.max(data)
    data = np.clip((data - low) / (high - low), 0, 1)
    colour_img = np.round(data[..., np.newaxis] *
                          np.array(perf_colour)[np.newaxis, np.newaxis, np.newaxis, :]).astype('uint8')
    if repeat is not None:
        colour_img = np.repeat(np.repeat(colour_img, repeat[0], axis=1), repeat[1], axis=2)
    return colour_img


def worker_init_fn(_):
    np.random.seed(torch.initial_seed() % 2**32)


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
