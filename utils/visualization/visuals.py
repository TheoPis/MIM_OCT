import os
import pathlib
from random import randint
from typing import cast, Union, List, Union, Optional, List, Tuple, Text, BinaryIO
from tqdm import tqdm
from PIL import Image
import einops
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import torch
import torch.nn as nn
from utils.logger import printlog
from utils.utils import create_new_directory, to_numpy, save_tensors_as_gif
from torchvision.utils import make_grid  # save_image
from torch.utils.data import DataLoader, Subset
from utils.defaults import DATASETS_INFO
import wandb

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
matplotlib.use('Agg')
known_grid_of_predictions_tags = ['_mae', '_mae_vol', '_multimae', '_multimae_cm', '_multimae_seq',
                                  '_multiseq_prolif', '_biomarker_detection', '_seg']


def tensor2pil_show_label_color(x):
    # x: (B, C, H, W), plots 1st element of the batch
    from PIL import Image
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)
    assert len(x.shape) == 4
    x = to_numpy(torch.clip(x*255, 0, 255)).astype(np.uint8)
    if x.shape[1] == 1:
        x = x[0, 0, ...]
    else:
        x = x[0]
        x = np.transpose(x, (1, 2, 0))
    # only show 1st image in the batch
    Image.fromarray(x).show()


def save_qualitative_results(loader: DataLoader,
                             num_images: int,
                             manager,
                             filename,
                             tag: str = '_mae',
                             use_wandb=False,
                             wandb_tag: str = 'val'):
    """
    :param loader: dataloader (train or validation)
    :param num_images: number of images to save
    :param manager: manager object (contains model, optimizer, log_dir etc)
    :param filename: name of visuals file
    :param tag: string that controls which grid_of_predictions function to use ('_mae', '_biomarker_detection', '_seg')
    :param use_wandb: whether to log images with wandb
    :param wandb_tag: string to name panel in wandb (e.x. 'val' or 'train')
    :return:
    """
    if tag not in known_grid_of_predictions_tags:
        printlog(f"tag {tag} not in {known_grid_of_predictions_tags}")
        return
    printlog(f"Generating qualitative results for {num_images} images ...")
    grid_of_predictions = globals()[f'grid_of_predictions{tag}']

    loader_fixed = loader_subset(loader, num_images, randomize=False)
    grid, wandb_dict = grid_of_predictions(manager, loader_fixed, num_images=num_images, use_wandb=use_wandb)

    loader_rand = loader_subset(loader, num_images, randomize=True)
    grid_shuffle, wandb_dict_shuffle = grid_of_predictions(manager, loader_rand, num_images=num_images, use_wandb=use_wandb)

    create_new_directory(manager.log_dir)  # fixme this is perhaps unnecessary as getting here means log_dir exists
    filename = os.path.join(manager.log_dir, filename)
    printlog(f"Saving images as {filename}")

    if isinstance(grid, torch.Tensor):
        grid = torch.concat([grid, grid_shuffle], dim=0)
        _ = save_image(grid, filename, nrow=grid.shape[0] // (num_images * 2))
    elif isinstance(grid, list):
        grid = [torch.concat([g1, g2], dim=0) for g1, g2 in zip(grid, grid_shuffle)]
        grid = [make_grid(g, g.shape[0] // (num_images * 2)) for g in grid]
        save_tensors_as_gif(grid, filename)
    else:
        grid.savefig(filename)

    if use_wandb and wandb_dict is not None and tag == '_seg':
        images_stacked = wandb_dict['imgs']  # (2*H, B*W, C)
        images_stacked_shuffle = wandb_dict_shuffle['imgs']  # (2*H, B*W, C)
        images = np.concatenate([images_stacked, images_stacked_shuffle], axis=1)  # (4*H, B*W, C)

        class_id_to_name = wandb_dict['class_id_to_name']

        lbls_preds_stacked = wandb_dict['lbls_preds']  # (2*H, B*W)
        lbls_preds_stacked_shuffle = wandb_dict_shuffle['lbls_preds']  # (2*H, B*W)
        lbls_preds = np.concatenate([lbls_preds_stacked, lbls_preds_stacked_shuffle], axis=1)  # (4*H, B*W)

        wandb_dict_merged = {f"{wandb_tag}_imgs": wandb.Image(images, masks={
            "predictions": {
                "mask_data": lbls_preds,
                "class_labels": class_id_to_name
            },
            "ground_truth": {
                "mask_data": lbls_preds,
                "class_labels": class_id_to_name
            }
        })}
        wandb.log(wandb_dict_merged, step=manager.global_step)
    return filename


def stack_batch_tensor(x: torch.Tensor, horrizontal=True):
    """ stacks images in a batched tensor (B,C,H,W) or (B,H,W) into a single image
    :arg x: tensor of shape (B,C,H,W) or (B,H,W)
    :arg horrizontal: stack orientation (default: True else vertical)
    :returns x: tensor of shape (H, B*W, C) or (B*H, W, C) (C dim can be omitted if x is 3D)
    """
    if len(x.shape) == 4:
        if horrizontal:
            # horizontal stack
            x = einops.rearrange(x, 'b c h w -> h b w c')
            x = einops.rearrange(x, 'h b w c -> h (b w) c')
        else:
            # vertical stack
            x = einops.rearrange(x, 'b c h w -> b h w c)')
            x = einops.rearrange(x, 'b h w c -> (b h) w c)')
    else:
        if horrizontal:
            # horizontal stack
            x = einops.rearrange(x, 'b h w -> h b w')
            x = einops.rearrange(x, 'h b w -> h (b w)')
        else:
            # vertical stack
            x = einops.rearrange(x, 'b h w -> (b h) w')
    return x


def loader_subset(loader: DataLoader, num_images: int, randomize=False) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    fixed_indices = range(0, lng - lng % num_images, lng // num_images)
    if randomize:
        overlap = True
        fixed_indices_set = set(fixed_indices)
        maxatt = 5
        cnt = 0
        while overlap and cnt < maxatt:
            indices = [randint(0, lng - 1) for _ in range(0, num_images)]
            overlap = len(set(indices).intersection(fixed_indices_set)) > 0
            cnt += 1
    else:
        indices = fixed_indices
    return DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False
    )


def save_image(tensor: Union[torch.Tensor, List[torch.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str] = None,
               **kwargs) -> Image:
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    return im


@torch.no_grad()
def grid_of_predictions_mae(manager, loader, num_images, use_wandb=False, **kwargs) \
        -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    imgs_: List[torch.Tensor] = []   # original images
    preds_: List[torch.Tensor] = []   # all patches that are predicted by the network (both masked and unmasked)
    masked_: List[torch.Tensor] = []  # masked image (i.e patches that are processed by the network)
    pasted_: List[torch.Tensor] = []  # combined original patches with predicted patches
    wandb_dict = {}
    # img_channels = loader.dataset.dataset.img_channels
    for idx, batch in enumerate(tqdm(loader)):
        # if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        images_b, metadata = batch
        images_b = images_b.cuda()  # Images/labels are (B,C,H,W)
        output, mask_tensor = manager.model(images_b, is_training=True)
        im_masked, im_pasted, pred = manager.post_process_output_img(images_b, output, mask_tensor)

        imgs_.append(images_b)  # channels first
        preds_.append(torch.einsum('bhwc->bchw', pred))
        masked_.append(torch.einsum('bhwc->bchw', im_masked))
        pasted_.append(torch.einsum('bhwc->bchw', im_pasted))
        # if idx == num_images == 1:
        #     break

    images = torch.cat(imgs_, dim=0).cpu()

    predictions = torch.cat(preds_, dim=0).cpu()
    masked = torch.cat(masked_, dim=0).cpu()
    pasted = torch.cat(pasted_, dim=0).cpu()

    grid = torch.cat([
        images[:, None].expand((-1, -1, 3, -1, -1)),  # B,1,C,H,W
        predictions[:, None].expand((-1, -1, 3, -1, -1)),
        masked[:, None].expand((-1, -1, 3, -1, -1)),
        pasted[:, None].expand((-1, -1, 3, -1, -1))
    ],
        dim=1
    )
    grid = torch.reshape(grid, (-1,) + grid.shape[2:])
    return grid, wandb_dict


@torch.no_grad()
def grid_of_predictions_mae_vol(manager, loader, num_images, use_wandb=False, **kwargs) \
        -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    imgs_: List[torch.Tensor] = []   # original images
    preds_: List[torch.Tensor] = []   # all patches that are predicted by the network (both masked and unmasked)
    masked_: List[torch.Tensor] = []  # masked image (i.e patches that are processed by the network)
    pasted_: List[torch.Tensor] = []  # combined original patches with predicted patches
    wandb_dict = {}
    # img_channels = loader.dataset.dataset.img_channels
    for idx, batch in enumerate(tqdm(loader)):
        # if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        images_b, metadata = batch
        images_b = images_b.cuda()  # Images/labels are (B,C,T,H,W)
        output, mask_tensor = manager.model(images_b, is_training=True)
        im_masked, im_pasted, pred = manager.post_process_output_video(images_b, output, mask_tensor)
        # all are (B,T,H,W,C)

        # imgs_.append(torch.unbind(images_b, dim=2))  # channels first
        # preds_.append(torch.unbind(torch.einsum('bthwc->btchw', pred), dim=1))
        # masked_.append(torch.unbind(torch.einsum('bthwc->btchw', im_masked), dim=1))
        # pasted_.append(torch.unbind(torch.einsum('bthwc->btchw', im_pasted), dim=1))
        imgs_.append(images_b)  # channels first
        preds_.append(torch.einsum('bthwc->btchw', pred))
        masked_.append(torch.einsum('bthwc->btchw', im_masked))
        pasted_.append(torch.einsum('bthwc->btchw', im_pasted))

        # if idx == num_images == 1:
        #     break
    grid = []
    imgs = torch.cat(imgs_, dim=0).cpu() # [(B',C,T,H,W)]
    predictions = torch.cat(preds_, dim=0).cpu()
    masked = torch.cat(masked_, dim=0).cpu()
    pasted = torch.cat(pasted_, dim=0).cpu()
    T = imgs.shape[2]
    for t in range(T):
        images_t = imgs[:, :, t, :, :]  # (B',C,H,W)
        predictions_t = predictions[:, t, :, :, :]  # (B',C,H,W)
        masked_t = masked[:, t, :, :, :]  # (B',C,H,W)
        pasted_t = pasted[:, t, :, :, :]  # (B',C,H,W)

        grid_t = torch.cat([
            images_t[:, None].expand((-1, -1, 3, -1, -1)),  # B,1,C,H,W
            predictions_t[:, None].expand((-1, -1, 3, -1, -1)),
            masked_t[:, None].expand((-1, -1, 3, -1, -1)),
            pasted_t[:, None].expand((-1, -1, 3, -1, -1))
        ],
            dim=1
        )  # (B,4,C,H,W)
        grid_t = torch.reshape(grid_t, (-1,) + grid_t.shape[2:])

        grid.append(grid_t)
    return grid, wandb_dict


@torch.no_grad()
def grid_of_predictions_multimae(manager, loader, num_images, use_wandb=False, **kwargs) \
        -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    imgs_: List[torch.Tensor] = []   # original images
    preds_: List[torch.Tensor] = []   # all patches that are predicted by the network (both masked and unmasked)
    masked_: List[torch.Tensor] = []  # masked image (i.e patches that are processed by the network)
    pasted_: List[torch.Tensor] = []  # combined original patches with predicted patches
    wandb_dict = {}
    # img_channels = loader.dataset.dataset.img_channels

    if hasattr(loader.dataset.dataset, 'modality'):
        modality = loader.dataset.dataset.modality
    else:
        raise ValueError(f'modality not found in dataset {loader.dataset.dataset.__class__.__name__}')

    # loader.dataset.dataset.use_ominvision_api = False  # turn off ominvision api
    # loader.dataset.dataset.return_metadata = False

    for idx, batch in enumerate(tqdm(loader)):
        # if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        if len(batch) == 2:
            images_b, metadata = batch
        else:
            images_b = batch

        images_b = images_b.cuda()  # Images/labels are (B,C,H,W)
        outs = manager.model(images_b, modality=modality, is_training=True)
        output, mask_tensor = outs['output'], outs['mask']
        im_masked, im_pasted, pred = manager.post_process_output_img(images_b, output, mask_tensor)
        imgs_.append(images_b)  # channels first
        preds_.append(torch.einsum('bhwc->bchw', pred))
        masked_.append(torch.einsum('bhwc->bchw', im_masked))
        pasted_.append(torch.einsum('bhwc->bchw', im_pasted))
        # if idx == num_images == 1:
        #     break

    images = torch.cat(imgs_, dim=0).cpu()

    predictions = torch.cat(preds_, dim=0).cpu()
    masked = torch.cat(masked_, dim=0).cpu()
    pasted = torch.cat(pasted_, dim=0).cpu()

    grid = torch.cat([
        images[:, None].expand((-1, -1, 3, -1, -1)),  # B,1,C,H,W
        predictions[:, None].expand((-1, -1, 3, -1, -1)),
        masked[:, None].expand((-1, -1, 3, -1, -1)),
        pasted[:, None].expand((-1, -1, 3, -1, -1))
    ],
        dim=1
    )
    grid = torch.reshape(grid, (-1,) + grid.shape[2:])
    return grid, wandb_dict


@torch.no_grad()
def grid_of_predictions_seg(manager, loader, num_images, use_wandb=False) \
        -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    imgs_: List[torch.Tensor] = []   # original images
    preds_: List[torch.Tensor] = []   # preds in rgb
    lbls_: List[torch.Tensor] = []  # lbl in rgb
    lbls_class_id_: List[torch.Tensor] = []  # lbl in class id
    preds_class_id_: List[torch.Tensor] = []  # lbl in class id
    # pasted_: List[torch.Tensor] = []  # combined original patches with predicted patches
    manager.model.eval()
    for idx, batch in enumerate(tqdm(loader)):
        # if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        images_b, labels_class_id_b, metadata = batch
        images_b = images_b.cuda()  # Images/labels are (B,C,H,W)
        if hasattr(manager.flat_model, 'is_multimodal'):
            if manager.flat_model.is_multimodal:
                output_b = manager.model(images_b, modality=manager.model.modality)
            else:
                output_b = manager.model(images_b)
        else:
            output_b = manager.model(images_b)

        pred_class_id_b = torch.argmax(nn.Softmax2d()(output_b), dim=1)  # contains train_ids
        output_b, labels_b = manager.get_preds_labels_rgb(output_b, labels_class_id_b)

        imgs_.append(images_b)  # channels first
        preds_.append(torch.einsum('bhwc->bchw', output_b))
        lbls_.append(torch.einsum('bhwc->bchw', labels_b))
        lbls_class_id_.append(labels_class_id_b)
        preds_class_id_.append(pred_class_id_b)

    manager.model.train()

    images = torch.cat(imgs_, dim=0).cpu()
    predictions = torch.cat(preds_, dim=0).cpu()
    lbls = torch.cat(lbls_, dim=0).cpu()

    grid = torch.cat([
        images[:, None].expand((-1, -1, 3, -1, -1)),  # B,1,C,H,W
        predictions[:, None].expand((-1, -1, 3, -1, -1)),
        lbls[:, None].expand((-1, -1, 3, -1, -1)),
    ],
        dim=1
    )

    grid = torch.reshape(grid, (-1,) + grid.shape[2:])
    wandb_dict = {}

    if use_wandb:
        # getting predictions and labels in class_id format (NOT rgb)

        lbls_class_id = torch.cat(lbls_class_id_, dim=0).cpu()
        preds_class_id = torch.cat(preds_class_id_, dim=0).cpu()

        # dictionary that maps integer class_id to class name
        class_id_to_name = DATASETS_INFO[manager.dataset].CLASS_INFO[manager.experiment][1]

        # batch stacked horrizontally
        # B,C,H,W -> H, B*W, C
        images_stacked = to_numpy(stack_batch_tensor(images))
        # debug: verify with PIL
        # Image.fromarray(images_stacked[..., 0] * 255).show()

        # replicate row to overlay preds and lbl
        images_stacked = np.concatenate([images_stacked, images_stacked], axis=0)  # (2*H, B*W, C)

        # preds_stacked = to_numpy(torch.cat([preds_class_id[i] for i in range(b)], dim=1))
        preds_stacked = to_numpy(stack_batch_tensor(preds_class_id))
        lbl_stacked = to_numpy(stack_batch_tensor(lbls_class_id))

        # combine rows to overlay preds and lbl
        lbls_preds_stacked = np.concatenate([lbl_stacked, preds_stacked], axis=0)  # (2*H, B*W)

        # to be passed to wandb.log()
        wandb_dict = {"imgs": images_stacked, "lbls_preds": lbls_preds_stacked, 'class_id_to_name': class_id_to_name}

    return grid, wandb_dict


@torch.no_grad()
def grid_of_predictions_biomarker_detection(manager, loader, num_images, use_wandb=False, **kwargs) -> \
        Tuple[torch.Tensor, dict]:
    manager.model.eval()
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(20, 20))
    class_names = DATASETS_INFO['OctBiom'].CLASS_NAMES[0]
    X_axis = np.arange(len(class_names))
    width = 0.4
    offset = 0.2
    wandb_dict = {}
    manager.model.eval()
    for idx, batch in enumerate(tqdm(loader)):
        images_b, labels_b, metadata = batch
        images_b = manager.un_norm(images_b)
        images_b = images_b.cuda()  # Images, labels are (B,C,H,W) (B,10)
        modality = manager.modality if hasattr(manager, 'modality') else None
        pred = manager.model(images_b, modality=modality)
        axes[idx, 0].imshow(to_numpy(einops.rearrange(images_b[0], 'c h w -> h w c')))
        axes[idx, 1].bar(X_axis - offset, to_numpy(torch.sigmoid(pred[0])), width, label='prediction')
        axes[idx, 1].bar(X_axis + offset, to_numpy(labels_b[0]), width, label='label')
        axes[idx, 1].set_xticks(X_axis, class_names)
        # axes[idx, 1].bar(class_names, to_numpy(torch.sigmoid(pred[0])))
        # axes[idx, 2].bar(class_names, to_numpy(labels_b[0]))
    manager.model.train()
    return fig, wandb_dict


def save_results_biomarker_detection(images: List[torch.Tensor],
                                     predictions: List[torch.Tensor],
                                     labels: List[torch.Tensor], manager):
    """function to save a figure with image-label-pred for OctBiom dataset"""

    assert len(images) == len(predictions) == len(labels)

    images_unbatched = []
    predictions_unbatched = []
    labels_unbatched = []
    # [[B,C,H,W]] -> [[1,C,H,W], ..., [1,C,H,W]]
    for image, predictions, label in zip(images, predictions, labels):
        images_unbatched.append(torch.split(image, 1, 0))
        predictions_unbatched.append((torch.split(predictions, 1, 0)))
        labels_unbatched.append(torch.split(label, 1, 0))
    for idx, (image, pred, label) in enumerate(zip(images_unbatched,
                                                                predictions_unbatched,
                                                                labels_unbatched)):
        save_results_single_biomarker_detection(image, pred, label, manager, idx)


def save_results_single_biomarker_detection(image, pred, label, manager, idx: int):
    image = manager.un_norm(image)
    # image preds and labels are (1,C,H,W) (1,10) (1,10)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    class_names = DATASETS_INFO['OctBiom'].CLASS_NAMES[0]
    X_axis = np.arange(len(class_names))
    width = 0.4
    offset = 0.2
    axes[0].imshow(to_numpy(einops.rearrange(image[0], 'c h w -> h w c')))
    axes[1].bar(X_axis - offset, to_numpy(torch.sigmoid(pred[0])), width, label='prediction')
    axes[1].bar(X_axis + offset, to_numpy(label[0]), width, label='label')
    axes[1].set_xticks(X_axis, class_names)
    save_path = pathlib.Path(manager.log_dir) / 'outputs' / manager.config['data']['split'][-1]
    create_new_directory(save_path)
    printlog("saving image-label-pred figure to {}".format(save_path / '{:06d}.png'.format(idx)))
    fig.savefig(save_path / '{:06d}.png'.format(idx))
    plt.close()
    del fig


@torch.no_grad()
def grid_of_predictions_multimae_cm(manager, loader, num_images, use_wandb=False, **kwargs) \
        -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    imgs_: List[torch.Tensor] = []   # original images
    preds_: List[torch.Tensor] = []   # all patches that are predicted by the network (both masked and unmasked)
    masked_: List[torch.Tensor] = []  # masked image (i.e patches that are processed by the network)
    pasted_: List[torch.Tensor] = []  # combined original patches with predicted patches
    wandb_dict = {}

    for idx, batch in enumerate(tqdm(loader)):
        # if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        # if len(batch) == 2:
        #     images_b, metadata = batch
        # else:
        #     images_b = batch

        modalities = list(batch.keys())
        # forward: one modality visible the other is masked # todo the names of the modality are unused so remove>?
        # outs: dict with keys: "cls_token", "output", "mask"
        for modality in modalities:
            batch[modality] = batch[modality].cuda()

        outs1 = manager.model(x=batch[modalities[0]], x_m=batch[modalities[1]], is_training=True,
                              modality=modalities[0], modality_m=modalities[1])

        outs2 = manager.model(x=batch[modalities[1]], x_m=batch[modalities[0]], is_training=True,
                              modality=modalities[1], modality_m=modalities[0])

        output1, mask_tensor1 = outs1['output'], outs1['mask']
        output2, mask_tensor2 = outs2['output'], outs2['mask']
        # Images/labels are (B,C,H,W)
        im_masked_1, im_pasted_1, pred_1 = manager.post_process_output_img(batch[modalities[1]], output1, mask_tensor1)
        im_masked_2, im_pasted_2, pred_2 = manager.post_process_output_img(batch[modalities[0]], output2, mask_tensor2)
        imgs_.append(batch[modalities[1]])  # channels first
        imgs_.append(batch[modalities[0]])  # channels first
        preds_.append(torch.einsum('bhwc->bchw', pred_1))
        preds_.append(torch.einsum('bhwc->bchw', pred_2))
        masked_.append(torch.einsum('bhwc->bchw', im_masked_1))
        masked_.append(torch.einsum('bhwc->bchw', im_masked_2))
        pasted_.append(torch.einsum('bhwc->bchw', im_pasted_1))
        pasted_.append(torch.einsum('bhwc->bchw', im_pasted_2))
        # if idx == num_images == 1:
        #     break

    images = torch.cat(imgs_, dim=0).cpu()

    predictions = torch.cat(preds_, dim=0).cpu()
    masked = torch.cat(masked_, dim=0).cpu()
    pasted = torch.cat(pasted_, dim=0).cpu()

    grid = torch.cat([
        images[:, None].expand((-1, -1, 3, -1, -1)),  # B,1,C,H,W
        predictions[:, None].expand((-1, -1, 3, -1, -1)),
        masked[:, None].expand((-1, -1, 3, -1, -1)),
        pasted[:, None].expand((-1, -1, 3, -1, -1))
    ],
        dim=1
    )
    grid = torch.reshape(grid, (-1,) + grid.shape[2:])
    return grid, wandb_dict


@torch.no_grad()
def grid_of_predictions_multimae_seq(manager, loader, num_images, use_wandb=False, **kwargs) -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    imgs_: List[torch.Tensor] = []   # original images
    preds_: List[torch.Tensor] = []   # all patches that are predicted by the network (both masked and unmasked)
    masked_: List[torch.Tensor] = []  # masked image (i.e patches that are processed by the network)
    pasted_: List[torch.Tensor] = []  # combined original patches with predicted patches
    wandb_dict = {}

    for idx, batch in enumerate(tqdm(loader)):
        # if 'Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'
        # if len(batch) == 2:
        #     images_b, metadata = batch
        # else:
        #     images_b = batch

        modalities = list(batch.keys())
        # forward: one modality visible the other is masked # todo the names of the modality are unused so remove>?
        # outs: dict with keys: "cls_token", "output", "mask"
        for modality in modalities:
            batch[modality] = batch[modality].cuda()

        outs1 = manager.model(x=batch, is_training=True)

        output1, mask_tensor1 = outs1['output'], outs1['mask']
        # Images/labels are (B,C,H,W)
        im_masked_1, im_pasted_1, pred_1 = manager.post_process_output_img(batch[modalities[1]], output1[modalities[1]],
                                                                           outs1[f'mask_{modalities[1]}'])
        im_masked_0, im_pasted_0, pred_0 = manager.post_process_output_img(batch[modalities[0]], output1[modalities[0]],
                                                                           outs1[f'mask_{modalities[0]}'])
        imgs_.append(batch[modalities[0]])  # channels first
        imgs_.append(batch[modalities[1]])  # channels first
        preds_.append(torch.einsum('bhwc->bchw', pred_0))
        preds_.append(torch.einsum('bhwc->bchw', pred_1))
        masked_.append(torch.einsum('bhwc->bchw', im_masked_0))
        masked_.append(torch.einsum('bhwc->bchw', im_masked_1))
        pasted_.append(torch.einsum('bhwc->bchw', im_pasted_0))
        pasted_.append(torch.einsum('bhwc->bchw', im_pasted_1))
        # if idx == num_images == 1:
        #     break

    images = torch.cat(imgs_, dim=0).cpu()

    predictions = torch.cat(preds_, dim=0).cpu()
    masked = torch.cat(masked_, dim=0).cpu()
    pasted = torch.cat(pasted_, dim=0).cpu()

    grid = torch.cat([
        images[:, None].expand((-1, -1, 3, -1, -1)),  # B,1,C,H,W
        predictions[:, None].expand((-1, -1, 3, -1, -1)),
        masked[:, None].expand((-1, -1, 3, -1, -1)),
        pasted[:, None].expand((-1, -1, 3, -1, -1))
    ],
        dim=1
    )
    grid = torch.reshape(grid, (-1,) + grid.shape[2:])
    return grid, wandb_dict


@torch.no_grad()
def grid_of_predictions_multiseq_prolif(manager, loader, num_images, use_wandb=False, **kwargs) -> Tuple[torch.Tensor, dict]:
    manager.model.eval()
    assert hasattr(manager, 'modes')
    assert hasattr(manager, 'modalities')

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(20, 20))
    class_names = DATASETS_INFO['DR'].CLASS_NAMES[0]
    X_axis = np.arange(len(class_names))
    width = 0.4
    offset = 0.2
    wandb_dict = {}

    for idx, batch in enumerate(tqdm(loader)):
        labels_b = batch['label'].cuda()
        data = {}  # move to gpu
        for modality in manager.modalities:
            mode = manager.modes[modality][0]  # only use the first mode for now fixme
            # note: batch[modality][mode] is a BCHW tensor and H,W are the same for all other modalities fixme
            data.update({modality: batch[modality][mode].cuda()})
        pred = manager.model(x=data, is_training=True)

        # concat imgs from different modalities

        images_b = torch.cat([data[modality] for modality in manager.modalities], dim=3)
        axes[idx, 0].imshow(to_numpy(einops.rearrange(images_b[0], 'c h w -> h w c')))
        axes[idx, 1].bar(X_axis - offset, to_numpy(torch.softmax(pred[0], 0)), width, label='prediction')
        axes[idx, 1].bar(X_axis + offset, to_numpy(labels_b[0]), width, label='label')
        axes[idx, 1].set_xticks(X_axis, class_names)

    manager.model.train()
    return fig, wandb_dict
