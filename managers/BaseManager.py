import wandb
import cv2
import os
import torch
import datetime
import warnings
import numpy as np
import pandas as pd
from typing import Union
from contextlib import nullcontext
from functools import partial
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from datasets import AROI, OctBiom, get_retouch_dataframes, DatasetFromDF, \
    ConcatDataset, DatasetDataloader, DRDataset, OCTIRSingle, OCTIRPaired, OCTID, OCTDL, \
    DatasetFromDFSub, RAVIR, OCT5K, OLIVES
from models import *
from losses import LossWrapper, CLIPLoss
from utils import Logger as Log
from utils import DATASETS_INFO, LRFcts, get_remapped_colormap, worker_init_fn, printlog, \
    un_normalise, do_nothing, parse_transform_lists, mask_to_colormap_batched_torch, \
    get_param_groups_with_stage_wise_lr_decay, set_seeds, get_param_groups_using_keys, PolyakAverager,\
    get_param_groups_weight_decay
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from .LoggingManager import LoggingManager


class BaseManager(LoggingManager):
    eligible_tasks = ['ssl', 'detection', 'segmentation']

    """Base Manager class, from which all model specific managers inherit"""
    def __init__(self, configuration):
        super().__init__(configuration)
        if self.parallel and self.config['mode'] == 'training':
            if self.debugging:
                os.environ['NCCL_DEBUG'] = 'INFO'
            return
        else:
            if self.use_wandb:
                self._setup_wandb()

        cudnn.benchmark = self.config['cudnn_benchmark'] if 'cudnn_benchmark' in self.config else True
        printlog(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')
        printlog(f"*** cudnn.enabled {cudnn.enabled}")
        printlog(f"*** cudnn.deterministic {cudnn.deterministic}")

        self._setup_precision()
        self.load_model()

        if self.config['mode'] == 'training' and not self.parallel:
            torch.manual_seed(self.config['seed'])
            printlog(f'*** setting torch.manual_seed to {self.config["seed"]} ***')
            self.load_data()
            self.load_loss()
            self.load_optimiser()
            self.init_ema()
            if self.config.get('load_checkpoint', None):
                chkpt_type = 'last' if self.config.get('load_last', False) else 'best'
                self.load_checkpoint(chkpt_type)

        elif self.config['mode'] == 'inference':
            cudnn.benchmark = self.config['cudnn'] if 'cudnn' in self.config else True
            printlog(f'*** setting cudnn.benchmark {cudnn.benchmark} ***')
            torch.manual_seed(self.config['seed'])
            self.init_ema()
            self.load_data()

        elif self.config['mode'] == 'demo_tsne':
            cudnn.benchmark = self.config['cudnn'] if 'cudnn' in self.config else True
            printlog(f'*** setting cudnn.benchmark {cudnn.benchmark} ***')
            torch.manual_seed(self.config['seed'])
            self.load_data()

        elif self.config['mode'] == 'grad_cam':
            # this mode loads only the validation dataset of a split
            cudnn.benchmark = self.config['cudnn'] if 'cudnn' in self.config else True
            printlog(f'*** setting cudnn.benchmark {cudnn.benchmark} ***')
            torch.manual_seed(self.config['seed'])
            self.init_ema()
            self.load_data()
            assert self.dataset == 'OctBiom', f'grad_cam only implemented for OctBiom instead got {self.dataset}'
            assert 'load_checkpoint' in self.config, 'load_checkpoint must be specified in config for inference mode'
            if 'load_last' in self.config:
                chkpt_type = 'last' if self.config['load_last'] else 'best'
                self.load_checkpoint(chkpt_type)
            else:
                self.load_checkpoint('best')
            self.grad_cam()

        elif self.config['mode'] == 'submission_inference':
            assert self.dataset == 'RETOUCH', f'config["mode"]="submission_inference" only implemented for RETOUCH' \
                                              f' instead got {self.dataset}'

            cudnn.benchmark = self.config['cudnn'] if 'cudnn' in self.config else True
            printlog(f'*** setting cudnn.benchmark {cudnn.benchmark} ***')
            torch.manual_seed(self.config['seed'])
            self.init_ema()
            self.load_data()
            assert 'load_checkpoint' in self.config, 'load_checkpoint must be specified in config for inference mode'
            if 'load_last' in self.config:
                chkpt_type = 'last' if self.config['load_last'] else 'best'
                printlog(f'*** loading {chkpt_type} checkpoint ***')
                self.load_checkpoint(chkpt_type)
            else:
                printlog(f'*** loading best checkpoint ***')
                self.load_checkpoint('best')
            self.submission_infer()
        else:
            raise NotImplementedError(f"Mode {self.config['mode']} not implemented")

    def init_ema(self):
        """
        we initialize ema_model (an identical copy of self.model)
        all processes
        however this is only updated for process rank=0 during training (config['mode'] == 'training')
        during validation we use the ema_model instead of self.model
        during inference (config['mode'] == 'inference') we use the ema_model instead of self.model
        Note: Obvious but important: self.model is passed by reference to PolyakAverager
              thus optimiser.step() will also update the weights of self.ema_model.model
        Note: self.ema_model(x) = ema_model.average_model(x)
        :return:
        """
        self.use_ema = self.config['train'].get('ema', False)
        if self.use_ema:
            printlog(f"*** Initializing ema : {self.use_ema} with decay {self.config['train'].get('ema_decay', 0.99)}")
            model_class = globals()[self.config['graph']['model']]
            average_model = model_class(config=self.config['graph'], experiment=self.experiment)
            self.ema_model = PolyakAverager(self.model, average_model, self.config['train'].get('ema_decay', 0.99))
            self.ema_model.average_model.to(self.device)
            self.ema_model.model.to(self.device)

    def _setup_wandb(self):
        if self.rank == 0:
            # only rank 0 process logs with wandb
            tags = self.get_wandb_tag()
            wandb_project = self.config['logging'].get('wandb_project', 'MOP_v2')
            wandb_notes = self.config.get('wandb_notes', None)
            if self.use_wandb:
                wandb.init(project=wandb_project,
                           config=self.config,
                           name=self.run_id.split('__')[-1],
                           entity='theodoros-pissas',
                           tags=tags,
                           notes=wandb_notes)
            printlog(f"notes {wandb_notes}")
            printlog(f"*** wandb init on {self.rank} ***")

    def _setup_precision(self):
        # precision
        self.precision = self.config['train'].get('precision', "float32")
        printlog(f"*** setting precision to {self.precision} ***")
        if self.precision == "float16":
            self.dtype = torch.float16
            self.use_autocast = True
            self.scaler = torch.cuda.amp.GradScaler()
        elif self.precision == "float32":
            self.dtype = torch.float32
            self.use_autocast = False
        elif self.precision == "bfloat16":
            self.dtype = torch.bfloat16
            self.use_autocast = True
        else:
            raise ValueError(f"Unsupported precision {self.precision}")

    def ddp_train(self):
        assert self.config['mode'] == 'training', f"mode must be training, got {self.config['mode']}"
        assert self.parallel, f"parallel must be True, got {self.parallel}"
        set_seeds(self.config['seed'])
        self.rank = 0
        # mp.set_start_method('spawn') # moved spawn in main.py (repeating this here causes error)
        q = mp.Queue()
        printlog(f"created queue : {q}")
        ps = []
        printlog(f"r{self.rank} : spawning {self.n_gpus} processes for DDP training")
        for rank in range(self.n_gpus):
            p = mp.Process(target=self.distributed_train_worker, args=(rank, q))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
        printlog(f"r{self.rank} : back to main process")
        ret = q.get()
        if isinstance(ret, dict):  # dict rank:metrics
            printlog(f"r{self.rank} : got {ret}")
            ret = ret[0]  # only rank 0 returns metrics
        return ret

    def distributed_train_worker(self, gpu, q=None):
        """process called by mp.Process which runs on each GPU -- to return a value use the queue"""
        Log.init(logfile_level="info",
                 stdout_level=None,
                 log_file=self.log_file,
                 rewrite=True)
        set_seeds(self.config['seed'])
        if self.rank == 0:
            printlog("Run ID: {}".format(self.run_id))

        self.device = torch.device(f'cuda:{self.allocated_devices[gpu]}')
        torch.cuda.set_device(self.device)

        cudnn.benchmark = self.config.get('cudnn_benchmark', True)
        # cudnn.benchmark = self.config['cudnn_enabled'] if 'cudnn_enabled' in self.config else True
        # cudnn.benchmark = self.config['cudnn_determinstic'] if 'cudnn_deterministic' in self.config else True
        printlog(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***') if self.rank == 0 else None
        printlog(f"*** cudnn.enabled {cudnn.enabled}") if self.rank == 0 else None
        printlog(f"*** cudnn.deterministic {cudnn.deterministic}") if self.rank == 0 else None

        assert self.batch_size % self.n_gpus == 0, f'Batch size {self.batch_size} must be divisible' \
                                                   f' by number of gpus {self.n_gpus}'

        self.batch_size = int(self.batch_size) // self.n_gpus
        self.rank = self.rank * self.n_gpus + gpu
        printlog(f'Process on GPU: {self.device} Rank: {self.rank}')
        dist.init_process_group(backend='nccl', world_size=self.world_size, rank=self.rank)
        self._setup_precision()
        self._setup_wandb()
        self.load_model()
        self.init_ema()  # no effect if ema not requested
        self.load_data()
        self.load_loss()
        self.load_optimiser()

        if 'load_checkpoint' in self.config:
            if 'load_last' in self.config:
                chkpt_type = 'last' if self.config['load_last'] else 'best'
                self.load_checkpoint(chkpt_type)
            else:
                self.load_checkpoint('best')
        ret = self.train()
        if self.rank == 0:
            q.put({self.rank: ret})

    def train(self):
        """Main training loop"""
        printlog(f"***** Training started from {self.epoch}+{self.start_epoch} @ step {self.global_step} *****\n")
        self.epoch = 0  # current run's epoch counter

        while (self.epoch + self.start_epoch) <= self.config['train']['epochs'] - 1:

            if (self.epoch + self.start_epoch) == int(0.9 * self.config['train']['epochs']):
                self.valid_freq = 1  # run validation every epoch from now on
                if self.rank == 0:
                    printlog(f'** train: validation to be run after every {self.valid_freq} epoch from now on')

            if self.parallel:
                if self.using_multi_dataset_training and isinstance(self.train_sampler, list)\
                        and self.config['data'].get('use_omnivision_api', False):
                    pass
                    # note: self.sampler.set_epoch is called inside DatasetDataloader
                    # when get_loader() is called after each epoch
                else:
                    self.train_sampler.set_epoch(self.epoch+self.start_epoch)

            if self.epoch == 0:  # total required time estimation
                t1 = datetime.datetime.now()

            # if not self.debugging:
            self.validate() if self.epoch == 0 and not self.debugging else None
            self.train_one_epoch()

            if self.epoch == 0:  # total required time estimation
                t = (datetime.datetime.now() - t1).total_seconds()
                printlog('** Approx. run time: {:.1f} h total / {:.2f} h '
                         'per epoch'.format(t * (self.config['train']['epochs']-self.start_epoch) / 3600, t / 3600))

            if (self.epoch + self.start_epoch) % self.valid_freq == 0:
                self.validate()

            elif (self.epoch + self.start_epoch) == self.config['train']['epochs']:
                self.validate()
                break
            if self.rank == 0:
                self.save_checkpoint('latest')

            self.epoch += 1

        if not self.rank == 0:  # stop
            return

        # outro steps that are task-dependent
        self.print_final_msg()
        self.write_info_json()

        if self.rank == 0:
            metrics = self.infer_last_best()
            metrics['seed'] = self.config['seed']
            # save metrics in log_dir
            self.write_dict_json(metrics, f'metrics_seed_{self.config["seed"]}.json')
            return metrics
        else:
            return None

    def print_final_msg(self):
        if self.config['task'] == 'segmentation':
            printlog("\n***** Training finished *****\n"
                     "Run ID: {}\n"
                     "     Best validation loss: {:.5f}\n".format(self.run_id, self.best_loss))
            msg_stra = "     Best mIoU        (tot"
            msg_strb = "     best loss mIoU   (tot "
            msg_strC = "     FINAL mIoU   (tot "
            msg1, msg2 = "", "{:.4f} ".format(self.metrics['best_miou'])
            msg3, msg4 = "", "{:.4f} ".format(self.metrics['best_loss_miou'])
            msg5, msg6 = "", "{:.4f} ".format(self.metrics['final_miou'])

            for categ in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]:
                msg1 += "/ {} ".format(categ)
                msg2 += "{:.4f} ".format(self.metrics['best_miou_{}'.format(categ)])
                msg3 += "/ {} ".format(categ)
                msg4 += "{:.4f} ".format(self.metrics['best_loss_miou_{}'.format(categ)])
                msg5 += "/ {} ".format(categ)
                msg6 += "{:.4f} ".format(self.metrics['final_miou_{}'.format(categ)])
            printlog(msg_stra + msg1 + ' ): ' + msg2 + '@ epoch {} (step {})'.format(
                *self.metrics['best_miou_epoch_step']))
            printlog(msg_strb + msg3 + ' ): ' + msg4 + '@ epoch {} (step {})'.format(
                *self.metrics['best_loss_epoch_step']))
            printlog(msg_strC + msg5 + ' ): ' + msg6 + '@ epoch {} (step {})'.format(self.config['train']['epochs'],
                                                                                     self.global_step))

            self.finalise()
            # if self.run_final_val and self.rank == 0:# only rank=0 process runs this if ddp
            #     printlog(f'starting validation with final model r {self.rank}')
            #     self.config.update({"load_checkpoint": self.run_id, "load_last": True, "tta": False})
            #     self.infer()

        elif self.config['task'] == 'pretraining':
            printlog("\n***** Training finished *****\n"
                     "Run ID: {}\n"
                     "     Best validation loss: {:.5f} {}\n".format(self.run_id,
                                                                     self.metrics['best_val_loss'],
                                                                     self.metrics['best_loss_epoch_step'])
                     )

        elif self.config['task'] == 'detection':
            printlog("\n***** Training finished *****\n"
                     "Run ID: {}\n"
                     "     Best validation loss: {:.5f}\n".format(self.run_id, self.best_loss))
            msg_stra = f"     Best ROC AUC    "
            msg_strb = f"     FINAL ROC AUC   "
            msg1, msg2 = "", "{:.4f} ".format(self.metrics['best_ROC AUC'])
            msg3, msg4 = "", "{:.4f} ".format(self.metrics['final_ROC AUC'])

            printlog(msg_stra + msg1 + ' : ' + msg2 + '@ epoch {} (step {})'.format(
                *self.metrics['best_ROC AUC_epoch_step']))
            printlog(msg_strb + msg3 + ' : ' + msg4 + '@ epoch {} (step {})'.format(
                *self.metrics['final_ROC AUC_epoch_step']))

            self.finalise()

    def inference(self):
        if self.dataset in ['OctBiom', 'OCTDL', 'OCTID', 'DR', 'OLIVES',
                            'RETOUCH', 'AROI', 'RAVIR', 'OCT5K']:
            assert 'load_checkpoint' in self.config, 'load_checkpoint must be specified in config for inference mode'
            metrics = self.infer_last_best()
            return metrics
        else:
            printlog(f"No inference mode implemented for or unknown dataset : {self.dataset}")
            return {}

    def infer_last_best(self):
        """
        Runs inference on the last checkpoint and the best checkpoint (with and without ema)
        :return:
        """
        metrics = {"best_ema": {"test": {}, "val": {}},
                   "last_ema": {"test": {}, "val": {}},
                   "best": {"test": {}, "val": {}},
                   "last": {"test": {}, "val": {}},
                   }  # store metrics

        if self.rank > 0:
            return  # only rank=0 process runs this if ddp

        if self.config['mode'] == 'inference':
            assert 'load_checkpoint' in self.config, f'missing "load_checkpoint" for mode {self.config["mode"]}'
            printlog(f"setting run_id to load_checkpoint {self.config['load_checkpoint']} (rank={self.rank})")
            self.run_id = self.config['load_checkpoint']

        if self.dataset in ['OctBiom', 'DR', 'OCTID', 'OCTDL', 'OLIVES']:
            printlog("*"*10)
            printlog(f'starting testing with final model (rank={self.rank})')
            self.config.update({"mode": "inference"})   # change mode
            test_split = 'test'
            if self.dataset == 'DR':
                test_split = 'test_refined'
            printlog(f"setting split to {test_split}")
            # hacky way to change split and build dataloader
            self.config["data"].update({"split": [test_split, test_split]})
            self.load_data()  # build dataloader of test split

            checkpoint_has_ema = self.use_ema and self.ema_model is not None
            printlog("checkpoint has ema: {}".format(checkpoint_has_ema))
            if checkpoint_has_ema:
                printlog(f"Found ema going to evaluate first without ema and then with ema ...")
                self.use_ema = False  # switch ema off

            # load best checkpoint (no ema)
            printlog(f"{'*' * 10}Testing best checkpoint ema={self.use_ema} {'*' * 10}")
            self.config.update({"load_checkpoint": self.run_id, "load_last": False})
            self.load_checkpoint('best')

            metrics_test, metrics_val, latex_line = self.infer(show_with_validation_metrics=True, chkpt_type='best')
            metrics["best"]["test"] = metrics_test
            metrics["best"]["val"] = metrics_val

            if checkpoint_has_ema:
                self.use_ema = True  # switch ema on
                printlog(f"\n{'*'*10}Testing best checkpoint ema={self.use_ema} {'*'*10}" )
                self.config.update({"load_checkpoint": self.run_id, "load_last": False})
                self.load_checkpoint('best')
                metrics_test, metrics_val, latex_line = self.infer(show_with_validation_metrics=True, chkpt_type='best')
                metrics["best_ema"]["test"] = metrics_test
                metrics["best_ema"]["val"] = metrics_val

            # load final checkpoint (no ema)
            self.use_ema = False  # switch ema off
            printlog(f"\n{'*' * 10}Testing best checkpoint ema={self.use_ema} {'*' * 10}")
            self.config.update({"load_checkpoint": self.run_id, "load_last": True})
            self.load_checkpoint('last')
            metrics_test, metrics_val, latex_line = self.infer(show_with_validation_metrics=True, chkpt_type='final')
            metrics["last"]["test"] = metrics_test
            metrics["last"]["val"] = metrics_val

            # infer with ema from final checkpoint
            if checkpoint_has_ema:
                self.use_ema = True  # switch ema on
                printlog(f"\n{'*'*10}Testing best checkpoint ema={self.use_ema} {'*'*10}")
                self.config.update({"load_checkpoint": self.run_id, "load_last": True})
                self.load_checkpoint('last')
                metrics_test, metrics_val, latex_line = self.infer(show_with_validation_metrics=True, chkpt_type='final')
                metrics["last_ema"]["test"] = metrics_test
                metrics["last_ema"]["val"] = metrics_val

        elif self.dataset in ['RETOUCH', 'AROI', 'RAVIR', 'OCT5K']:
            # we assume that we DO NOT use ema for these segmentation datasets
            # we use final checkpoints here
            if self.config['mode'] == 'inference':
                if 'load_last' in self.config:
                    chkpt_type = 'last' if self.config['load_last'] else 'best'
                    self.load_checkpoint(chkpt_type)
                else:
                    self.load_checkpoint('best')
                mious, _, latex_line = self.infer()
                warnings.warn("best* metrics are only valid when run at the end of training:"
                              "only take into account final* metrics printed below")

            metric_names = self.metrics_per_dataset[self.dataset]
            mious_last = {f'{m}': self.metrics[f'final_{m}'].item() for m in metric_names}
            mious_best = {f'{m}': self.metrics[f'best_{m}'].item() for m in metric_names}
            metrics["last"]["val"].update(mious_last)
            metrics["best"]["val"].update(mious_best)
        else:
            printlog(f"no last inference_last_best for dataset : {self.dataset}")
            metrics = {}
        self.write_info_json()
        printlog("*" * 10)
        return metrics

    def load_data(self):
        """Creates a dict containing the training and validation data loaders, loaded into self.data"""
        # Create dataloaders

        if self.config['mode'] == 'inference':
            train_df, valid_df = self.get_seg_dataframes()
            _, valid_loader = self.get_dataloaders(train_df, valid_df, 'default')
            self.data_loaders = {'valid_loader': valid_loader}
            return

        train_df, valid_df = self.get_seg_dataframes()
        train_loader, valid_loader = self.get_dataloaders(train_df, valid_df, 'default')

        # fixme workaround: everything below here is based on the assumption of a single dataset thus we just return
        if (self.dataset == 'MKEKI' and not self.config['data'].get('paired', False)) or \
                (self.dataset == 'OCTIR' and not self.config['data'].get('paired', False)
                 and type(self.config['data']['modality']) == list):
            # only MKEKI unpaired requires this modification to standard dataloader structure
            self.using_multi_dataset_training = True
            self.data_loaders = {'train_loader': train_loader, 'valid_loader': valid_loader}
            return

        self.data_loaders = {'train_loader': train_loader, 'valid_loader': valid_loader}
        self.batches_per_epoch = len(train_loader)

    def get_seg_dataframes(self):
        """Create pd.dataframes only for datasets that use them for splitting else returns None, None"""
        # datasets using pandas dataframes for splitting
        # add other datasets if needed
        datasets_with_pd = ['RETOUCH']
        # note: supported datasets that are splitted using info given in their implementations in MOP/datasets/
        # ['KEKI', 'OctBiom', 'IACL', 'AROI', 'MKEKI', 'DR', "OCTIR", "OCTID", "OCTDL", "RAVIR", 'OCT5K']

        if self.dataset in datasets_with_pd:
            printlog(f"Loading dataframes for {self.dataset} ...")
            train, valid = get_retouch_dataframes(self.config)
            return train, valid
        else:
            printlog(f"No dataframes for {self.dataset} ...")
            return None, None

    @staticmethod
    def _sanity_check_multidaset_config(config):
        printlog("Sanity check for config.data ...")
        printlog("********************************")
        assert 'modality' in config, 'config.data must contain modality key'
        assert isinstance(config['modality'], list), f'config.data.modality must ' \
                                                     f'be a list instead got {type(config["modality"])}'
        num_modalities = len(config['modality'])
        printlog(f"num_modalities {num_modalities} : {config['modality']}")
        required_list_keys = ['transforms', 'transform_values', 'img_channels', 'mode']
        for key in required_list_keys:
            assert key in config.keys(), f"config.data must contain {key} key"
            assert isinstance(config[key], list), f"config.data.{key} must be a list instead got {type(key)}"
            assert len(config[key]) == num_modalities, f"config.data.{key} must have the same length ({num_modalities})" \
                                               f" as config.data.modality instead got {len(key)}"
            printlog(f"Key: {key}:{config[key]}")
        printlog("Finished sanity check for config.data ...")
        printlog("*****************************************")

    @staticmethod
    def is_a_multimodal_dataset(dataset, config):
        return dataset == 'MKEKI' or (dataset == 'DR' and len(config['data']['modality']) == 2) or \
               (dataset == 'OCTIR' and len(config['data']['modality']) == 2 and type(config['data']['modality']) == list) or \
               (dataset == 'OLIVES' and len(config['data']['modality']) == 2)

    def get_dataloaders(self,
                        train_df: Union[pd.DataFrame, None],
                        valid_df: Union[pd.DataFrame, None],
                        mode: str = 'default', **kwargs):
        """
        Creates the training and validation segmentation datasets and dataloaders from the config
        :param train_df: optional
        :param valid_df: optional
        :param mode: dataloader mode
        :param kwargs:
        :return: dict('train_loader': Dataloader, 'valid_loader': Dataloader)
        """
        transforms_dict = dict()
        parser = partial(parse_transform_lists, dataset=self.dataset, experiment=self.experiment)
        ############################ parse transforms lists from config ################################################
        if self.is_a_multimodal_dataset(self.dataset, self.config):
            # note: multimodal case : transforms_dict['train'] and transforms_dict['valid'] are dictionaries
            # with keys being the modalities and values being lists of transforms (functions)
            self._sanity_check_multidaset_config(self.config['data'])
            transforms_dict['train'] = dict()
            transforms_dict['valid'] = dict()
            modalities = self.config['data']['modality']
            for modality, tdict, tvalues, tdict_val, tvalues_val in zip(modalities,
                                                                        self.config['data']['transforms'],
                                                                        self.config['data']['transform_values'],
                                                                        self.config['data']['transforms_val'],
                                                                        self.config['data']['transform_values_val']):

                printlog(f"Parsing transform_lists for {modality} ...")
                transforms_dict['train'][modality] = parser(tdict, tvalues)
                transforms_dict['valid'][modality] = parser(tdict_val, tvalues_val)

                # Dataset transforms console output
                img_transforms = [str(type(item).__name__) for item in transforms_dict['train'][modality]['img'] if
                                  not (isinstance(item, ToPILImage) or isinstance(item, ToTensor))]
                common_transforms = [str(type(item).__name__) for item in transforms_dict['train'][modality]['common']]
                printlog("Dataset transforms: {}".format(img_transforms + common_transforms))

            if any(['torchvision_normalise' in d for d in self.config['data']['transforms']]):
                self.un_norm = un_normalise  # for un-normalizing images for visualization
            else:
                self.un_norm = do_nothing

        else:
            # single-modality dataset
            # note: transforms_dict['train'] and transforms_dict['valid'] are lists of transforms (functions)
            transforms_dict['train'] = parser(self.config['data']['transforms'],
                                              self.config['data']['transform_values'])
            transforms_dict['valid'] = parser(self.config['data']['transforms_val'],
                                              self.config['data']['transform_values_val'])
            if 'torchvision_normalise' in self.config['data']['transforms']:
                self.un_norm = un_normalise  # for un-normalizing images for visualization
            else:
                self.un_norm = do_nothing
            # Dataset transforms console output
            img_transforms = [str(type(item).__name__) for item in transforms_dict['train']['img'] if
                              not (isinstance(item, ToPILImage) or isinstance(item, ToTensor))]
            common_transforms = [str(type(item).__name__) for item in transforms_dict['train']['common']]
            printlog("Dataset transforms: {}".format(img_transforms + common_transforms))

        data_path = self.config['data_path']
        real_num_classes = DATASETS_INFO[self.dataset].NUM_CLASSES[self.experiment]  # irrelevant for pretraining
        printlog(f'num classes {real_num_classes} exp {self.experiment}')
        num_workers = int(self.config['data']['num_workers'])

        ########################################### OCTIR single-modality pretraining  #################################
        if self.dataset == 'OCTIR' and isinstance(self.config['data']['modality'], str):
            metafile = self.config['data']['metafile']
            ood_file = self.config['data'].get('ood_file', None)
            max_per_modality_per_patient = self.config['data'].get('max_per_modality_per_patient', 4)
            path_to_metafile = os.path.join(data_path, metafile)
            path_to_ood_file = os.path.join(data_path, ood_file) if ood_file is not None else None
            get_paths_method = self.config['data'].get('get_paths_method', 'naive')  # same for all modalities
            stratification_method = self.config['data'].get('stratification_method', None)  # same for all modalities
            modality = self.config['data']['modality']
            split = self.config['data']['split']
            printlog(f"Preparing OCTIR Single modality {modality}"
                     f" with metafile: {metafile}\n"
                     f" ood_file: {ood_file}\n"
                     f" get_paths_method: {get_paths_method}\n"
                     f" stratification_method: {stratification_method}")

            train_split = split[0] if not self.debugging else 'train'
            valid_split = split[1] if not self.debugging else 'val'

            train_set = OCTIRSingle(root=data_path / 'resized',
                                    metafile=path_to_metafile,
                                    transforms_dict=transforms_dict['train'],
                                    split=train_split,
                                    modality=modality,
                                    mode=self.config['data']['mode'],
                                    debug=False,
                                    # keep_every_nth_slice=keep_every_nth_slice,
                                    img_channels=self.config['data'].get('img_channels', 1),
                                    return_metadata=True,
                                    use_omnivision_api=False,
                                    get_paths_method=get_paths_method,
                                    stratification_method=stratification_method,
                                    max_per_modality_per_patient=max_per_modality_per_patient,
                                    keep_every_nth_patient=self.config['data'].get('keep_every_nth_patient', 1),
                                    ood_modality_to_patient_to_filename_to_score=path_to_ood_file,
                                    log_dir=self.log_dir,
                                    use_kermany=self.config['data'].get('use_kermany', False)
                                    )

            valid_set = OCTIRSingle(root=data_path / 'resized',
                                    metafile=path_to_metafile,
                                    debug=False,
                                    # keep_every_nth_slice=keep_every_nth_slice,
                                    img_channels=self.config['data'].get('img_channels', 1),
                                    split=valid_split,
                                    modality=modality,
                                    mode=self.config['data']['mode'],
                                    transforms_dict=transforms_dict['valid'],
                                    get_paths_method=get_paths_method,
                                    stratification_method=stratification_method,
                                    max_per_modality_per_patient=max_per_modality_per_patient,
                                    ood_modality_to_patient_to_filename_to_score=path_to_ood_file,
                                    log_dir=self.log_dir)

            self.train_sampler = None
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            # self.batches_per_epoch = len(train_loader)
            # todo figure out what to do with valid_loader should not be created in every worker
            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}"
                     "\n".format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        ########################################### OCTIR multimodal PAIRED pretraining  ###############################
        if self.dataset == 'OCTIR' and isinstance(self.config['data']['modality'], list) and self.config['data']['paired']:
            # self.train_sampler = []
            # self._sanity_check_multidaset_config(self.config['data'])
            metafile = self.config['data']['metafile']
            ood_files = self.config['data'].get('ood_file', [None] * len(self.config['data']['modality']))
            max_per_modality_per_patient = self.config['data'].get('max_per_modality_per_patient',
                                                                   [100] * len(self.config['data']['modality']))
            max_per_modality_per_patient_val = self.config['data'].get('max_per_modality_per_patient_val',
                                                                          [100] * len(self.config['data']['modality']))

            cap_dates_per_patient = self.config['data'].get('cap_dates_per_patient',
                                                            [None] * len(self.config['data']['modality']))
            cap_dates_per_patient_val = self.config['data'].get('cap_dates_per_patient_val',
                                                                [None] * len(self.config['data']['modality']))

            path_to_metafile = os.path.join(data_path, metafile)
            get_paths_method = self.config['data'].get('get_paths_method', 'naive')  # same for all modalities
            stratification_method = self.config['data'].get('stratification_method', None)  # same for all modalities

            printlog(f"Preparing MKEKI with metafile: {metafile}\n"
                     f"ood_files: {ood_files}\n"
                     f"get_paths_method: {get_paths_method}\n"
                     f"stratification_method: {stratification_method}"
                     f"max_per_modality_per_patient: {max_per_modality_per_patient}"
                     f"max_per_modality_per_patient_val: {max_per_modality_per_patient_val}"
                     f"cap_dates_per_patient: {cap_dates_per_patient}"
                     f"cap_dates_per_patient_val: {cap_dates_per_patient_val}")

            train_split = 'train'
            valid_split = 'val'
            train_set = OCTIRPaired(root=data_path / 'resized',
                                    metafile=path_to_metafile,
                                    transforms_dict=transforms_dict['train'],
                                    split=train_split,
                                    modalities=self.config['data']['modality'],
                                    modes=self.config['data']['mode'],
                                    debug=False,
                                    img_channels=self.config['data'].get('img_channels', (3, 3)),
                                    return_metadata=True,
                                    get_paths_method=get_paths_method,
                                    stratification_method=stratification_method,
                                    cap_dates_per_patient=cap_dates_per_patient,
                                    max_per_modality_per_patient=max_per_modality_per_patient,
                                    log_dir=self.log_dir)

            valid_set = OCTIRPaired(root=data_path / 'resized',
                                    metafile=path_to_metafile,
                                    transforms_dict=transforms_dict['valid'],
                                    split=valid_split,
                                    modalities=self.config['data']['modality'],
                                    modes=self.config['data']['mode'],
                                    debug=False,
                                    img_channels=self.config['data'].get('img_channels', (3, 3)),
                                    return_metadata=True,
                                    get_paths_method=get_paths_method,
                                    stratification_method=stratification_method,
                                    cap_dates_per_patient=cap_dates_per_patient_val,
                                    max_per_modality_per_patient=max_per_modality_per_patient_val,
                                    log_dir=self.log_dir)

            self.train_sampler = None
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            # self.batches_per_epoch = len(train_loader)
            # todo figure out what to do with valid_loader should not be created in every worker
            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}"
                     "\n".format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers,
                                                          self.grad_accumulation_steps))
            return train_loader, valid_loader

        ########################################### OCTIR multimodal UN-PAIRED pretraining  ############################
        if self.dataset == 'OCTIR' and isinstance(self.config['data']['modality'], list) and not self.config['data']['paired']:
            if self.grad_accumulation_steps > 1:
                raise NotImplementedError("Gradient accumulation is not implemented for multimodal unpaired training")
            n_modalities = len(self.config['data']['modality'])
            self.train_sampler = []

            path_to_metafiles = [os.path.join(data_path, metafile) for metafile in self.config['data']['metafile']]
            max_per_modality_per_patient = self.config['data'].get('max_per_modality_per_patient', [1] * n_modalities)
            get_paths_method = self.config['data'].get('get_paths_method', ['stratified', 'stratified'])
            stratification_method = self.config['data'].get('stratification_method',[None]*n_modalities)

            printlog(f"Preparing unpaired OCTIR with metafile: {path_to_metafiles}\n"
                     f"ood_files: {None}\n"
                     f"get_paths_method: {get_paths_method}\n"
                     f"stratification_method: {stratification_method}")
            train_sets_loaders = []  # storing DatasetDataLoader objects for training sets
            valid_loaders = dict()  # storing DataLoaders for validation sets : dict[modality] = DataLoader
            native_repeat_factors = self.config['data'].get('native_repeat_factors', [1] * n_modalities)
            for i, (modality, split), in enumerate(zip(self.config['data']['modality'], self.config['data']['split'])):
                train_split = split[0] if not self.debugging else 'train'
                valid_split = split[1] if not self.debugging else 'val'

                train_set = OCTIRSingle(root=data_path / 'resized',
                                        metafile=path_to_metafiles[i],
                                        transforms_dict=transforms_dict['train'][modality],
                                        split=train_split,
                                        modality=modality,
                                        mode=self.config['data']['mode'][i],
                                        debug=False,
                                        # keep_every_nth_slice=keep_every_nth_slice,
                                        img_channels=self.config['data'].get('img_channels', [1, 1])[i],
                                        return_metadata=True,
                                        use_omnivision_api=True,
                                        get_paths_method=get_paths_method[i],
                                        stratification_method=stratification_method[i],
                                        max_per_modality_per_patient=max_per_modality_per_patient[i],
                                        ood_modality_to_patient_to_filename_to_score=None,
                                        log_dir=self.log_dir,
                                        use_kermany=self.config['data'].get('use_kermany', False)
                                        )

                valid_set = OCTIRSingle(root=data_path / 'resized',
                                        metafile=path_to_metafiles[i],
                                        debug=False,
                                        # keep_every_nth_slice=keep_every_nth_slice,
                                        img_channels=self.config['data'].get('img_channels', 1),
                                        split=valid_split,
                                        modality=modality,
                                        mode=self.config['data']['mode'][i],
                                        transforms_dict=transforms_dict['valid'][modality],
                                        get_paths_method=get_paths_method[i],
                                        stratification_method=stratification_method[i],
                                        max_per_modality_per_patient=max_per_modality_per_patient[i],
                                        ood_modality_to_patient_to_filename_to_score=None,
                                        log_dir=self.log_dir)
                if self.parallel:
                    sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                              rank=self.rank,
                                                                              num_replicas=self.n_gpus)
                    self.train_sampler.append(sampler)
                else:
                    sampler = None

                train_dataset_dataloader = DatasetDataloader(train_set,
                                                             batch_size=self.batch_size,
                                                             # we set num_workers = cpus requested,
                                                             # equally distributed among modalities
                                                             num_workers=num_workers//n_modalities,
                                                             worker_init_fn=worker_init_fn,
                                                             # setting to True may speed up a bit
                                                             # but it almost always leads to dataloader crashes
                                                             persistent_workers=False,
                                                             sampler=sampler,
                                                             output_key=modality)

                train_sets_loaders.append(train_dataset_dataloader)

                valid_dataloader = DataLoader(valid_set,
                                              batch_size=self.valid_batch_size,
                                              num_workers=num_workers,
                                              worker_init_fn=worker_init_fn)

                valid_loaders[modality] = valid_dataloader
                printlog(f'KEKI {modality} dataset train on [{train_split}] valid on [{valid_split}]')
            # self.batches_per_epoch = len(train_loader)
            # todo figure out what to do with valid_loader should not be created in every worker

            train_concat_loader = ConcatDataset(train_sets_loaders,
                                                repeat_factors=self.config['data'].get('repeat_factors', None),
                                                batching_strategy='use_one',
                                                max_steps='sum')

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers (per modality {}): {}\n".format(num_workers//n_modalities,
                                                                                      self.batch_size, num_workers)
                     )
            return train_concat_loader, valid_loaders
        ########################################### DR training (single/multi-modal) ###################################
        if self.dataset == 'DR':
            train_split = self.config['data']['split'][0]
            valid_split = self.config['data']['split'][1] if len(self.config['data']['split']) > 1 else 'test'
            train_set = DRDataset(root=data_path,
                                  split=train_split,
                                  transforms_dict=transforms_dict['train'],
                                  modalities=self.config['data']['modality'],
                                  modes=self.config['data']['mode'],
                                  use_condor=self.config['data'].get('use_condor', False),
                                  max_condor_dates=self.config['data'].get('max_condor_dates', 3))

            if valid_split == 'test_refined':  # testing only
                max_dates = 2
            else:
                max_dates = 3  # no effect on validation set

            valid_set = DRDataset(root=data_path, split=valid_split,
                                  transforms_dict=transforms_dict['valid'],
                                  modalities=self.config['data']['modality'],
                                  modes=self.config['data']['mode'],
                                  use_condor=self.config['data'].get('use_condor', False),
                                  max_condor_dates=max_dates)

            printlog(f'DR dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)
            else:
                self.train_sampler = None

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     .format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'OLIVES':
            train_split = self.config['data']['split'][0]
            valid_split = self.config['data']['split'][1] if len(self.config['data']['split']) > 1 else 'test'
            train_set = OLIVES.init_from_config(data_path, train_split, transforms_dict['train'], self.config['data'])
            valid_set = OLIVES.init_from_config(data_path, valid_split, transforms_dict['valid'], self.config['data'])
            printlog(f'OLIVES dataset train on [{train_split}] valid on [{valid_split}]')
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)
            else:
                self.train_sampler = None

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     .format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'OctBiom':
            train_split = self.config['data']['split'][0]
            valid_split = self.config['data']['split'][1] if len(self.config['data']['split']) > 1 else 'val'
            train_set = OctBiom(root=data_path, hdf5_file=self.config['data']['hdf5_file'],
                                transforms_dict=transforms_dict['train'], split=train_split,  # fixme
                                img_channels=self.config['data'].get('img_channels', 3))

            valid_set = OctBiom(root=data_path, hdf5_file=self.config['data']['hdf5_file'],
                                transforms_dict=transforms_dict['valid'], split=valid_split,
                                img_channels=self.config['data'].get('img_channels', 3))

            printlog(f'OctBiom dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)
            else:
                self.train_sampler = None

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     .format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'OCTID':
            train_split = self.config['data']['split'][0]
            valid_split = self.config['data']['split'][1] if len(self.config['data']['split']) > 1 else 'val'
            train_set = OCTID(root=data_path, split=train_split, transforms_dict=transforms_dict['train'],
                              img_channels=self.config['data'].get('img_channels', 3))

            valid_set = OCTID(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'],
                              img_channels=self.config['data'].get('img_channels', 3))

            printlog(f'OCTID dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)
            else:
                self.train_sampler = None

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     .format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'OCTDL':
            train_split = self.config['data']['split'][0]
            valid_split = self.config['data']['split'][1] if len(self.config['data']['split']) > 1 else 'val'
            train_set = OCTDL(root=data_path, split=train_split, transforms_dict=transforms_dict['train'],
                              img_channels=self.config['data'].get('img_channels', 3))

            valid_set = OCTDL(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'],
                              img_channels=self.config['data'].get('img_channels', 3))

            printlog(f'OCTDL dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)
            else:
                self.train_sampler = None

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set,
                                      batch_size=self.valid_batch_size,
                                      num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     .format(len(train_set), len(valid_set)))

            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'IACL':
            train_split = self.config['data']['split'][0]
            train_set = IACL(root=data_path, split=train_split, transforms_dict=transforms_dict['train'])
            valid_split = self.config['data']['split'][1]
            valid_set = IACL(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'])
            if self.save_outputs and self.config['mode'] in ['inference', 'training']:
                valid_set.return_filename = True
            printlog(f'IACL dataset train on [{train_split}] valid on [{valid_split}]')

            self.train_sampler = None
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)

            train_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            # todo figure out what to do with valid_loader should not be created in every worker
            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)
            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'RETOUCH':
            # if mode == 'default':
            is_submission_inference = self.config['mode'] == 'submission_inference'
            if not is_submission_inference:
                train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                          dataset=self.dataset, data_path=data_path, debug=self.debugging,
                                          config=self.config)
                valid_set = DatasetFromDF(valid_df, self.experiment, transforms_dict['valid'],
                                          dataset=self.dataset, data_path=data_path, debug=self.debugging,
                                          config=self.config)
            else:
                train_set = DatasetFromDFSub(train_df, self.experiment, transforms_dict['train'],
                                             dataset=self.dataset, data_path=data_path, debug=self.debugging,
                                             config=self.config)
                valid_set = DatasetFromDFSub(valid_df, self.experiment, transforms_dict['valid'],
                                             dataset=self.dataset, data_path=data_path, debug=self.debugging,
                                             config=self.config)

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)
            else:
                self.train_sampler = None

            train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None, num_workers=num_workers,
                                      worker_init_fn=worker_init_fn)
            # self.batches_per_epoch = len(train_loader)
            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)
            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'AROI':

            train_split = self.config['data']['split'][0]
            train_set = AROI(root=data_path, split=train_split, transforms_dict=transforms_dict['train'],
                             img_channels=self.config['data'].get('img_channels', 1))
            valid_split = self.config['data']['split'][1]
            valid_set = AROI(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'],
                             img_channels=self.config['data'].get('img_channels', 1))
            if self.save_outputs and self.config['mode'] in ['inference', 'training']:
                valid_set.return_filename = True

            printlog(f'AROI dataset train on [{train_split}] valid on [{valid_split}]')
            self.train_sampler = None
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'RAVIR':
            train_split = self.config['data']['split'][0]
            train_set = RAVIR(root=data_path, split=train_split, transforms_dict=transforms_dict['train'],
                              img_channels=self.config['data'].get('img_channels', 1))
            valid_split = self.config['data']['split'][1]
            valid_set = RAVIR(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'],
                              img_channels=self.config['data'].get('img_channels', 1))
            if self.save_outputs and self.config['mode'] in ['inference', 'training']:
                valid_set.return_filename = True

            printlog(f'RAVIR dataset train on [{train_split}] valid on [{valid_split}]')
            self.train_sampler = None
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        if self.dataset == 'OCT5K':
            train_split = self.config['data']['split'][0]
            train_set = OCT5K(root=data_path, split=train_split, transforms_dict=transforms_dict['train'],
                              img_channels=self.config['data'].get('img_channels', 1))
            valid_split = self.config['data']['split'][1]
            valid_set = OCT5K(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'],
                              img_channels=self.config['data'].get('img_channels', 1))
            if self.save_outputs and self.config['mode'] in ['inference', 'training']:
                valid_set.return_filename = True

            printlog(f'{self.dataset} dataset train on [{train_split}] valid on [{valid_split}]')
            self.train_sampler = None
            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                                     rank=self.rank,
                                                                                     num_replicas=self.n_gpus)

            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                     "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                     "              Number of workers: {}\n"
                     "              GradAccum: {}".format(self.batch_size, num_workers, self.grad_accumulation_steps))
            return train_loader, valid_loader

        else:
            raise NotImplementedError(f'Unknown dataset {self.dataset}')

    def load_model(self):
        """Loads the model into self.model"""
        model_class = globals()[self.config['graph']['model']]
        self.config['graph'].update({'internal_checkpoint_dir': self.config.get('internal_pretrained_path', None)})
        self.config['graph'].update({'external_checkpoint_dir': self.config.get('external_checkpoints', None)})
        self.config['graph'].update({'task': self.config.get('task')})
        self.model = model_class(config=self.config['graph'], experiment=self.experiment)
        out_stride = self.model.out_stride

        if hasattr(self.model, 'projector_model'):
            self.return_features = self.model.projector_model is not None and self.config['mode'] == 'training'
        else:
            self.return_features = False

        if self.parallel:  # ddp wrapper
            self.model.cuda()
            printlog(f"using sync batch norm : {self.config['graph']['sync_bn']}")

            if self.config['graph']['sync_bn']:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            elif self.config['graph']['phase'] == 'linear_probing':
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            else:
                printlog(f"using sync batch norm : {False}")

            # todo check if gradient_as_bucket_view = True slows down
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.device],
                                                                   find_unused_parameters=self.config.get(
                                                                       'find_unused_parameters', False),
                                                                   gradient_as_bucket_view=True)
        else:
            self.model = self.model.to(self.device)

        num_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        printlog(f"Using model '{self.config['graph']['model']}' with backbone '{self.config['graph']['backbone']}' "
                 f"with output stride {out_stride} : trainable parameters {num_train_params}")

    @property
    def flat_model(self):
        """returns self.model without the ddp wrapper if ddp else just returns self.model"""
        if self.parallel:
            return self.model.module
        else:
            return self.model

    @staticmethod
    def unpack_batch(batch: tuple) -> dict:
        """ Unpacks batch into dictionary with keys 'img', 'lbl', 'metadata' """
        batch_dict = dict()
        if len(batch) == 2:
            img, lbl = batch
            metadata = None
        else:
            img, lbl, metadata = batch
        batch_dict['img'] = img
        batch_dict['lbl'] = lbl
        batch_dict['metadata'] = metadata
        return batch_dict

    def load_loss(self):
        """Load loss function"""
        assert 'loss' in self.config
        if self.config['loss']['name'] == 'manager':
            printlog(f"loss is handled internaly by manager: {self.config['manager']}")
            assert hasattr(self, 'get_loss')
            return None

        elif self.config['loss']['name'] == 'RegressionLoss':
            # todo STAGE regression approach
            raise NotImplementedError("Implement for stage - make task specific")

        elif self.config['loss']['name'] == 'sigmoidBCE':
            use_weights = self.config['loss'].get('use_weights', False) \
                            and hasattr(self.data_loaders[self.train_schedule[self.epoch]].dataset, 'weights')
            printlog(f"loss is sigmoidBCE with use_weights {use_weights}")
            self.loss_weights = None
            if use_weights:
                self.loss_weights = self.data_loaders[self.train_schedule[self.epoch]].dataset.weights
                printlog(f"    loss weights {self.loss_weights}")
            self.loss = nn.BCEWithLogitsLoss(pos_weight=self.loss_weights)
            self.loss = self.loss.to(self.device)

        elif self.config['loss']['name'] == 'BCE':
            # if requested use_weights and dataset has weights then use them
            use_weights = self.config['loss'].get('use_weights', False) \
                          and hasattr(self.data_loaders['train_loader'].dataset, 'weights')
            printlog(f"loss is BCE with use_weights {use_weights}")
            self.loss_weights = None
            if use_weights:
                self.loss_weights = self.data_loaders['train_loader'].dataset.weights
                printlog(f"loss weights {self.loss_weights}")
            self.loss = nn.CrossEntropyLoss(weight=self.loss_weights)
            self.loss = self.loss.to(self.device)

        elif self.config['loss']['name'] == 'CLIPLoss':
            printlog(f"loss is CLIPLoss")
            printlog(f"world_size: {self.n_gpus} , rank: {self.rank}")
            self.loss = CLIPLoss(world_size=self.n_gpus, rank=self.rank)
        else:
            # segmentation losses only
            self.config['loss']['experiment'] = self.experiment
            self.config['loss']['device'] = str(self.device)
            loss_class = globals()[self.config['loss']['name']]
            if self.config['loss']['name'] == 'CrossEntropyLoss':
                ignore_index_in_loss = len(
                    DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1 \
                    if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else -100
                self.loss = CrossEntropyLoss(ignore_index=ignore_index_in_loss)
            else:
                self.loss = loss_class(self.config['loss'])

            self.loss = self.loss.to(self.device)

            if isinstance(self.loss, LossWrapper):
                printlog(f"Loaded loss: {self.loss.info_string} rank : {self.rank}")
            else:
                printlog(f"Loaded loss function: {loss_class} rank : {self.rank}")

    def load_optimiser(self):
        """Set optimiser and if required, learning rate schedule"""
        params = self.model.parameters()
        # if linear_probing then restrict params passed to optimiser to head and potentially modality_tokens
        if hasattr(self.model, 'train_head_only'):
            if self.model.train_head_only:
                if hasattr(self.model, 'train_modality_tokens'):
                    if self.model.train_modality_tokens:
                        modality_tokens_names = ['OCT_token', 'IR_token']
                        params = [(p_name, p) for p_name, p in self.model.named_parameters() if 'head' in p_name
                                  or any([mt in p_name for mt in modality_tokens_names])]
                        printlog(f"Trainable params: {[p[0] for p in params]}")  # their names
                        params = [p[1] for p in params]  # their values

                else:
                    params = [(p_name, p) for p_name, p in self.model.named_parameters() if 'head' in p_name]
                    printlog(f"Trainable params: {[p[0] for p in params]}")  # their names
                    params = [p[1] for p in params]  # their values

        if hasattr(self.model, 'train_fpn_only'):  # seg only
            if self.model.train_fpn_only:
                # module = recursive_search_submodule(self.model, 'head')
                params = [(p_name, p) for p_name, p in self.model.named_parameters() if 'fpn' in p_name
                          or 'segmentation_head' in p_name ]
                printlog(f"Trainalbe params: {[p[0] for p in params]}")
                params = [p[1] for p in params]  # their values

        if 'optim' not in self.config['train']:
            printlog('defaulting to adam optimiser')
            self.config['train']['optim'] = 'Adam'
            self.optimiser = torch.optim.Adam(params, lr=self.config['train']['learning_rate'])
        else:
            if 'opt_keys' in self.config['train']:
                params = get_param_groups_using_keys(self.model, self.config)
            elif 'stage_wise_lr' in self.config['train']:
                params = get_param_groups_with_stage_wise_lr_decay(self.model, self.config)
            if 'no_weight_decay_names' in self.config['train']:
                params = get_param_groups_weight_decay(self.model, self.config)

            if self.config['train']['optim'] == 'SGD':
                wd = self.config['train']['weight_decay'] if 'weight_decay' in self.config['train'] else 0.0005
                momentum = self.config['train']['momentum'] if 'momentum' in self.config['train'] else 0.9
                self.optimiser = torch.optim.SGD(params,
                                                 lr=self.config['train']['learning_rate'],
                                                 momentum=momentum,
                                                 weight_decay=wd)
            elif self.config['train']['optim'] == 'Adam':
                self.optimiser = torch.optim.Adam(params, lr=self.config['train']['learning_rate'])

            elif self.config['train']['optim'] == 'AdamW':
                wd = self.config['train'].get('weight_decay', 0.01)
                betas = self.config['train'].get('betas', [0.9, 0.999])
                self.optimiser = torch.optim.AdamW(params,
                                                   lr=self.config['train']['learning_rate'],
                                                   betas=betas,
                                                   weight_decay=wd)
            else:
                raise ValueError(f"optimizer {self.config['train']['optim']} not recognized")

        if self.config['train']['lr_batchwise']:  # Replace given lr_restarts with numbers in batches instead of epochs
            use_grad_accumulation = self.grad_accumulation_steps > 1
            if self.using_multi_dataset_training:
                if use_grad_accumulation:
                    raise NotImplementedError('grad accumulation not implemented for multi dataset training')
                dummy_loader = self.data_loaders['train_loader'].get_loader(epoch=0)
                batches_per_epoch = [len(dummy_loader)]*self.config['train']['epochs']
            else:
                if use_grad_accumulation:
                    num_batches_per_epoch_mult = 1 / self.grad_accumulation_steps
                else:
                    num_batches_per_epoch_mult = 1

                batches_per_epoch = [int(len(self.data_loaders[self.train_schedule[e]]) * num_batches_per_epoch_mult)
                                     for e in range(self.config['train']['epochs'])]
            lr_total_steps = np.sum(batches_per_epoch)
        else:
            lr_restart_steps = self.config['train']['lr_restarts']
            lr_total_steps = self.config['train']['epochs']

        lr_function = LRFcts(self.config['train'], lr_total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lr_function)
        # if number then clip else set to high number
        self.grad_norm_clip = self.config['train'].get('grad_norm_clip', 10000000)

        printlog(f"*** \n"
                 f"lr_schedule: '{self.config['train']['lr_fct']}'\n"
                 f"  total steps {lr_total_steps}\n"
                 f"  grad_accumulation_step: {self.grad_accumulation_steps}\n"
                 f"  batchwise_schedule {self.config['train']['lr_batchwise']}\n"
                 f"  grad_norm_clip: {self.grad_norm_clip}\n"
                 f"***")
        # self.plot_learning_schedule(lr_total_steps, lr_function)

    def plot_learning_schedule(self, lr_total_steps, lr_function):
        import matplotlib.pyplot as plt
        lrs = []
        for i in range(lr_total_steps):
            lrs.append(lr_function(i))
        plt.plot(lrs)
        # plt.show()
        plt.savefig(os.path.join(self.log_dir, 'lr_funct.png'))
        plt.close()

    def get_grad_context(self, accumulated_iter):
        # if we are not using DDP, then we a call of backward is always
        if self.parallel:
            return nullcontext()
        # if we are accumulating gradients, we only sync at the last accumulation step
        sync_gradients = (accumulated_iter + 1) % self.grad_accumulation_steps == 0
        # if the model is DDP, but we are not at the last accumulation step then we use no_sync as context
        # otherwise we use the default context
        if (not sync_gradients) and self.parallel:
            grad_context = self.model.no_sync()
        else:
            grad_context = nullcontext()
        if self.debugging: printlog(f"grad_contex: {self.global_step}== {grad_context}")
        return grad_context

    def train_one_epoch(self):
        """Train the model for one epoch"""
        raise NotImplementedError

    def forward_train_step(self, img, lbl, reduce_batch=True, **kwrargs):
        """Forward pass for training"""
        raise NotImplementedError

    def forward_val_step(self, img, lbl, skip_loss=False, **kwrargs):
        """Forward pass for validation"""
        raise NotImplementedError

    def validate(self):
        """Validate the model on the validation data"""
        raise NotImplementedError

    def infer(self, **kwargs):
        """Inference on the test data"""
        raise NotImplementedError

    def submission_infer(self, **kwargs):
        """Inference on the external data"""
        raise NotImplementedError

    def post_process_output(self, **kwargs):
        """Validate the model on the validation data"""
        raise NotImplementedError

    def get_preds_labels_rgb(self, output, lbl):
        """ map network output and labels from mask --> rgb format for visualization
            - by default applies softmax across dim=1 and then argmax across dim=1 to get a prediction from the output
        :arg
        output: network output in shape (B, num_classes, H, W)
        lbl: label mask in shape (B, H, W)
        :returns debug_pred of shape (B, H, W, 3), debug_lbl of shape (B, H, W, 3)
        """
        assert self.config['task'] == 'segmentation', 'get_preds_labels_rgb only implemented for segmentation task'
        pred = torch.argmax(nn.Softmax2d()(output), dim=1)  # contains train_ids
        # # todo this is a debug only
        # pred = torch.cat([pred,pred],dim=0)

        int_to_rgb = get_remapped_colormap(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], self.dataset)
        mask_to_rgb = partial(mask_to_colormap_batched_torch,
                              colormap=int_to_rgb,
                              experiment=self.experiment,
                              dataset=self.dataset,
                              from_network=True)

        debug_pred = mask_to_rgb(pred)
        debug_lbl = mask_to_rgb(lbl)
        return debug_pred, debug_lbl

    def grad_cam(self):
        """ Grad-CAM
        """
        from pytorch_grad_cam import GradCAM, HiResCAM, EigenGradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from PIL import Image
        # from torchvision.models import resnet50
        from torchvision import transforms
        model = self.ema_model.average_model

        # Create an input tensor image for your model
        # Note: input_tensor can be a batch tensor with several images!
        # path_to_img = "C:\\Users\\thopis\\Documents\\datasets\\DR\\data\\images\\5b4c84a2-3f38-49bc-b04b-a26024c67f3a.png"
        # rgb_img = Image.open(path_to_img).convert('RGB').resize((416, 416))
        # input_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)

        def reshape_transform(tensor, height=26, width=26):
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                              height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        target_layers = [model.backbone.encoder.blocks[-1].mlp]
        # Construct the CAM object once, and then re-use it on many images:
        cam = HiResCAM(model=model.backbone, target_layers=target_layers, use_cuda=True,
                       reshape_transform=reshape_transform)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.
        for b_ind, batch in enumerate(self.data_loaders['valid_loader']):
            if len(batch) == 2:
                img, lbl = batch
            else:
                img, lbl, metadata = batch

            if lbl[0][6] == 1:
                rgb_img = transforms.ToPILImage()(img[0])
                # ind = 4
                ind = 6
                # c = 'Drusen'
                c = 'ERM'
                # for ind, c in enumerate(DATASETS_INFO[self.dataset].CLASS_NAMES[self.experiment]):
                targets = [ClassifierOutputTarget(ind)]

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=img, targets=targets)
                # grayscale_cam = grayscale_cam * (1.0 * (grayscale_cam > 0.98))
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(np.array(rgb_img)/255, grayscale_cam, image_weight=0.8, use_rgb=True)
                # bgr to rgb
                visualization = visualization[..., ::-1]
                cv2.imwrite(f'{self.log_dir}\\HGradCAM_{b_ind}_{c}.jpg', visualization)
