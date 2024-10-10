import pathlib
import json
import torch
from typing import List
from torch.distributed import destroy_process_group
import datetime
import numpy as np
from utils import DATASETS_INFO,  get_log_name, get_rank, printlog
import os
from utils import Logger as Log
from collections import OrderedDict


def is_metric_higher_is_better(metric: str):
    """determine whether a model improves as it gets a higher value"""
    if metric in ['SMAPE', "MAE", "val_loss"]:
        return False
    else:
        return True


class LoggingManager:
    model_selection_metrics = {"OCTIR": {"pretraining": "val_loss"},
                               "OctBiom": {"detection": "ROC AUC"},
                               "AROI": {"segmentation": "miou"},
                               "OCT5K": {"segmentation": "miou"},
                               "RAVIR": {"segmentation": "miou"},
                               "RETOUCH": {"segmentation": "miou"},
                               "DR": {"classification": "ROC AUC"},
                               "OCTID": {"classification" "ROC AUC"},
                               "OCTDL": {"classification": "ROC AUC"},
                               "OLIVES": {"detection": "MF1",
                                          "regression": "SMAPE"}
                               }

    metrics_per_dataset_per_task = {

        "RETOUCH": {"segmentation": ["miou_Fluids", "miou_IRF", "miou_SRF", "miou_PED"]},
        "AROI": {"segmentation": ["miou_Layers", "miou_Fluid", "miou_IRF", "miou_SRF", "miou_PED"]},
        "OCT5K": {"segmentation": ["Layers", "vitreous", 'ILM', 'OPL', 'IS_OS', 'IB_RPE', 'OBRPE']},
        "RAVIR": {"segmentation": ["miou"]},

        "OCTID": {"classification": ["ROC AUC", "MF1", "mF1", "MP", "mP", "accuracy"]},
        "OCTDL": {"classification": ["ROC AUC", "MF1", "mF1", "MP", "mP", "accuracy"]},
        "DR": {"classification": ["ROC AUC", "AP", "accuracy"]},

        "OctBiom": {"detection": ["ROC AUC", "MF1", "mF1", "MAP", "mAP"]},
        "OLIVES": {"detection": ["ROC AUC", "MF1", "mF1", "MAP", "mAP"],
                   "regression": ["SMAPE", "R2", "MAE"]},

        "OCTIR": ["val_loss"]
    }

    def __init__(self, configuration):
        self.rank = 0
        self.config = configuration
        self.parallel = self.config['parallel']
        self.debugging = self.config['debugging']
        self.start_epoch = 0
        self.epoch = 0
        self.best_loss = 1e10
        self.best_roc_auc = 0
        self.global_step = 0
        self.valid_freq = self.config["logging"].get('valid_freq', 1)
        self.dataset = self.config['data']['dataset']
        self.config['graph'].update({'dataset': self.dataset})
        self.experiment = self.config['data']['experiment']

        self.metrics = {'best_val_loss': 1e10,
                        'final_val_loss': 1e10,
                        'final_epoch_step': [0, 0],  # [latest training epoch, latest step]
                        'best_loss_epoch_step': [0, 0],
                        }

        self.model_selection_metric = self.model_selection_metrics[self.dataset][self.config['task']]
        self.model_selection_metric_higher_is_better = is_metric_higher_is_better(self.model_selection_metric)
        self.metrics_per_dataset = {self.dataset: self.metrics_per_dataset_per_task[self.dataset][self.config['task']]}

        self.metrics.update({f'best_{self.model_selection_metric}': 0.0,
                             f'best_{self.model_selection_metric}_epoch_step': [0, 0]})

        if self.config['task'] == 'segmentation':
            self.metrics.update({'final_miou': 0, 'final_miou_step': 0})
            for category in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][-1]:
                self.metrics.update({f'best_miou_{category}': 0.0})  # iou of category @best_miou

        assert f'best_{self.model_selection_metric}' in self.metrics.keys(), \
            f"Selection metric [{self.model_selection_metric}] for [{self.dataset}] not in metrics {self.metrics.keys()}"

        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0])
        self.model = None
        self.tta_model = None
        self.data_loaders = {}
        self.loss = None
        self.batch_size = self.config['data']['batch_size']
        self.grad_accumulation_steps = self.config['train'].get('grad_accumulation_step', 1)
        self.grad_norm_clip = self.config['train'].get('grad_norm_clip', 0)
        self.scaler = None

        self.batches_per_epoch = -1
        self.valid_batch_size = self.config['data'].get('valid_batch_size', 1)
        self.un_norm = None  # placeholder for a function that undoes torchvision_normalize (or other normalization)
        self.checkpoint_name = 'unspecified'
        self.using_multi_dataset_training = False  # flag for whether multiple datasets/modalities are used for training

        # ema
        # this is a switch to turn on/off EMA for inference.
        # If true during training ema_model is computed
        self.use_ema = False

        # this where the ema of self.model with be stored
        self.ema_model = None

        if self.config['mode'] in ['inference', 'grad_cam']:
            printlog(f"Mode {self.config['mode']}: setting batch size to {self.valid_batch_size}")
            self.valid_batch_size = self.batch_size

        if 'loss' in self.config:
            self.config['loss'].update({'dataset': self.dataset})
            self.config['loss'].update({'experiment': self.experiment})
        self.optimiser = None
        self.scheduler = None
        self.train_schedule = {}
        self.save_dir_path = None  # path to where pseudo labelled data are saved
        self.save_outputs = False
        self.use_wandb = self.config['logging'].get('wandb', False)
        printlog(f"Using Wandb to log {self.use_wandb}")
        for i in range(self.config['train']['epochs']):
            self.train_schedule.update({i: 'train_loader'})  # pre-fill

        # Print debugging state in Console
        if self.debugging:
            print("\n\n* * * * * DEBUGGING ACTIVE * * * * * \n\n")
            print(f"** changing num_workers to 0 from {self.config['data']['num_workers']}")
            self.config['data']['num_workers'] = 0

        # Identify run
        self.date = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
        if 'load_checkpoint' in self.config and self.config['mode'] is not 'training':
            self.run_id = self.config['load_checkpoint']
        else:
            self.run_id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
            if 'name' in self.config:
                if self.debugging:
                    self.run_id = 'debug_' + self.run_id
                self.run_id = '__'.join((self.run_id, get_log_name(self.config)))

        self.log_dir = pathlib.Path(self.config['log_path']) / self.run_id
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)

        # very long filenames not supported on windows ... thanks Bill
        mode_tag = f"_{self.config['mode']}"
        if self.config['mode'] == 'inference':
            log_filename = f"{self.date}" + mode_tag
        else:
            log_filename = f"{self.run_id}{mode_tag}"
        self.log_file = str(self.log_dir / log_filename)
        if self.config['mode'] == 'submission_inference':
            self.log_file = self.log_dir / 'log_submission'
        printlog(f'loggging in {self.log_file}')
        Log.init(logfile_level="info",
                 stdout_level=None,
                 log_file=self.log_file,
                 rewrite=True)
        self.run_final_val = self.config.get('run_final_val', False)
        printlog(f'going to run tta val after training {self.run_final_val}')
        printlog("Run ID: {} {}".format(self.run_id, self.config['mode']))
        if self.config['mode'] in ['inference', 'training', 'submission_inference']:
            self.save_outputs = self.config['save_outputs'] if 'save_outputs' in self.config else False
            printlog(f'going to save inference outputs ') if self.save_outputs else None

        # Set cuda flag
        if torch.cuda.is_available() and not self.config['cuda']:
            printlog("CUDA device available, but not used")
        if self.config['cuda'] and not torch.cuda.is_available():
            printlog("CUDA device required, but not available - using CPU instead")
        self.cuda = torch.cuda.is_available() & self.config['cuda']

        # cuda device ids identification
        local_devices_count = torch.cuda.device_count()
        printlog(f'available_devices {local_devices_count}')
        if not self.parallel:
            assert len(self.config['gpu_device']) == 1
            self.config['gpu_device'] = self.config['gpu_device'][0]
        elif len(self.config['gpu_device']) == 1:
            printlog(f'ddp requested but only 1 gpu device requested {self.config["gpu_device"]}')
            local_devices_count = torch.cuda.device_count()
            printlog(f'setting it to all available cuda devices {local_devices_count}')
            self.config['gpu_device'] = [i for i in range(local_devices_count)]

        if self.cuda:
            if self.parallel:
                import random
                import time
                random.seed(time.process_time())
                assert(local_devices_count > 1), 'parallel was set to True but devices are {}<2'.format(local_devices_count)
                printlog(f'available_devices {local_devices_count}')
                self.device = torch.device('cuda')
                self.rank = 0  # init
                self.n_gpus = len(self.config['gpu_device']) if isinstance(self.config['gpu_device'], list) else local_devices_count
                self.allocated_devices = [0, 1, 2, 3, 4, 5, 6, 7] if isinstance(self.config['gpu_device'], list) else self.config['gpu_device']
                self.world_size = self.n_gpus  # one machine only
                printlog("Program will run on *****Multi-GPU-CUDA, devices {}*****".format(self.n_gpus))
                """ Initialize the distributed environment. """
                port = int(f'29{random.randint(0, 9)}{random.randint(0, 9)}{random.randint(0, 9)}')
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = str(port)
                printlog(f"DDP: MASTER_ADDR: {os.environ['MASTER_ADDR']} MASTER_PORT: {os.environ['MASTER_PORT']}")
            else:
                self.n_gpus = 1
                self.device = torch.device('cuda')
                torch.cuda.set_device(self.config['gpu_device'])
                printlog("Program will run on *****GPU-CUDA, device {}*****".format(self.config['gpu_device']))
        else:
            self.n_gpus = 0
            self.device = torch.device('cpu')
            printlog("Program will run on *****CPU*****")

        self.empty_cache = self.config['empty_cache'] if 'empty_cache' in self.config else False
        printlog(f'Empty cache after model.forward(): {self.empty_cache}')

    def train_logging(self, **kwargs):
        pass

    def valid_logging(self, **kwargs):
        pass

    @staticmethod
    def remove_module_prefix(state_dict):
        ret_state_dict = OrderedDict()
        for key in state_dict:
            if 'module' in key:
                ret_state_dict['.'.join(key.split('.')[1:])] = state_dict[key]
            else:
                ret_state_dict[key] = state_dict[key]
        return ret_state_dict

    def get_wandb_tag(self) -> List[str]:
        """ Generate a list of wandb tags according to config
        :return: tags [list]
        """
        modality = self.config['data']['modality']
        tag = []
        graph = self.config.get('graph', None)
        phase = graph.get('phase', 'scratch')  # can be 'pretraining', 'finetuning', 'scratch', 'linear_probing'
        tag.append(phase)
        # add a tag for each modality
        if type(modality) == str:
            tag.append(modality)
        else:
            tag.extend(modality)
        tag.append(self.config['data']['dataset'])  # add dataset as tag
        # add a tag for certain backbone settings
        backbone_settings = graph.get('backbone_settings', None)
        if backbone_settings:
            if backbone_settings.get('use_modality_token', False):
                tag.append('modality_token')
        return tag

    def save_checkpoint(self, save_as='specific'):
        """
        Saves a checkpoint in given self.log_dir
        :param save_as: type of checkpoint to save options are the following:
        'best': save the best checkpoint based on the model selection metric
        'latest': save the latest checkpoint after each epoch and remove the one from the previous epoch
        'specific': save a checkpoint for a specific epoch/step and keep it forever. Its name is formatted as:
        'chkpt_epoch_{:04d}_step_{:07d}.pt'.format(state['epoch'], state['global_step'])

        """

        assert save_as in ['best', 'latest', 'specific'], f'invalid save_as argument {save_as} ' \
                                                          f'choose from [best, latest, specific]'

        base_path = self.log_dir / 'chkpts'
        if not base_path.is_dir():
            base_path.mkdir()

        state = {
            'global_step': self.global_step-1,  # -1 as train_logging increments by +1 and then valid_logging is called
            'epoch': self.start_epoch + self.epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics
        }

        if self.epoch + self.start_epoch not in [0, self.config['train']['epochs'] - 1]:
            state.update({'optimiser_state_dict': self.optimiser.state_dict()})
            if self.scaler is not None:
                state.update({'scaler_state_dict': self.scaler.state_dict()})
            if self.scheduler is not None:
                state.update({'scheduler_state_dict': self.scheduler.state_dict()})

        if self.use_ema and self.ema_model is not None:
            state.update({'model_ema_state_dict': self.ema_model.average_model.state_dict()})

        if save_as == 'best':
            # for when we save the best checkpoint
            name = 'chkpt_best.pt'

        elif save_as == 'latest':
            # we save a checkpoint after each epoch and remove the one from the previous epoch
            name_prev_prefix = 'chkpt_epoch_{:04d}'.format(state['epoch'] - 1)
            for f in base_path.iterdir():
                if name_prev_prefix in f.name:
                    name_prev = f.name
                    # if previously saved checkpoint's epoch is at multiple of config['logging']['checkpoint_epoch']
                    # then do not remove it
                    if not ((self.epoch + self.start_epoch - 1) % self.config['logging']['checkpoint_epoch'] == 0):
                        os.remove(base_path / name_prev)
                        printlog(f"previous latest checkpoint removed: {name_prev}")
                    else:
                        printlog(f"previous latest checkpoint NOT removed: {name_prev}")
            name = 'chkpt_epoch_{:04d}_step_{:07d}.pt'.format(state['epoch'], state['global_step'])
        elif save_as == 'specific':
            # for when we save a checkpoint for a specific epoch/step and keep it forever
            name = 'chkpt_epoch_{:04d}_step_{:07d}.pt'.format(state['epoch'], state['global_step'])
        else:
            raise ValueError(f'Invalid save_as argument {save_as} choose from [best, latest, specific]')

        torch.save(state, base_path / name)
        printlog(f"new {save_as} Checkpoint saved: {name}")

    def load_checkpoint(self, chkpt_type):
        """Load a model and model state from a checkpoint
        :param chkpt_type: 'best' or 'last'
        :return:
        """
        checkpoint_list = [f.name for f in (self.log_dir / 'chkpts').iterdir()]
        checkpoint_list.sort()
        name = 'chkpt_best.pt'
        if chkpt_type == 'best':
            # n = self.log_dir / 'chkpts' / 'chkpt_best.pt'
            if 'chkpt_best.pt' in checkpoint_list:
                printlog(f"Found checkpoint 'best' in checkpoints_list {checkpoint_list}")
            elif 'chkpt_epoch_' in checkpoint_list[-1]:
                printlog("No chkpt of type 'best': found checkpoint of type 'last' ('chkpt_epoch_*') instead")
                name = checkpoint_list[-1]
            else:
                raise ValueError(f'Neither chkpt of type "best" nor of type "last" was found'
                                 f' in chekpoints_list {checkpoint_list}')
        elif chkpt_type == 'last':
            if 'chkpt_epoch_' in checkpoint_list[-1]:
                name = checkpoint_list[-1]
            else:
                raise ValueError("No checkpoint of type 'last' found.")

        path = self.log_dir / 'chkpts' / name
        self.checkpoint_name = name.split('.pt')[0]
        # print(torch.cuda.current_device())
        # this is required if checkpoint trained on one device and now is loaded on a different device+
        # https://github.com/pytorch/pytorch/issues/15541
        if self.parallel:
            if self.n_gpus <= 4:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            else:
                map_location = 'cuda:%d' % self.rank

        else:
            map_location = 'cuda:{}'.format(self.config['gpu_device'])
        printlog(f' loading checkpoint with map_location: {map_location}')

        checkpoint = torch.load(str(path), map_location)

        if not self.parallel:
            printlog(f'removing module prefix from checkpoint (rank={self.rank})')
            checkpoint['model_state_dict'] = self.remove_module_prefix(checkpoint['model_state_dict'])

        ret = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        printlog(f' model -- loading_state_dict \n :{ret}')
        if self.use_ema:
            ret_ema = self.ema_model.average_model.load_state_dict(checkpoint['model_ema_state_dict'], strict=False)
            printlog(f'ema_model.average_model -- loading_state_dict \n :{ret_ema}')

        if self.config['mode'] == 'training':
            printlog(f'loading optimizer_state_dict')
            printlog(f'loading scheduler_state_dict')
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.metrics = checkpoint.get('metrics', self.metrics)
            if 'best_val_loss' in self.metrics:
                self.best_loss = self.metrics['best_val_loss']

            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.scheduler.lr_lambdas[0].start_step = self.global_step  # set learning rate scheduler to current step
            if 'scaler' in checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            printlog(f'start_epoch :{self.start_epoch} global_step: {self.global_step}')
        elif self.config['mode'] == 'inference' and chkpt_type=='best':
            printlog(f'loading inference_state_dict')
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.metrics = checkpoint.get('metrics', self.metrics)
            self.metrics.update({'epoch': self.epoch, 'global_step': self.global_step})
            printlog(f'Best checkpoint from epoch :{self.epoch} global_step: {self.global_step}')
        printlog(f"rank : {get_rank()} Checkpoint loaded: {path} type: {chkpt_type}")

    def finalise(self):
        """Saves info, resets main variables"""
        config_text = self.write_info_json()
        if self.parallel:
            destroy_process_group()

    def write_info_json(self, tag=''):
        config = self.config.copy()
        config['run_id'] = self.run_id
        config['best_loss'] = self.best_loss
        metrics = self.metrics.copy()
        for k in metrics.keys():
            if isinstance(self.metrics[k], np.ndarray) or isinstance(metrics[k], torch.Tensor):
                # noinspection PyUnresolvedReferences
                metrics[k] = metrics[k].tolist()
        config['metrics'] = metrics
        # Save config to json
        config_text = json.dumps(config, indent=4, sort_keys=True, default=self.default)
        with open(self.log_dir / f'info{tag}.json', 'w') as json_file:
            json_file.write(config_text)
        return config_text

    def write_dict_json(self, config: dict, filename='inference'):
        """write a json to log dir"""
        config['run_id'] = self.run_id
        for k in config:
            if isinstance(config[k], np.ndarray) or isinstance(config[k], torch.Tensor):
                # noinspection PyUnresolvedReferences
                config[k] = config[k].tolist()
        # Save config to json
        config_text = json.dumps(config, indent=4, sort_keys=True, default=self.default)
        with open(self.log_dir / f'{filename}.json', 'w') as json_file:
            json_file.write(config_text)
        return config_text

    @staticmethod
    def default(obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, pathlib.WindowsPath) or isinstance(obj, pathlib.PosixPath):
            return str(obj)
        raise TypeError(f'Not serializable {obj}')

    def get_latex_line(self, metrics_test, metrics_val=None):
        """ utility function to output copy-paste-able latex line for tables """
        metrics_names = self.metrics_per_dataset[self.dataset]
        printlog(f"getting latex line for {self.dataset} for metrics {metrics_names}")
        print(metrics_test)
        # first two columns empty for model and init
        latex_line = '&  '
        if self.dataset == 'OctBiom':
            assert metrics_val is not None
            for metric in metrics_names:
                latex_line += ' & {:.1f} / {:.1f}'.format(metrics_test[metric]*100, metrics_val[metric]*100)
        elif self.dataset in ['OCTDL', 'OCTID']:
            for metric in metrics_names:
                latex_line += ' & {:.1f}'.format(metrics_test[metric]*100)
        elif self.dataset in ['RETOUCH']:
            for metric in metrics_names:
                # latex_line += ' & {:.1f} ({:.1f})'
                latex_line += ' & {:.1f}'.format(metrics_test['categories'][metric.split('miou')[-1][1:]]*100)
        elif self.dataset in ['AROI', 'OCT5K']:
            for metric in metrics_names:
                latex_line += ' & {:.1f}'.format(metrics_test['categories'][metric]*100)
        elif self.dataset in ['DR']:
            for metric in metrics_names:
                latex_line += ' & {:.1f}'.format(metrics_test[metric] * 100)
        elif self.dataset in ['OLIVES']:
            for metric in metrics_names:
                latex_line += ' & {:.1f}'.format(metrics_test[metric] * 100)

        latex_line += '\\'
        return latex_line
