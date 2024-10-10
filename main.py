import os
import importlib
import wandb
import numpy as np
from typing import Dict, List
from utils import parse_config, str2bool
from utils.parsing.cmd_parser import get_cmd_parser, override_config_with_args
os.environ['NCCL_P2P_DISABLE'] = str(1)  # needed for a100 ddp
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
# uncomment for debugging when ddp has issues with unused params in forward
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
# import torch
# torch.autograd.set_detect_anomaly(True) # uncomment only for debugging this slows down. a lot.


def run_inference(config: Dict, manager_cls):
    """run inference on a single seed and return metrics
    """
    config['seed'] = config['seed'][0]
    assert config['parallel'] is False, 'parallel inference not supported'
    manager = manager_cls(config)
    metrics = manager.inference()
    manager.write_dict_json(metrics, 'metrics_inference.json')


def run_training(config: Dict, manager_cls):
    """Train with multiple seeds and return metrics averaged across seeds.
    If a single seed is provided in config['seeds'] then just train once and return metrics.

    Each call of manager.train() returns a Dict[seed, metrics] we call "metrics":
    metrics is a 4-level nested dict of the following structure:
    seed->chkpt_type->split->metric_name: metric_value

    seed:{"best_ema": {"test": {Dict[metric_name, metric_value]},
                       "val": {...}},
          "last_ema": {"test": {...},
                       "val": {...}},
          "best": {"test": {...},
                   "val": {...}},
          "last": {"test": {...},
                   "val": {...}}
         }
    return mean metrics across seeds for each combination of split (e.g. test/val) and chkpt_type (e.g. best/last)

    """
    # extracting some necessary information
    seeds = config['seed']  # list of seeds
    assert type(seeds) == list, f'seed must be a list instead got {seeds} of type : {type(seeds)}'
    use_ema = config['train'].get('ema', False)  # if True report metrics w & w/o ema
    metrics = {str(seed): None for seed in seeds}  # store return metrics for each seed
    print(f"Training with seeds: {config['seed']}")
    if config['parallel']:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')  # done here because doing it for every process gives error
    for seed in seeds:
        config_s = config.copy()  # reset config
        config_s['seed'] = seed
        # Instantiating a manager includes: loading the model, the dataloaders, the optimizer & scheduler, logging
        manager = manager_cls(config_s)

        #### START OF TRAINING ####
        metrics[str(seed)] = manager.ddp_train() if config['parallel'] else manager.train()
        #### END OF TRAINING ####

        wandb.finish() if config_s['logging']['wandb'] else None
    if config['task'] not in ['pretraining']:
        metrics = average_metrics(metrics,
                                  manager.metrics_per_dataset[manager.dataset],
                                  manager.dataset,
                                  seeds,
                                  use_ema)
        manager.write_dict_json(metrics, f'average_metrics_across_{len(seeds)}_seeds.json')
    print(f'Average metrics for seeds: {seeds}')
    print(metrics)


def average_metrics(metrics: Dict,
                    metric_names: List[str],
                    dataset: str,
                    seeds: List[int],
                    use_ema: bool) -> Dict:

    chkpt_types = ['best', 'last', 'best_ema', 'last_ema'] if use_ema else ['best', 'last']
    splits = ['val'] if dataset in ['RETOUCH', 'OCT5K', 'AROI'] else ['test']
    for chkpt_type in chkpt_types:
        for split in splits:
            for metric_name in metric_names:
                mean = np.mean([metrics[str(seed)][chkpt_type][split][metric_name] for seed in seeds])
                std = np.std([metrics[str(seed)][chkpt_type][split][metric_name] for seed in seeds])
                # round to 4 decimal places
                metrics[f'{split}_avg_' + metric_name + '_' + chkpt_type] = round(mean, 4)
                metrics[f'{split}_std_' + metric_name + '_' + chkpt_type] = round(std, 4)
    return metrics


if __name__ == '__main__':
    # parse config, cmd args and determine manager class
    parser = get_cmd_parser()
    args = parser.parse_args()
    config_ = parse_config(args.config, args.user, args.device, args.dataset, args.parallel)
    override_config_with_args(args, config_)
    manager_module_name = 'managers.' + config_['manager'] + '_Manager'
    manager_name = config_['manager'] + 'Manager'
    module = importlib.import_module(manager_module_name)
    manager_class = getattr(module, manager_name)

    if config_['mode'] == 'training':
        run_training(config_, manager_class)
    elif config_['mode'] == 'inference':
        run_inference(config_, manager_class)

