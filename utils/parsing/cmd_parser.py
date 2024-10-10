import argparse
from typing import Dict
from utils import str2bool


def get_cmd_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default='configs/FCN_train_config.json',
                        help='Set path to configuration files, e.g. '
                             'python main.py --config configs/FCN_train_config.json.')

    parser.add_argument('-u', '--user', type=str, default='c',
                        help='Select user to set correct data / logging paths for your system, e.g. '
                             'python main.py --user theo')

    parser.add_argument('-d', '--device', nargs="+", type=int, default=-1,
                        help='Select GPU device to run the experiment one.g. --device 3')

    parser.add_argument('-s', '--dataset', type=str, default=-1, required=False,
                        help='Select dataset to run the experiment one.g. --device 3')

    parser.add_argument('-p', '--parallel', action='store_true',
                        help='whether to use distributed training')

    parser.add_argument('-debug', '--debugging', action='store_true',
                        help='sets manager into debugging mode e.x --> cts is run with val/val split')

    parser.add_argument('-cdnb', '--cudnn_benchmark', type=str, default=None, required=False,
                        help='if added in args then uses cudnn benchmark set to True '
                             'else uses config '
                             'else sets it to True by default')

    parser.add_argument('-cdne', '--cudnn_enabled', type=str, default=None, required=False,
                        help='if added in args then uses cudnn enabled set to True '
                             'else uses config '
                             'else sets it to True by default')

    parser.add_argument('-vf', '--valid_freq', type=int, default=None, required=False,
                        help='sets how often to run validation')

    parser.add_argument('-w', '--workers', type=int, default=None, required=False,
                        help='workers for dataloader per gpu process')

    parser.add_argument('-ec', '--empty_cache', action='store_true',
                        help='whether to empty cache (per gpu process) after each forward step to avoid OOM --'
                             ' this is useful in DCV2_ms or DCV3/ms')

    parser.add_argument('-m', '--mode', type=str, default=None, required=False,
                        help='mode setting e.x training, inference (see BaseManager for others)')

    parser.add_argument('-bs', '--batch_size', type=int, default=None, required=False,
                        help='batch size -- the number given is then divided by n_gpus if ddp')

    parser.add_argument('-ga', '--grad_accumulation', type=int, default=None, required=False,
                        help='grad accumulation steps -- batch_size per gpu = batch_size * grad_accumulation_steps')

    parser.add_argument('-ep', '--epochs', type=int, default=None, required=False,
                        help='training epochs -- overrides config')

    parser.add_argument('-so', '--save_outputs', action='store_true',
                        help='whether to save outputs for submission cts')

    parser.add_argument('-rfv', '--run_final_val', action='store_true',
                        help='whether to run validation with special settings'
                             ' at the end of training (ex using tta or sliding window inference)')

    parser.add_argument('-tta', '--tta', action='store_true',
                        help='whether to tta_val at the end of training')

    parser.add_argument('-tsnes', '--tsne_scale', type=int, default=None, required=False,
                        help=' stride of feats on which to apply tsne must be [4,8,16,32]')

    parser.add_argument('-lr', '--learning_rate', type=float, default=None, required=False,
                        help=' peak learning rate')

    parser.add_argument('-wd', '--weight_decay', type=float, default=None, required=False,
                        help=' weight decay')

    parser.add_argument('-seed', '--seed', type=int, nargs="+", default=None, required=False,
                        help=' single random seed or list of random seeds')

    # loss args for convenience
    parser.add_argument('--loss', '-l', choices=[None, 'ce', 'ms', 'ms_cs'], default=None, required=False,
                        help=f'choose loss overriding config (refer to config for other options except {"[ce, ms, ms_cs]"}')

    parser.add_argument('--wandb', '-wb', action='store_true')

    return parser


def override_config_with_args(args, config: Dict):
    print(f'\n*** overriding config with args {args} ***')

    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
        print(f'overriding lr to {args.learning_rate}')

    if args.weight_decay:
        config['train']['weight_decay'] = args.weight_decay
        print(f'overriding wd to {args.weight_decay}')

    if args.seed:
        config['seed'] = args.seed

    if args.save_outputs:
        config['save_outputs'] = True
    if args.run_final_val:
        config['run_final_val'] = True
        print('going to run tta val at the end of training')
    if args.empty_cache:
        config['empty_cache'] = True
        print('emptying cache')
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
        print(f'bsize {args.batch_size}')
    if args.grad_accumulation is not None:
        config['train']['grad_accumulation'] = args.grad_accumulation
        print(f'grad_accumulation {args.grad_accumulation}')

    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
        print(f'epochs : {args.epochs}')
    if args.tta:
        config['tta'] = True
        print(f'tta set to {config["tta"]}')
    if args.debugging:
        config['debugging'] = True
    if args.valid_freq is not None:
        config['logging']['valid_freq'] = args.valid_freq
    if args.workers is not None:
        config['data']['num_workers'] = args.workers
        print(f'workers {args.workers}')
    if args.mode is not None:
        config['mode'] = args.mode
        print(f'mode {args.mode}')
    if args.cudnn_benchmark is not None:
        config['cudnn_benchmark'] = str2bool(args.cudnn_benchmark)
    if args.cudnn_enabled is not None:
        config['cudnn_enabled'] = str2bool(args.cudnn_enabled)
    if args.wandb:
        config['logging']['wandb'] = True

    # # switch it off if debug
    if config['debugging']:
        config['logging']['wandb'] = False

    # config['logging']['wandb'] = False
    if isinstance(config['seed'], int):
        config['seed'] = [config['seed']]

    print(f'requested device ids:  {config["gpu_device"]}')

    # override config
    config['parallel'] = args.parallel
    print("*** finished overriding config with args ***\n")
