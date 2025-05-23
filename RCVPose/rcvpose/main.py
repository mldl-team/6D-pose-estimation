import torch
from utils import get_config, get_log_dir, str2bool
from data_loader import get_loader
from train import Trainer
import warnings
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

resume = ''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Parameters to set
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'test'],
        help="Whether to train or test the model"
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help="Which GPU to use (−1 for CPU)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='lm',
        choices=['lm', 'lmo', 'ycb'],
        help="Which dataset to run on: 'lm', 'lmo' or 'ycb'"
    )
    parser.add_argument(
        '--root_dataset',
        type=str,
        default='./datasets/LINEMOD',
        help="Path to the root of the dataset"
    )
    parser.add_argument(
        '--resume_train',
        type=str2bool,
        default=False,
        help="Whether to resume training from a checkpoint"
    )
    parser.add_argument(
        '--optim',
        type=str,
        default='Adam',
        choices=['Adam', 'SGD'],
        help="Optimizer to use"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Batch size for training/testing"
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default='ape',
        help="Class name (for LINEMOD/LINEMOD-O)"
    )
    parser.add_argument(
        '--initial_lr',
        type=float,
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        '--kpt_num',
        type=int,
        default=1,
        help="Number of keypoints"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='ckpts/',
        help="Directory where checkpoints are stored"
    )
    parser.add_argument(
        '--demo_mode',
        action='store_true',
        help="If set, will display demo visualizations"
    )
    parser.add_argument(
        '--test_occ',
        action='store_true',
        help="If set, will test with occlusions"
    )
    parser.add_argument(
    '--using_ckpts',
    action='store_true',
    default=False,
    help='(unused) placeholder to satisfy downstream code'
    )
    opts = parser.parse_args()

    # load config and attach to opts
    cfg = get_config()[1]
    opts.cfg = cfg

    # prepare logging / tensorboard
    if opts.mode == 'train':
        opts.out = get_log_dir(f"{opts.dataset}/{opts.class_name}Kp{opts.kpt_num}", cfg)
        print('Output logs: ', opts.out)
        vis = SummaryWriter(logdir=f"{opts.out}/tbLog/")
    else:
        vis = None

    # data loader
    data = get_loader(opts)

    # initialize trainer
    trainer = Trainer(data, opts, vis)

    # run
    if opts.mode == 'test':
        trainer.Test()
    else:
        trainer.Train()
