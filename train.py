import argparse
from parse_config import ConfigParser 
import numpy as np
from utils import util
import dataset.downloader as download
import torch

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser):
    """main code"""
    pass

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Multi-task learner for' +
                        ' the joint learning of depth and segmentation.')
    args.add_argument('-c', '--config', default=None, type=str, 
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, 
                        help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    
    downloader = config.init_obj('data_downloader', download)



