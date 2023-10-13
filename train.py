import argparse
from parse_config import ConfigParser 
import numpy as np
from utils import util
import dataset.downloader as download
import data_loader.data_loaders as module_data
import torch
import model.model as module_architecture
import model.loss as model_loss
import model.metric as module_metric

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser):
    """main code"""
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model, encoder, decoder = config.init_obj('architecture', module_architecture)
    logger.info(model)

    architecture = {'model': model, 'encoder': encoder, 'decoder': decoder}

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # prepare for (multi-device) GPU training
    device, device_ids = util.prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    print(f'device = {device}')

    masks = config['data_loader']['args']['masks']
    
    criterion = {f'{msk}': getattr(model_loss, config[f'loss_{msk}']['type'])
                    (ignore_index=config[f'loss_{msk}']['ignore']).cuda() for msk in masks} 

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = {coder: filter(lambda p: p.requires_grad, architecture[coder].parameters()) 
                                        for coder in ['encoder', 'decoder']}

    optimizer = {f'optimizer_{coder}': config.init_obj(f'optimizer_{coder}', torch.optim, trainable_params[coder])
                                        for coder in ['encoder', 'decoder']}


    milestones = np.arange(1, config['trainer']['epochs'], config['lr_scheduler']['step_size'])
    gamma = config['lr_scheduler']['gamma']

    lr_scheduler = {f'lr_scheduler_{coder}': getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])
            (optimizer[f'optimizer_{coder}'], milestones=milestones, gamma=gamma) for coder in ['encoder', 'decoder']}


    print(lr_scheduler)


    # trainer = Trainer(model, criterion, metrics, optimizer,
    #                   config=config,
    #                   device=device,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    # trainer.train()





if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Multi-task learner for' +
                        ' the joint learning of depth and segmentation.')
    args.add_argument('-c', '--config', default=None, type=str, 
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, 
                        help='indices of GPUs to enable (default: all)')
    args.add_argument('--download', default=False, type=bool, 
                        help='bool parameter if the data is to be downloaded or not (default: False)')

    config = ConfigParser.from_args(args)

    main(config=config)
