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
from trainer.trainer import Trainer 


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
    data_loader = config.init_obj('data_loader_train', module_data)
    valid_data_loader = config.init_obj('data_loader_test', module_data)

    print(f'len(data_loader.dataset) = {len(data_loader.dataset)}')
    print(f'len(valid_data_loader.dataset) = {len(valid_data_loader.dataset)}')

    # build model architecture, then print to console
    model, encoder, decoder = config.init_obj('architecture', module_architecture)
    logger.info(model)

    architecture = {'model': model, 'encoder': encoder, 'decoder': decoder}

    # prepare for (multi-device) GPU training
    device, device_ids = util.prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    masks = config['masks']

    criterion = {f'{msk}': getattr(model_loss, config[f'loss_{msk}']['type'])
                    (ignore_index=config[f'loss_{msk}']['ignore']).cuda() for msk in masks}
    criterions = list(criterion.values())

    metrics = [config.init_obj(met, module_metric) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = {coder: filter(lambda p: p.requires_grad, architecture[coder].parameters()) 
                                        for coder in ['encoder', 'decoder']}

    optimizer = {f'optimizer_{coder}': config.init_obj(f'optimizer_{coder}', torch.optim, trainable_params[coder])
                                        for coder in ['encoder', 'decoder']}
    optimizers = list(optimizer.values())

    # Setting up the lr_scheduler  
    milestones = np.arange(1, config['trainer']['epochs'], config['lr_scheduler']['step_size'])
    gamma = config['lr_scheduler']['gamma']

    lr_scheduler = {f'lr_scheduler_{coder}': getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])
            (optimizer[f'optimizer_{coder}'], milestones=milestones, gamma=gamma) for coder in ['encoder', 'decoder']}
    lr_schedulers = list(lr_scheduler.values())

    # Starting the training loop 

    trainer = Trainer(model, 
                      criterions, 
                      metrics, 
                      optimizers,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_schedulers)

    trainer.train()



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
