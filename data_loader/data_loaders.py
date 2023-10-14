from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import util
from dataset.dataset import CustomNYUv2Dataset
from utils.util import Normalise, RandomCrop, ToTensor, RandomMirror
import numpy as np

def get_transforms(mode='traini', mg_scale=None, depth_scale=None, crop_size=None, mean=None, std=None):
    """ Augmentation parameters and functions. """
    img_scale = 1. / 255
    depth_scale = 5000.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]     

    mean = np.array(mean).reshape((1, 1, 3))
    std = np.array(std).reshape((1, 1, 3))

    normalise_params = [img_scale, mean, std, depth_scale,]

    if mode == 'train':
        crop_size = 400
        trsfm = transforms.Compose([RandomMirror(), 
                                    RandomCrop(crop_size), 
                                    Normalise(*normalise_params), 
                                    ToTensor()])
    elif mode == 'test':
        trsfm = transforms.Compose([Normalise(*normalise_params),
                                    ToTensor()])
    else:
        print('[INFO] mode given is not supported!')
        return None

    return trsfm


class CustomNYUv2DataLoader(BaseDataLoader):
    """
    NYUv2 data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, masks=None, shuffle=True, validation_split=0.0, num_workers=1, training=True, mode='train'):
        # Data
        self.data_dir = data_dir
        self.masks = masks

        print(f'mode = {mode}')
        trsfm = get_transforms(mode=mode)

        self.dataset = CustomNYUv2Dataset(self.data_dir, masks=self.masks, mode=mode, trsfm=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
