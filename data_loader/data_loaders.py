from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset.dataset import CustomNYUv2Dataset
from utils.util import Normalise, RandomCrop, ToTensor, RandomMirror, get_transforms
import numpy as np

class CustomNYUv2DataLoader(BaseDataLoader):
    """
    NYUv2 data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, masks=None, shuffle=True, 
                        validation_split=0.0, num_workers=1, training=True, mode='train'):
        # Data
        self.data_dir = data_dir
        self.masks = masks
        trsfm = get_transforms(mode=mode)
        self.dataset = CustomNYUv2Dataset(self.data_dir, masks=self.masks,
                                                                 mode=mode, trsfm=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
