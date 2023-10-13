from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import util
from dataset.dataset import CustomNYUv2Dataset
from utils.util import Normalise, RandomCrop, ToTensor, RandomMirror
import numpy as np

class CustomNYUv2DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, masks=None, shuffle=True, validation_split=0.0, num_workers=1, training=True, url=None):
        # Data
        self.data_dir = data_dir
        self.masks = masks

        # Augmentation parameters and functions.
        self._img_scale = 1. / 255
        self._depth_scale = 5000.
        self._crop_size = 400
        self._mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self._std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

        self._normalise_params = [self._img_scale,  self._mean, self._std, self._depth_scale,]

        trsfm_train = transforms.Compose([RandomMirror(), 
                                          RandomCrop(self._crop_size), 
                                          Normalise(*self._normalise_params), 
                                          ToTensor()])

        trsfm_val = transforms.Compose([Normalise(*self._normalise_params),
                                        ToTensor()])

        self.train_dataset = CustomNYUv2Dataset(self.data_dir, masks=self.masks, 
                                                mode='train', trsfm=trsfm_train)
        self.valid_dataset = CustomNYUv2Dataset(self.data_dir, masks=self.masks, 
                                                mode='test', trsfm=trsfm_val)

        super().__init__(self.train_dataset, self.valid_dataset, batch_size, 
                                        shuffle, validation_split, num_workers)
