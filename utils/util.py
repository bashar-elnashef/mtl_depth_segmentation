import os
import zipfile
import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torchvision import transforms
import torch 
import torch.nn as nn
import gdown
import numpy as np
import cv2

KEYS_TO_DTYPES = {
    "seg40": torch.long,
    "mask": torch.long,
    "depth": torch.float,
    "normals": torch.float,
}

# TODO: Should be moved to a different script.  
def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """3x3 Convolution: Depthwise: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    "1x1 Convolution: Pointwise"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)

def batchnorm(num_features):
    """https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html"""
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                            padding=int(kernel_size / 2.), groups=groups, bias=False),
                            batchnorm(out_channels),
                            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                            padding=int(kernel_size / 2.), groups=groups, bias=False),
                            batchnorm(out_channels))
                        
class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (scale * channel - mean) / std

    Args:
        scale (float): Scaling constant.
        mean (sequence): Sequence of means for R,G,B channels respecitvely.
        std (sequence): Sequence of standard deviations for R,G,B channels
            respecitvely.
        depth_scale (float): Depth divisor for depth annotations.

    """
    def __init__(self, scale, mean, std, depth_scale=1.0):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):
        sample["image"] = (self.scale * sample["image"] - self.mean) / self.std
        if "depth" in sample:
            sample["depth"] = sample["depth"] / self.depth_scale
        return sample
        
class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        crop_size (int): Desired output size.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        sample["image"] = image[top : top + new_h, left : left + new_w]
        for msk_key in msk_keys:
            sample[msk_key] = sample[msk_key][top : top + new_h, left : left + new_w]
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        # swap color axis because: numpy image: H x W x C and torch image: C X H X W
        sample["image"] = torch.from_numpy(image.transpose((2, 0, 1)))
        for msk_key in msk_keys:
            sample[msk_key] = torch.from_numpy(sample[msk_key]).to(
                KEYS_TO_DTYPES[msk_key]
            )
        return sample
        
class RandomMirror(object):
    """Randomly flip the image and the mask"""
    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        if do_mirror := np.random.randint(2):
            sample["image"] = cv2.flip(image, 1)
            for msk_key in msk_keys:
                scale_mult = [-1, 1, 1] if "normal" in msk_key else 1
                sample[msk_key] = scale_mult * cv2.flip(sample[msk_key], 1)
        return sample


class AverageMeter:
    """Simple running average estimator.
    Args:
      momentum (float): running average decay.
    """
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.avg = 0
        self.val = None

    def update(self, val):
        """Update running average given a new value.
        The new running average estimate is given as a weighted combination \
        of the previous estimate and the current value.
        Args:
          val (float): new value
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1.0 - self.momentum)
        self.val = val

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def fast_cm(preds, gt, n_classes):
    """Computing confusion matrix faster.
    Args:
      preds (Tensor) : predictions (either flatten or of size (len(gt), top-N)).
      gt (Tensor) : flatten gt.
      n_classes (int) : number of classes.
    Returns:
      Confusion matrix (Tensor of size (n_classes, n_classes)).
    """
    cm = np.zeros((n_classes, n_classes),dtype=np.int_)
    #print(gt.shape)
    #i,a,p, n = gt.shape[0]

    for i in range(gt.shape[0]):
        a = gt[i]
        p = preds[i]
        cm[a, p] += 1
    return cm

def compute_iu(cm):
    """Compute IU from confusion matrix.
    Args:
      cm (Tensor) : square confusion matrix.
    Returns:
      IU vector (Tensor).
    """
    pi = 0
    gi = 0
    ii = 0
    denom = 0
    n_classes = cm.shape[0]
    # IU is between 0 and 1, hence any value larger than that can be safely ignored
    default_value = 2
    IU = np.ones(n_classes) * default_value
    for i in range(n_classes):
        pi = sum(cm[:, i])
        gi = sum(cm[i, :])
        ii = cm[i, i]
        denom = pi + gi - ii
        if denom > 0:
            IU[i] = ii / denom
    return IU


def get_transforms(mode='train', mg_scale=None, depth_scale=None, crop_size=None, mean=None, std=None):
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

