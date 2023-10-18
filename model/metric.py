import os
import random
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import fast_cm, compute_iu

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

# TODO: Need to check this class.
class MeanIoU:
    """Mean-IoU computational block for semantic segmentation.
    Args:
      num_classes (int): number of classes to evaluate.
    Attributes:
      name (str): descriptor of the estimator.
    """

    def __init__(self, num_classes=40):
        if isinstance(num_classes, (list, tuple)):
            num_classes = num_classes[0]
        assert isinstance(
            num_classes, int
        ), f"Number of classes must be int, got {num_classes}"
        self.num_classes = num_classes
        self.name = "MeanIoU"
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, pred, gt):
        idx = gt < self.num_classes
        pred_dims = len(pred.shape)
        assert (pred_dims - 1) == len(
            gt.shape
        ), "Prediction tensor must have 1 more dimension that ground truth"
        if pred_dims == 3:
            class_axis = 0
        elif pred_dims == 4:
            class_axis = 1
        else:
            raise ValueError(f"{pred_dims}-dimensional input is not supported")
        assert (
            pred.shape[class_axis] == self.num_classes
        ), f"Dimension {class_axis} of prediction tensor must be equal to the number of classes"
        pred = pred.argmax(axis=class_axis)
        self.cm += fast_cm(
            pred[idx].astype(np.uint8), gt[idx].astype(np.uint8), self.num_classes
        )

    def val(self):
        return np.mean([iu for iu in compute_iu(self.cm) if iu <= 1.0])


class RMSE:
    """Root Mean Squared Error computational block for depth estimation.
    Args:
      ignore_val (float): value to ignore in the target
                          when computing the metric.
    Attributes:
      name (str): descriptor of the estimator.
    """

    def __init__(self, ignore_val=0):
        self.ignore_val = ignore_val
        self.name = "RMSE"
        self.reset()

    def reset(self):
        self.num = 0.0
        self.den = 0.0

    def update(self, pred, gt):
        assert (
            pred.shape == gt.shape
        ), "Prediction tensor must have the same shape as ground truth"
        pred = np.abs(pred)
        idx = gt != self.ignore_val
        diff = (pred - gt)[idx]
        self.num += np.sum(diff ** 2)
        self.den += np.sum(idx)

    def val(self):
        return np.sqrt(self.num / self.den)