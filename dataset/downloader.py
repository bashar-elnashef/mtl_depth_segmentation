import os
import zipfile
import json
import sys
import h5py
import torch
import gdown
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage import io
from scipy.io import loadmat
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


class NYUV2Downloader(object):
    def __init__(self, url, destination=None, download=False, 
                    procesess=False, exist_ok=True, save_colored=True):
        self._url = url
        self._procesess = procesess
        self._save_colored = save_colored
        
        # Setup path to data folder
        self._image_path = Path(destination)
        self._data_mat_path = self._image_path / 'nyu_labeled.mat'

        # if self._data_mat_path.is_file(): download=False
        self._download = download
        # Download dataset
        if self._download: self._download_dataset()

        self._splits_path = Path(f'{destination}/splits.mat')
        # if self._data_mat_path.is_file(): download=False
        self._splits = loadmat(str(self._splits_path))

        self._img_dirs = ['image', 'seg40', 'seg13', 'depth'] 
        if self._save_colored: 
            self._cmap = self._colormap()
            self._img_dirs += ['colored_13', 'colored_40', 'colored_depth']
        self._make_img_dirs(exist_ok=exist_ok)

        if self._procesess: self._procesess_dataset()

    def _make_img_dirs(self, exist_ok=True):
        self._paths = {img_dir: self._image_path / img_dir for img_dir in self._img_dirs}
        
        for path in self._paths.values():
            # path.mkdir(parents=True, exist_ok=exist_ok)
            for split in ['train', 'test']:
                split_path = path / split 
                split_path.mkdir(parents=True, exist_ok=exist_ok)

    def _download_dataset(self, exist_ok=True):
        # Setup path to data folder
        self._image_path.mkdir(parents=True, exist_ok=exist_ok)
        _zip_file_path = self._image_path / 'NYUv2.zip'
        try:
            gdown.download(self._url, str(_zip_file_path), quiet=False)
        except: raise Exception("[INFO] Could not download the data.")

        with zipfile.ZipFile(_zip_file_path, "r") as zip_ref:
            print(f"[INFO] Unzipping {_zip_file_path} data...")
            zip_ref.extractall(self._image_path)
        # Remove .zip file
        if remove_source:
            print(f"[INFO] Removing the source file {_zip_file_path} from folder...")
            os.remove(_zip_file_path)


    def _procesess_dataset(self, save_colored=True):
        
        with h5py.File(self._data_mat_path, 'r') as fr:
            self._extract_labels(np.array(fr["labels"]), save_colored=save_colored )
            self._extract_depths(np.array(fr["depths"]), save_colored=save_colored)
            self._extract_images(np.array(fr["images"]))


    def _extract_images(self, imgs):
        print("Extracting images...")
        imgs = imgs.transpose(0, 3, 2, 1)
        for split in ['train', 'test']:
            idxs = self._splits[split+'Ndxs'].reshape(-1)
            for idx in tqdm(idxs):
                img = imgs[idx-1]
                path = os.path.join(str(self._paths['image']), split, '%05d.png' % (idx))
                io.imsave(path, img)

    
    def _extract_depths(self, depths, save_colored=False):
        depths = depths.transpose(0, 2, 1)
        print("Extracting depths...")
        depths = (depths*1e3).astype(np.uint16)

        for split in ['train', 'test']:
            os.makedirs(os.path.join(str(self._paths['depth']), split), exist_ok=True)
            idxs = self._splits[split+'Ndxs'].reshape(-1)
            for idx in tqdm(idxs):
                depth = depths[idx-1]
                path = os.path.join(str(self._paths['depth']), split, '%05d.png' % (idx))
                io.imsave(path, depth, check_contrast=False)

                if save_colored:
                    norm = plt.Normalize()
                    colored = plt.cm.jet(norm(depth))
                    colored_path = os.path.join(str(self._paths["colored_depth"]), split, '%05d.png' % (idx))
                    plt.imsave(colored_path, colored)

    def _extract_labels(self, labels, save_colored=True):
        labels = labels.transpose([0, 2, 1])

        # Load data
        mapping40 = loadmat(f'{str(self._image_path)}/classMapping40.mat')['mapClass'][0]
        mapping40 = np.insert(mapping40, 0, 0)

        mapping13 = loadmat(f'{str(self._image_path)}/class13Mapping.mat')['classMapping13'][0][0][0][0]
        mapping13 = np.insert(mapping13, 0, 0)

        labels_40 = mapping40[labels]
        labels_13 = mapping13[labels_40].astype('uint8')

        labels_40 = labels_40.astype('uint8') - 1
        labels_13 = labels_13.astype('uint8') - 1

        print("Extracting labels (40 classes)...")

        for split in ['train', 'test']:
            idxs = self._splits[split+'Ndxs'].reshape(-1)
            for idx in tqdm(idxs):
                lbl = labels_40[idx-1]
                path = os.path.join(str(self._paths['seg40']), split, '%05d.png' % (idx))
                io.imsave(path, lbl, check_contrast=False)
                if self._save_colored:
                    colored_lbl = self._cmap[lbl+1]
                    colored_path = os.path.join(str(self._paths['colored_40']), split, '%05d.png' % (idx))
                    io.imsave(colored_path, colored_lbl)

        print("Extracting labels (13 classes)...")

        for split in ['train', 'test']:
            idxs = self._splits[split+'Ndxs'].reshape(-1)
            for idx in tqdm(idxs):
                lbl = labels_13[idx-1]
                path = os.path.join(str(self._paths['seg13']), split, '%05d.png' % (idx))
                io.imsave(path, lbl, check_contrast=False)
                if self._save_colored:
                    colored_lbl = self._cmap[lbl+1]
                    colored_path = os.path.join(str(self._paths['colored_13']), split, '%05d.png' % (idx))
                    io.imsave(colored_path, colored_lbl)

    def _colormap(self, N=256, normalized=False):
        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self._bitget(c, 0) << 7-j)
                g = g | (self._bitget(c, 1) << 7-j)
                b = b | (self._bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap

        return cmap

    def _bitget(self, byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    