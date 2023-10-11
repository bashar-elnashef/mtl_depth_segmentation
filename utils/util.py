import os
import zipfile
import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import gdown

def download_nyu2_dataset(url, destination=None, remove_source=True, download=False):
    """Downloads the nyu_depth_v2_labeled.mat and unpack in the destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.
    """
    # Setup path to data folder
    image_path = Path(destination)
    output = f"{destination}/nyu_depth_v2_labeled.mat" 

    print(f'image_path.is_dir() = {image_path.is_dir()}')
    # If the image folder doesn't exist, download it and prepare it...

    if image_path.is_dir():
        download = not any(image_path.iterdir())
    else: 
        image_path.mkdir(parents=True, exist_ok=True)
        download = True
        
    if download:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        print(url)

        if url is not None:
            gdown.download(url, output, quiet=False)
        else:
            print("[INFO] Could not download the data, the given url are None.")
            return


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
