from torch.utils.data import Dataset
from utils import util
from pathlib import Path
from PIL import Image

class CustomNYUv2Dataset(Dataset):

    def __init__(self, data_path='data', masks=None, mode='train', trsfm=None):
        """ Custom dataset for NYUv2 dataset with image, depth and seg40 sets."""
        if masks is None:
            masks = ['seg40', 'depth']
        self.masks = masks
        self._mask_num = len(masks)

        self._data_path = Path(data_path)
        self._image_paths = [list(self._data_path.glob(f'{msk}/{mode}/*.png')) for msk in self.masks]
        self._image_paths.append(list(self._data_path.glob(f'image/{mode}/*.png')))
        self._paths = [[*r] for r in zip(*self._image_paths)]

        self.trsfm = trsfm

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        _paths = self._paths[idx]
        sample = {
            msk: np.array(Image.open(_paths[i]))
            for i, msk in enumerate(self.masks)
        }

        sample['image'] = np.array(Image.open(_paths[-1]))

        for msk in self.masks:
            err_msg = f'{msk} mask must be encoded without colourmap' 
            assert len(sample[msk].shape) == 2, err_msg

        if self.trsfm:
            # TODO: initialise trsfms with mask names beforehand, seems stupid for it to be here. 
            sample["names"] = self.masks
            sample = self.trsfm(sample)
            # --> the names key should be removed by the transformation
            if "names" in sample:
                del sample["names"]
        return sample