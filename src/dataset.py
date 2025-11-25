from torch.utils.data import Dataset
from PIL import Image
import os


class FishyDataset(Dataset):
    """
    Custom Pytorch Dataset class for load images from Fish4Knowledge Dataset
    """
    def __init__(self, img_dir, annotation_file, transforms):
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # TODO change depending on dataset format 
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        img = Image.open(img_path)
        label = self.annotation_file[idx]
        if self.transforms:
            img = self.transforms(img)

        return img, label
