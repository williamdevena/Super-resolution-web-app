import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

#from utils import costants


class SuperResolutionDataset(Dataset):
    """
    Custom PyTorch dataset used to train models in the
    Image Super Resolution  framework.
    """

    def __init__(self, hr_path, lr_path, transform_both, transform_hr, transform_lr):
        self.transform_both = transform_both
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.list_couples_hr_lr = self.build_list_couples_hr_lr()


    def build_list_couples_hr_lr(self):
        """
        Builds a list of tuples that contain the path of a HR image and
        the path of the corresponding LR image.

        Returns:
            list_couples_hr_lr (List): contains tuples that contain
            the path of a HR image and the path of the corresponding LR image, in
            the following form (hr_path, lr_path).
        """
        hr_images = os.listdir(self.hr_path)
        list_couples_hr_lr = []

        for hr_image_name in hr_images:
            if ".png" in hr_image_name:
                id_num = hr_image_name.split(".")[0]
                lr_image_name = id_num +"x4(x8).png"
                hr_image_path = os.path.join(self.hr_path, hr_image_name)
                lr_image_path = os.path.join(self.lr_path, lr_image_name)
                list_couples_hr_lr.append((hr_image_path, lr_image_path))

        list_couples_hr_lr.sort(key=lambda x: int(x[0].split("/")[-1].split(".")[0]))

        return list_couples_hr_lr


    def __len__(self):
        return len(self.list_couples_hr_lr)


    def __getitem__(self, idx):
        hr_image_name, _ = self.list_couples_hr_lr[idx]
        image = np.array(Image.open(hr_image_name))
        image = self.transform_both(image=image)["image"]
        hr_image = self.transform_hr(image=image)["image"]
        lr_image = self.transform_lr(image=image)["image"]

        return hr_image, lr_image