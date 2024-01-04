import logging
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src import super_resolution_dataset
from utils import costants, metrics


def testing_generator(model, test_dataloader, device, image_folder):
    """
    Tests a model on the test set a returns the metrics.

    Args:
        - model (esrgan.Generator): generator model
        - test_dataloader (DataLoader): test dataloader
        - device (str): "cuda" or "cpu", it indicates whether we want
       to use cpu or gpu.
        - image_folder (str): path of the folder where to save the
        output images

    Returns:
        - tot_lpips (float): average LPIPS metric
        - tot_ssim (float): average SSIM metric
        - tot_psnr (float): average PSNR metric
    """
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    model.to(device)
    model.eval()
    len_dataloader = len(test_dataloader)
    pbar = tqdm(test_dataloader)
    tot_lpips = 0
    tot_ssim = 0
    tot_psnr = 0
    pbar.set_description("Testing")
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images = hr_images.squeeze(0)
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)
        fake_hr_images = model(lr_images)
        fake_hr_images = fake_hr_images.squeeze(0)
        fake_hr_images = fake_hr_images.to(device)

        fake_hr_images = (fake_hr_images+1)/2

        pil_fake_image = transforms.ToPILImage()(fake_hr_images)
        image_path = os.path.join(image_folder, f"{idx}.png")
        pil_fake_image.save(image_path)

        lpips, ssim, psnr = metrics.calculate_metrics(hr_images, fake_hr_images, device)
        tot_lpips += lpips
        tot_ssim += ssim
        tot_psnr += psnr

    tot_lpips /= len(test_dataloader)
    tot_ssim /= len(test_dataloader)
    tot_psnr /= len(test_dataloader)

    return tot_lpips, tot_ssim, tot_psnr


def build_test_dataloader():
    """
    Builds and returns the test dataloder

    Returns:
        - test_dataloader (DataLoader): test dataloader
    """
    transform_both = A.Compose([
        #A.RandomCrop(HR, HR)
        ]
    )

    transform_hr = A.Compose([
        A.Resize(width=1020, height=768, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
        ]
    )

    transform_lr = A.Compose([
        A.Resize(width=255, height=192, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
        ]
    )


    test_ds = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_TEST,
        lr_path=costants.LR_TEST,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )

    test_dataloader = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

    return test_dataloader


