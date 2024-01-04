"""
This file contains the baselines used
"""
import logging
import os

import cv2
import numpy as np
import skimage
import torch
from PIL import Image
from tqdm import tqdm

from utils import costants, logging_utilities, metrics


def baselines():
    """
    Runs the baselines and produces metrics.

    Args: None

    Returns: None
    """
    original_ds_path = costants.ORIGINAL_DS_TEST
    test_ds_path = costants.TRACK2_TEST

    ## BILINEAR INTERPOLATION
    avg_ssim, avg_lpips, avg_psnr = test_baseline_upscaling(
        method="bilinear",
        test_ds_path=test_ds_path,
        original_ds_path=original_ds_path,
        path_upscaled_folder="./upsampling/bilinear_interp")
    logging_utilities.print_name_stage_project("BILINEAR INTERPOLATION")
    logging.info(f"AVG SSIM: {avg_ssim}\nAVG LPIPS: {avg_lpips}\nAVG PSNR: {avg_psnr}")

    ## NEAREST NEIGHBOUR
    avg_ssim, avg_lpips, avg_psnr = test_baseline_upscaling(
        method="nn",
        test_ds_path=test_ds_path,
        original_ds_path=original_ds_path,
        path_upscaled_folder="./upsampling/nn")
    logging_utilities.print_name_stage_project("NEAREST NEIGHBOUR")
    logging.info(f"AVG SSIM: {avg_ssim}\nAVG LPIPS: {avg_lpips}\nAVG PSNR: {avg_psnr}")



def test_baselines(method, test_dataloader, image_folder):
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    len_dataloader = len(test_dataloader)
    pbar = tqdm(test_dataloader)
    tot_lpips = 0
    tot_ssim = 0
    tot_psnr = 0
    pbar.set_description("Testing")
    for idx, (hr_image, lr_image) in enumerate(pbar):
        hr_image = np.array(hr_image.squeeze(0).permute(1,2,0))
        lr_image = np.array(lr_image.squeeze(0).permute(1,2,0))

        size = (hr_image.shape[1], hr_image.shape[0])
        if method=="bilinear":
            fake_hr_image = bilinear_interpolation_upscaling(lr_image, size)
        elif method=="nn":
            fake_hr_image = nearest_neighbour_upscaling(lr_image, size)

        image_path = os.path.join(image_folder, f"{idx}.png")
        Image.fromarray(np.uint8(fake_hr_image*255)).convert('RGB').save(image_path)

        lpips, ssim, psnr = metrics.calculate_metrics(hr_image, fake_hr_image, device='cpu')
        tot_lpips += lpips
        tot_ssim += ssim
        tot_psnr += psnr

    tot_lpips /= len(test_dataloader)
    tot_ssim /= len(test_dataloader)
    tot_psnr /= len(test_dataloader)

    return tot_lpips, tot_ssim, tot_psnr



def test_baseline_upscaling(method, test_ds_path, original_ds_path, path_upscaled_folder):
    """
    Applies a baseline method (bilinear interpolation or nearest neighbour)
    for upscaling to a folder of images, writes the results in a new folder
    and finally calculates some metrics.

    Args:
        - method (str): "bilinear" or "nn", denotes the method used
        - test_ds_path (str): path of the folder with low resolution images
        - original_ds_path (str): path of the folder with original high
        resolution images
        - path_upscaled_folder (str): path of the new folder with the upscaled images

    Returns:
        - avg_ssim (float): average SSIM metric calculated on all the images
        - avg_lpips (float): average LPIPS metric calculated on all the images
    """
    files = os.listdir(test_ds_path)
    tot_ssim = 0
    tot_lpips = 0
    tot_psnr = 0

    if not os.path.exists(path_upscaled_folder):
        os.makedirs(path_upscaled_folder)

    for img_name in files:
        ## LOW RESOLUTION IMAGE
        low_res_img_path = os.path.join(test_ds_path, img_name)
        low_res_img = cv2.imread(low_res_img_path)

        ## ORIGINAL HIGH RESOLUTION IMAGE
        original_img_name = "".join([img_name.split("x")[0], ".png"])
        original_img_path = os.path.join(original_ds_path, original_img_name)
        original_img = cv2.imread(original_img_path)
        original_shape = original_img.shape

        ## UPSCALED HIGH RESOLUTION IMAGE
        size = (original_shape[1], original_shape[0])

        if method=="bilinear":
            upscaled_img = bilinear_interpolation_upscaling(low_res_img, size)
        elif method=="nn":
            upscaled_img = nearest_neighbour_upscaling(low_res_img, size)
        else:
            raise ValueError(f"The parameter 'method' of the function 'test_baseline_upscaling' has an invalid value {method}.")

        ## METRICS
        # ssim_score = skimage.metrics.structural_similarity(original_img, upscaled_img, channel_axis=2)
        # lpips_distance = metrics.calculate_lpips_distance(original_img, upscaled_img)
        lpips, ssim, psnr = metrics.calculate_metrics(original_img=original_img, upscaled_img=upscaled_img)

        ## SAVES THE UPSCALED IMAGE
        cv2.imwrite(os.path.join(path_upscaled_folder, original_img_name), upscaled_img)

        tot_ssim += ssim
        tot_lpips += lpips
        tot_psnr += psnr

    avg_ssim = tot_ssim/len(files)
    avg_lpips = tot_lpips/len(files)
    avg_psnr = tot_psnr/len(files)
    return avg_ssim, avg_lpips, avg_psnr



def bilinear_interpolation_upscaling(img, size):
    """
    Upscales an image using bilinear interpolation.

    Args:
        - img (np.ndarray): low resolution image
        - size (tuple): dimensionsof the high resolution
        image we want to produce

    Returns:
        - upscaled_img (np.ndarray): upscaled image
    """
    upscaled_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    return upscaled_img


def nearest_neighbour_upscaling(img, size):
    """
    Upscales an image using nearest neighbour.

    Args:
        - img (np.ndarray): low resolution image
        - size (tuple): dimensionsof the high resolution
        image we want to produce

    Returns:
        - upscaled_img (np.ndarray): upscaled image
    """
    upscaled_img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    return upscaled_img
