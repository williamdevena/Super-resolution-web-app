
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models import esrgan
from src import baselines, super_resolution_dataset, test
#from src.super_resolution_dataset import SuperResolutionDataset
from utils import costants, logging_utilities, preprocessing


def main():
    """
    Executes the all the stages of the project
    """

    ## INITIAL CONFIGURATIONS
    if not os.path.exists("./project_log"):
        os.mkdir("./project_log")
    if not os.path.exists("./output_images"):
        os.mkdir("./output_images")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("project_log/assignment.log"),
            logging.StreamHandler()
        ]
    )
    logging.info((('-'*70)+'\n')*5)
    logging_utilities.print_name_stage_project("IMAGE SUPER RESOLUTION")
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"DATE AND TIME OF EXCUTION: {dt_string}")


    ### TESTING
    test_dataloader = test.build_test_dataloader()
    device="cpu"

    ### BASELINES TESTING
    ### NN
    tot_lpips, tot_ssim, tot_psnr = baselines.test_baselines(method="nn",
                                                             test_dataloader=test_dataloader,
                                                             image_folder="./output_images/nn_baseline")
    logging.info(
        f"\nBASELINE NEAREST-NEIGHBOR METRICS\nPSNR: {tot_psnr}\nSSIM: {tot_ssim}\nLPIPS: {tot_lpips}"
    )
    ## BILINEAR INTERPOLATION
    tot_lpips, tot_ssim, tot_psnr = baselines.test_baselines(method="nn",
                                                             test_dataloader=test_dataloader,
                                                             image_folder="./output_images/bilinear_baseline")
    logging.info(
        f"\nBASELINE BILINEAR INTERPOLATION METRICS\nPSNR: {tot_psnr}\nSSIM: {tot_ssim}\nLPIPS: {tot_lpips}"
    )


    ### CNN TESTING
    model = esrgan.Generator(upsample_algo="nearest", num_blocks=2)
    model.load_state_dict(torch.load("./models_weights/L1_model.pt",
                                     map_location=torch.device(device=device)))
    tot_lpips, tot_ssim, tot_psnr = test.testing_generator(model=model,
                                                           test_dataloader=test_dataloader,
                                                           device=device,
                                                           image_folder="./output_images/cnn_l1")
    logging.info(
        f"\nCNN METRICS\nPSNR: {tot_psnr}\nSSIM: {tot_ssim}\nLPIPS: {tot_lpips}"
    )

    ### GAN TESTING
    model = esrgan.Generator(upsample_algo="nearest", num_blocks=2)
    model.load_state_dict(torch.load("./models_weights/gan.pt",
                                     map_location=torch.device(device=device)))
    tot_lpips, tot_ssim, tot_psnr = test.testing_generator(model=model,
                                                           test_dataloader=test_dataloader,
                                                           device=device,
                                                           image_folder="./output_images/gan")
    logging.info(
        f"\nGAN METRICS\nPSNR: {tot_psnr}\nSSIM: {tot_ssim}\nLPIPS: {tot_lpips}"
    )





if __name__ == "__main__":
    main()
