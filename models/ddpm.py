import logging
from typing import Tuple, Union

import torch
from tqdm import tqdm

from utils import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) for image resolution enhancement.

    This class implements the DDPM algorithm, which progressively enhances image resolution
    by applying a series of noise reduction steps.
    """

    def __init__(self, noise_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 img_size: int = 256,
                 device: str = "cuda") -> None:
        """
        Initializes the DDPM model with specified parameters.

        Args:
            noise_steps (int): The number of steps for the noise schedule.
            beta_start (float): The initial value of beta in the noise schedule.
            beta_end (float): The final value of beta in the noise schedule.
            img_size (int): The size of the images (in pixels).
            device (str): The device for computation ('cuda' or 'cpu').
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def prepare_noise_schedule(self) -> torch.Tensor:
        """
        Prepares the noise schedule used in the DDPM process.
        Creates a one-dimensional tensor of size self.noise_steps whose
        values are evenly spaced from self.beta_start to self.beta_end, inclusive.

        Returns:
            torch.Tensor: A tensor representing the noise schedule.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)


    def noise_images(self, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies noise to the images based on the given timestep.

        Args:
            x (torch.Tensor): The input images.
            t (int): The current timestep.

        Returns:
            tuple: A tuple containing the noised images and the applied noise.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon


    def sample_timesteps(self, n: int) -> torch.Tensor:
        """
        Samples random timesteps for the DDPM process.

        Args:
            n (int): The number of timesteps to sample.

        Returns:
            torch.Tensor: Randomly sampled timesteps.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sample(self, model: torch.nn.Module, n: int) -> torch.Tensor:
        """
        Generates new images using the DDPM model.

        Args:
            model (torch.nn.Module): The DDPM model.
            n (int): The number of images to generate.

        Returns:
            torch.Tensor: The generated images.
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()

        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x





if __name__ == '__main__':
    pass