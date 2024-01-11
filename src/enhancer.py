from typing import Union

import numpy as np
from torchsr.models import ninasr_b0, ninasr_b1, ninasr_b2
from torchvision.transforms.functional import to_tensor


class Enhancer():
    """
    Class responsible for the image enhancement.
    """
    def __init__(self, scale: int, model: Union[ninasr_b0, ninasr_b1, ninasr_b2]) -> None:
        self.scale = scale
        self.model = model
        self.model = self.model(scale=self.scale, pretrained=True)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Image resolution enhancement.

        Args:
            image (np.ndarray): original image.

        Returns:
            np.ndarray: enhanved image.
        """
        image = image / 255
        image_tensor = to_tensor(image).unsqueeze(0).float()
        self.model = self.model.float()
        enhanced_image_tensor = self.model(image_tensor)
        enhanced_image = enhanced_image_tensor.squeeze(0)
        enhanced_image = enhanced_image.permute(1,2,0).detach().numpy()

        return enhanced_image

