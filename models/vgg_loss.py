import torch
from torchvision.models import vgg19


class VGGLoss(torch.nn.Module):
    """
    VGGLoss computes the perceptual loss using features extracted by a VGG19 network.

    This loss function is often used in training generative models to improve the perceptual
    quality of the generated images by comparing high-level image features.

    Attributes:
        device (str): The device (e.g., 'cuda' or 'cpu') on which the VGG model is loaded.
        without_activation (bool): Flag to determine if the activation layer is included.

    Args:
        device (str): The device to run the VGG model on.
        without_activation (bool): If True, the VGG model is used without the final activation layer.
    """

    def __init__(self, device: str, without_activation: bool) -> None:
        super().__init__()

        if without_activation:
            self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)
        else:
            self.vgg = vgg19(pretrained=True).features[:36].eval().to(device)

        self.loss = torch.nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VGGLoss.

        Args:
            input (torch.Tensor): The generated image tensor.
            target (torch.Tensor): The target image tensor.

        Returns:
            torch.Tensor: The calculated loss based on the VGG features of the input and target.
        """
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)