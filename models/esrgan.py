from typing import List, Tuple, Union

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A convolutional block that combines a convolutional layer with an activation function.

    Attributes:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        use_act (bool): Whether to use an activation function after the convolution.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        use_act (bool): Flag to indicate the use of an activation function.
    """
    # def __init__(self, in_channels, out_channels, use_act):
    #     super().__init__()
    def __init__(self, in_channels: int, out_channels: int, use_act: bool) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.use_act=use_act

        self.cnn = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.act = nn.LeakyReLU(0.2) if self.use_act else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): The input tensor to the convolutional block.

        Returns:
            torch.Tensor: The output tensor after applying the convolution and activation function.
        """
        out = self.cnn(x)
        out = self.act(out)

        return out




class UpSampleBlock(nn.Module):
    """
    Upsample block for enlarging the spatial dimensions of images.

    This block combines upsampling (with a specified algorithm), convolution, and activation.

    Attributes:
        upsample_algo (str): The algorithm used for upsampling (e.g., 'nearest', 'bilinear').
        in_channels (int): Number of channels in the input image.
        scale_factor (int): The multiplier for increasing image size.
    """

    def __init__(self, upsample_algo: str, in_channels: int, scale_factor: int = 2) -> None:
        super().__init__()
        self.upsample_algo=upsample_algo
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=self.upsample_algo)
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.act = nn.LeakyReLU(0.2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpSampleBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Upsampled output tensor.
        """
        out = self.upsample(x)
        out = self.conv(out)
        out = self.act(out)

        return out



class DenseResidualBlock(nn.Module):
    """
    Dense Residual Block for image feature extraction.

    Consists of several convolutional blocks, each adding its output to the cumulative input of subsequent blocks.

    Attributes:
        in_channels (int): Number of channels in the input image.
        channels (int): Number of output channels for intermediate convolutions.
        residual_beta (float): Scaling factor for the residual connection.
    """

    def __init__(self, in_channels: int, channels: int = 32, residual_beta: float = 0.2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()


        for idx in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels=self.in_channels + self.channels*idx,
                    out_channels=self.channels if idx<=3 else self.in_channels,
                    use_act=True if idx <=3 else False
                )
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DenseResidualBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with applied dense residual connections.
        """
        new_inputs = x

        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.concat([new_inputs, out], dim=1)

        return self.residual_beta*out + x



class RRDB(nn.Module):
    """
    Residual-In-Residual Dense Block (RRDB) used in ESRGAN models for enhanced feature extraction.

    This block employs a series of Dense Residual Blocks to deepen the model without
    increasing computational complexity significantly.

    Attributes:
        in_channels (int): Number of channels in the input image.
        residual_beta (float): Scaling factor for the residual connections.
    """

    def __init__(self, in_channels: int, residual_beta: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.residual_beta = residual_beta

        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels=self.in_channels)
                                    for _ in range(3)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RRDB.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with applied dense residual blocks.
        """
        return self.rrdb(x)*self.residual_beta + x




class Generator(nn.Module):
    """
    The Generator module of the ESRGAN.

    This module generates high-resolution images from low-resolution inputs
    using a series of convolutional, residual, and upsampling blocks.

    Attributes:
        upsample_algo (str): The algorithm used for upsampling.
        in_channels (int): Number of channels in the input images.
        num_channels (int): Base number of channels used in convolutional layers.
        num_blocks (int): Number of RRDB blocks.
    """

    def __init__(self, upsample_algo: str,
                 in_channels: int = 3,
                 num_channels: int = 64,
                 num_blocks: int = 23):
        super().__init__()
        super().__init__()
        self.upsample_algo=upsample_algo
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        self.initial = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.residuals = nn.Sequential(*[RRDB(in_channels=self.num_channels)
                                        for _ in range(self.num_blocks)])

        self.conv = nn.Conv2d(in_channels=self.num_channels,
                              out_channels=self.num_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        self.upsamples = nn.Sequential(
            UpSampleBlock(upsample_algo=self.upsample_algo, in_channels=self.num_channels),
            UpSampleBlock(upsample_algo=self.upsample_algo, in_channels=self.num_channels),
            #UpSampleBlock(upsample_algo=self.upsample_algo, in_channels=self.num_channels),

        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The high-resolution output tensor.
        """
        initial = self.initial(x)
        out = self.residuals(initial)
        out = self.conv(out)
        out = out + initial
        out = self.upsamples(out)
        out = self.final(out)

        return out






class Discriminator(nn.Module):
    """
    The Discriminator module of the ESRGAN.

    This module discriminates between high-resolution images and generated images,
    aiding the training of the Generator.

    Attributes:
        in_channels (int): Number of channels in the input images.
        features (List[int]): List of features specifying the number of channels in each convolutional layer.

    Args:
        in_channels (int, optional): Input channels. Default is 3.
        features (List[int], optional): List of features for each convolutional block. Default is [64, 128, 256, 512].
    """

    def __init__(self, in_channels: int = 3, features: List[int] = [64, 128, 256, 512]):
        super().__init__()

        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    # kernel_size=3,
                    # stride=1 + idx % 2,
                    # padding=1,
                    use_act=True,
                ),
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            #nn.Linear(128 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing the discriminator's assessment.
        """
        x = self.blocks(x)

        return self.classifier(x)