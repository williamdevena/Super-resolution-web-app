import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Convolutional block composed of a convolution and
    activation function.
    """
    def __init__(self, in_channels, out_channels, use_act):
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


    def forward(self, x):
        out = self.cnn(x)
        out = self.act(out)

        return out




class UpSampleBlock(nn.Module):
    """
    Upsample block composed of a upsampling operation, convolution
    and activation function.
    """
    def __init__(self, upsample_algo, in_channels, scale_factor=2):
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


    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.act(out)

        return out



class DenseResidualBlock(nn.Module):
    """
    Dense Residual Block.
    """
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
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


    def forward(self, x):
        new_inputs = x

        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.concat([new_inputs, out], dim=1)

        return self.residual_beta*out + x



class RRDB(nn.Module):
    """
    Residual-In-Residual Dense block.
    """
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.residual_beta = residual_beta

        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels=self.in_channels)
                                    for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x)*self.residual_beta + x




class Generator(nn.Module):
    def __init__(self, upsample_algo, in_channels=3, num_channels=64, num_blocks=23):
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


    def forward(self, x):
        initial = self.initial(x)
        out = self.residuals(initial)
        out = self.conv(out)
        out = out + initial
        out = self.upsamples(out)
        out = self.final(out)

        return out






class Discriminator(nn.Module):
    def __init__(self, in_channels=3,
                 #features=[64, 64, 128, 128, 256, 256, 512, 512]
                 features=[64, 128, 256, 512]
                 ):
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


    def forward(self, x):
        x = self.blocks(x)

        return self.classifier(x)