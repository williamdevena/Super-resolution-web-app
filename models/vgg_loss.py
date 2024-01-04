from torch import nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    """
    VGG loss function (perceptual loss) used to train the generator
    in the adversarial framework.
    """
    def __init__(self, device, without_activation):
        super().__init__()
        if without_activation:
            self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)
        else:
            self.vgg = vgg19(pretrained=True).features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)