import cv2
from torchsr.datasets import Div2K

from torchvision.transforms.functional import to_pil_image, to_tensor

class Enhancer():
    def __init__(self, scale, model):
        self.scale = scale
        self.model = model
        self.model = self.model(scale=self.scale, pretrained=True)

    def enhance(self, image):
        image = image / 255
        #print(image)
        image_tensor = to_tensor(image).unsqueeze(0).float()
        self.model = self.model.float()
        enhanced_image_tensor = self.model(image_tensor)
        enhanced_image = enhanced_image_tensor.squeeze(0)
        enhanced_image = enhanced_image.permute(1,2,0).detach().numpy()

        return enhanced_image

