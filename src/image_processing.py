import cv2
from torchsr.datasets import Div2K
from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor


def enhance(image):
    #print(image)
    enhanced_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(enhanced_image)

    return enhanced_image






def enhanceResolution(image):


