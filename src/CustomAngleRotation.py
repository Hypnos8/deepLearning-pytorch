import random
from torchvision.transforms import functional

class CustomAngleRotation:
    def __init__(self, angles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return functional.rotate(x, angle)
