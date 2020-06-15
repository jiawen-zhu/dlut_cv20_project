from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch
import numpy as np
# imagenet100_32X32
# normalize = transforms.Normalize(mean=[0.484, 0.460, 0.411],
#                                  std=[0.260, 0.253, 0.271])
normalize = transforms.Normalize(mean=[0.484, 0.460, 0.411],
                                 std=[0.260, 0.253, 0.271])

def train_val_transforms():
    """
        You can modify the train_transforms to try different image preprocessing methods when training model
    """
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.2)),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),


        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        # Cutout(n_holes=1, length=2),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms

def test_transforms():
    """
        You can modify the function to try different image fusion methods when evaluating the trained model
    """
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return trans

def test_transforms_ori():
    """
        You can modify the function to try different image fusion methods when evaluating the trained model
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return trans

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
