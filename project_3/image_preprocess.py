from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch
# imagenet100_32X32
normalize = transforms.Normalize(mean=[0.484, 0.460, 0.411],
                                 std=[0.260, 0.253, 0.271])

def train_val_transforms():
    """
        You can modify the train_transforms to try different image preprocessing methods when training model
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
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
        transforms.ToTensor(),
        normalize,
    ])
    return trans
