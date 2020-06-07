import torch.nn as nn
import torch

'''
modified to fit dataset size
'''

__all__ = ['AlexNet_BN']

class AlexNet_BN(nn.Module):

  def __init__(self, num_classes=100):
    super(AlexNet_BN, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(192),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(128, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(inplace=True),
      nn.Linear(128, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x