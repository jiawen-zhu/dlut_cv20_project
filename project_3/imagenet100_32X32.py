from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset


class ImageNet100(VisionDataset):
    """`ImageNet100 _ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            'train_data' or 'val_data'  exists .
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    train_data = 'train_data'
    val_data = 'val_data'
    test = 'test_data'

    def __init__(self, root, train=True, evaluate=False, transform=None, target_transform=None,):

        super(ImageNet100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train
        if self.train:
            data_dir = os.path.join(root, self.train_data)
        else:
            data_dir = os.path.join(root, self.val_data)

        d = self.unpickle(data_dir)
        self.data = d['data']
        self.targets = d['labels']

        self.data = np.dstack(( self.data[:, :1024],  self.data[:, 1024:2048],  self.data[:, 2048:]))
        self.data = self.data.reshape((self.data.shape[0], 32, 32, 3))


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageNet100_Test(VisionDataset):
    """`ImageNet100 _ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            'train_data' or 'val_data'  exists .
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    test_data = 'test_data'

    def __init__(self, root, transform=None, target_transform=None):

        super(ImageNet100_Test, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        data_dir = os.path.join(root, self.test_data)

        d = self.unpickle(data_dir)
        self.data = d['data']
        self.data = np.dstack((self.data[:, :1024], self.data[:, 1024:2048], self.data[:, 2048:]))
        self.data = self.data.reshape((self.data.shape[0], 32, 32, 3))

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.data)