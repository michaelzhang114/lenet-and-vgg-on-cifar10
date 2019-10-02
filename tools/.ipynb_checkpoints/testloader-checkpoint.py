"""
sz114

This is adapted form dataloader.py and loads the test dataset instead.


"""


import os
import os.path
import numpy as np
import sys
import torch

from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data import Dataset as VisionDataset
from tools.utils import check_integrity, download_and_extract_archive

import shutil

class TEST_SET():
    """TEST SET from Kaggle

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):


        self.url = "https://www.dropbox.com/s/1tk8nv0b57o1lr8/cifar10-batches-images-test.tar.gz?dl=0"
        self.filename = "cifar10-batches-images-test.tar.gz"
        self.transform = transform
        self.target_transform = target_transform

        self.root = root
        if download:
            self.download()
        self.data = []
        self.targets = 0
        self.train = train

        
        img_name = os.path.join(root, "cifar10-batches-images-test.npy")
        target_name = os.path.join(root, "cifar10_train_val/cifar10-batches-labels-val.npy")

        self.data = np.load(img_name)
        self.targets = np.load(target_name)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #img, target = self.data[index], self.targets[index]
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
        
        target = 0
        
        return img, target

    def __len__(self):
        return len(self.data)