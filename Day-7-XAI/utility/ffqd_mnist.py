import os

import numpy as np
from torch.utils.data import Dataset


def label_for_id(i):
    return \
        ['airplane', 'apple', 'wine bottle', 'car', 'mouth', 'pineapple', 'umbrella', 'pear', 'mustache', 'smiley'][
            i]


class FFQD_Dataset(Dataset):
    def __init__(self, data_source, train=True, transform=None, target_transform=None):
        super().__init__()
        self.img_labels = np.load(os.path.join(data_source, f'ffqd_mnist_00_{"train" if train else "test"}_labels.npy'))
        self.images = np.load(os.path.join(data_source, f'ffqd_mnist_00_{"train" if train else "test"}_images.npy'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = self.img_labels[idx].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
