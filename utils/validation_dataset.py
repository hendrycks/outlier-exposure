import torch
import numpy as np


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)


class PartialFolder(torch.utils.data.Dataset):
    def __init__(self, parent_ds, perm, length):
        self.parent_ds = parent_ds
        self.perm = perm
        self.length = length
        super(PartialFolder, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[self.perm[i]]


def validation_split_folder(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    num_train = int(len(dataset) * (1 - val_share))
    num_val = len(dataset) - num_train

    perm = np.asarray(range(len(dataset)))
    np.random.seed(0)
    np.random.shuffle(perm)

    train_perm, val_perm = perm[:num_train], perm[num_train:]

    return PartialFolder(dataset, train_perm, num_train), PartialFolder(dataset, val_perm, num_val)
