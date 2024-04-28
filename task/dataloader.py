# -*- coding: utf-8 -*-
# @Author: Frank
# @Date:   2022-10-22 08:38:41
# @Last Modified by:   Frank
# @Last Modified time: 2024-03-15 17:28:21
import yacs.config
import torch
import pandas as pd
import torchvision.datasets as datasets
from .dataset import create_dataset
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return [x[1] for x in dataset.imgs]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples






def create_dataloader(config):

    train_dataset1,train_dataset2,val_dataset = create_dataset(config)

    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                               batch_size=config.dataloader.batch_size,
                                               shuffle=True,
                                               num_workers=config.dataloader.workers,
                                               pin_memory=False)


    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                               batch_size=config.dataloader.batch_size,
                                               shuffle=False,
                                               num_workers=config.dataloader.workers,
                                               sampler = ImbalancedDatasetSampler(train_dataset2),
                                               pin_memory=False)
                                               
                                               
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.dataloader.batch_size,
                                             shuffle=False,
                                             num_workers=config.dataloader.workers,
                                             pin_memory=False)   

    return train_loader1, train_loader2, val_loader
