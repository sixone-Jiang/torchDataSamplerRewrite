import os
import PIL
from PIL import Image
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import random, cv2, pickle

# make the shuffle of the dataset, just when load batch:
'''
    exp:
        data = [0, 1, 2, 3, 4]
        batch_size = 2
        shuffle = True
    get_batchs:
        keep the order of the data, but shuffle the order of the batchs
        [2, 3], [0, 1], [4]
'''

class FixBase(Dataset):
    def __init__(self, num_data=100):

        self.data = list(range(num_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
from torch.utils.data import BatchSampler, Sampler, SequentialSampler
from typing import Iterator, List, Union, Iterable

class outerBatchSampler(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool, outer_shuffle=True) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.outer_shuffle = outer_shuffle
        #random.shuffle(self.sampler)


    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            batch_all = []
            while True:
                try:

                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    batch_all.append(batch)
                except StopIteration:
                    break

            if self.outer_shuffle:
                random.shuffle(batch_all)

            for batch in batch_all:
                yield batch

        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
            
    
if __name__=='__main__':
    dataset = FixBase(11)
    sampler = SequentialSampler(dataset)
    batch_sampler = outerBatchSampler(sampler, 2, True)
    dataloader = DataLoader(dataset, batch_size=1, batch_sampler=batch_sampler, num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, batch)
