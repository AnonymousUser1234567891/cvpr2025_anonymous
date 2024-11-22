from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .caltech_dataset import NCaltech101

    
class NCaltech101DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_txt,
            val_txt,
            classPath,
            batch_size=32,
            num_workers=4,
            num_events=20000,
            median_length=20000,
            resize_width=224,
            resize_height=224,
            representation=None,
            augmentation=False,
            pad_frame_255=False,
            EventCLIP=False,
        ):
        super().__init__()
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.classPath = classPath
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_events = num_events
        self.median_length = median_length
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.representation = representation
        self.augmentation = augmentation
        self.pad_frame_255 = pad_frame_255
        self.EventCLIP = EventCLIP

    def setup(self, stage=None):
        self.train_dataset = NCaltech101(
            txtPath=self.train_txt,
            classPath=self.classPath,
            num_events=self.num_events,
            median_length=self.median_length,
            resize_width=self.resize_width,
            resize_height=self.resize_height,
            representation=self.representation,
            augmentation=self.augmentation,
            pad_frame_255=self.pad_frame_255,
            EventCLIP=self.EventCLIP,
            mode='train'
        )
        self.val_dataset = NCaltech101(
            txtPath=self.val_txt,
            classPath=self.classPath,
            num_events=self.num_events,
            median_length=self.median_length,
            resize_width=self.resize_width,
            resize_height=self.resize_height,
            representation=self.representation,
            augmentation=self.augmentation,
            pad_frame_255=self.pad_frame_255,
            EventCLIP=self.EventCLIP,
            mode='val'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )