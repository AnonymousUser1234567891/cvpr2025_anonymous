from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .imagenet_dataset import PretrainImageNetDataset
from .caltech_dataset import NCaltech101
from .mnist import NMINIST

dataset_config = {
    'N-imagenet': {
        'reshape': True, 
        'reshape_method': 'no_sample', 
        'loader_type': 'reshape_then_acc_count_pol', 
        'slice': {
            'slice_events': True, 
            'slice_length': 30000, 
            'slice_method': 'random', 
            'slice_augment': False, 
            'slice_augment_width': 0, 
            'slice_start': 0, 
            'slice_end': 30000
        }, 
        'height': 224, 
        'width': 224, 
        'augment': True, 
        'augment_type': 'base_augment', 
        'persistent_workers': True, 
        'pin_memory': True, 
        'train': {
            'type': 'PretrainImageNetDataset', 
            'root': '/mnt/Data_3/event/', 
            'file': './configs/train_list_zero.txt', 
            'label_map': '/mnt/Data_3/event/N_ImageNet/extracted_train', 
            'emb_path': '/mnt/Data_3/event/ImageNet_CLIP/emb_train'
        }, 
        'eval': {
            'type': 'PretrainImageNetDataset', 
            'root': '/mnt/Data_3/event/', 
            'file': './configs/val_list_zero.txt', 
            'label_map': '/mnt/Data_3/event/N_ImageNet/extracted_val', 
            'emb_path': '/mnt/Data_3/event/ImageNet_CLIP/emb_val'
        }, 
        'num_workers': 7, 
        'batch_size': 4, 
        'point_level_aug': False, 
        'view_augmentation': {
            'name': 'Ours', 
            'view1': {
                'crop_min': 0.45
            }, 
            'view2': {
                'crop_min': 0.45
            }
        }
    }
}

train_text_file_lst = {
    '1-shot': './configs/train_list_1_shot.txt',
    '2-shot': './configs/train_list_2_shot.txt',
    '5-shot': './configs/train_list_5_shot.txt',
    '10-shot': './configs/train_list_10_shot.txt',
    '20-shot': './configs/train_list_20_shot.txt',
    'all': './configs/train_list_all_cls.txt'
}

train_batch_size = {
    '1-shot': 20,
    '2-shot': 40,
    '5-shot': 100,
    '10-shot': 128,
    '20-shot': 128,
    'all': 4
}


def create_dataloader(
        dataset,
        dataset_opt,
        num_process
    ):
    phase = dataset_opt['phase']
    
    collate_fn = None
    if phase == 'train':
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True
        )
        TrainLoader = DataLoader(
            dataset, 
            batch_size=dataset_opt["batch_size"], 
            num_workers=dataset_opt["num_workers"], 
            drop_last=True, 
            pin_memory=dataset_opt["pin_memory"], 
            persistent_workers=dataset_opt["persistent_workers"], 
            collate_fn = collate_fn, 
            sampler = sampler
        )
        return TrainLoader
    elif phase== 'test' or phase== 'eval' :
        if len(dataset) % num_process == 0:
            sampler = torch.utils.data.DistributedSampler(
                dataset, shuffle=False,
            )
        else:
           sampler = torch.utils.data.RandomSampler(dataset)
        TestLoader = DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"], 
            num_workers=dataset_opt["num_workers"], 
            drop_last=False, 
            pin_memory= dataset_opt["pin_memory"],
            persistent_workers=dataset_opt["persistent_workers"], 
            collate_fn = collate_fn, 
            sampler = sampler
        )
        return TestLoader
    else:
        raise AttributeError('Mode not provided')


class Data_prepare(pl.LightningDataModule):
    def __init__(self, ds, num_process, ft=None):
        if ds == 'N-imagenet':
            self.args = dataset_config[ds]
            if ft is not None:
                self.args['train']['file'] = train_text_file_lst[ft]
                self.args['batch_size'] = train_batch_size[ft]
                if ft == 'all':
                    self.args['eval']['file'] = './configs/val_list_all_cls.txt'
        elif ds == 'N-imagenet-1000':
            self.args = dataset_config['N-imagenet']
            self.args['train']['file'] = './configs/train_1000_zero.txt'
            self.args['train']['label_map'] = '/mnt/Data_3/event/tmp/extracted_train'
            self.args['train']['emb_path'] = '/mnt/Data_3/event/ImageNet_CLIP/emb_train'
            self.args['eval']['file'] = './configs/val_1000_zero.txt'
            self.args['eval']['label_map'] = '/mnt/Data_3/event/tmp/extracted_val'
            self.args['eval']['emb_path'] = '/mnt/Data_3/event/ImageNet_CLIP/emb_val'
            self.args['batch_size'] = 32
        else:
            NotImplementedError
        self.num_process = num_process
        print(f'Load file: {self.args["train"]["file"]}')
        print(f'Load file: {self.args["eval"]["file"]}')

    def prepare_data(self):
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage):
        args = self.args
        dataset_args = deepcopy(args)
        dataset_args.update(dataset_args["train"])
        dataset_args.update(dataset_args["slice"])
        dataset_args['phase'] = 'train'
        del dataset_args["train"]
        del dataset_args["slice"]

        self.train_args = dataset_args
        self.train_dataset = PretrainImageNetDataset(dataset_args)

        dataset_args = deepcopy(args)
        dataset_args.update(dataset_args["eval"])
        dataset_args.update(dataset_args["slice"])
        dataset_args['phase'] = 'eval'
        del dataset_args["eval"]
        del dataset_args["slice"]

        self.val_args = dataset_args
        self.val_dataset = PretrainImageNetDataset(dataset_args)

    def train_dataloader(self):
        if hasattr(self, "train_loader"):
            return self.train_loader
        args = self.args
        train_loader = create_dataloader(
            self.train_dataset,
            self.train_args,
            self.num_process
        )
        self.train_loader = train_loader
        return train_loader

    def val_dataloader(self):
        if hasattr(self, "eval_loader"):
            return self.eval_loader
        args = self.args
        eval_loader = create_dataloader(
            self.val_dataset,
            self.val_args,
            self.num_process
        )
        self.eval_loader = eval_loader
        return eval_loader
    
    
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

        
class NMNISTDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_txt,
            val_txt, 
            classPath, 
            batch_size,
            num_workers=4,
            num_events=10000, 
            median_length=10000,
            frame=6, 
            resize_width=224, 
            resize_height=224, 
            representation=None,
            augmentation=False, 
            pad_frame_255=False
        ):
        super().__init__()
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.classPath = classPath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_events = num_events
        self.median_length = median_length
        self.frame = frame
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.representation = representation
        self.augmentation = augmentation
        self.pad_frame_255 = pad_frame_255

    def setup(self, stage=None):
        self.train_dataset = NMINIST(
            txtPath=self.train_txt,
            classPath=self.classPath,
            num_events=self.num_events,
            median_length=self.median_length,
            frame=self.frame,
            resize_width=self.resize_width,
            resize_height=self.resize_height,
            representation=self.representation,
            augmentation=self.augmentation,
            pad_frame_255=self.pad_frame_255
        )
        self.val_dataset = NMINIST(
            txtPath=self.val_txt,
            classPath=self.classPath,
            num_events=self.num_events,
            median_length=self.median_length,
            frame=self.frame,
            resize_width=self.resize_width,
            resize_height=self.resize_height,
            representation=self.representation,
            augmentation=self.augmentation,
            pad_frame_255=self.pad_frame_255
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