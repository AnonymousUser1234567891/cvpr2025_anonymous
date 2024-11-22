from torch.utils.data import Dataset
from .augmentation import get_augmentation, RandAug
import torchvision.transforms as transform
from torchvision.transforms import functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch
import random, time
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, copy
from PIL import Image

import clip

_, preprocess = clip.load("ViT-B/32")

    
def load_labels(path):
    with open(path, "r") as f:
        labels = json.load(f)
    return labels

json_path = "data/MNIST/NMNIST_classnames.json"
labels = load_labels(json_path)
text_inputs = torch.cat(
    [
        clip.tokenize(f"The digit {label}") 
            for label in labels.values()
    ]
)


class NMINIST(Dataset):
    def __init__(
            self, 
            txtPath, 
            classPath, 
            num_events=20000, 
            median_length=100000,
            frame=6, 
            resize_width=224, 
            resize_height=224, 
            representation=None,
            augmentation=False, 
            pad_frame_255=False,
            mode="train"
        ):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        self.augmentation = augmentation
        self.width, self.height = resize_width, resize_height
        self.representation = representation
        self.frame = frame
        self.num_events = num_events
        self.median_length = median_length
        self.pad_frame_255 = pad_frame_255
        tf = open(classPath, "r")
        self.classnames_dict = json.load(tf)  # class name idx start from 0
        self.mode = mode

    def __len__(self):
        return self.length

    def get_image(self, path):
        img = Image.open(path)
        img = np.array(img)
        img = np.stack([img, img, img], axis=-1)
        img = Image.fromarray(img)
        img = preprocess(img)
        return img

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        event_stream_path, image_path = self.files[idx].split('\t')
        label_str = event_stream_path.split('/')[-2]
        label_idx = int(label_str)

        events = self.load_ATIS_bin(event_stream_path)
        events_stream = np.array([events['x'], events['y'], events['t'], events['p']]).transpose()  # nx4,(x,y,t,p)

        img = self.get_image(image_path[:-1])
        event_stream = self.get_events(events_stream)

        data = {
            "img": img,
            "event": event_stream,
            "emb": -1,
            "label": label_idx,
        }
        return data

    def base_augment(self, events):
        if self.mode == "train":
            events = torch.from_numpy(events).float()
            # events = random_time_flip(events, resolution=(self.height, self.width))
            events = random_shift_events(events)
            events = add_correlated_events(events)
            events = np.array(events)
        return events

    def get_events(self, events):
        max_ = 2000
        if events.shape[0] > max_:
            start = np.random.randint(0, events.shape[0] - max_)
            events = events[start : start + max_]

        if self.mode == "train":
            events = self.base_augment(events)

        W = events[:, 0].max() + 1
        H = events[:, 1].max() + 1
        W = int(W)
        H = int(H)
        pos = events[events[:, 3] == 1]
        neg = events[events[:, 3] == 0]

        pos_count = np.bincount(
            pos[:, 0].astype(np.int64) + pos[:, 1].astype(np.int64) * W, 
            minlength=H * W
        ).reshape(H, W)
        neg_count = np.bincount(
            neg[:, 0].astype(np.int64) + neg[:, 1].astype(np.int64) * W, 
            minlength=H * W
        ).reshape(H, W)
        
        events = np.stack([pos_count, neg_count], axis=-1)
        events = torch.from_numpy(events)
        events = events.permute(2, 0, 1)
        events = events.float()
        events = torch.clamp(events, 0, 10)

        events = events / (events.amax([1,2],True) + 1)
        if self.mode == "train":
            events = RandAug()(events)
        events = transform.Resize((224, 224))(events)

        return events

    def load_ATIS_bin(self, file_name):
        '''
        :param file_name: path of the aedat v3 file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict
        This function is written by referring to https://github.com/jackd/events-tfds .
        Each ATIS binary example is a separate binary file consisting of a list of events. Each event occupies 40 bits as described below:
        bit 39 - 32: Xaddress (in pixels)
        bit 31 - 24: Yaddress (in pixels)
        bit 23: Polarity (0 for OFF, 1 for ON)
        bit 22 - 0: Timestamp (in microseconds)
        '''
        with open(file_name, 'rb') as bin_f:
            raw_data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
            x = raw_data[0::5]
            y = raw_data[1::5]
            rd_2__5 = raw_data[2::5]
            p = (rd_2__5 & 128) >> 7
            t = ((rd_2__5 & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        return {'t': t, 'x': x, 'y': y, 'p': p}

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        random.shuffle(self.files)
        return len(self.files)


def random_shift_events(event_tensor, max_shift=20, resolution=(224, 224)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    event_tensor[:, 0] += x_shift
    event_tensor[:, 1] += y_shift

    valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
    event_tensor = event_tensor[valid_events]

    return event_tensor

def add_correlated_events(event, xy_std = 1.5, ts_std = 0.001, add_noise=0):
    if event.size(0) < 1000:
        return event
    to_add = np.random.randint(min(100, event.size(0)-1),min(5000,event.size(0)))
    event_new = torch.cat((
        event[:,[0]] + torch.normal(0, xy_std,size = (event.size(0),1)),
        event[:,[1]] + torch.normal(0, xy_std,size = (event.size(0),1)),
        event[:,[2]] + torch.normal(0, ts_std,size = (event.size(0),1)),
        event[:,[3]]
        ),-1)
    
    idx = np.random.choice(np.arange(event_new.size(0)), size=to_add, replace=False)
    event_new = event_new[idx]
    event_new[:,[0]] = torch.clip(event_new[:,[0]],0,event[:,[0]].max())
    event_new[:,[1]] = torch.clip(event_new[:,[1]],0,event[:,[1]].max())
    
    event = torch.cat((event,event_new))
    return event[event[:,2].argsort(descending = False)]  