import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os
from .augmentation import get_augmentation, RandAug
from functools import partial

# import clip
import model.clip as clip

from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transform


_, preprocess = clip.load("ViT-B/32")


def load_imagenet_labels(path):
    with open(path, "r") as f:
        labels = json.load(f)
    return labels


json_path = "data/Caltech/Caltech101_classnames.json"
labels = load_imagenet_labels(json_path)
text_inputs = torch.cat(
    [
        clip.tokenize(f"a point cloud image of a {label}") 
            for label in labels.keys()
    ]
)

class NCaltech101(Dataset):
    def __init__(
        self,
        txtPath,
        classPath,
        num_events=20000,
        median_length=100000,
        resize_width=224,
        resize_height=224,
        representation=None,
        augmentation=False,
        pad_frame_255=False,
        EventCLIP=False,
        mode='train'
    ):
        super(NCaltech101, self).__init__() 
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        self.augmentation = augmentation
        self.width, self.height = resize_width, resize_height
        self.representation = representation
        self.num_events = num_events
        self.median_length = median_length
        self.pad_frame_255 = pad_frame_255
        self.EventCLIP = EventCLIP
        tf = open(classPath, "r")
        self.classnames_dict = json.load(tf)
        self.mode = mode
    
    def __len__(self):
        return self.length

    def get_image(self, path):
        img = Image.open(path)
        img = img.convert("L")
        img = np.array(img)
        img = np.stack([img, img, img], axis=-1)
        img = Image.fromarray(img)
        img = preprocess(img)
        return img

    def __getitem__(self, idx):
        event_stream_path, image_path = self.files[idx].split("\t")
        label_str = event_stream_path.split("/")[-2]
        label_idx = int(self.classnames_dict[label_str])
        
        raw_data = np.fromfile(open(event_stream_path, "rb"), dtype=np.uint8)
        raw_data = np.int32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        events_stream = np.array([all_x, all_y, all_ts, all_p]).transpose()

        real_n, _ = events_stream.shape
        real_num_frame = int(real_n / self.num_events)
        N, _ = events_stream.shape
        num_frame = int(N / self.num_events)

        events_stream = self.get_events(events_stream)
        image = self.get_image(image_path[:-1])
        
        data = {
            "img": image,
            "event": events_stream,
            "label": label_idx,
        }
        return data
        
    def base_augment(self, events):
        if self.mode == "train":
            events = torch.from_numpy(events)
            # events = random_time_flip(events, resolution=(self.height, self.width))
            events = random_shift_events(events)
            events = add_correlated_events(events)
            events = np.array(events)
        return events
        
    def get_events(self, events):
        # event = self.loader(event) 
        
        if events.shape[0] > 30000:
            start = np.random.randint(0, events.shape[0] - 30000)
            events = events[start:start+30000]

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
        events = torch.clamp(events, 0, 20)
        
        events = events / (events.amax([1,2],True) + 1)
        if self.mode == "train":
            events = RandAug()(events)
        
        events = transform.Resize((224, 224))(events)
        
        return events
    
    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        random.shuffle(self.files)
        return len(self.files)


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


def random_shift_events(event_tensor, max_shift=20, resolution=(224, 224)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    event_tensor[:, 0] += x_shift
    event_tensor[:, 1] += y_shift

    valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
    event_tensor = event_tensor[valid_events]

    return event_tensor