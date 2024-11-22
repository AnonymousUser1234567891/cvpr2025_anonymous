import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms as transforms
import os
from .augmentation import get_augmentation, RandAug
from functools import partial

# import clip
# import model.clip as clip
import clip

from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2

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
        # img = img.convert("L")
        # img = np.array(img)
        # img = np.stack([img, img, img], axis=-1)
        # img = Image.fromarray(img)
        # img = preprocess(img)
        img = transforms.ToTensor()(img)
        img = transforms.Resize((224, 224))(img)
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
        # events_stream, pad_flag = self.pad_event_stream(
            # events_stream, median_length=self.median_length)
        N, _ = events_stream.shape
        num_frame = int(N / self.num_events)

#  ---------------------------------------------------
        events_stream = self.get_events(events_stream)
        image = self.get_image(image_path[:-1])
        
        data = {
            "img": image,
            "event": events_stream,
            "emb": -1,
            "label": label_idx,
        }
        return data
#  ---------------------------------------------------
        
        all_frame = []
        for i in range(num_frame):
            events_tmp = events_stream[i*self.num_events : (i + 1)*self.num_events, :]

            events_image = self.generate_event_image(
                events_tmp, 
                (self.height, self.width),
                self.representation)
            all_frame.append(events_image)

        if self.augmentation and random.random() > 0.5:
                # print("flip along x")
                all_frame = [cv2.flip(all_frame[i], 1) for i in range(len(all_frame))]
                # image = cv2.flip(image, 1)
        all_frame = np.array(all_frame)
        events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W
        
        events_data = events_data.squeeze()
        # events_data = np.mean(events_data, axis=1)
        # events_data = events_data[:, 0]

        # plt.imshow(
        #     events_data.transpose(1, 2, 0), 
        #     vmin=events_data.min(), 
        #     vmax=events_data.max())
        # plt.title(f"Label: {label_str}")
        # plt.savefig(f"events_data.png")
        # plt.clf()
        # import pdb; pdb.set_trace()

        image = self.get_image(image_path[:-1])
        data = {
            "img": image,
            "event": events_data,
            "emb": -1,
            "label": label_idx,
        }
        return data
#----------------------------------------------------------------
       
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
        # events = self.base_augment(events)

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
        events = transforms.Resize((224, 224))(events)
        # if self.mode == "train":
            # events = RandAug()(events)
        
        return events

    def generate_event_image(self, events, shape, representation):
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = x.max() + 1
        h_event = y.max() + 1
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + w_event * y[p == 0], 1)

        gray_scale = 1 - (img_pos.reshape((h_event, w_event, 1)) + img_neg.reshape((h_event, w_event, 1))) * [127,127,127] / 255
        gray_scale = np.clip(gray_scale, 0, 255)

        # scale
        scale = H * 1.0 / h_event
        scale2 = W * 1.0 / w_event
        gray_scale = cv2.resize(gray_scale, dsize=None, fx=scale2, fy=scale)
        return gray_scale
    
    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        random.shuffle(self.files)
        return len(self.files)

    def pad_event_stream(self, event_stream, median_length = 104815):
        """
        pad event stream along n dim with 0
        so that event streams in one batch have the same dimension
        """
        # max_length = 428595
        pad_flag = False
        (N, _) = event_stream.shape
        if N < median_length:
            n = median_length - N
            pad = np.ones((n, 4))
            event_stream = np.concatenate((event_stream, pad), axis=0)
            pad_flag = True
        else:
            event_stream = event_stream[:median_length, :]
        return event_stream, pad_flag

    def generate_event_image_EventBind(self, events, shape, representation):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}.
        x and y correspond to image coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = x.max() + 1
        h_event = y.max() + 1
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + w_event * y[p == 0], 1)
        # if representation == 'rgb':
            # gray_scale = 1 - (img_pos.reshape((h_event, w_event, 1))* [0, 255, 255] + img_neg.reshape((h_event, w_event, 1)) * [255,255,0]) / 255
        # elif representation == 'gray_scale':
        
        gray_scale = 1 - (img_pos.reshape((h_event, w_event, 1)) + img_neg.reshape((h_event, w_event, 1))) * [127,127,127] / 255
        gray_scale = np.clip(gray_scale, 0, 255)

        # scale
        scale = H * 1.0 / h_event
        scale2 = W * 1.0 / w_event
        gray_scale = cv2.resize(gray_scale, dsize=None, fx=scale2, fy=scale)
        return gray_scale

#     def __getitem__(self, idx):
#         event_stream_path, image_path = self.files[idx].split("\t")
#         label_str = event_stream_path.split("/")[-2]
#         label_idx = int(self.classnames_dict[label_str])
        
#         raw_data = np.fromfile(open(event_stream_path, "rb"), dtype=np.uint8)
#         raw_data = np.int32(raw_data)
#         all_y = raw_data[1::5]
#         all_x = raw_data[0::5]
#         all_p = (raw_data[2::5] & 128) >> 7
#         all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
#         events_stream = np.array([all_x, all_y, all_ts, all_p]).transpose()

#         real_n, _ = events_stream.shape
#         real_num_frame = int(real_n / self.num_events)
#         # events_stream, pad_flag = self.pad_event_stream(
#             # events_stream, median_length=self.median_length)
#         N, _ = events_stream.shape
#         num_frame = int(N / self.num_events)

#         # import pickle
#         # with open(f"samples/events_stream_{idx}.pkl", "wb") as f:
#         #     pickle.dump(events_stream, f)

#         # events_stream = self.get_events(events_stream)
#         # image = self.get_image(image_path[:-1])
        
#         # data = {
#         #     "img": image,
#         #     "event": events_stream,
#         #     "emb": -1,
#         #     "label": label_idx,
#         # }
#         # return data
# #  ---------------------------------------------------
        
#         all_frame = []
#         for i in range(num_frame):
#             events_tmp = events_stream[i*self.num_events : (i + 1)*self.num_events, :]

#             events_image = self.generate_event_image(
#                 events_tmp, 
#                 (self.height, self.width),
#                 self.representation)
#             all_frame.append(events_image)

#         if self.augmentation and random.random() > 0.5:
#                 # print("flip along x")
#                 all_frame = [cv2.flip(all_frame[i], 1) for i in range(len(all_frame))]
#                 # image = cv2.flip(image, 1)
#         all_frame = np.array(all_frame)
#         events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W
        
#         events_data = events_data.squeeze()
#         # events_data = np.mean(events_data, axis=1)
#         # events_data = events_data[:, 0]

#         # plt.imshow(
#         #     events_data.transpose(1, 2, 0), 
#         #     vmin=events_data.min(), 
#         #     vmax=events_data.max())
#         # plt.title(f"Label: {label_str}")
#         # plt.savefig(f"events_data.png")
#         # plt.clf()
#         # import pdb; pdb.set_trace()

#         image = self.get_image(image_path[:-1])
#         data = {
#             "img": image,
#             "event": events_data,
#             "emb": -1,
#             "label": label_idx,
#         }
#         return data


# def random_time_flip(event_tensor, resolution=(224, 224), p=0.5):
#     if np.random.random() < p:
#         event_tensor = torch.flip(event_tensor, [0])
#         event_tensor[:, 2] = event_tensor[0, 2] - event_tensor[:, 2]
#         event_tensor[:, 3] = - event_tensor[:, 3]  # Inversion in time means inversion in polarity
#     return event_tensor


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