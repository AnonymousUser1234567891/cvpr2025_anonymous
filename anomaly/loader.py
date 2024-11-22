import torch
import pickle
from torch.utils.data import Dataset

import os
import cv2
import numpy as np


class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.video_list = {}
        with open(os.path.join(root_dir, 'Temporal_Anomaly_Annotation.txt')) as f:
            labels = f.readlines()
            for label in labels:
                name, cls, st, en, st2, en2 = label.split()
                self.video_list[name] = [cls, st, en, st2, en2]
        self.update_video_list(root_dir)    
        self.transform = transform
    
    def update_video_list(self, root_dir):
        update_lists = []
        root_dir = os.path.join(root_dir, 'videos')
        
        for label in os.listdir(root_dir):
            for video in os.listdir(os.path.join(root_dir, label)):
                if video in self.video_list.keys():
                    update_list = []
                    update_list.append(os.path.join(root_dir, label, video))
                    update_list.append(self.video_list[video])            
                    update_lists.append(update_list)
        self.video_list = update_lists
    
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, labels = self.video_list[idx]
        label, st, en, st2, en2 = labels
        
        frames = self.load_video(video_path)
        # if self.transform:
            # frames = self.transform(frames)
        frame_label = torch.zeros(len(frames))
        frame_label[int(st):int(en)] = 1
        if int(st2) != -1:
            frame_label[int(st2):int(en2)] = 1

        # print(f"start: {st}, end: {en}")
        # print(f"start2: {st2}, end2: {en2}")
        return np.array(frames), label, frame_label

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

        
class UCFCrime_embedding(Dataset):
    def __init__(self, root_dir):
        self.root_dir = [
            os.path.join(root_dir, file) for file in os.listdir(root_dir)
        ]
        
    def __len__(self):
        return len(self.root_dir)
    
    def __getitem__(self, idx):
        path = self.root_dir[idx]
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
            lbl = pickle.load(f)
        return embeddings, lbl


class XD_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.join(root_dir, 'videos')
        self.video_list = []
        with open(os.path.join(root_dir, 'labels.txt')) as f:
            labels = f.readlines()
            for label in labels:
                line = label.split()
                self.video_list.append([line[0]+'.mp4'] + [int(vl) for vl in line[1:]])
        self.transform = transform
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        line = self.video_list[idx]
        video_path = line[0]
        rang = line[1:]
        
        frames = self.load_video(os.path.join(self.root_dir, video_path))
        labels = self.create_label(len(frames), rang)
        
        return np.array(frames), 'None', labels
    
    def create_label(self, length, rang):
        labels = torch.zeros(length)
        for i in range(0, len(rang), 2):
            labels[rang[i]:rang[i+1]] = 1
        return labels
    
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames    
    
        
class XD_embedding(Dataset):
    def __init__(self, root_dir):
        self.root_dir = [
            os.path.join(root_dir, file) for file in os.listdir(root_dir)
        ]
    
    def __len__(self):
        return len(self.root_dir)
    
    def __getitem__(self, idx):
        path = self.root_dir[idx]
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
            lbl = pickle.load(f)
        return embeddings, lbl
    
    
class shang_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = os.path.join(root_dir, 'frames')
        self.video_list = []
        with open(os.path.join(root_dir, 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt')) as f:
            labels = f.readlines()
            for label in labels:
                line = label.split()
                self.video_list.append([line[0]] + [line[1]] + [int(vl) for vl in line[2:]])
        self.video_list = self.update_video_list(root_dir)

    def __len__(self):
        return len(self.video_list)
    
    def update_video_list(self, root_dir):
        update_lists = []
        root_dir = os.path.join(root_dir, 'frames')
        
        for video_name in self.video_list:
            if video_name[0].split('.')[0] in os.listdir(root_dir):
                update_lists.append(video_name)
        return update_lists
                
    
    def __getitem__(self, idx):
        line = self.video_list[idx]
        video_path = line[0]
        class_name = line[1]
        rang = line[2:]
        
        frames = self.load_video(os.path.join(self.root_dir, video_path))
        labels = self.create_label(len(frames), rang)
        
        return np.array(frames), class_name, labels
    
    def create_label(self, length, rang):
        labels = torch.zeros(length)
        for i in range(0, len(rang), 2):
            labels[rang[i]:rang[i+1]] = 1
        return labels
    
    def load_video(self, video_path):
        video_path = video_path.split('.')[0]
        
        frames = []
        for frame in sorted(os.listdir(video_path)):
            frame = cv2.imread(os.path.join(video_path, frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames
    
    
class shang_embedding(Dataset):
    def __init__(self, root_dir):
        self.root_dir = [
            os.path.join(root_dir, file) for file in os.listdir(root_dir)
        ]
    
    def __len__(self):
        return len(self.root_dir)
    
    def __getitem__(self, idx):
        path = self.root_dir[idx]
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)
            lbl = pickle.load(f)
        return embeddings, lbl