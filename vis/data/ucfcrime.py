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

        
def generate_event_image(frames, threshold=2):
    frames = np.array(frames)  
    num_frames, height, width, _ = frames.shape
    event_images = []
    
    for i in range(1, num_frames):
        diff = cv2.absdiff(frames[i], frames[i-1])
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, event_image = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        event_images.append(event_image)

    return torch.tensor(event_images).sum(dim=0)