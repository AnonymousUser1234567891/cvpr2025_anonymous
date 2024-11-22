import os
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import cv2
import pickle

import pandas as pd

import torch
import torch.nn.functional as F
from models.helpers import Normalize, LearnableLogitScaling

from utils import load_weights, processing
from PIL import Image

from models import imagebind_model
from models.imagebind_model import ModalityType
from models import data as imagebinddata
import torch.nn as nn

import sys
sys.path.append('..')

import clip


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query, key):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        logits = torch.matmul(query, key.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

        
class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split=64):
        self.transform = transform  

        self.rgb_path = []
        self.depth_path = []
        self.event_path = []

        start = 0
        for path in os.listdir(root_dir):
            rgb_dir = os.path.join(root_dir, path, 'rgb/frames')
            depth_dir = os.path.join(root_dir, path, 'depth/frames')
            event_dir = os.path.join(root_dir, path, 'events/frames_white')
                    
            for rgb_file in sorted(os.listdir(rgb_dir)):
                if not rgb_file.endswith('.png'):
                    continue
                else:
                    if start % split == 0:
                        rgb_path = os.path.join(rgb_dir, rgb_file)
                        
                        depth_path = os.path.join(depth_dir, rgb_file)
                
                        event_file = rgb_file.replace('frame', 'events')
                        event_path = os.path.join(event_dir, event_file)

                        self.rgb_path.append(rgb_path)
                        self.depth_path.append(depth_path)
                        self.event_path.append(event_path)        
                    start += 1

    def __getitem__(self, idx):
        rgb = self.rgb_path[idx]
        depth = self.depth_path[idx]
        if self.transform:
            depth = Image.open(depth)
            depth = self.transform(depth)
            
        event = Image.open(self.event_path[idx])
        event = np.array(event)
        return rgb, depth, event
    
    def __len__(self):
        return len(self.rgb_path)


def retrieve_similar_images(event_embedding, image_embedding, top_k=1):
    event_embedding = torch.nn.functional.normalize(event_embedding, dim=1)
    image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)

    similarity_matrix = torch.mm(event_embedding, image_embedding.T)  # Shape (N, M)
    top_k_indices = torch.topk(similarity_matrix, top_k, dim=1).indices  # Only the indices needed
    predictions = top_k_indices.tolist()

    return predictions


def calculate_precision_recall_at_k(prediction, gt, k_values=[1, 5, 10]):
    precision_at_k = {k: 0.0 for k in k_values}
    recall_at_k = {k: 0.0 for k in k_values}

    num_samples = prediction.size(0)

    for i in range(num_samples):
        true_label = gt[i].item()  # ground truth index for this sample
        pred_indices = prediction[i]  # top-10 predicted indices for this sample

        for k in k_values:
            top_k_pred = pred_indices[:k]
            relevant_retrieved = 1 if true_label in top_k_pred else 0
            precision_at_k[k] += relevant_retrieved / k
            recall_at_k[k] += relevant_retrieved  # Since we have only one relevant item per sample

    precision_at_k = {k: precision_at_k[k] / num_samples for k in k_values}
    recall_at_k = {k: recall_at_k[k] / num_samples for k in k_values}

    return {"precision": precision_at_k, "recall": recall_at_k}


def calculate_mean_reciprocal_rank(predictions, gt):
    reciprocal_ranks = []

    for i in range(len(predictions)):
        true_label = gt[i].item()
        pred_indices = predictions[i].tolist()

        if true_label in pred_indices:
            rank = pred_indices.index(true_label) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr


def calculate_mean_average_precision_at_k(predictions, gt, k=10):
    average_precisions = []
    for i in range(len(predictions)):
        true_label = gt[i]
        pred_indices = predictions[i][:k]  

        num_relevant = 0
        precision_sum = 0.0
        for j in range(len(pred_indices)):
            if pred_indices[j] == true_label:
                num_relevant += 1
                precision_sum += num_relevant / (j + 1)
        if num_relevant > 0:
            average_precision = precision_sum / num_relevant
        else:
            average_precision = 0.0
        average_precisions.append(average_precision)
    map_k = sum(average_precisions) / len(average_precisions)
    return map_k


def main(args):
    event_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    event_model, preprocess = clip.load(args.backbone, device=event_device, jit=False)
    event_model = load_weights(event_model, args.ckpt_path)
    event_model.eval()
    event_model = event_model.float()
    
    imagebind = imagebind_model.imagebind_huge(pretrained=True)
    imagebind.eval()
    imagebind.to(event_device)

    adapter = nn.Sequential(
        nn.Linear(768, 1024),
        Normalize(dim=-1),
        LearnableLogitScaling(logit_scale_init=5.0, learnable=True),   
    ).to(event_device)
    adapter.to(event_device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    criterion = InfoNCELoss()

    depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, ], [0.5, ]
        )
    ])
    event_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    depth_traindataset = DepthDataset(args.train_dir, transform=depth_transform)
    depth_valdataset = DepthDataset(args.val_dir, transform=depth_transform)

    train_loader = torch.utils.data.DataLoader(
        depth_traindataset,
        batch_size=20, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        depth_valdataset, 
        batch_size=20, 
        shuffle=True,
        drop_last=True
    )
    
    for epoch in range(args.epochs):
        for rgb, depth, event in tqdm(train_loader):
            inputs = {
                ModalityType.VISION: imagebinddata.load_and_transform_vision_data(rgb, event_device),
                ModalityType.DEPTH: depth.to(event_device),
            }
            event = (1 - event.float().mean(-1)/255.)
            event = event.to(event_device)
            event = event_transform(event)
            event = torch.stack([event, event, event], dim=1)

            with torch.no_grad():
                embeddings = imagebind(inputs)
                event_embedding = event_model.encode_image(event)
            
            event_embedding = adapter(event_embedding)
            loss = criterion(embeddings[ModalityType.VISION], event_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds, gt = [], []
            for rgb, depth, event in val_loader:
                inputs = {
                    ModalityType.VISION: imagebinddata.load_and_transform_vision_data(rgb, event_device),
                    ModalityType.DEPTH: depth.to(event_device),
                }
                event = (1 - event.float().mean(-1)/255.)
                event = event.to(event_device)
                event = event_transform(event)
                event = torch.stack([event, event, event], dim=1)

                embeddings = imagebind(inputs)
                event_embedding = event_model.encode_image(event)
                event_embedding = adapter(event_embedding)

                retrieval_pred = retrieve_similar_images(
                    embeddings[ModalityType.DEPTH], 
                    event_embedding, 
                    top_k=10
                )
                preds.extend(torch.tensor(retrieval_pred))
                gt.extend(torch.tensor([idx for idx in range(len(event))]))

            preds = torch.stack(preds).cpu()
            gt = torch.stack(gt).cpu()
            results = calculate_precision_recall_at_k(preds, gt, k_values=[1, 5, 10])

            for k, v in results['recall'].items():
                print(f'Recall@{k}: {v*100:.2f}')
                
            mrr = calculate_mean_reciprocal_rank(preds, gt)
            print(f'MRR: {mrr*100:.2f}')
            mAP = calculate_mean_average_precision_at_k(preds, gt, k=1)
            print(f'mAP@1: {mAP*100:.2f}')
            mAP = calculate_mean_average_precision_at_k(preds, gt, k=5)
            print(f'mAP@5: {mAP*100:.2f}')
            mAP = calculate_mean_average_precision_at_k(preds, gt, k=10)
            print(f'mAP@10: {mAP*100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/vitl.pt')
    parser.add_argument('--backbone', type=str, default='ViT-L/14')
    parser.add_argument('--train_dir', type=str, default='dense/train')
    parser.add_argument('--val_dir', type=str, default='dense/test')
    parser.add_argument('--lr', type=float, default=1e-5)  
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
        
    main(args)
