import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import argparse
from tqdm import tqdm

import clip

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from loader import UCFCrimeDataset, XD_dataset, shang_dataset
from utils import generate_event_image
from sklearn.metrics import roc_auc_score

import pandas as pd


def load_model(args):
    if args.model == 'ViT-B/32':
        model, preprocess = clip.load('ViT-B/32')

        state_dict = torch.load('ViT-B.pt')['checkpoint']
        new_state_dict = {}
        for key in state_dict.keys():
            if 'encoder_k' in key:
                new_state_dict[key.replace('encoder_k.', '')] = state_dict[key]
        model.load_state_dict(new_state_dict)
        print(f'Model loaded {args.model}')
    elif args.model == 'ViT-L/14':
        model, preprocess = clip.load('ViT-L/14')
        
        state_dict = torch.load('ViT-L.pt')['checkpoint']
        new_state_dict = {}
        for key in state_dict.keys():
            if 'encoder_k' in key:
                new_state_dict[key.replace('encoder_k.', '')] = state_dict[key]
        model.load_state_dict(new_state_dict)
        print(f'Model loaded {args.model}')
    else:
        raise ValueError('Invalid model name')
    return model, preprocess


def auroc(preds, gt):
    preds = torch.sigmoid(preds).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    return roc_auc_score(gt, preds)


def load_dataset(args):
    if args.dataset == 'UCFCrime':
        dataset = UCFCrimeDataset(root_dir=args.root_dir)
    elif args.dataset == 'XD':
        dataset = XD_dataset(root_dir=args.root_dir)
    elif args.dataset == 'shang':
        dataset = shang_dataset(root_dir=args.root_dir)
    else:
        raise ValueError('Invalid dataset')
    return dataset


def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    dataset = load_dataset(args)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False
    )
    model, _ = load_model(args)
    classes = [
        # Abnormal Classes
        'Abuse', 'Arson', 'Burglary', 'Fighting', 
        'RoadAccidents', 'Shooting', 'Stealing', 'Arrest',  
        'Assault',  'Explosion', 'Robbery',        
        'Shoplifting', 'Vandalism', 
        # Normal Classes
        "Peace", "Calm", "Quiet", "Normalcy", 
        "Routine", "Stability", "Tranquility", "Serenity", 
        "Nothing", "Order", "Normal", "Nothing",
        "Safe", "Slience"
    ]
    len_ab = 13
    if args.dataset == 'shang':
        classes = [
            # Abnormal Classes
            "chasing", "push", "monocycle", "throwing_object", "vaudeville",
            "fighting", "car", "running", "stoop", "robbery",
            "vehicle", "skateboard", "jumping", "fall", "circuit",
            # Normal Classes
            "street", "walking", "waiting", "standing", "crossing",
            "sitting", "gathering", "office", "shopping", "family",
            "commuting"
        ]
        len_ab = 15
    text = clip.tokenize(
        [f"a photo of {cls}" for cls in classes]
    ).cuda()
    print(f'Lenght of dataset: {len(dataset)}')

    preds, gt = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            predictions, labels = [], []
            
            frames = batch[0].squeeze()
            frame_label = batch[2][0]

            for idx in range(0, len(frames)-args.stack_size, args.stack_size):
                event = generate_event_image(
                    frames[idx : idx + args.stack_size], 
                    threshold=args.threshold
                )
                event = event / event.max()
                event = torch.stack([event, event, event])
                event = transform(event)
                event = event.cuda().unsqueeze(0)
                
                logits_per_image, _ = model(event, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                predictions.append(probs[0][:len_ab].sum())
                lbl = frame_label[idx : idx + args.stack_size].mean()
                lbl = 1 if lbl > 0.5 else 0
                labels.append(lbl)
            predictions = torch.tensor(predictions)
            labels = torch.tensor(labels)
            mask = ~torch.isnan(predictions)
            predictions = predictions[mask]
            labels = labels[mask]
            preds.append(predictions)
            gt.append(labels)
    preds = torch.cat(preds)
    gt = torch.cat(gt)
    print(f'Final AUROC: {auroc(preds, gt)}')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--root_dir', type=str, default='/mnt/Data_3/UCFCrime_raw')
    parser.add_argument('--stack_size', type=int, default=16)
    parser.add_argument('--threshold', type=int, default=25)
    parser.add_argument('--clamp', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='UCFCrime', choices=['UCFCrime', 'XD', 'shang'])

    args = parser.parse_args()

    if args.model == 'ViT-B/32':
        args.ckpt = '../checkpoints/vitb.pt'
    elif args.model == 'ViT-L/14':
        args.ckpt = '../checkpoints/vitl.pt'
    
    if args.dataset == 'UCFCrime':
        args.root_dir = '/mnt/Data_3/UCFCrime_raw'
    elif args.dataset == 'XD':
        args.root_dir = '/mnt/Data_3/xdviolence_raw'
    elif args.dataset == 'shang':
        args.root_dir = '/mnt/Data_3/shang_raw'
    print(args)
    main(args)