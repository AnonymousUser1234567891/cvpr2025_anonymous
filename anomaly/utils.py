import cv2
import numpy as np
import torch

import clip
from sklearn.metrics import roc_auc_score


def auroc(preds, gt):
    preds = torch.sigmoid(preds).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    return roc_auc_score(gt, preds)


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