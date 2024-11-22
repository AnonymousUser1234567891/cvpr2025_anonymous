import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('..')
import os
import clip
from PIL import Image
import numpy as np
import cv2


def load_weights(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    state_dict = {
        k.replace('encoder_k.', ''): v 
            for k, v in state_dict['checkpoint'].items()
                if 'encoder_k' in k
    }
    model.load_state_dict(state_dict)
    return model    


def processing(events):
    events = events - 0.5    
    events *= 2
    events = events.mean(1)
    events = torch.stack([events, events, events], dim=1)
    return events