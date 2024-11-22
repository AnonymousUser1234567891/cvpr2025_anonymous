import os
from tqdm import tqdm
import argparse
from tqdm import tqdm
import pickle

import pandas as pd

import torch
import torch.nn.functional as F
from models.helpers import Normalize, LearnableLogitScaling

from utils import load_weights, processing

from models import imagebind_model
from models.imagebind_model import ModalityType
from models import data as imagebinddata
import torch.nn as nn

import sys
sys.path.append('..')

import clip

from data.caltech_dataset2 import labels as label_dict
from data.caltech_dataset2 import NCaltech101

label_dict = {
    v: k for k, v in label_dict.items()
}


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

        
class SoundDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.items = [
           "airplane", "helicopter", "clock_alarm", "cat", "car_horn",
            "insects", "engine", "keyboard_typing", "sheep", "dog"
        ]
        
        self.category_dict = {
            "airplane": "airplanes",
            "helicopter": "helicopter",
            "clock_alarm": "watch",
            "cat": "wild_cat",
            "car_horn": "car_side",
            "insects": ["ant", "butterfly", "dragonfly"],
            "engine": "ferry",
            "keyboard_typing": "laptop",
            "sheep": "llama",
            "dog": "dalmatian"
        }
        expanded_category_dict = {}
        for key, values in self.category_dict.items():
            if isinstance(values, list):
                for value in values:
                    expanded_category_dict[value] = value
            else:
                expanded_category_dict[key] = values
        self.data['category'] = self.data['category'].map(expanded_category_dict).fillna(self.data['category'])
        
    def __getitem__(self, idx):
        sound_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 3]
        return sound_path, label
    
    def __len__(self):
        return len(self.data)


def random_sampling(audio_dict):
    audio_embedding = []
    for key in audio_dict:
        audio_embedding.append(audio_dict[key][torch.randint(0, len(audio_dict[key]), (1,))])
    return torch.cat(audio_embedding)


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


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    event_model, preprocess = clip.load(args.backbone, device=device, jit=False)
    event_model = load_weights(event_model, args.ckpt_path)
    
    train_dataset = NCaltech101(
        txtPath = 'data/Caltech/Caltech101_train.txt',
        classPath = 'data/Caltech/Caltech101_classnames.json',
        mode = 'train'
    )
    val_dataset = NCaltech101(
        txtPath = 'data/Caltech/Caltech101_val.txt',
        classPath = 'data/Caltech/Caltech101_classnames.json',
        mode = 'val'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=20, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    
    sound_dataset = SoundDataset(
        csv_path = 'ESC-50-master/meta/esc50.csv',
        root_dir = 'ESC-50-master/audio'
    )
    sound_loader = torch.utils.data.DataLoader(
        sound_dataset, batch_size=20, shuffle=False
    )
    
    imagebind = imagebind_model.imagebind_huge(pretrained=True)
    imagebind.eval()
    imagebind.to(device)

    audio_embeddings, audio_texts = [], []
    for sound_path, labels in sound_loader:
        text_list = [f"A {i}" for i in labels]
        
        inputs = {
            ModalityType.AUDIO: imagebinddata.load_and_transform_audio_data(sound_path, device),
            ModalityType.TEXT: imagebinddata.load_and_transform_text(text_list, device),
        }
        with torch.no_grad():
            embeddings = imagebind(inputs)

        audio_embeddings.append(embeddings['audio'])
        audio_texts.append(text_list)
        
    audios = torch.cat(audio_embeddings)
    audio_texts = [text for texts in audio_texts for text in texts]
    with open('audio_embeddings.pkl', 'wb') as f:
        pickle.dump(audios, f)
        pickle.dump(audio_texts, f)

    select_key = [
        "A airplanes",
        "A helicopter",
        "A watch",
        "A wild_cat",
        "A car_side",
        "A insects",
        "A ferry",
        "A laptop",
        "A llama",
        "A dalmatian"
    ]
    
    with open('audio_embeddings.pkl', 'rb') as f:
        audios = pickle.load(f)
        audio_texts = pickle.load(f)

        audio_dict = {}
        for key in set(audio_texts):
            if key in select_key:
                audio_dict[key] = []
        
        for i, text in enumerate(audio_texts):
            if text in select_key:    
                audio_dict[text].append(audios[i])
        
        for key in audio_dict:
            audio_dict[key] = torch.stack(audio_dict[key])

    event_model.eval()
    event_model = event_model.float()
    adapter = nn.Sequential(
        nn.Linear(768, 1024),
        Normalize(dim=-1),
        LearnableLogitScaling(logit_scale_init=5.0, learnable=True),   
    ).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-5)
    criterion = InfoNCELoss()
    
    for epoch in tqdm(range(args.epochs)):
        print(f'Epoch {epoch}')
        for batch in train_loader:
            img = batch['img']
            event = batch['event']
            event = processing(event)
            labels = batch['label']

            text_list = [f"A {label_dict[int(i)]}" for i in labels]        
            
            inputs = {
                ModalityType.VISION: imagebinddata.load_and_transform_vision_data(img, device),
                ModalityType.TEXT: imagebinddata.load_and_transform_text(text_list, device),
            }
            
            with torch.no_grad():
                embeddings = imagebind(inputs)
                event_embedding = event_model.encode_image(event.to(device))
            event_embedding = adapter(event_embedding)
            
            loss = criterion(embeddings[ModalityType.VISION], event_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        preds, gt = [], []    
        with torch.no_grad():
            for batch in val_loader:
                img = batch['img']
                event = batch['event']
                event = processing(event)
                labels = batch['label']
                texts = [f"A {label_dict[int(i)]}" for i in labels]
                
                with torch.no_grad():
                    event_embedding = event_model.encode_image(event.to(device))
                event_embedding = adapter(event_embedding)

                audio_embedding = random_sampling(audio_dict)
                
                similarity_matrix = torch.mm(
                    torch.nn.functional.normalize(event_embedding, dim=1).to(device), 
                    torch.nn.functional.normalize(audio_embedding, dim=1).to(device).T
                ).squeeze()
                top_k_indices = torch.topk(similarity_matrix.unsqueeze(0), k=10, dim=1).indices
                predictions = top_k_indices.tolist()
                
                preds.append(torch.tensor(predictions))
                if texts[0] in ['A ant', 'A butterfly', 'A dragonfly']:
                    texts[0] = 'A insects'
                gt.append(list(audio_dict.keys()).index(texts[0]))
            
            preds = torch.stack(preds).cpu().squeeze()
            gt = torch.tensor(gt).cpu()
            
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
        torch.save(adapter.state_dict(), f'sound_ckpt_{epoch+1}.pt')

    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/vitl.pt')
    parser.add_argument('--backbone', type=str, default='ViT-L/14')
    parser.add_argument('--dataset', type=str, default='N-imagenet', choices=['N-imagenet', 'N-caltech'])
    parser.add_argument('--ft', type=str, default='all')
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()
        
    main(args)
