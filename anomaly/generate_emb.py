import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import clip
import pickle

from loader import UCFCrimeDataset, XD_dataset
from utils import generate_event_image


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


def load_dataset(args):
    if args.dataset == 'UCFCrime':
        dataset = UCFCrimeDataset(root_dir=args.root_dir)
    elif args.dataset == 'XD':
        dataset = XD_dataset(root_dir=args.root_dir)
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
    model, preprocess = load_model(args)
    model.eval()
    
    save_index = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            frames = batch[0].squeeze()
            label = batch[1][0]
            frame_label = batch[2][0]

            for idx in range(0, len(frames) - args.stack_size, args.stack_size):
                event = generate_event_image(
                    frames[idx: idx + args.stack_size],
                    threshold=args.threshold
                )
                event = torch.clamp(event, 0, args.clamp)
                event = event / event.max()
                event = torch.stack([event, event, event])
                event = transform(event)
                event = event.cuda().unsqueeze(0)

                image_features = model.encode_image(event)
                embedding = image_features.cpu().squeeze()
                lbl = frame_label[idx: idx + args.stack_size].mean()
                
                if torch.isnan(embedding).any():
                    continue
                else:
                    output_path = os.path.join(args.output_dir, f"{save_index}.pkl")
                    save_index += 1
                    with open(output_path, 'wb') as f:
                        pickle.dump(embedding, f)
                        pickle.dump(lbl, f)
                    print(f"Saved embeddings for video {label} to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--stack_size', type=int, default=16)
    parser.add_argument('--threshold', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default='./embeddings')
    parser.add_argument('--clamp', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='UCFCrime', choices=['UCFCrime', 'XD', 'shang'])

    args = parser.parse_args()
    if args.dataset == 'UCFCrime':
        args.root_dir = '/mnt/Data_3/UCFCrime_raw'
    elif args.dataset == 'XD':
        args.root_dir = '/mnt/Data_3/xdviolence_raw'

    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset, 
        f'threshold_{args.threshold}_clamp_{args.clamp}_stack_{args.stack_size}'
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(args)
    main(args)