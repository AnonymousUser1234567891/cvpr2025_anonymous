import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from utils import load_weights, load_dataloader, processing

import sys
sys.path.append('..')
import CLIP.clip as clip

from utils import interpret, save_text_relevance, save_image_relevance


def main(args):
    device = 'cpu'
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = load_weights(model, args.ckpt_path)
    train_loader, val_loader = load_dataloader(args)
    
    texts = ["Peacock standing on green grass with chicks"]
    text = clip.tokenize(texts).to(device)
    
    if args.dataset == 'N-imagenet':
        pass
    elif args.dataset == 'N-caltech':
        train_loader.dataset.files = sorted(train_loader.dataset.files)
        val_loader.dataset.files = sorted(val_loader.dataset.files)
        
    for idx in tqdm(range(len(val_loader.dataset))):
        batch = val_loader.dataset[idx]
        events = batch['event'].unsqueeze(0)
        events_raw = batch['event'].unsqueeze(0)
        events_raw = events_raw.squeeze(0).mean(0)
        img = batch['img'].unsqueeze(0)
        events = processing(events)

        R_text, R_image = interpret(
            model=model, 
            image=events, 
            texts=text, 
            device=device
        )

        batch_size = text.shape[0]
        for i in range(batch_size):
            save_text_relevance(
                texts[i], 
                text[i], 
                R_text[i], 
                save_path=f"text_relevance.png"
            )
            save_image_relevance(
                R_image[i], 
                events, 
                orig_image=None, 
                save_path=f"image_relevance_{i}.png"
            )
        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(events_raw , cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0)
        plt.savefig(f"results.png", bbox_inches='tight')
        plt.clf()
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/vitb.pt')
    parser.add_argument('--image', type=str, help='Path to the image')
    parser.add_argument('--dataset', type=str, default='N-imagenet', choices=['N-imagenet', 'N-caltech'])
    parser.add_argument('--save_path', type=str, default='visualization')
    parser.add_argument('--ft', type=str, default='all')

    args = parser.parse_args()
    if args.dataset == 'N-caltech':
        args.train_txt = 'data/Caltech/Caltech101_train.txt'
        args.val_txt = 'data/Caltech/Caltech101_val.txt'
        args.classPath = 'data/Caltech/Caltech101_classnames.json'
        args.batch_size = 1
        args.num_events = 200000
        args.median_length = 200000

    args.save_path = os.path.join(args.save_path, args.dataset)
    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)
        
    main(args)
