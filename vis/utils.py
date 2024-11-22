import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.prepare import Data_prepare, NCaltech101DataModule, NMNISTDataModule
import sys
sys.path.append('..')
import os
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2

from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def load_weights(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    state_dict = {
        k.replace('encoder_k.', ''): v 
            for k, v in state_dict['checkpoint'].items()
                if 'encoder_k' in k
    }
    model.load_state_dict(state_dict)
    return model    


def load_data_module(args, num_process):
    if args.dataset == 'N-imagenet':
        return Data_prepare(args.dataset, num_process, ft=args.ft)
    elif args.dataset == 'N-caltech':
        return NCaltech101DataModule(
            args.train_txt,
            args.val_txt,
            args.classPath,
            args.batch_size,
            num_events=args.num_events,
            median_length=args.median_length,
        )
    elif args.dataset == 'N-mnist':
        return NMNISTDataModule(
            args.train_txt,
            args.val_txt,
            args.classPath,
            args.batch_size,
            representation=args.representation,
            num_events=args.num_events,
            median_length=args.median_length,
        )
    else:
        raise ValueError('Dataset not found')


def load_dataloader(args):
    data_module = load_data_module(args, num_process=1)
    data_module.setup('validate')
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    return train_dataloader, val_dataloader


def processing(events):
    events = events - 0.5    
    events *= 2
    events = events.mean(1)
    events = torch.stack([events, events, events], dim=1)
    return events


from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


start_layer = -1
start_layer_text = -1


def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cpu() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
        start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance

def save_image_relevance(image_relevance, image, orig_image, save_path="image_relevance.png"):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.subplot(1, 3, 3)
    plt.imshow(vis)
    plt.axis('off')
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def save_text_relevance(text, text_encoding, R_text, save_path="text_relevance.png"):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten().cpu().numpy()
    
    # 토큰화된 텍스트 가져오기
    text_tokens = _tokenizer.encode(text)
    text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    
    # 시각화 플롯 생성
    # plt.figure(figsize=(12, 1))
    bars = plt.barh(text_tokens_decoded, text_scores, color="skyblue")

    # plt.barh(text_tokens_decoded, text_scores, color="skyblue")
    plt.gca().invert_yaxis()  # 텍스트가 상단에 오도록 순서 반전
    plt.xlabel("Relevance Score")
    plt.title("Text Relevance")
    plt.tight_layout()
    for bar, score in zip(bars, text_scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{score:.2f}', 
                 ha='left', va='center', fontsize=8)
    
    # 이미지로 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()