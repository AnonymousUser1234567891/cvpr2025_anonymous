import argparse
from tqdm import tqdm

import torch
from utils import load_weights, processing

import sys
sys.path.append('..')
from data.prepare import NCaltech101DataModule

import clip
from data.caltech_dataset import labels as label_dict
label_dict = {
    v: k for k, v in label_dict.items()
}


def load_data_module(args, num_process):
    return NCaltech101DataModule(
            args.train_txt,
            args.val_txt,
            args.classPath,
            args.batch_size,
            num_events=args.num_events,
            median_length=args.median_length)


def load_dataloader(args):
    data_module = load_data_module(args, num_process=1)
    data_module.setup('validate')
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    return train_dataloader, val_dataloader


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
    event_model, _ = clip.load(args.backbone, device=event_device, jit=False)
    event_model = load_weights(event_model, args.ckpt_path)
    event_model.eval()
    event_model = event_model.float()

    image_text_model, _ = clip.load(args.backbone)
    image_text_model = image_text_model.float().to(event_device)
    image_text_model.eval()
    _, val_loader = load_dataloader(args)

    gt, preds = [], []
    for batch in tqdm(val_loader):
        img = batch['img']
        event = batch['event']
        event = processing(event)

        text_list = [label_dict[int(i)] for i in batch['label']]
        text_list = torch.cat(
            [
                clip.tokenize(f"a point cloud image of a {text}")
                    for text in text_list
            ]
        )
        with torch.no_grad():
            event_embedding = event_model.encode_image(event.to(event_device))
            image_embedding = image_text_model.encode_image(img.to(event_device))
            _ = image_text_model.encode_text(text_list.to(event_device)).float()
   
        retrieval_pred = retrieve_similar_images(image_embedding, event_embedding, top_k=10)
        preds.extend(torch.tensor(retrieval_pred))
        gt.extend(torch.tensor([idx for idx in range(len(batch['label']))]))
        
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/vitl.pt')
    parser.add_argument('--backbone', type=str, default='ViT-L/14')
    parser.add_argument('--dataset', type=str, default='N-caltech')
    parser.add_argument('--ft', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()
    if args.dataset == 'N-caltech':
        args.train_txt = 'data/Caltech/Caltech101_train.txt'
        args.val_txt = 'data/Caltech/Caltech101_val.txt'
        args.classPath = 'data/Caltech/Caltech101_classnames.json'
        args.num_events = 100
        args.median_length = 100
    main(args)
