# Expanding Event Modality Applications through a Robust CLIP-Based Encoder


# Data Prepare
## N-ImageNet

    Path
    |---ImageNet
    |     |--- train
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
    |     |--- val
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
    |---N_ImageNet
    |     |--- extracted_train
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
    |     |--- extracted_val
    |     |      |--- n01843383
    |     |      |--- n02281406
    |     |      |--- n02814860
    |     |      |--- ....
 
Check the data/prepare.py to put the right path for dataset.

configs/ .txt files should be changed.

## N-Caltech
    Path
    |---Caltech-101
    |     |--- accordion
    |     |--- butterfly
    |     |--- ....
    |---N_Caltech101
    |     |--- accordion
    |     |--- butterfly
    |     |--- ....

Check the data/Caltech/ .txt files should be changed.
 
## N-MNIST
    Path
    |---MNIST
    |     |--- Train
    |     |      |--- 0
    |     |      |--- 1
    |     |      |--- 2
    |     |      |--- ....
    |     |--- Test
    |     |      |--- 0
    |     |      |--- 1
    |     |      |--- 2
    |     |      |--- ....
    |---N_MNIST
    |     |--- Train
    |     |      |--- 0
    |     |      |--- 1
    |     |      |--- 2
    |     |      |--- ....
    |     |--- Test
    |     |      |--- 0
    |     |      |--- 1
    |     |      |--- 2
    |     |      |--- ....
 
Check the data/MNIST/ .txt files should be changed.

 
# Pre training

- Dataset in [N-imagenet, N-imagenet-1000]
- foundation in [ViT-B/32, ViT-L/14]

    python main.py --dataset "Dataset" --foundation "foundation"

# FineTuning

- Dataset in [N-imagenet, N-imagenet-1000, N-caltech, N-mnist]
- foundation in [ViT-B/32, ViT-L/14]
- ckpt_path "Pre-training .pt file"
- test_mode : for evaluating only
- ft in [1-shot, 2-shot, 5-shot, all]

    python finetune.py --dataset "Dataset" --foundation "foundation" --ckpt_path "ckpt_path" -- ft "ft"

# Event Video Anomaly Detection
    [Model Directory](https://github.com/AnonymousUser1234567891/cvpr2025_anonymous/tree/main/model)
    
# Event

# VIS
