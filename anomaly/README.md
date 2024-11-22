# Data Prepare

## UCFCrime
    Path
    |---videos
    |     |--- Abuse
    |     |--- Arson
    |     |--- ....
    |---Temporal_Anomaly_Annotation.txt

## XD-Violence
    Path
    |---videos
    |     |--- video1.mp4
    |     |--- video2.mp4
    |---labels.txt

## Shanghaitech
    Path
    |---frames
    |     |--- 01_0014
    |     |      |--- 001.jpg
    |     |      |--- 002.jpg
    |     |      |--- 003.jpg
    |     |      |--- ....
    |     |--- 01_0015
    |     |      |--- 001.jpg
    |     |      |--- 002.jpg
    |     |      |--- 003.jpg
    |     |      |--- ....
    |     |--- ...
    |---Temporal_Anomaly_Annotation_for_Testing_Videos.txt

# Evaluation

model in [ViT-B/32, ViT-L/14]

dataset in ['UCFCrime', 'XD', 'shang']

clamp in default 10

threshold int default 25

stack_size int default 16 

    python main.py --model "model" --dataset "dataset" --clamp "clamp" --threshold "threshold" --stack_size "stack size"