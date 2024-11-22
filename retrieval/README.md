# Data Prepare

## ESC-50 (audio)

    ESC-50-master
    |---meta
    |     |---esc50.csv
    |---audio
    |     |--- audio1.wav
    |     |--- audio2.wav
    |     |--- audio3.wav
    |     |--- ....
 

## DENSE (dense)

    dense
    |---train
    |   |---0
    |   |   |---depth
    |   |   |     |---data
    |   |   |     |---frames
    |   |   |---event
    |   |   |     |--- data
    |   |   |     |--- frames_white
    |   |   |---rgb
    |   |   |     |--- frames
    |   |---1
    |   |---2
    |   |---...
    |---test
    |   |---0
    |   |   |---depth
    |   |   |     |---data
    |   |   |     |---frames
    |   |   |---event
    |   |   |     |--- data
    |   |   |     |--- frames_white
    |   |   |---rgb
    |   |   |     |--- frames
    |   |---1
    |   |---2
    |   |---...


## Evaluation

prepare imagenet ckpt and put 

.checkpoints/imagebind_hugh.pth

put bpe folder from imagebind

    # for text
    python event_image_text.py --batch_size "batch size" --backbone "backbone" --ckpt_path "ckpt path"

    # for sound
    python event_image_sound.py

    # for depth
    python evebt_depth_image.py
