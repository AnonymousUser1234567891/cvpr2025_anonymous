import os
import json
import argparse


def load_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def extract_dict(txts, cls_list):
    dicts = {}
    for key in cls_list:
        dicts[key] = []
    for line in txts:
        cls = line.split('/')[-2]
        if cls in dicts:
            dicts[cls].append(line)
    return dicts


def extract_samples(dicts, n_samples):
    samples = []
    
    if n_samples == 'all':
        for key in dicts:
            samples += dicts[key]
        return samples    
    else:
        for key in dicts:
            samples += dicts[key][:n_samples]
    return samples


def save_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)


def check_n_classes(txts):
    classes = []
    for line in txts:
        cls = line.split('/')[-2]
        if cls not in classes:
            classes.append(cls)
    print(len(classes))


def main(args):
    with open(args.cls_list, 'r') as f:
        cls_list = json.load(f)
        train_classes = list(cls_list.keys())
        val_classes = list(cls_list.keys())
    
    train_txts = load_txt(args.train_list)
    val_txts = load_txt(args.val_list)

    train_dict = extract_dict(train_txts, train_classes)
    val_dict = extract_dict(val_txts, val_classes)    

    train_list = extract_samples(train_dict, args.few_shot)
    val_list = extract_samples(val_dict, 'all')
    
    if args.few_shot == 'all':
        save_txt(f'Caltech101_train_zero_2.txt', train_list)
    else:
        save_txt(f'Caltech101_train_{args.few_shot}_shot_2.txt', train_list)
    save_txt(f'Caltech101_val_zero.txt_2', val_list)    
    
    print('Train')
    check_n_classes(train_list)
    print(f'Train Total samples: {len(train_list)}')
    print('Val')
    check_n_classes(val_list)
    print(f'Text Total samples: {len(val_list)}')
    #-------------------------------------------------------------------

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cls_list", default='Caltech101_classnames.json', type=str)
    parser.add_argument("--train_list", default='Caltech101_train.txt', type=str)
    parser.add_argument("--val_list", default='Caltech101_val.txt', type=str)
    parser.add_argument('--few_shot', default=0, type=int)

    args = parser.parse_args()
    main(args)