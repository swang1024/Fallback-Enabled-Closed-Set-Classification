import os
import base64
import re
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import INaturalist
from torchvision import transforms
import random
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPModel
from transformers import MllamaTextModel, AutoProcessor
import torch
import seaborn as sns
from collections import Counter
from scipy import stats

# Reproducibility: fix RNGs for deterministic behaviour where possible
set_random_seed(2025)

# This script analyzes LLM-generated image class predictions and
# corresponding summaries for different datasets (DomainNet, VisDA, INaturalist).
# It computes per-class accuracy for known vs unknown classes and
# several aggregated statistics (H-score, mean/wgt accuracies, ratios).

# Configuration: choose dataset and model/version metadata used to load CSVs
dataset = "VisDA"
model_name = "qwen"
version = 'v13'
load_scores = False
print(dataset, version, model_name)

if dataset == "DomainNet":
    # DomainNet dataset configuration
    domain = 0
    print("domain", domain)
    random_seed = 1
    # maximum number of samples to read from CSV files (slice later)
    num_samples = 400
    # full DomainNet class names (source label space)
    class_list = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', \
        'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', \
        'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', \
        'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', \
        'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', \
        'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', \
        'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', \
        'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', \
        'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', \
        'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', \
        'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', \
        'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', \
        'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', \
        'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer',\
        'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', \
        'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', \
        'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', \
        'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', \
        'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', \
        'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', \
        'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', \
        'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']
    # read predictions & summaries produced by the LLM (or other sources)
    df = pd.read_csv(f'llm_data/{dataset}_target_domain{domain}_{model_name}_{version}_summary_pred.csv')[:num_samples]
    # direct predictions (no summary) used as a baseline/comparison
    df_direct = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)[:num_samples]

    # Partition classes into target-private, shared, and source-private classes
    tgt_priv_class_list = list(df_direct[df_direct['private'] == True]['ground truth'].unique())
    shared_class_list = list(df_direct[df_direct['private'] == False]['ground truth'].unique())
    src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
    print(tgt_priv_class_list, shared_class_list, src_priv_class_list)
    # tgt_priv_class_list = ['wine glass', 'parrot', 'violin', 'train', 'triangle', 'nose', 'rain', 'teddy-bear', 'onion', 'pickup truck', 'roller coaster', 'octopus', 'sailboat', 'submarine', 'rhinoceros', 'panda', 'windmill', 'squiggle', 'river', 'tree', 'soccer ball', 'smiley face', 'string bean', 'school bus', 'penguin', 'snowman', 'wine bottle', 'sandwich', 'palm tree', 'watermelon', 'suitcase', 'snowflake', 'tiger', 'rake', 'toaster', 'pear', 'pond', 'shorts', 'pool', 'pliers', 'raccoon', 'sun', 'rollerskates', 'speedboat', 'pig', 'square', 'pants', 'owl', 'strawberry', 'saw', 'zigzag', 'skateboard', 'star', 'spider', 'sheep', 'scissors', 'police car', 'saxophone', 'picture frame', 'teapot', 'sword', 'van', 'pencil', 'trombone', 'zebra', 'necklace', 'paintbrush', 'paper clip', 'piano', 'umbrella', 'tent', 'trumpet', 'rifle', 'tooth', 'snail', 'rabbit', 'postcard', 'parachute', 'swan', 'pizza', 'whale', 'sea turtle', 'toothbrush', 'ocean', 'streetlight', 'scorpion', 'screwdriver', 'tornado', 'pillow', 'pineapple', 'spoon', 'sink', 'truck', 'power outlet', 'snorkel', 'remote control', 'spreadsheet', 'tractor', 'squirrel', 'traffic light', 'vase', 'passport', 'snake', 'skull', 'shark', 'peanut', 'stethoscope', 'stop sign', 'see saw', 'skyscraper', 'shoe', 'paint can', 'radio', 'sweater', 'stairs', 'peas', 'potato', 'table', 'swing set', 'telephone', 'yoga', 'waterslide', 'toilet', 'popsicle', 'rainbow', 'shovel', 't-shirt', 'steak', 'purse', 'stereo', 'oven', 'television', 'toothpaste', 'octagon', 'wheel', 'wristwatch', 'toe', 'sock', 'sleeping bag', 'stove', 'underwear', 'tennis racquet', 'washing machine', 'stitches', 'syringe'] 
    # shared_class_list = ['crab', 'bottlecap', 'cup', 'dog', 'duck', 'grapes', 'garden', 'fireplace', 'guitar', 'baseball bat', 'harp', 'dolphin', 'golf club', 'castle', 'butterfly', 'bread', 'bowtie', 'arm', 'compass', 'asparagus', 'drums', 'animal migration', 'church', 'bridge', 'grass', 'bicycle', 'hat', 'fish', 'camera', 'basket', 'dragon', 'hexagon', 'flashlight', 'bus', 'hedgehog', 'helicopter', 'beach', 'chandelier', 'donut', 'angel', 'elephant', 'flamingo', 'backpack', 'hand', 'bush', 'bed', 'eye', 'face', 'beard', 'cloud', 'binoculars', 'bucket', 'dumbbell', 'hamburger', 'baseball', 'bathtub', 'apple', 'brain', 'airplane', 'The Mona Lisa', 'bench', 'headphones', 'garden hose', 'cat', 'camel', 'The Eiffel Tower', 'cello', 'cow', 'giraffe', 'flip flops', 'basketball', 'cake', 'fire hydrant', 'bird', 'barn', 'The Great Wall of China', 'crayon', 'frog', 'canoe', 'anvil', 'cruise ship', 'bee', 'hockey stick', 'cell phone', 'coffee cup', 'frying pan', 'carrot', 'clock', 'flying saucer', 'envelope', 'bear', 'blueberry', 'chair', 'aircraft carrier', 'birthday cake', 'flower', 'bracelet', 'hockey puck', 'ant', 'goatee', 'dishwasher', 'bat', 'feather', 'fork', 'circle', 'axe', 'campfire', 'bandage', 'firetruck', 'clarinet', 'ear', 'ambulance', 'foot', 'banana', 'cactus', 'boomerang', 'broom', 'candle', 'book', 'alarm clock', 'calendar', 'diving board', 'elbow', 'finger', 'door', 'broccoli', 'eyeglasses', 'cannon', 'fence', 'cookie', 'camouflage', 'crown', 'blackberry', 'eraser', 'dresser', 'helmet', 'calculator', 'couch', 'crocodile', 'car', 'ceiling fan', 'diamond', 'bulldozer', 'hammer', 'drill', 'cooler', 'belt', 'computer', 'fan', 'floor lamp'] 
    # src_priv_class_list = ['horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']
    # Map class name to integer id for downstream numeric ops
    class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 
    src_class_list = shared_class_list + src_priv_class_list
    tgt_class_list = shared_class_list + tgt_priv_class_list
    # src_class_list_new is the source-known label set with optional context applied later
    src_class_list_new = src_class_list
elif dataset == "VisDA":
    # VisDA dataset configuration (smaller example set)
    domain = 1
    random_seed = 1
    num_samples = 20000
    class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
    df = pd.read_csv(f'llm_data/{dataset}_target_domain{domain}_{model_name}_{version}_summary_pred.csv', index_col=False)[:num_samples]
    df_direct = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)[:num_samples]

    # Similar partitioning logic as DomainNet above
    tgt_priv_class_list = list(df_direct[df_direct['private'] == True]['ground truth'].unique())
    shared_class_list = list(df_direct[df_direct['private'] == False]['ground truth'].unique())
    src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
    class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 
    src_class_list = shared_class_list + src_priv_class_list
    tgt_class_list = shared_class_list + tgt_priv_class_list
    src_class_list_new = src_class_list
elif dataset == "INaturalist":
    # INaturalist requires access to torchvision's dataset to enumerate labels.
    domain = 2
    num_samples = 18000

    # different random seed for label splits
    random_seed = 2

    # Minimal transform for dataset inspection (we don't use images here, only labels)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    target_type = 'phylum'
    print("target type", target_type)
    # Load the INaturalist validation set to extract category names and ids
    dataset_ = INaturalist(
        root='/hpc/group/carin/sw361/data/',
        version='2021_valid',
        target_type=target_type,
        transform=transform,
        download=True,
    )

    # Build a list of category names by iterating until an IndexError/ValueError occurs
    label_names_list = []
    label_ids_list = []
    i = 0
    while True:
        try:
            name = dataset_.category_name(target_type, i)
            label_names_list.append(name)
            label_ids_list.append(i)
            i += 1
        except (IndexError, ValueError):
            break

    print(f"Total label categories: {len(label_names_list)}")
    print("label categories:", label_names_list, label_ids_list)

    # Shuffle label ids deterministically for splitting known/unknown classes
    print("seed", random_seed)
    random.Random(random_seed).shuffle(label_ids_list)
    if target_type == 'phylum':
        shared_class_num = 5
        source_private_class_num = 4
    elif target_type == 'class':
        shared_class_num = 20
        source_private_class_num = 15
    target_private_class_num = len(label_ids_list) - shared_class_num - source_private_class_num

    # Partition id lists for shared/source/target private classes
    shared_class_ids_list = label_ids_list[:shared_class_num]
    source_private_class_ids_list = label_ids_list[shared_class_num:shared_class_num+source_private_class_num]
    target_private_class_ids_list = label_ids_list[shared_class_num+source_private_class_num:]
    source_classes = shared_class_ids_list + source_private_class_ids_list
    target_classes = shared_class_ids_list + target_private_class_ids_list

    # Map ids to human-readable class names
    shared_class_list = [label_names_list[i] for i in shared_class_ids_list]
    src_priv_class_list = [label_names_list[i] for i in source_private_class_ids_list]
    tgt_priv_class_list = [label_names_list[i] for i in target_private_class_ids_list]
    src_class_list = [label_names_list[i] for i in source_classes]
    src_class_list_new = src_class_list
    
    print("known class", src_class_list_new)
    
    tgt_class_list = [label_names_list[i] for i in target_classes]
    
    # Class name -> integer id mapping used for numeric evaluation
    class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 

    # Read precomputed LLM summary predictions and direct predictions for INaturalist
    df = pd.read_csv(f'llm_data/{target_type}_{dataset}_target_domain0_{model_name}_{version}_randomseed{random_seed}_summary_pred.csv', index_col=False)[:num_samples]
    df_direct = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8_randomseed{random_seed}.csv", index_col=False)[:num_samples]

# Extract ground-truth and private/known flags from LLM summary dataframe
gt_labels = list(df['ground truth'].values)

# Build a human-readable source class list with optional context prefix
src_class_list_w_context = ["An image of " + cls for cls in src_class_list_new]

def process_preds(row):
    """Normalize the predicted class name string from CSV rows.

    Some predictions include escaped quotes; this removes single-quotes and
    ensures we have a clean string for comparisons.
    """
    return str(row['predicted class name']).replace('\'', '')

# Apply normalization to the direct-prediction dataframe and the summary dataframe
df_direct['predicted class name'] = df_direct.apply(process_preds, axis=1)
llm_direct_preds = list(df_direct['predicted class name'].values)
df['predicted class name'] = df.apply(process_preds, axis=1)

# Convert ground truth labels to integer ids using class_dict for numeric ops
gt = [class_dict[lbl] for lbl in gt_labels]

# Identify indices for known/unknown examples according to the 'private' flag
known_idx = df_direct.index[df_direct['private'] == False].tolist()
known_in_idx = df_direct.index[(df_direct['private'] == False) & (df_direct['predicted class name'].isin(src_class_list_new))]
known_outside_idx = df_direct.index[(df_direct['private'] == False) & (~df_direct['predicted class name'].isin(src_class_list_new))]
print("known", "all: ", len(known_idx), "outside: ", len(known_outside_idx), "in: ", len(known_in_idx))

unknown_idx = df_direct.index[df_direct['private'] == True].tolist()
unknown_in_idx = df_direct.index[(df_direct['private'] == True) & (df_direct['predicted class name'].isin(src_class_list_new))]
unknown_outside_idx = df_direct.index[(df_direct['private'] == True) & (~df_direct['predicted class name'].isin(src_class_list_new))]
print("unknown", "all: ", len(unknown_idx), "outside: ", len(unknown_outside_idx), "in: ", len(unknown_in_idx))

llm_direct_preds_known_in = list(df_direct.iloc[known_in_idx]['predicted class name'].values)
llm_summary_preds_known_in = list(df.iloc[known_in_idx]['predicted class name'].values)

# Indices where direct and summary agree (for known-in examples)
kwn_in_pred_known_idx = [kwn_idx for kwn_idx,  direct_pred, summary_pred in zip(known_in_idx, llm_direct_preds_known_in, llm_summary_preds_known_in) if direct_pred == summary_pred]
# Indices where they disagree
kwn_in_pred_unknown_idx = [kwn_idx for kwn_idx,  direct_pred, summary_pred in zip(known_in_idx, llm_direct_preds_known_in, llm_summary_preds_known_in) if direct_pred != summary_pred]
print("known pred known", len(kwn_in_pred_known_idx))

# For unknown examples predicted as known classes, compute various correctness stats
llm_direct_preds_unknown_in = list(df_direct.iloc[unknown_in_idx]['predicted class name'].values)
llm_summary_preds_unknown_in = list(df.iloc[unknown_in_idx]['predicted class name'].values)
ukn_llm_direct_preds_outside = list(df_direct.iloc[unknown_outside_idx]['predicted class name'].values)

# unknown_in_idx: indices of unknown examples predicted as source-known classes
# incorrect_idx collects examples where direct and summary agree (i.e., both predicted a known class)
ukn_in_pred_known_idx = [ukn_idx for ukn_idx,  direct_pred, summary_pred in zip(unknown_in_idx, llm_direct_preds_unknown_in, llm_summary_preds_unknown_in) if direct_pred == summary_pred]
print("unknown pred known", len(ukn_in_pred_known_idx))

# Positive predictive value (PPV) approximated as true-known-over-(true-known + false-known)
print("ppv", len(kwn_in_pred_known_idx) / (len(kwn_in_pred_known_idx) + len(ukn_in_pred_known_idx)), 'fp', len(ukn_in_pred_known_idx), 'tp', len(kwn_in_pred_known_idx))
print("npv", (len(unknown_idx) - len(ukn_in_pred_known_idx)) / ((len(known_idx) - len(kwn_in_pred_known_idx)) + (len(unknown_idx) - len(ukn_in_pred_known_idx))), 'fn', (len(known_idx) - len(kwn_in_pred_known_idx)), 'tn', (len(unknown_idx) - len(ukn_in_pred_known_idx)))

##############################
# make boxplots with accuracy of known classes and unknown classes 
kwn_per_class_num, kwn_per_class_correct = [], []
ukn_per_class_num, ukn_per_class_correct = [], []
kwn_label_list = []
ukn_label_list = []
cls_lst = tgt_class_list
print("cls_lst", cls_lst)
cls_lst_cls_num = []
per_cls_acc = []
for i, label in enumerate(cls_lst):
    # check if pred_label is shared or not
    if label not in tgt_priv_class_list:
        # print(label)
        label_idx = df_direct.index[df_direct['ground truth'] == label]
        known_in_idx = df_direct.index[(df_direct['ground truth'] == label) & (df_direct['predicted class name'].isin(src_class_list_new))]
        llm_direct_preds = list(df_direct.iloc[known_in_idx]['predicted class name'].values)
        llm_summary_preds = list(df.iloc[known_in_idx]['predicted class name'].values)

        correct_idx = [kwn_idx for kwn_idx,  direct_pred, summary_pred in zip(known_in_idx, llm_direct_preds, llm_summary_preds) if direct_pred == summary_pred and df_direct.iloc[kwn_idx]['ground truth'] == label]
        num_corr = len(correct_idx)
        kwn_per_class_num.append(len(label_idx))
        kwn_per_class_correct.append(num_corr)
        kwn_label_list.append(label)
    else:
        # print("unknown", label)
        label_idx = df_direct.index[df_direct['ground truth'] == label]
        unknown_in_idx = df_direct.index[(df_direct['ground truth'] == label) & (df_direct['predicted class name'].isin(src_class_list_new))]
        llm_direct_preds = list(df_direct.iloc[unknown_in_idx]['predicted class name'].values)
        llm_summary_preds = list(df.iloc[unknown_in_idx]['predicted class name'].values)

        incorrect_idx = [ukn_idx for ukn_idx, direct_pred, summary_pred in zip(unknown_in_idx, llm_direct_preds, llm_summary_preds) if direct_pred == summary_pred and df_direct.iloc[ukn_idx]['ground truth'] == label]
        num_corr = len(label_idx) - len(incorrect_idx)
        ukn_per_class_num.append(len(label_idx))
        ukn_per_class_correct.append(num_corr)
        ukn_label_list.append(label)
    cls_lst_cls_num.append(len(label_idx))
    per_cls_acc.append(num_corr / (1e-5+len(label_idx)))
kwn_per_class_acc = [x / (y + 1e-5) for x, y in zip(kwn_per_class_correct, kwn_per_class_num)]
print(kwn_per_class_acc)
print(kwn_per_class_num)
print(kwn_label_list)
ukn_per_class_acc = [x / (y + 1e-5) for x, y in zip(ukn_per_class_correct, ukn_per_class_num)]
print(ukn_per_class_acc)
print(ukn_per_class_num)
print(ukn_label_list)
known_acc = sum(kwn_per_class_acc) / len(kwn_per_class_acc)
unknown_acc = sum(ukn_per_class_correct) / sum(ukn_per_class_num)
h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
print("h-score", h_score)

h_score_all = []
for ukn_acc in ukn_per_class_acc:
    h_score_tmp = 2 * known_acc * ukn_acc / (known_acc + ukn_acc + 1e-5)
    h_score_all.append(h_score_tmp)

print("h score all", h_score_all)
# fig = plt.figure(figsize=(20, 20))
# ax = plt.subplot(111)
# # cls_lst_cls_num = [12388+15132, 27+43, 23+17, 753+937, 37+33, 94+116, 22+18, 18947+23233, 369+471] 
# # cls_lst_cls_num = [150, 17483, 13, 30, 682, 96, 11420, 14, 10112]
# cls_lst_cls_num = [17713, 28, 11386, 366, 23, 10210, 151, 15, 108]
# ax.bar(np.arange(len(cls_lst_cls_num)), cls_lst_cls_num, label=cls_lst)
# plt.yticks(rotation=0, fontsize=10)
# plt.title(f'{dataset}', fontsize=30)
# plt.xticks(rotation=45, fontsize=20)
# ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Sample Size', fontsize=30)
# plt.savefig(f"{dataset}_Sample_Size_{model_name}_randomseed{random_seed}.png")

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# ax.bar(np.arange(len(per_cls_acc)), per_cls_acc, label=cls_lst)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'{dataset} {model_name}', fontsize=30)
# ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
# plt.xticks(rotation=45, fontsize=30)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"phylum_{dataset}_Accuracy_{model_name}_barplot_randomseed{random_seed}.png")

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
qwen_kwn_acc = kwn_per_class_acc
qwen_ukn_acc = ukn_per_class_acc
# print(len(qwen_kwn_acc), len(qwen_ukn_acc))
# ax.boxplot([qwen_kwn_acc, qwen_ukn_acc], labels=['llama known', 'llama unknown'])
# plt.yticks(rotation=45, fontsize=20)
# plt.title(f'{dataset} Known vs Unknown', fontsize=30)
# ax.xaxis.set_label_position('bottom')
# # ax.set_xticks(ticks=np.arange(2), labels=['known', 'unknown'])
# plt.xticks(rotation=45, fontsize=10)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"phylum_{dataset}_Accuracy_boxplot_randomseed{random_seed}.png")

kwn_sample_size = kwn_per_class_num
ukn_sample_size = ukn_per_class_num
mean_qwen_kwn = sum(qwen_kwn_acc) / len(qwen_kwn_acc)
mean_qwen_ukn = sum(qwen_ukn_acc) / len(qwen_ukn_acc)
print(mean_qwen_kwn, mean_qwen_ukn)
print("------------")

wgt_qwen_kwn = sum([x * y for x, y in zip(kwn_sample_size, qwen_kwn_acc)])  / sum(kwn_sample_size)
wgt_qwen_ukn = sum([x * y for x, y in zip(ukn_sample_size, qwen_ukn_acc)])  / sum(ukn_sample_size)
print(wgt_qwen_kwn, wgt_qwen_ukn)
print("------------")

ratio_mean_qwen = mean_qwen_kwn / mean_qwen_ukn
print(ratio_mean_qwen)

ratio_wgt_mean_qwen = wgt_qwen_kwn / wgt_qwen_ukn
print(ratio_wgt_mean_qwen)
