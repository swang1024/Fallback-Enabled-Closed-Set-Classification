import os
import base64
from openai import OpenAI
from dataset.dataset import SFUniDADataset_BLIP, SFUniDADataset, INaturalist_UniDA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import INaturalist
from torchvision import transforms
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import pandas as pd
import json
from pydantic import BaseModel
import torch.multiprocessing as mp
import random
import httpx, certifi
mp.set_sharing_strategy('file_system')

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
args = build_args()

dataset = "INaturalist (Phylum)"

set_random_seed(2025)
dataset == 'INaturalist'
version = "v13"
target_type = 'phylum'
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
])
# Load the validation dataset 
dataset_ = INaturalist(
    root='/hpc/group/carin/sw361/data/',
    version='2021_valid',
    target_type=target_type,
    transform=transform,
    download=True,
)
# Find out how many phylum categories there are by inspecting one sample
# We'll scan through increasing IDs until category_name errors out or repeats
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

# randomly split the label_ids_list into two
random_seed = 2
print("seed", random_seed)
random.Random(random_seed).shuffle(label_ids_list)
if target_type == 'phylum':
    shared_class_num = 5
    source_private_class_num = 4
elif target_type == 'class':
    shared_class_num = 20
    source_private_class_num = 15
target_private_class_num = len(label_ids_list) - shared_class_num - source_private_class_num
shared_class_ids_list = label_ids_list[:shared_class_num]
source_private_class_ids_list = label_ids_list[shared_class_num:shared_class_num+source_private_class_num]
target_private_class_ids_list = label_ids_list[shared_class_num+source_private_class_num:]
source_classes = shared_class_ids_list + source_private_class_ids_list
target_classes = shared_class_ids_list + target_private_class_ids_list
known_class_list = [label_names_list[i] for i in source_classes]
target_class_list = [label_names_list[i] for i in target_classes]

print("known_class_list", known_class_list, flush=True)
print("target_class_list", target_class_list, flush=True)

target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data/', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)

# --- Plot sample counts per class in the target split ---------------------------------
# Count how many samples belong to each class in `target_dataset` and plot a bar chart
try:
    import matplotlib.pyplot as plt
    from collections import Counter

    # counts = Counter()
    # # iterate over the dataset (may load images) to collect label counts
    # for i in range(len(target_dataset)):
    #     item = target_dataset[i]
    #     # expected __getitem__ -> (image, label) or (image, label, ...)
    #     if isinstance(item, (list, tuple)) and len(item) >= 2:
    #         lbl = item[1]
    #     else:
    #         # unexpected format; skip
    #         continue
    #     # if label is a tensor, get int value
    #     try:
    #         lbl_idx = int(lbl)
    #     except Exception:
    #         lbl_idx = lbl
    #     # map index to human-readable name when possible
    #     if isinstance(lbl_idx, int) and 0 <= lbl_idx < len(label_names_list):
    #         name = label_names_list[lbl_idx]
    #     else:
    #         name = str(lbl_idx)
    #     counts[name] += 1

    # Prepare data in the same order as `target_class_list`
    x = list(target_class_list)
    # y = [counts.get(name, 0) for name in x]
    y = []

    # Simple bar plot
    plt.figure(figsize=(max(10, len(x) * 0.25), 6))
    bars = plt.bar(range(len(x)), y, color='tab:blue', edgecolor='black')
    plt.xticks(range(len(x)), x, rotation=90, fontsize=10)
    plt.ylabel('Sample count', fontsize=12)
    plt.title(f'Sample counts per class — {dataset} ({version})', fontsize=14)
    plt.tight_layout()

    # annotate counts above bars
    for rect, val in zip(bars, y):
        height = rect.get_height()
        if height > 0:
            plt.text(rect.get_x() + rect.get_width() / 2., height + max(1, max(y) * 0.01), f'{val}', ha='center', va='bottom', fontsize=8)

    out_png = Path('phylum_target_class_counts.png')
    plt.savefig(out_png, dpi=200)
    print(f"Saved class counts plot to {out_png}")
except Exception as e:
    print('Could not create class counts plot:', e)
# ------------------------------------------------------------------------------------