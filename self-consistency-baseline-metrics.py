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
dataset = "DomainNet"
model_name = "qwen"
version = 'v8_sc'
load_scores = False
print(dataset, version, model_name)
args = build_args()

def sanitize_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)

# You can keep aliases ("llama3.2-vision", "qwen-2.5-7B-VL") or full model IDs.
# Override with MODEL_NAMES env var, e.g., MODEL_NAMES="llama3.2-vision"
model_names_env = os.getenv("MODEL_NAMES", "").strip()
if model_names_env:
    model_names = [m.strip() for m in model_names_env.split(",") if m.strip()]
else:
    model_names = [
        # "gemini-2.0-flash",
        # "gpt-4o-mini",
        "llama3.2-vision",
        # "qwen-2.5-7B-VL",
    ]
num_consistency_samples  = 3
num_rephrases = 3
if dataset == "DomainNet":
    # DomainNet dataset configuration
    domain = 0
    print("domain", domain)
    random_seed = 1
    # maximum number of samples to read from CSV files (slice later)
    num_samples = 20000
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
    # safe_name = sanitize_model_name(model_name)
    safe_name = "qwen-2.5-7B-VL"
    output_path = os.path.join(
                "/hpc/group/carin/sw361/ChatGPT_exp/llm_data",
                "{}_target_domain{}_{}_{}_k{}.csv".format(
                    dataset,
                    args.t_idx,
                    safe_name,
                    f"{version}_self-consistency",
                    num_consistency_samples,
                ),
            )
    # output_path = "/hpc/group/carin/sw361/ChatGPT_exp/llm_data/{}_target_domain{}_{}_{}_rephrase_rephrase-k{}.csv".format(
    #         dataset,
    #         args.t_idx,
    #         safe_name,
    #         version,
    #         num_rephrases,
    #     )
    # Read precomputed LLM summary predictions and direct predictions for INaturalist
    df_direct = pd.read_csv(output_path, index_col=False)[:num_samples]

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
    # safe_name = sanitize_model_name(model_name)
    safe_name = "llama3.2-vision"
    output_path = os.path.join(
                "/hpc/group/carin/sw361/ChatGPT_exp/llm_data",
                "{}_target_domain{}_{}_{}_k{}.csv".format(
                    dataset,
                    args.t_idx,
                    safe_name,
                    f"{version}_self-consistency",
                    num_consistency_samples,
                ),
            )
    # output_path = "/hpc/group/carin/sw361/ChatGPT_exp/llm_data/{}_target_domain{}_{}_{}_rephrase_rephrase-k{}.csv".format(
    #         dataset,
    #         args.t_idx,
    #         safe_name,
    #         version,
    #         num_rephrases,
    #     )
    # Read precomputed LLM summary predictions and direct predictions for INaturalist
    df_direct = pd.read_csv(output_path, index_col=False)[:num_samples]

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
    domain = 1
    num_samples = 20000

    # different random seed for label splits
    random_seed = 0

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

    # safe_name = sanitize_model_name(model_name)
    safe_name = "qwen-2.5-7B-VL"
    output_path = os.path.join(
                "/hpc/group/carin/sw361/ChatGPT_exp/llm_data",
                "{}_{}_target_domain{}_{}_{}_k{}.csv".format(
                    target_type,
                    dataset,
                    args.t_idx,
                    safe_name,
                    f"{version}_self-consistency",
                    num_consistency_samples,
                ),
            )
    # output_path = "/hpc/group/carin/sw361/ChatGPT_exp/llm_data/{}_{}_target_domain{}_{}_{}_rephrase_rephrase-k{}.csv".format(
    #         target_type,
    #         dataset,
    #         args.t_idx,
    #         safe_name,
    #         version,
    #         num_rephrases,
    #     )
    # Read precomputed LLM summary predictions and direct predictions for INaturalist
    df_direct = pd.read_csv(output_path, index_col=False)[:num_samples]

# Extract ground-truth and private/known flags from LLM summary dataframe
gt_labels = list(df_direct['ground truth'].values)

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

# Convert ground truth labels to integer ids using class_dict for numeric ops
gt = [class_dict[lbl] for lbl in gt_labels]

def normalize_bool_value(value, column_name):
    """Convert CSV bool-like values to Python bool with explicit validation."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise ValueError(f"Missing value in column '{column_name}'.")
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, (float, np.floating)) and value.is_integer():
        if int(value) in (0, 1):
            return bool(int(value))

    val = str(value).strip().lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if val in mapping:
        return mapping[val]
    raise ValueError(f"Invalid boolean value '{value}' in column '{column_name}'.")


if 'unknown' not in df_direct.columns:
    raise KeyError("Column 'unknown' is required in the CSV file.")

df_direct['private'] = df_direct['private'].apply(lambda x: normalize_bool_value(x, 'private'))
df_direct['unknown'] = df_direct['unknown'].apply(lambda x: normalize_bool_value(x, 'unknown'))

# Binary unknown-classification accuracy on unknown samples only (private == True)
unknown_sample_mask = df_direct['private'] == True
if unknown_sample_mask.sum() == 0:
    raise ValueError("No unknown samples found where private == True.")
unknown_binary_acc = (
    df_direct.loc[unknown_sample_mask, 'private'] == df_direct.loc[unknown_sample_mask, 'unknown']
).mean()

# Per-class closed-set accuracy and macro/weighted summaries for known vs unknown classes
total_dataset_size = len(df_direct)
per_class_stats = []

for label, class_df in df_direct.groupby('ground truth'):
    class_size = len(class_df)
    private_values = class_df['private'].unique()
    if len(private_values) != 1:
        raise ValueError(f"Inconsistent 'private' values found within class '{label}'.")

    class_is_unknown = bool(private_values[0])
    if not class_is_unknown:
        # For known classes, compute closed-set accuracy (predicted class name matches ground truth)
        class_acc = (class_df['predicted class name'] == class_df['ground truth']).mean()
    else:
        class_acc = (class_df['private'] == class_df['unknown']).mean()

    per_class_stats.append({
        'label': label,
        'class_size': class_size,
        'is_unknown': class_is_unknown,
        'accuracy': class_acc,
    })

known_class_stats = [x for x in per_class_stats if not x['is_unknown']]
unknown_class_stats = [x for x in per_class_stats if x['is_unknown']]

if len(known_class_stats) == 0:
    raise ValueError("No known classes found.")
if len(unknown_class_stats) == 0:
    raise ValueError("No unknown classes found.")

mean_known_acc = sum(x['accuracy'] for x in known_class_stats) / len(known_class_stats)
mean_unknown_acc = sum(x['accuracy'] for x in unknown_class_stats) / len(unknown_class_stats)

weighted_mean_known_acc = sum(
    x['accuracy'] * (x['class_size'] / total_dataset_size) for x in known_class_stats
)
weighted_mean_unknown_acc = sum(
    x['accuracy'] * (x['class_size'] / total_dataset_size) for x in unknown_class_stats
)

# H-score of mean known-class accuracy and unknown binary-classification accuracy
h_score = 2 * mean_known_acc * unknown_binary_acc / (mean_known_acc + unknown_binary_acc + 1e-5)

print(f"Unknown binary classification accuracy (private==True): {unknown_binary_acc:.6f}")
print(f"Mean known-class accuracy: {mean_known_acc:.6f}")
print(f"Weighted mean known-class accuracy (class_size / total_dataset_size): {weighted_mean_known_acc:.6f}")
print(f"Mean unknown-class accuracy: {mean_unknown_acc:.6f}")
print(f"Weighted mean unknown-class accuracy (class_size / total_dataset_size): {weighted_mean_unknown_acc:.6f}")
print(f"H-score (mean known accuracy vs unknown binary accuracy): {h_score:.6f}")
