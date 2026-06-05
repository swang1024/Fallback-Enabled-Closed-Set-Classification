import os
import base64
from openai import OpenAI
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import numpy as np
import pandas as pd
import json
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPModel
import torch
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

domain = 1
print("domain", domain)
df = pd.read_csv(f'target_domain{domain}_4o-mini_v10_summary.csv')

gt_unknown = list(df['private'].values)
pred_summary= list(df['summary'].values)
gt_labels = list(df['ground truth'].values)
tgt_priv_class_list = list(df[df['private'] == True]['ground truth'].unique())
shared_class_list = list(df[df['private'] == False]['ground truth'].unique())

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
'mailbox', 'geographical map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', \
'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']

src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
print(src_priv_class_list)
# print(len(src_priv_class_list))

class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 

src_class_list = shared_class_list + src_priv_class_list
src_class_list_w_context = ["An image of " + cls for cls in src_class_list]

# Load the tokenizer and model for the CLIP text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize the sentence and convert it to tensor format
summary_token = tokenizer(pred_summary, return_tensors="pt", truncation=True, padding=True)
src_class_list_w_context_tokens = tokenizer(src_class_list_w_context, return_tensors="pt", truncation=True, padding=True)
src_class_list_tokens = tokenizer(src_class_list, return_tensors="pt", truncation=True, padding=True)

# Compute the embeddings using the text encoder
with torch.no_grad():
    # Get the text embeddings using the CLIP model's text encoder
    sentence_embedding = model.get_text_features(**summary_token)
    class_list_w_context_embeddings = model.get_text_features(**src_class_list_w_context_tokens)
    class_list_embeddings = model.get_text_features(**src_class_list_tokens)

# Normalize the embeddings for cosine similarity
# sentence_embedding = sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)
class_list_w_context_embeddings = class_list_w_context_embeddings / class_list_w_context_embeddings.norm(dim=-1, keepdim=True)
class_list_embeddings = class_list_embeddings / class_list_embeddings.norm(dim=-1, keepdim=True)

# # print(sentence_embedding.size(), class_list_embeddings.size())

# Compute cosine similarity: dot product of the normalized vectors
similarity_scores = (sentence_embedding @ class_list_w_context_embeddings.T).squeeze(0)

cos_sim_df = pd.DataFrame(similarity_scores)
cos_sim_df.to_csv(f"Domain {domain} similarity_score_add_context_to_labels_change_map.csv", header=False, index=False)

# find the most similar label for each label 
# and save the results in a dictionary
# class_list_similarity_scores = (class_list_embeddings @ class_list_embeddings.T).squeeze(0)
# class_list_similarity_scores.fill_diagonal_(-torch.inf)
# most_similar_label = {}
# most_sim_lbl = np.argmax(class_list_similarity_scores, axis=1)
# for lbl, most_sim in enumerate(most_sim_lbl):
#     most_similar_label[src_class_list[lbl]] = src_class_list[most_sim]
#     # most_similar_label[lbl] = most_sim
# print("most_similar_label", most_similar_label)

class_list_w_context_similarity_scores = (class_list_w_context_embeddings @ class_list_w_context_embeddings.T).squeeze(0)
class_list_w_context_similarity_scores.fill_diagonal_(-torch.inf)
most_similar_label_w_context = {}
most_sim_lbl = np.argmax(class_list_w_context_similarity_scores, axis=1)
for lbl, most_sim in enumerate(most_sim_lbl):
    # most_similar_label_w_context[src_class_list[lbl]] = src_class_list[most_sim]
    most_similar_label_w_context[lbl] = most_sim.cpu()
print("most_similar_label_w_context", most_similar_label_w_context)

similarity_scores = pd.read_csv(f"Domain {domain} similarity_score_add_context_to_labels_change_map.csv", header=None, index_col=False).to_numpy()

preds_all = np.argmax(similarity_scores, axis=1)
gt = [class_dict[lbl] for lbl in gt_labels]

# ratio of highest cos score / sum of all cos scores
# for known
known_idx = df.index[df['private'] == False].tolist()

preds_known = preds_all[known_idx]
most_sim_lbl_known = [most_similar_label_w_context[pred] for pred in preds_known]
known_ratios = [np.exp(similarity_scores[idx, i]) / np.sum(np.exp(similarity_scores[idx])) for idx, i, _ in zip(known_idx, preds_known, most_sim_lbl_known)]
# known_ratios = [np.exp(similarity_scores[idx, i]) / (np.exp(similarity_scores[idx, i]) + np.exp(similarity_scores[idx, j])) for idx, i, j in zip(known_idx, preds_known, most_sim_lbl_known)]
known_median = np.median(known_ratios)
# known_ratios = np.max(similarity_scores[known_idx], axis=1) 
# known_ratios = np.sum(np.exp(similarity_scores[known_idx]), axis=1)

# for unknown
unknown_idx = df.index[df['private'] == True].tolist()
preds_unknown = preds_all[unknown_idx]
most_sim_lbl_unknown = [most_similar_label_w_context[pred] for pred in preds_unknown]
unknown_ratios = [np.exp(similarity_scores[idx, i]) / np.sum(np.exp(similarity_scores[idx])) for idx, i, _ in zip(unknown_idx, preds_unknown, most_sim_lbl_unknown)]
# unknown_ratios = [np.exp(similarity_scores[idx, i]) / (np.exp(similarity_scores[idx, i]) + np.exp(similarity_scores[idx, j])) for idx, i, j in zip(unknown_idx, preds_unknown, most_sim_lbl_unknown)]
unknown_median = np.median(unknown_ratios)
# unknown_ratios = np.max(similarity_scores[unknown_idx], axis=1)
# unknown_ratios = np.sum(np.exp(similarity_scores[unknown_idx]), axis=1)
ratios = [known_ratios, unknown_ratios]
# ratios = [known_correct_ratios, known_incorrect_ratios, unknown_ratios]

fig = plt.figure(figsize=(32, 14))
ax = plt.subplot(111)
sns.histplot(ratios, bins=50, stat='probability', common_norm=False)
plt.yticks(rotation=0, fontsize=20)
plt.title(f'Domain {domain}', fontsize=30)
plt.legend(['unknown', 'known'], fontsize=30)
plt.axvline(known_median, c='blue')
plt.axvline(unknown_median, c='red')
# labels, title and ticks
ax.set_xlabel("Ratio", fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=45, fontsize=20)
ax.xaxis.tick_bottom()
ax.set_ylabel('Density', fontsize=30)
plt.savefig("known_unknown_add_label_context_change_map.png")
