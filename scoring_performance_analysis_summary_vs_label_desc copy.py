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

domain = 1
print("domain", domain)
df = pd.read_csv(f'target_domain{domain}_4o-mini_v13_summary.csv')

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
'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', \
'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']

src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
print(src_priv_class_list)
print(len(src_priv_class_list))

class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 

src_class_list = shared_class_list + src_priv_class_list

label_desc_df = pd.read_csv(f'label_visual_features.csv')
label_desc = []
for cls in src_class_list:
    label_desc.append(label_desc_df[label_desc_df['class label'] == cls]['description'].values[0])

# Load the tokenizer and model for the CLIP text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize the sentence and convert it to tensor format
summary_token = tokenizer(pred_summary, return_tensors="pt", truncation=True, padding=True)
src_class_list_tokens = tokenizer(label_desc, return_tensors="pt", truncation=True, padding=True)

# Compute the embeddings using the text encoder
with torch.no_grad():
    # Get the text embeddings using the CLIP model's text encoder
    sentence_embedding = model.get_text_features(**summary_token)
    class_list_embeddings = model.get_text_features(**src_class_list_tokens)

# Normalize the embeddings for cosine similarity
sentence_embedding = sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)
class_list_embeddings = class_list_embeddings / class_list_embeddings.norm(dim=-1, keepdim=True)

# print(sentence_embedding.size(), class_list_embeddings.size())

# Compute cosine similarity: dot product of the normalized vectors
similarity_scores = (sentence_embedding @ class_list_embeddings.T).squeeze(0)

cos_sim_df = pd.DataFrame(similarity_scores)
cos_sim_df.to_csv(f"Domain {domain} similarity_score_summary_v13_vs_label_desc.csv", header=False, index=False)

similarity_scores = pd.read_csv(f"Domain {domain} similarity_score_summary_v13_vs_label_desc.csv", header=None, index_col=False).to_numpy()

preds_all = np.argmax(similarity_scores, axis=1)
gt = [class_dict[lbl] for lbl in gt_labels]

# ratio of highest cos score / sum of all cos scores
# for known
known_idx = df.index[df['private'] == False]
known_ratios = np.exp(np.max(similarity_scores[known_idx], axis=1)) / np.sum(np.exp(similarity_scores[known_idx]), axis=1)
# for unknown
unknown_idx = df.index[df['private'] == True].tolist()
unknown_ratios = np.exp(np.max(similarity_scores[unknown_idx], axis=1)) / np.sum(np.exp(similarity_scores[unknown_idx]), axis=1)
ratios = [known_ratios, unknown_ratios]

fig = plt.figure(figsize=(32, 14))
ax = plt.subplot(111)
sns.histplot(ratios, bins=50, stat='probability')
plt.yticks(rotation=0)
plt.title(f'Domain {domain}', fontsize=20)
plt.legend(['known', 'unknown'])
# labels, title and ticks
ax.set_xlabel("Ratio", fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.tick_bottom()
ax.set_ylabel('Density', fontsize=20)
# ax.set_xlim(0.005, 0.015)
plt.savefig("known_unknown_exp_ratio_summary_vs_label_desc.png")


# # plot confusion matrix with y axis being the true labels 
# # and x axis being the predicted labels (source domain labels, which are in the list)
# matrix = confusion_matrix(gt, preds_all, labels=np.arange(350))
# matrix_np = matrix[:, :len(src_class_list)]
# matrix_df = pd.DataFrame(matrix_np)
# matrix_df.to_csv(f"Domain {domain} confusion_matrix.csv", header=False, index=False)

# fig = plt.figure(figsize=(16, 14))
# ax= plt.subplot()
# sns.heatmap(matrix[:, :len(src_class_list)], annot=True, ax = ax, cmap="crest"); #annot=True to annotate cells
# # labels, title and ticks
# ax.set_xlabel('Predicted', fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=90)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('True', fontsize=20)
# plt.yticks(rotation=0)
# plt.title(f'Domain {domain} Confusion Matrix', fontsize=20)
# plt.savefig(f"Domain {domain} confusion_matrix.png")

def get_elements_by_indices(data_list, index_list):
    """
    Returns a new list containing elements from data_list at the positions specified in index_list.

    Args:
        data_list: The original list to extract elements from.
        index_list: A list of integers representing the indices of the desired elements.

    Returns:
        A new list containing the extracted elements.
    """
    return [data_list[i] for i in index_list]

accuracy_all_indirect = []
most_common_all = []
for idx, cls in enumerate(shared_class_list):
    indexes = df.index[df['ground truth'] == cls].to_list()
    class_idx = src_class_list.index(cls)
    # print(cls, idx, preds_all[indexes])
    most_common = Counter(preds_all[indexes]).most_common()
    # print("most common", most_common, [src_class_list[i] for i in list(list(zip(*most_common))[0])])
    acc = np.sum(preds_all[indexes] == class_idx) / len(preds_all[indexes])
    # print(cls, "total:", len(preds_all[indexes]), "most common", [(src_class_list[cls], list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))])
    # print()
    most_common_all.append([cls, len(preds_all[indexes]), acc, [(src_class_list[cls], list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))]])
    accuracy_all_indirect.append(acc)

indirect_most_common = pd.DataFrame(most_common_all, columns=["label name", "total num", "pred acc", "most common pred"])
indirect_most_common.to_csv(f"indirect_most_common_predicted_summary_vs_label_desc.csv")

print("indirect acc", np.mean(accuracy_all_indirect))

df_direct = pd.read_csv(f"target_domain1_4o-mini_v8.csv", index_col=False)
def process_row(row):
    if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
        return row['predicted class name'][1:-1]
    else:
        return row['predicted class name']

df_direct['predicted class name'] = df_direct.apply(process_row, axis=1)

accuracy_all_direct = []
most_common_direct_all = []
for cls in shared_class_list:
    df_indirect_cls = df_direct[df_direct['ground truth'] == cls]
    most_common = Counter(list(df_indirect_cls['predicted class name'].values)).most_common()
    print(most_common)
    df_indirect_cls_correct = df_indirect_cls[df_indirect_cls['ground truth'] == df_indirect_cls['predicted class name']]
    acc = len(df_indirect_cls_correct) / len(df_indirect_cls)
    most_common_direct_all.append([cls, len(df_indirect_cls), acc, [(cls, list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))] ])
    accuracy_all_direct.append(acc)

direct_most_common = pd.DataFrame(most_common_direct_all, columns=["label name", "total num", "pred acc", "most common pred"])
# direct_most_common.to_csv(f"direct_most_common_predicted.csv")

print("direct acc", np.mean(accuracy_all_direct))

fig = plt.figure(figsize=(38, 14))
ax = plt.subplot(111)
# plt.bar(np.arange(len(shared_class_list)), accuracy_all_indirect)
# plt.yticks(rotation=0)
# plt.title(f'Domain {domain}', fontsize=20)
# plt.legend(['indirect labeling'])
# # labels, title and ticks
# ax.set_xlabel("class index", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=90)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=20)
# plt.savefig("indirect_labeling.png")

barWidth = 0.3
br1 = np.arange(len(shared_class_list)) 
br2 = [x + barWidth for x in br1] 

plt.bar(br1, accuracy_all_direct, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Direct Labeling') 
plt.bar(br2, accuracy_all_indirect, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Indirect Labeling') 

plt.xlabel('class index', fontweight ='bold', fontsize = 15) 
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15) 
# plt.xticks([r + 1/2 * barWidth for r in range(len(shared_class_list))])
plt.legend()
plt.savefig("direct_vs_indirect_labeling_summary_vs_label_desc.png")

# figure out the classes with accuracy difference more than 10%
# get the accuracy difference, and sort them from high to low in a csv file with one column being the label name and the other being the difference
acc_diff = [x - y for x, y in zip(accuracy_all_direct, accuracy_all_indirect)]
zipped_diff = list(zip(shared_class_list, acc_diff, direct_most_common["most common pred"], indirect_most_common["most common pred"]))
print(sorted(zipped_diff, key=lambda x: np.abs(x[1]), reverse=True))
zipped_diff = sorted(zipped_diff, key=lambda x: np.abs(x[1]), reverse=True)
df_acc_diff = pd.DataFrame(zipped_diff, columns=["label name", "accuracy difference (direc - indirect)", "direct's pred", "indirect's pred"])
df_acc_diff.to_csv(f"Domain {domain} direct_vs_indirect_acc_diff_summary_vs_label_desc.csv")

