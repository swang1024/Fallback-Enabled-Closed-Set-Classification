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
from scipy import stats

set_random_seed(2025)

domain = 1
mean_centering = False
print("domain", domain, "mean_centering", mean_centering)
df = pd.read_csv(f'target_domain{domain}_4o-mini_v14_summary.csv')[:500]

gt_unknown = list(df['private'].values)
pred_summary = list(df['summary'].values)[:500]
gt_labels = list(df['ground truth'].values)
tgt_priv_class_list = list(df[df['private'] == True]['ground truth'].unique())
shared_class_list = list(df[df['private'] == False]['ground truth'].unique())

# def remove_str(row):
#     return row['summary'].replace('The image depicts ', '').replace('The image shows ', '').replace('The image features ', '').replace('The image illustrates ', '').replace('The image illustrates ', '')
# df['cleaned summary']= df.apply(remove_str, axis=1)
# df.to_csv(f'target_domain{domain}_4o-mini_v10_summary_w_cleaned_summary.csv')
# print(df['cleaned summary'])
# clean_pred_summary = list(df['cleaned summary'].values)

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

class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 

src_class_list = shared_class_list + src_priv_class_list
src_class_list_w_context = ["An image of " + cls for cls in src_class_list]
# src_class_list_w_context = ["A photo of " + cls for cls in src_class_list]

label_desc_df = pd.read_csv(f'label_visual_features.csv')
label_desc = []
for cls in src_class_list:
    label_desc.append(label_desc_df[label_desc_df['class label'] == cls]['description'].values[0])

llm_pred_desc_df = pd.read_csv(f'llm_pred_visual_features.csv')
llm_pred_desc = list(llm_pred_desc_df['description'].values)

df_direct = pd.read_csv(f"target_domain1_4o-mini_v8.csv", index_col=False)
def process_row(row):
    if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
        return row['predicted class name'][1:-1]
    else:
        return row['predicted class name']

df_direct['predicted class name'] = df_direct.apply(process_row, axis=1)
llm_preds = list(df_direct['predicted class name'].values)[:500]
# llm_preds_w_context = ["An image of " + cls for cls in llm_preds]
# llm_preds_w_context = ["A photo of " + cls for cls in llm_preds]
llm_preds_w_context = llm_pred_desc

# Load the tokenizer and model for the CLIP text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize the sentence and convert it to tensor format
summary_token = tokenizer(pred_summary, return_tensors="pt", truncation=True, padding=True)
src_class_list_w_context_tokens = tokenizer(src_class_list_w_context, return_tensors="pt", truncation=True, padding=True)
# src_class_list_tokens = tokenizer(src_class_list, return_tensors="pt", truncation=True, padding=True)
src_class_list_tokens = tokenizer(label_desc, return_tensors="pt", truncation=True, padding=True)
llm_preds_w_context_token = tokenizer(llm_preds_w_context, return_tensors="pt", truncation=True, padding=True)

# Compute the embeddings using the text encoder
with torch.no_grad():
    # Get the text embeddings using the CLIP model's text encoder
    sentence_embedding = model.get_text_features(**summary_token)
    class_list_w_context_embeddings = model.get_text_features(**src_class_list_w_context_tokens)
    # class_list_w_context_embeddings = model.get_text_features(**src_class_list_tokens)
    llm_preds_w_context_embedding = model.get_text_features(**llm_preds_w_context_token)

# Normalize the embeddings for cosine similarity
if mean_centering:
    mean_embedding = torch.mean(sentence_embedding, dim=0)
    mean_centered_sentence_embedding = sentence_embedding - mean_embedding
    sentence_embedding = mean_centered_sentence_embedding / mean_centered_sentence_embedding.norm(dim=-1, keepdim=True)

    mean_class_list_w_context_embedding = torch.mean(class_list_w_context_embeddings, dim=0)
    # mean_centered_class_list_w_context_embedding = class_list_w_context_embeddings - mean_class_list_w_context_embedding + mean_embedding
    mean_centered_class_list_w_context_embedding = class_list_w_context_embeddings - mean_class_list_w_context_embedding
    class_list_w_context_embeddings = mean_centered_class_list_w_context_embedding / mean_centered_class_list_w_context_embedding.norm(dim=-1, keepdim=True)
else:
    sentence_embedding = sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)
    class_list_w_context_embeddings = class_list_w_context_embeddings / class_list_w_context_embeddings.norm(dim=-1, keepdim=True)
    llm_preds_w_context_embedding = llm_preds_w_context_embedding / llm_preds_w_context_embedding.norm(dim=-1, keepdim=True)

# Compute cosine similarity: dot product of the normalized vectors
print(sentence_embedding.size())
print(llm_preds_w_context_embedding.size())
similarity_scores_w_all_lbls = (sentence_embedding @ class_list_w_context_embeddings.T).squeeze(0)

cos = torch.nn.CosineSimilarity(dim=1) 
similarity_scores_w_llm_preds = cos(sentence_embedding, llm_preds_w_context_embedding).cpu().numpy()

cos_sim_df = pd.DataFrame(similarity_scores_w_all_lbls)
cos_sim_w_llm_df = pd.DataFrame(list(zip(llm_preds, similarity_scores_w_llm_preds)), columns=['llm preds', 'scores'])

if mean_centering:  
    cos_sim_df.to_csv(f"Domain {domain} similarity_score_add_context_to_labels_mean_centering_independent.csv", header=False, index=False)
else:
    cos_sim_df.to_csv(f"Domain {domain} similarity_score_add_context_to_labels_desc_v14.csv", header=False, index=False)
    cos_sim_w_llm_df.to_csv(f"Domain {domain} similarity_score_add_context_to_labels_w_llm_preds_desc_v14.csv", header=True, index=False)

# class_list_w_context_similarity_scores = (class_list_w_context_embeddings @ class_list_w_context_embeddings.T).squeeze(0)
# class_list_w_context_similarity_scores.fill_diagonal_(-torch.inf)
# most_similar_label_w_context = {}
# most_sim_lbl = np.argmax(class_list_w_context_similarity_scores, axis=1)
# for lbl, most_sim in enumerate(most_sim_lbl):
#     # most_similar_label_w_context[src_class_list[lbl]] = src_class_list[most_sim]
#     most_similar_label_w_context[lbl] = most_sim.cpu()
# print("most_similar_label_w_context", most_similar_label_w_context)

if mean_centering:
    similarity_scores_w_all_lbls = pd.read_csv(f"Domain {domain} similarity_score_add_context_to_labels_mean_centering_independent.csv", header=None, index_col=False).to_numpy()
else:
    similarity_scores_w_all_lbls = pd.read_csv(f"Domain {domain} similarity_score_add_context_to_labels_desc_v14.csv", header=None, index_col=False).to_numpy()
    similarity_scores_w_llm_preds = pd.read_csv(f"Domain {domain} similarity_score_add_context_to_labels_w_llm_preds_desc_v14.csv", index_col=False)

preds_all = np.argmax(similarity_scores_w_all_lbls, axis=1)
gt = [class_dict[lbl] for lbl in gt_labels]

# ratio of highest cos score / sum of all cos scores
# for known
df_concat = pd.concat([df, similarity_scores_w_llm_preds], axis=1)
known_idx = df_concat.index[df_concat['private'] == False].tolist()
known_outside = df_concat[(df_concat['private'] == False) & (~df_concat['llm preds'].isin(src_class_list))]
print("known outside", len(known_outside))
known_in_idx = df_concat.index[(df_concat['private'] == False) & (df_concat['llm preds'].isin(src_class_list))]
# print("known", len(known_idx), len(known_outside), len(known_in_idx))
preds_known = preds_all[known_in_idx]
# most_sim_lbl_known = [most_similar_label_w_context[pred] for pred in preds_known]

tau = 1
# known_ratios = [np.exp(list(similarity_scores_w_llm_preds['scores'].values)[idx] / tau) / np.sum(np.exp(similarity_scores_w_all_lbls[idx]) / tau) for idx in known_idx]
# known_ratios = [stats.percentileofscore(similarity_scores_w_all_lbls[idx], list(similarity_scores_w_llm_preds['scores'].values)[idx], 'weak') / 100 for idx in known_in_idx]

known_ratios = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(known_in_idx, preds_known)]
# print(known_ratios)
equal_to_one = sum([1 for i in known_ratios if round(i, 5) >= 1])
smaller_than_one = sum(1 for i in known_ratios if round(i, 5) < 1)
print("known", equal_to_one, smaller_than_one)
# print("known", equal_to_one, len(known_ratios))

# larger_known_preds = [list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx] for idx, i in zip(known_in_idx, preds_known) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) > 1]
# print("known larger", larger_known_preds)
# print("known larger", sum([True for pred in larger_known_preds if pred in class_list]), len(larger_known_preds))

corr_known, corr_unknown = [], []
lbl_corr_known = []

# for threshold in np.arange(0.90, 1.05, 0.01):
threshold = 1.0
equal_known_preds = [list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx] for idx, i in zip(known_in_idx, preds_known) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == threshold]
# print("known equal", sum(equal_known_preds), len(equal_known_preds))

# smaller_known_preds = [list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx] for idx, i in zip(known_in_idx, preds_known) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) < threshold]
# print("known smaller", sum(smaller_known_preds), len(smaller_known_preds))

# print("known", len(equal_known_preds), len(known_idx), len(equal_known_preds) / len(known_idx))
# known_ratios_small_scale = [np.exp(similarity_scores_w_all_lbls[idx, i]) / (np.exp(similarity_scores_w_all_lbls[idx, i]) + np.exp(similarity_scores_w_all_lbls[idx, j])) for idx, i, j in zip(known_idx, preds_known, most_sim_lbl_known)]
# known_median = np.median(known_ratios)
# known_median_small_scale = np.median(known_ratios_small_scale)
# known_ratios = np.max(similarity_scores[known_idx], axis=1) 
# known_ratios = np.sum(np.exp(similarity_scores[known_idx]), axis=1)

# for unknown
unknown_idx = df.index[df['private'] == True].tolist()
unknown_outside = df_concat[(df_concat['private'] == True) & (~df_concat['llm preds'].isin(src_class_list))]
print("unknown outside", len(unknown_outside))
unknown_in_idx = df_concat.index[(df_concat['private'] == True) & (df_concat['llm preds'].isin(src_class_list))]
# print("unknown inside", len(unknown_in_idx))
preds_unknown = preds_all[unknown_in_idx]
# most_sim_lbl_unknown = [most_similar_label_w_context[pred] for pred in preds_unknown]
# unknown_ratios = [np.exp(list(similarity_scores_w_llm_preds['scores'].values)[idx] / tau) / np.sum(np.exp(similarity_scores_w_all_lbls[idx]) / tau) for idx in unknown_idx]
# unknown_ratios = [stats.percentileofscore(similarity_scores_w_all_lbls[idx], list(similarity_scores_w_llm_preds['scores'].values)[idx], 'weak') / 100 for idx in unknown_in_idx]

unknown_ratios = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(unknown_in_idx, preds_unknown)]
# u_larger_than_one = sum(1 for i in unknown_ratios if round(i, 5) > 1)
u_equal_to_one = sum(1 for i in unknown_ratios if round(i, 5) >= 1)
u_smaller_than_one = sum(1 for i in unknown_ratios if round(i, 5) < 1)
print("unknown", u_equal_to_one, u_smaller_than_one)

# larger_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) > 1]
# print("unknown larger", larger_unknown_preds)
# print("unknown larger", sum([True for pred in larger_unknown_preds if pred in class_list]), len(larger_unknown_preds))

equal_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_in_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == threshold]
# print("unknown equal", sum([True for pred in equal_unknown_preds if pred in class_list]), len(equal_unknown_preds))

# smaller_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_in_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) < threshold]
# print("unknown smaller", sum([True for pred in smaller_unknown_preds if pred in class_list]), len(smaller_unknown_preds))

print("unknown", 1 - len(equal_unknown_preds) / len(unknown_idx))
print("known", sum(equal_known_preds) / len(known_idx))
known_acc = sum(equal_known_preds) / len(known_idx)
unknown_acc = 1 - len(equal_unknown_preds) / len(unknown_idx)
h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
print("h score", h_score)
print("v14 desc")

# corr_known.append(len(equal_known_preds) / len(known_idx))
# corr_unknown.append(1 - len(equal_unknown_preds) / len(unknown_idx))
# lbl_corr_known.append(sum(equal_known_preds) / len(known_idx))

# fig = plt.figure(figsize=(32, 14))
# ax = plt.subplot(111)
# plt.plot(np.arange(0.9, 1.05, 0.01), corr_known)
# plt.plot(np.arange(0.9, 1.05, 0.01), corr_unknown)
# plt.plot(np.arange(0.9, 1.05, 0.01), lbl_corr_known)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'Domain {domain}', fontsize=30)
# plt.legend(['accuracy known', 'accuracy unknown', 'lbl prediction accuracy known'], fontsize=30)
# # labels, title and ticks
# # ax.set_xlim(0.004, 0.009)
# ax.set_xlabel("Threshold", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=45, fontsize=20)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig("Threshold.png")

# unknown_ratios_small_scale = [np.exp(similarity_scores_w_all_lbls[idx, i]) / (np.exp(similarity_scores_w_all_lbls[idx, i]) + np.exp(similarity_scores_w_all_lbls[idx, j])) for idx, i, j in zip(unknown_idx, preds_unknown, most_sim_lbl_unknown)]
# unknown_median = np.median(unknown_ratios)
# unknown_median_small_scale = np.median(unknown_ratios_small_scale)
# unknown_ratios = np.max(similarity_scores[unknown_idx], axis=1)
# unknown_ratios = np.sum(np.exp(similarity_scores[unknown_idx]), axis=1)
# ratios = [known_ratios, unknown_ratios]
# ratios_small_scale = [known_ratios_small_scale, unknown_ratios_small_scale]
# ratios = [known_correct_ratios, known_incorrect_ratios, unknown_ratios]

# fig = plt.figure(figsize=(32, 14))
# ax = plt.subplot(111)
# sns.histplot(ratios, bins=50, stat='probability', common_norm=False)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'Domain {domain}', fontsize=30)
# plt.legend(['unknown', 'known'], fontsize=30)
# plt.axvline(known_median, c='blue')
# plt.axvline(unknown_median, c='red')
# # labels, title and ticks
# # ax.set_xlim(0.004, 0.009)
# ax.set_xlabel("Ratio", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=45, fontsize=20)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Density', fontsize=30)
# if mean_centering:
#     plt.savefig("known_unknown_add_label_context_mean_centering_independent.png")
# else:
#     plt.savefig("known_unknown_add_label_context_w_llm_preds_ori.png")

# fig = plt.figure(figsize=(32, 14))
# ax = plt.subplot(111)
# sns.histplot(ratios_small_scale, bins=50, stat='probability', common_norm=False)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'Domain {domain}', fontsize=30)
# plt.legend(['unknown', 'known'], fontsize=30)
# plt.axvline(known_median_small_scale, c='blue')
# plt.axvline(unknown_median_small_scale, c='red')
# # labels, title and ticks
# ax.set_xlim(0.5, 0.7)
# ax.set_xlabel("Ratio", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=45, fontsize=20)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Density', fontsize=30)
# if mean_centering:
#     plt.savefig("known_unknown_add_label_context_denom_small_scale_mean_centering_independent.png")
# else:
#     plt.savefig("known_unknown_add_label_context_denom_small_scale.png")

# accuracy_all_indirect = []
# most_common_all = []
# for idx, cls in enumerate(shared_class_list):
#     indexes = df.index[df['ground truth'] == cls].to_list()
#     class_idx = src_class_list.index(cls)
#     # print(cls, idx, preds_all[indexes])
#     most_common = Counter(preds_all[indexes]).most_common()
#     # print("most common", most_common, [src_class_list[i] for i in list(list(zip(*most_common))[0])])
#     acc = np.sum(preds_all[indexes] == class_idx) / len(preds_all[indexes])
#     # print(cls, "total:", len(preds_all[indexes]), "most common", [(src_class_list[cls], list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))])
#     # print()
#     most_common_all.append([cls, len(preds_all[indexes]), acc, [(src_class_list[cls], list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))]])
#     accuracy_all_indirect.append(acc)

# indirect_most_common = pd.DataFrame(most_common_all, columns=["label name", "total num", "pred acc", "most common pred"])
# if mean_centering:
#     indirect_most_common.to_csv(f"indirect_most_common_predicted_add_label_context_mean_centering_independent.csv")
# else:
#     indirect_most_common.to_csv(f"indirect_most_common_predicted_add_label_context.csv")

# print("indirect acc", np.mean(accuracy_all_indirect))

# # figure out the classes with accuracy difference more than 10%
# # get the accuracy difference, and sort them from high to low in a csv file with one column being the label name and the other being the difference
# acc_diff = [x - y for x, y in zip(accuracy_all_direct, accuracy_all_indirect)]
# zipped_diff = list(zip(shared_class_list, acc_diff, direct_most_common["most common pred"], indirect_most_common["most common pred"]))
# print(sorted(zipped_diff, key=lambda x: np.abs(x[1]), reverse=True))
# zipped_diff = sorted(zipped_diff, key=lambda x: np.abs(x[1]), reverse=True)
# df_acc_diff = pd.DataFrame(zipped_diff, columns=["label name", "accuracy difference (direc - indirect)", "direct's pred", "indirect's pred"])
# if mean_centering:
#     df_acc_diff.to_csv(f"Domain {domain} direct_vs_indirect_acc_diff_add_label_context_mean_centering_independent.csv")
# else:
#     df_acc_diff.to_csv(f"Domain {domain} direct_vs_indirect_acc_diff_add_label_context.csv")