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

# Load llm generated image summary
dataset = "VisDA"
load_scores = True
print(dataset)

if dataset == "DomainNet":
    domain = 1
    version='v10'
    num_samples=20000
    mean_centering = False
    print("domain", domain, "mean_centering", mean_centering)
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
    if domain == 0 or domain == 2:
        df1 = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v10_first_half.csv')
        df2 = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v10_second_half.csv')
        # df = pd.concat([df1, df2], axis=0)[:29478]
        df = pd.concat([df1, df2], axis=0)[:50000]
        df = df.reset_index(drop=True)
        # df = df.set_index('idx')
    else:
        # df = pd.read_csv(f'target_domain{domain}_4o-mini_v10_summary.csv')
        df = pd.read_csv(f'target_domain{domain}_4o-mini_v10_summary.csv')[:num_samples]
        # df3 = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v10_3.csv')
        # df = pd.concat([df1, df2], axis=0)[:15000]
        # df = df.reset_index(drop=True)
    df_direct_ = pd.read_csv(f"target_domain{domain}_4o-mini_v8.csv", index_col=False)[:num_samples]
    df_direct = df_direct_.reset_index(drop=True)
elif dataset == "VisDA":
    domain = 1
    version='v10'
    model_name = "4o-mini"
    num_samples = 4000
    class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
    df = pd.read_csv(f'{dataset}_target_domain{domain}_{model_name}_summary_{version}_reordered.csv', index_col=False)[:num_samples]
    df_direct = pd.read_csv(f"{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)[:num_samples]
    # df = df.sort_values(by='idx')
    # df_direct = df_direct.sort_values(by='idx')

# df_preordered = df_[df_['img url'].isin(df_direct['img url'])]
# df = df_preordered.set_index('img url').loc[df_direct['img url']].reset_index()
# df['private_'] = df_direct['private']
# df = df.drop('private', axis=1)
# new_order = ["idx", "ground truth", "private_", "summary", "img url"]
# df = df[new_order]
# df.to_csv(os.path.join("VisDA_target_domain1_4o-mini_summary_v10_reordered.csv"), index=False)
# print(df.head())

gt_unknown = list(df['private'].values)
pred_summary = list(df['summary'].values)
gt_labels = list(df['ground truth'].values)
tgt_priv_class_list = list(df[df['private'] == True]['ground truth'].unique())
shared_class_list = list(df[df['private'] == False]['ground truth'].unique())

src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 
src_class_list = shared_class_list + src_priv_class_list
print(src_class_list)
# Add context to the labels
src_class_list_w_context = ["An image of " + cls for cls in src_class_list]

# count_yes = 0
# count_no = 0
# for i in range(len(df_direct)):
#     if list(df_direct['idx'].values)[i] == list(df['idx'].values)[i]:
#         print("yes")
#         count_yes += 1

# print(count_yes, count_no)

def process_row(row):
    if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
        return row['predicted class name'][1:-1]
    else:
        return row['predicted class name']
df_direct['predicted class name'] = df_direct.apply(process_row, axis=1)
llm_preds = list(df_direct['predicted class name'].values)
llm_preds_w_context = ["An image of " + cls for cls in llm_preds]

if not load_scores:
    # Load the tokenizer and model for the CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Tokenize the sentence and convert it to tensor format
    summary_token = tokenizer(pred_summary, return_tensors="pt", truncation=True, padding=True)
    src_class_list_w_context_tokens = tokenizer(src_class_list_w_context, return_tensors="pt", truncation=True, padding=True)
    llm_preds_w_context_token = tokenizer(llm_preds_w_context, return_tensors="pt", truncation=True, padding=True)

    # Compute the embeddings using the text encoder
    with torch.no_grad():
        # Get the text embeddings using the CLIP model's text encoder
        sentence_embedding = model.get_text_features(**summary_token)
        class_list_w_context_embeddings = model.get_text_features(**src_class_list_w_context_tokens)
        llm_preds_w_context_embedding = model.get_text_features(**llm_preds_w_context_token)

    # Normalize the embeddings for cosine similarity
    sentence_embedding = sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)
    class_list_w_context_embeddings = class_list_w_context_embeddings / class_list_w_context_embeddings.norm(dim=-1, keepdim=True)
    llm_preds_w_context_embedding = llm_preds_w_context_embedding / llm_preds_w_context_embedding.norm(dim=-1, keepdim=True)

    # Compute cosine similarity: dot product of the normalized vectors
    similarity_scores_w_all_lbls = (sentence_embedding @ class_list_w_context_embeddings.T).squeeze(0)

    cos = torch.nn.CosineSimilarity(dim=1) 
    similarity_scores_w_llm_preds = cos(sentence_embedding, llm_preds_w_context_embedding).cpu().numpy()

    cos_sim_df = pd.DataFrame(similarity_scores_w_all_lbls)
    cos_sim_w_llm_df = pd.DataFrame(list(zip(llm_preds, similarity_scores_w_llm_preds)), columns=['llm preds', 'scores'])

    cos_sim_df.to_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{version}.csv", header=False, index=False)
    cos_sim_w_llm_df.to_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{version}.csv", header=True, index=False)
    similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{version}.csv", header=None, index_col=False).to_numpy()
    similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{version}.csv", index_col=False)
else:
    if dataset == "DomainNet":
        similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels.csv", header=None, index_col=False).to_numpy()
        similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds.csv", index_col=False)
    else:
        # similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels.csv", header=None, index_col=False).to_numpy()
        # similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds.csv", index_col=False)
        similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{version}.csv", header=None, index_col=False).to_numpy()
        similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{version}.csv", index_col=False)

preds_all = np.argmax(similarity_scores_w_all_lbls, axis=1)
gt = [class_dict[lbl] for lbl in gt_labels]

# ratio of highest cos score / sum of all cos scores
# for known
df.reset_index(drop=True, inplace=True)
similarity_scores_w_llm_preds.reset_index(drop=True, inplace=True)
df_concat = pd.concat([df, similarity_scores_w_llm_preds], axis=1)
print("---------", df_concat.head())
known_idx = df_concat.index[df_concat['private'] == False].tolist()

known_outside = df_concat[(df_concat['private'] == False) & (~df_concat['llm preds'].isin(src_class_list))]
print("known outside", len(known_outside))
known_in_idx = df_concat.index[(df_concat['private'] == False) & (df_concat['llm preds'].isin(src_class_list))]
# print("known", len(known_idx), len(known_outside), len(known_in_idx))
preds_known = preds_all[known_in_idx]
# most_sim_lbl_known = [most_similar_label_w_context[pred] for pred in preds_known]

# tau = 1
# known_ratios = [np.exp(list(similarity_scores_w_llm_preds['scores'].values)[idx] / tau) / np.sum(np.exp(similarity_scores_w_all_lbls[idx]) / tau) for idx in known_idx]
# known_ratios = [stats.percentileofscore(similarity_scores_w_all_lbls[idx], list(similarity_scores_w_llm_preds['scores'].values)[idx], 'weak') / 100 for idx in known_in_idx]

known_ratios = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(known_in_idx, preds_known)]
# print(known_ratios)
equal_to_one = sum([1 for i in known_ratios if round(i, 5) == 1])
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
u_equal_to_one = sum(1 for i in unknown_ratios if round(i, 5) == 1)
u_smaller_than_one = sum(1 for i in unknown_ratios if round(i, 5) < 1)
print("unknown", u_equal_to_one, u_smaller_than_one)

# larger_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) > 1]
# print("unknown larger", larger_unknown_preds)
# print("unknown larger", sum([True for pred in larger_unknown_preds if pred in class_list]), len(larger_unknown_preds))

equal_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_in_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == threshold]
# print("unknown equal", sum([True for pred in equal_unknown_preds if pred in class_list]), len(equal_unknown_preds))

# smaller_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_in_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) < threshold]
# print("unknown smaller", sum([True for pred in smaller_unknown_preds if pred in class_list]), len(smaller_unknown_preds))

##############################
# make boxplots with accuracy of known classes and unknown classes for VisDA
kwn_per_class_num, kwn_per_class_correct = [], []
ukn_per_class_num, ukn_per_class_correct = [], []
cls_lst = ['horse', 'car', 'aeroplane', 'bicycle', 'bus', 'knife', 'truck', 'train', 'skateboard']
cls_lst_cls_num = []
per_cls_acc = []
for i, label in enumerate(cls_lst):
    # check if pred_label is shared or not
    if label not in tgt_priv_class_list:
        label_idx = df_concat.index[df_concat['ground truth'] == label]
        known_in_idx = df_concat.index[(df_concat['ground truth'] == label) & (df_concat['llm preds'].isin(src_class_list))]
        preds_known_cls = preds_all[known_in_idx]
        correct_idx = [idx for idx, i in zip(known_in_idx, preds_known_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1 and list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx]]
        num_corr = len(correct_idx)
        kwn_per_class_num.append(len(label_idx))
        kwn_per_class_correct.append(len(correct_idx))
    else:
        print(label)
        label_idx = df_concat.index[df_concat['ground truth'] == label]
        unknown_in_idx = df_concat.index[(df_concat['ground truth'] == label) & (df_concat['llm preds'].isin(src_class_list))]
        preds_unknown_cls = preds_all[unknown_in_idx]
        incorrect_idx = [idx for idx, i in zip(unknown_in_idx, preds_unknown_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1]
        num_corr = len(label_idx) - len(incorrect_idx)
        ukn_per_class_num.append(len(label_idx))
        ukn_per_class_correct.append(len(label_idx) - len(incorrect_idx))
    cls_lst_cls_num.append(len(label_idx))
    per_cls_acc.append(num_corr/ len(label_idx))
kwn_per_class_acc = [x / (y + 1e-5) for x, y in zip(kwn_per_class_correct, kwn_per_class_num)]
print(kwn_per_class_acc)
ukn_per_class_acc = [x / (y + 1e-5) for x, y in zip(ukn_per_class_correct, ukn_per_class_num)]
print(ukn_per_class_acc)
known_acc = sum(kwn_per_class_acc) / len(kwn_per_class_acc)
unknown_acc = sum(ukn_per_class_correct) / sum(ukn_per_class_num)
h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
print("h-score", h_score)

fig = plt.figure(figsize=(20, 14))
ax = plt.subplot(111)
model_name = "glc"
kwn_per_class_acc = [0.897,	0.755, 0.637, 0.461, 0.889,	0.951]
ukn_per_class_acc = [0.65190706, 0.70774315, 0.65663302]
ax.boxplot([kwn_per_class_acc, ukn_per_class_acc], labels=['known classes', 'unknown classes'])
plt.yticks(rotation=0, fontsize=20)
plt.title(f'VisDA {model_name}', fontsize=30)
# labels, title and ticks
# ax.set_xlim(0.004, 0.009)
# ax.set_xticklabels(tgt_priv_class_list)
# ax.set_xlabel("class name", fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=0, fontsize=30)
ax.xaxis.tick_bottom()
ax.set_ylabel('Accuracy', fontsize=30)
plt.savefig(f"VisDA_Accuracy_{model_name}_boxplot.png")

fig = plt.figure(figsize=(20, 14))
ax = plt.subplot(111)
ax.bar(np.arange(len(cls_lst_cls_num)), cls_lst_cls_num, label=cls_lst)
plt.yticks(rotation=0, fontsize=20)
plt.title(f'VisDA {model_name}', fontsize=30)
ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
plt.xticks(rotation=45, fontsize=30)
ax.xaxis.tick_bottom()
ax.set_ylabel('Sample Size', fontsize=30)
plt.savefig(f"VisDA_Sample_Size_{model_name}.png")

fig = plt.figure(figsize=(20, 14))
ax = plt.subplot(111)
ax.bar(np.arange(len(per_cls_acc)), per_cls_acc, label=cls_lst)
plt.yticks(rotation=0, fontsize=20)
plt.title(f'VisDA {model_name}', fontsize=30)
ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
plt.xticks(rotation=45, fontsize=30)
ax.xaxis.tick_bottom()
ax.set_ylabel('Accuracy', fontsize=30)
plt.savefig(f"VisDA_Accuracy_{model_name}_barplot.png")
#################################

print("unknown", 1 - len(equal_unknown_preds) / len(unknown_idx))
print("known", sum(equal_known_preds) / len(known_idx))
known_acc = sum(equal_known_preds) / len(known_idx)
unknown_acc = 1 - len(equal_unknown_preds) / len(unknown_idx)
h_score_tot = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
print("h-score old wrong one", h_score_tot)
# print(f"{version} desc w/ {num_samples} samples")

h_score_all = []
for cls in tgt_priv_class_list:  
    unknown_cls_idx = df_concat.index[(df_concat['private'] == True) & (df_concat['ground truth'] == cls)]
    preds_unknown_cls = preds_all[unknown_cls_idx]
    most_common = Counter(preds_unknown_cls).most_common()
    print("most common", most_common, [src_class_list[i] for i in list(list(zip(*most_common))[0])])
    equal_unknown_cls_preds = [list(df_concat['llm preds'].values)[idx] for idx, i in zip(unknown_cls_idx, preds_unknown_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1]
    
    # unknown_ratios = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(unknown_cls_idx, preds_unknown_cls)]
    # equal_to_one = sum([1 for i in unknown_ratios if round(i, 5) == 1])
    # print(cls, equal_to_one / len(unknown_ratios))

    # unknown_cls_preds = [i for idx, i in zip(unknown_cls_idx, preds_unknown_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1]
    # most_common = Counter(unknown_cls_preds).most_common()
    # print("most common when llm pred == argmax lbl", most_common, [src_class_list[i] for i in list(list(zip(*most_common))[0])])

    # unknown_cls_preds = [i for idx, i in zip(unknown_cls_idx, preds_unknown_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) < 1]
    # most_common = Counter(unknown_cls_preds).most_common()
    # print("most common when llm pred != argmax lbl", most_common, [src_class_list[i] for i in list(list(zip(*most_common))[0])])

    print("unknown", 1 - len(equal_unknown_cls_preds) / len(unknown_cls_idx))
    print("known", sum(kwn_per_class_acc) / len(kwn_per_class_acc))
    # known_acc = sum(equal_known_preds) / len(known_idx)
    known_acc = sum(kwn_per_class_acc) / len(kwn_per_class_acc)
    unknown_acc = 1 - len(equal_unknown_cls_preds) / len(unknown_cls_idx)
    h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    print(f"{cls}'s h-score", h_score)
    print(f"{version} desc w/ {num_samples} samples")
    h_score_all.append(h_score)

# fig = plt.figure(figsize=(40, 14))
# ax = plt.subplot(111)
# plt.plot(np.arange(len(h_score_all)), [h_score_tot]*len(h_score_all))
# plt.bar(np.arange(len(h_score_all)), h_score_all, label=tgt_priv_class_list)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'Domain {domain}', fontsize=30)
# # labels, title and ticks
# # ax.set_xlim(0.004, 0.009)
# # ax.set_xticklabels(tgt_priv_class_list)
# plt.xticks(ticks=np.arange(len(h_score_all)), labels=tgt_priv_class_list)
# ax.set_xlabel("class name", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=90, fontsize=10)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('H-score', fontsize=30)
# plt.savefig(f"{dataset} H-scores.png")

# h_score_visda = [0.86184, 0.79880, 0.54636]

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# ax.boxplot([h_score_visda, h_score_all], labels=['visda', 'domainnet'])
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'H-scores', fontsize=30)
# # labels, title and ticks
# # ax.set_xlim(0.004, 0.009)
# # ax.set_xticklabels(tgt_priv_class_list)
# # ax.set_xlabel("class name", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=0, fontsize=30)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('H-score', fontsize=30)
# plt.savefig(f"H-scores.png")
