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
        df = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v10_2.csv')
        # df3 = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v10_3.csv')
        # df = pd.concat([df1, df2], axis=0)[:15000]
        # df = df.reset_index(drop=True)
elif dataset == "VisDA":
    domain = 1
    num_samples = 4000
    version = "v13"
    class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
    df = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v13.csv', index_col=False)[:num_samples]
    df_direct = pd.read_csv(f"{dataset}_target_domain{domain}_4o-mini_v8.csv", index_col=False)[:num_samples]
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
gt_labels = list(df['ground truth'].values)
tgt_priv_class_list = list(df[df['private'] == True]['ground truth'].unique())
shared_class_list = list(df[df['private'] == False]['ground truth'].unique())

print("gt", gt_labels)

src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 
src_class_list = shared_class_list + src_priv_class_list

# Add context to the labels
src_class_list_w_context = ["An image of " + cls for cls in src_class_list]

if dataset == "DomainNet":
    # Load predicted labels from llm
    df_direct_ = pd.read_csv(f"{dataset}_target_domain{domain}_4o-mini_v8.csv", index_col=False)[:12907]
    # df_direct1 = df_direct_[:12384]
    # df_direct2 = df_direct_[12385:29479]

    # df_direct1 = df_direct_[:11556]
    # df_direct2 = df_direct_[11557:50001]

    # domain 1
    # df_direct1 = df_direct_[:14914]
    # df_direct2 = df_direct_[14915:15001]
    # df_direct = pd.concat([df_direct1, df_direct2], axis=0)
    df_direct = df_direct_.reset_index(drop=True)
else:
    domain = 1
    # df_direct = pd.read_csv(f"{dataset}_target_domain{domain}_4o-mini_v8.csv", index_col=False) 
    # df_direct = df_direct.sort_values(by='idx')
    # print(df_direct.head())

# count_yes = 0
# count_no = 0
# for i in range(len(df_direct)):
#     if list(df_direct['idx'].values)[i] == list(df['idx'].values)[i]:
#         print("yes")
#         count_yes += 1

# print(count_yes, count_no)

# def process_preds(row):
#     if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
#         return row['predicted class name'][1:-1]
#     else:
#         return row['predicted class name']

def process_preds(row):
    return re.sub('\'', '', row['predicted class name'])
df_direct['predicted class name'] = df_direct.apply(process_preds, axis=1)
llm_preds = list(df_direct['predicted class name'].values)
print("llm preds", llm_preds)
llm_preds_w_context = ["An image of " + cls for cls in llm_preds]

def process_summary(row):
    if "main object of the " in row['summary']:
        return row['summary'].split("main object of the ")[1]
    else:
        return row['summary']
df['summary'] = df.apply(process_summary, axis=1)
pred_summary = list(df['summary'].values)

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
    similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{version}.csv", header=None, index_col=False).to_numpy()
    similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{version}.csv", index_col=False)

preds_all = np.argmax(similarity_scores_w_all_lbls, axis=1)
print("argmax preds", [src_class_list[i] for i in preds_all])
gt = [class_dict[lbl] for lbl in gt_labels]
    
# ratio of highest cos score / sum of all cos scores
# for known
df.reset_index(drop=True, inplace=True)
similarity_scores_w_llm_preds.reset_index(drop=True, inplace=True)
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

print("unknown", 1 - len(equal_unknown_preds) / len(unknown_idx))
print("known", sum(equal_known_preds) / len(known_idx))
known_acc = sum(equal_known_preds) / len(known_idx)
unknown_acc = 1 - len(equal_unknown_preds) / len(unknown_idx)
h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
print("h score", h_score)
print(f"{version} desc w/ {num_samples} samples")

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