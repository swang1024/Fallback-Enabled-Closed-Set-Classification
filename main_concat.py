import os
# import base64
# from openai import OpenAI
# from dataset.dataset import SFUniDADataset
# from torch.utils.data.dataloader import DataLoader
# import tqdm
# from config.model_config import build_args
from net_utils import set_random_seed
# from pathlib import Path
import numpy as np
import pandas as pd
# import json
# from pydantic import BaseModel
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from transformers import CLIPTokenizer, CLIPModel
# import torch
import seaborn as sns
from collections import Counter
# from scipy import stats
from sklearn.metrics import confusion_matrix

set_random_seed(2025)

# Load llm generated image summary
# df1 = pd.read_csv(f'VisDA_target_domain1_4o-mini_summary_v10_sorted.csv')

# dataset = "VisDA"
# domain = 1
# class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
# similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels.csv", header=None, index_col=False).to_numpy()
# df = pd.read_csv(f'VisDA_target_domain1_4o-mini_v8.csv')[:len(similarity_scores_w_all_lbls)]

# preds_all = np.argmax(similarity_scores_w_all_lbls, axis=1)
# accuracy_all_direct = []
# most_common_all = []
# for idx, cls in enumerate(class_list):
#     indexes = df.index[df['ground truth'] == cls].to_list()
#     class_idx = class_list.index(cls)
#     # print(cls, idx, preds_all[indexes])
#     most_common = Counter(preds_all[indexes]).most_common()
#     # print("most common", most_common, [src_class_list[i] for i in list(list(zip(*most_common))[0])])
#     acc = np.sum(preds_all[indexes] == class_idx) / len(preds_all[indexes])
#     # print(cls, "total:", len(preds_all[indexes]), "most common", [(src_class_list[cls], list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))])
#     # print()
#     most_common_all.append([cls, len(preds_all[indexes]), acc, [(src_class_list[cls], list(list(zip(*most_common))[1])[i]) for i, cls in enumerate(list(list(zip(*most_common))[0]))]])
#     accuracy_all_direct.append(acc)

# print("direct acc", accuracy_all_indirect)

# def process_row(row):
#     if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
#         return row['predicted class name'][1:-1]
#     else:
#         return row['predicted class name']
# df['predicted class name'] = df.apply(process_row, axis=1)
# tgt_list = ['skateboard', 'train', 'truck']
# for idx, cls in enumerate(tgt_list):
#     print(cls)
#     print(Counter(list(df[df['ground truth'] == cls]['predicted class name'].values)).most_common())

# cm=confusion_matrix(list(df2['private'].values), list(df2['unknown'].values))

# fig = plt.figure(figsize=(16, 14))
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, cmap="crest"); #annot=True to annotate cells
# # labels, title and ticks
# ax.set_xlabel('Predicted', fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=90)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('True', fontsize=20)
# plt.yticks(rotation=0)
# plt.title(f'VisDA Confusion Matrix', fontsize=20)
# plt.savefig(f"VisDA_confusion_matrix.png")

# # df_concat = pd.concat([df1, df2], axis=0)
# df2.to_csv(os.path.join("VisDA_target_domain1_4o-mini_v8_sorted.csv"), index=False)
# print(len(list(df2['idx'].values)))

# df1 = pd.read_csv(f'target_domain1_4o-mini_v10_summary.csv')
# df2 = pd.read_csv(f'DomainNet_target_domain1_4o-mini_summary_v10_2.csv')

# df3 = pd.read_csv(f'target_domain1_4o-mini_v8_1.csv')
# df4 = pd.read_csv(f'DomainNet_target_domain1_4o-mini_v8_2.csv')

# print(len(df1))
# print(len(df2))

# print(len(df3))
# print(len(df4))

# print(df1.iloc[len(df1)-1])

# print(df2.iloc[len(df2)-1])

# idx1 = set(list(df1['idx'].values))
# idx2 = set(list(df2['idx'].values))
# intersection_set = idx1.intersection(idx2)
# intersection_list = list(intersection_set)

# print(intersection_list)

dataset = "VisDA"
domain = 1
model_name = "4o-mini"
df = pd.read_csv(f'llm_data/{dataset}_target_domain{domain}_{model_name}_summary_v10.csv')
# print(df.head())

df_direct = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)

df1 = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_qwen_v8.csv", index_col=False)

df_direct['idx'] = pd.Categorical(df_direct['idx'], categories=df1['idx'], ordered=True)
df_direct = df_direct.sort_values('idx')
df_direct.to_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8_reordered.csv", index=False)

df['idx'] = pd.Categorical(df['idx'], categories=df1['idx'], ordered=True)
df = df.sort_values('idx')
df.to_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_summary_v10_reordered.csv", index=False)