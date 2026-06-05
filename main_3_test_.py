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

# # Load llm generated image summary
dataset = "VisDA"
# load_scores = False
# print(dataset)
# model_name = "llama"
# print(model_name)

# args = build_args()

# print(args.target_data_dir)
# target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
# target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

# target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
# print(len(target_train_dataloader))

# if dataset == "DomainNet":
#     domain = 1
#     version='v10'
#     num_samples=14000
#     mean_centering = False
#     print("domain", domain, "mean_centering", mean_centering)
#     class_list = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', \
#         'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', \
#         'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', \
#         'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', \
#         'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', \
#         'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', \
#         'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', \
#         'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', \
#         'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', \
#         'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', \
#         'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', \
#         'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', \
#         'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', \
#         'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer',\
#         'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', \
#         'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', \
#         'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', \
#         'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', \
#         'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', \
#         'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', \
#         'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', \
#         'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']
#     # if domain == 0 or domain == 2:
#     #     df1 = pd.read_csv(f'{dataset}_target_domain{domain}_{model_name}_summary_v10_first_half.csv')
#     #     df2 = pd.read_csv(f'{dataset}_target_domain{domain}_{model_name}_summary_v10_second_half.csv')
#     #     # df = pd.concat([df1, df2], axis=0)[:29478]
#     #     df = pd.concat([df1, df2], axis=0)[:50000]
#     #     df = df.reset_index(drop=True)
#     #     # df = df.set_index('idx')
#     # else:
#     #     # df = pd.read_csv(f'target_domain{domain}_4o-mini_v10_summary.csv')
#     #     df = pd.read_csv(f'{dataset}_target_domain{domain}_{model_name}_summary_v10.csv')[:num_samples]
#     #     # df3 = pd.read_csv(f'{dataset}_target_domain{domain}_4o-mini_summary_v10_3.csv')
#     #     # df = pd.concat([df1, df2], axis=0)[:15000]
#     #     # df = df.reset_index(drop=True)
#     df = pd.read_csv(f'llm_data/{dataset}_target_domain{domain}_{model_name}_summary_v10.csv')[:num_samples]
#     df_direct_ = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)[:num_samples]
#     df_direct = df_direct_.reset_index(drop=True)
# elif dataset == "VisDA":
#     domain = 1
#     version='v10'
#     print("model name", model_name)
#     num_samples = 4000
#     class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
#     df = pd.read_csv(f'llm_data/{dataset}_target_domain{domain}_{model_name}_summary_{version}.csv', index_col=False)[:num_samples]
#     df_direct = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)[:num_samples]
    
# gt_unknown = list(df['private'].values)
# pred_summary = list(df['summary'].values)
# gt_labels = list(df['ground truth'].values)
# tgt_priv_class_list = list(df[df['private'] == True]['ground truth'].unique())
# shared_class_list = list(df[df['private'] == False]['ground truth'].unique())

# src_priv_class_list = [cls for cls in class_list if cls not in shared_class_list]
# class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 
# src_class_list = shared_class_list + src_priv_class_list
# print(src_class_list)
# # Add context to the labels
# src_class_list_w_context = ["An image of " + cls for cls in src_class_list]

# def process_row(row):
#     if row['predicted class name'][0] == '\'' and row['predicted class name'][-1] == '\'':
#         return row['predicted class name'][1:-1]
#     else:
#         return row['predicted class name']
# df_direct['predicted class name'] = df_direct.apply(process_row, axis=1)
# llm_preds = list(df_direct['predicted class name'].values)
# llm_preds_w_context = ["An image of " + cls for cls in llm_preds]

# if not load_scores:
#     # Load the tokenizer and model for the CLIP text encoder
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

#     # Tokenize the sentence and convert it to tensor format
#     summary_token = tokenizer(pred_summary, return_tensors="pt", truncation=True, padding=True)
#     src_class_list_w_context_tokens = tokenizer(src_class_list_w_context, return_tensors="pt", truncation=True, padding=True)
#     llm_preds_w_context_token = tokenizer(llm_preds_w_context, return_tensors="pt", truncation=True, padding=True)

#     # Compute the embeddings using the text encoder
#     with torch.no_grad():
#         # Get the text embeddings using the CLIP model's text encoder
#         sentence_embedding = model.get_text_features(**summary_token)
#         class_list_w_context_embeddings = model.get_text_features(**src_class_list_w_context_tokens)
#         llm_preds_w_context_embedding = model.get_text_features(**llm_preds_w_context_token)

#     # Normalize the embeddings for cosine similarity
#     sentence_embedding = sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)
#     class_list_w_context_embeddings = class_list_w_context_embeddings / class_list_w_context_embeddings.norm(dim=-1, keepdim=True)
#     llm_preds_w_context_embedding = llm_preds_w_context_embedding / llm_preds_w_context_embedding.norm(dim=-1, keepdim=True)

#     # Compute cosine similarity: dot product of the normalized vectors
#     similarity_scores_w_all_lbls = (sentence_embedding @ class_list_w_context_embeddings.T).squeeze(0)

#     cos = torch.nn.CosineSimilarity(dim=1) 
#     similarity_scores_w_llm_preds = cos(sentence_embedding, llm_preds_w_context_embedding).cpu().numpy()

#     cos_sim_df = pd.DataFrame(similarity_scores_w_all_lbls)
#     cos_sim_w_llm_df = pd.DataFrame(list(zip(llm_preds, similarity_scores_w_llm_preds)), columns=['llm preds', 'scores'])

#     cos_sim_df.to_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{model_name}_{version}.csv", header=False, index=False)
#     cos_sim_w_llm_df.to_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{model_name}_{version}.csv", header=True, index=False)
#     similarity_scores_w_all_lbls = pd.read_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{model_name}_{version}.csv", header=None, index_col=False).to_numpy()
#     similarity_scores_w_llm_preds = pd.read_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{model_name}_{version}.csv", index_col=False)
# else:
#     if dataset == "DomainNet":
#         similarity_scores_w_all_lbls = pd.read_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels.csv", header=None, index_col=False).to_numpy()
#         similarity_scores_w_llm_preds = pd.read_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds.csv", index_col=False)
#     else:
#         # similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels.csv", header=None, index_col=False).to_numpy()
#         # similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds.csv", index_col=False)
#         similarity_scores_w_all_lbls = pd.read_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{version}.csv", header=None, index_col=False).to_numpy()
#         similarity_scores_w_llm_preds = pd.read_csv(f"similarity_score_files/{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{version}.csv", index_col=False)

# preds_all = np.argmax(similarity_scores_w_all_lbls, axis=1)
# gt = [class_dict[lbl] for lbl in gt_labels]
    
# # ratio of highest cos score / sum of all cos scores
# ################## for known ####################
# df.reset_index(drop=True, inplace=True)
# similarity_scores_w_llm_preds.reset_index(drop=True, inplace=True)
# df_concat = pd.concat([df, similarity_scores_w_llm_preds], axis=1)

# known_idx = df_concat.index[df_concat['private'] == False].tolist()
# known_outside = df_concat[(df_concat['private'] == False) & (~df_concat['llm preds'].isin(src_class_list))]
# print("known outside", len(known_outside))
# known_in_idx = df_concat.index[(df_concat['private'] == False) & (df_concat['llm preds'].isin(src_class_list))]
# # print("known", len(known_idx), len(known_outside), len(known_in_idx))
# preds_known = preds_all[known_in_idx]
# known_ratios = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(known_in_idx, preds_known)]
# # print(known_ratios)
# equal_to_one = sum([1 for i in known_ratios if round(i, 5) == 1])
# smaller_than_one = sum(1 for i in known_ratios if round(i, 5) < 1)
# print("known", equal_to_one, smaller_than_one)
# # print("known", equal_to_one, len(known_ratios))

# corr_known, corr_unknown = [], []
# lbl_corr_known = []

# threshold = 1.0
# equal_known_preds = [list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx] for idx, i in zip(known_in_idx, preds_known) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == threshold]

# #################### for unknown ###################
# unknown_idx = df.index[df['private'] == True].tolist()
# unknown_outside = df_concat[(df_concat['private'] == True) & (~df_concat['llm preds'].isin(src_class_list))]
# print("unknown outside", len(unknown_outside))
# unknown_in_idx = df_concat.index[(df_concat['private'] == True) & (df_concat['llm preds'].isin(src_class_list))]
# # print("unknown inside", len(unknown_in_idx))
# preds_unknown = preds_all[unknown_in_idx]
# unknown_ratios = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(unknown_in_idx, preds_unknown)]
# # u_larger_than_one = sum(1 for i in unknown_ratios if round(i, 5) > 1)
# u_equal_to_one = sum(1 for i in unknown_ratios if round(i, 5) == 1)
# u_smaller_than_one = sum(1 for i in unknown_ratios if round(i, 5) < 1)
# print("unknown", u_equal_to_one, u_smaller_than_one)
# equal_unknown_preds = [list(similarity_scores_w_llm_preds['llm preds'].values)[idx] for idx, i in zip(unknown_in_idx, preds_unknown) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == threshold]

# ##############################
# # make boxplots with accuracy of known classes and unknown classes for VisDA
# kwn_per_class_num, kwn_per_class_correct = [], []
# ukn_per_class_num, ukn_per_class_correct = [], []
# if dataset == "VisDA":
#     cls_lst = ['horse', 'car', 'aeroplane', 'bicycle', 'bus', 'knife', 'truck', 'train', 'skateboard']
#     tgt_priv = ['truck', 'train', 'skateboard']
# else:
#     cls_lst = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', \
#     'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', \
#     'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', \
#     'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', \
#     'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', \
#     'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', \
#     'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', \
#     'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', \
#     'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', \
#     'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', \
#     'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', \
#     'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', \
#     'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', \
#     'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'necklace', 'nose', \
#     'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', \
#     'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', \
#     'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', \
#     'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', \
#     'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', \
#     'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', \
#     'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', \
#     'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', \
#     'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', \
#     'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', \
#     'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', \
#     'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

#     tgt_priv = ['necklace', 'nose', \
#     'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', \
#     'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', \
#     'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', \
#     'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', \
#     'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', \
#     'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', \
#     'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', \
#     'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', \
#     'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', \
#     'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', \
#     'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', \
#     'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
# cls_lst_cls_num = []
# per_cls_acc = []
# for i, label in enumerate(cls_lst):
#     # check if pred_label is shared or not
#     if label in tgt_priv:
#         # print(label)
#         label_idx = df_concat.index[df_concat['ground truth'] == label]
#         unknown_in_idx = df_concat.index[(df_concat['ground truth'] == label) & (df_concat['llm preds'].isin(src_class_list))]
#         preds_unknown_cls = preds_all[unknown_in_idx]
#         incorrect_idx = [idx for idx, i in zip(unknown_in_idx, preds_unknown_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1]
#         num_corr = len(label_idx) - len(incorrect_idx)
#         ukn_per_class_num.append(len(label_idx))
#         ukn_per_class_correct.append(len(label_idx) - len(incorrect_idx))
#     else:
#         label_idx = df_concat.index[df_concat['ground truth'] == label]
#         known_in_idx = df_concat.index[(df_concat['ground truth'] == label) & (df_concat['llm preds'].isin(src_class_list))]
#         preds_known_cls = preds_all[known_in_idx]
#         correct_idx = [idx for idx, i in zip(known_in_idx, preds_known_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1 and list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx]]
#         num_corr = len(correct_idx)
#         kwn_per_class_num.append(len(label_idx))
#         kwn_per_class_correct.append(len(correct_idx))
#     cls_lst_cls_num.append(len(label_idx))
#     per_cls_acc.append(num_corr/ (len(label_idx) + 1e-5))
# kwn_per_class_acc = [x / (y + 1e-5) for x, y in zip(kwn_per_class_correct, kwn_per_class_num)]
# print(kwn_per_class_acc)
# print(kwn_per_class_num)
# ukn_per_class_acc = [x / (y + 1e-5) for x, y in zip(ukn_per_class_correct, ukn_per_class_num)]
# print(ukn_per_class_acc)
# print(ukn_per_class_num)
# known_acc = sum(kwn_per_class_acc) / len(kwn_per_class_acc)
# unknown_acc = sum(ukn_per_class_correct) / sum(ukn_per_class_num)
# h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
# print("h-score", h_score)

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# ax.boxplot([kwn_per_class_acc, ukn_per_class_acc], labels=['known classes', 'unknown classes'])
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'{dataset} {model_name}', fontsize=30)
# # labels, title and ticks
# # ax.set_xlim(0.004, 0.009)
# # ax.set_xticklabels(tgt_priv_class_list)
# # ax.set_xlabel("class name", fontsize=20)
# ax.xaxis.set_label_position('bottom')
# plt.xticks(rotation=0, fontsize=30)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"{dataset}_Accuracy_{model_name}_boxplot.png")

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# ax.bar(np.arange(len(cls_lst_cls_num)), cls_lst_cls_num, label=cls_lst)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'{dataset} {model_name}', fontsize=30)
# ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
# plt.xticks(rotation=45, fontsize=30)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Sample Size', fontsize=30)
# plt.savefig(f"{dataset}_Sample_Size_{model_name}.png")

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# ax.bar(np.arange(len(per_cls_acc)), per_cls_acc, label=cls_lst)
# plt.yticks(rotation=0, fontsize=20)
# plt.title(f'{dataset} {model_name}', fontsize=30)
# ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
# plt.xticks(rotation=45, fontsize=30)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"{dataset}_Accuracy_{model_name}_barplot.png")
################################

# print("unknown", 1 - len(equal_unknown_preds) / len(unknown_idx))
# print("known", sum(equal_known_preds) / len(known_idx))
# known_acc = sum(equal_known_preds) / len(known_idx)
# unknown_acc = 1 - len(equal_unknown_preds) / len(unknown_idx)
# h_score_tot = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
# print("h-score old wrong one", h_score_tot)
# print(f"{version} desc w/ {num_samples} samples")

############################## VisDA ##########################################
fig = plt.figure(figsize=(20, 14))
ax = plt.subplot(111)
model_name = "glc"
markers = ['^', 's', '*']
colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']
glc_kwn_acc = [0.889, 0.461, 0.897, 0.755, 0.637, 0.951]
glc_ukn_acc = [0.65663302, 0.7077431, 0.65190706]
glc_kwn_x = [0] * 6
glc_ukn_x = [1] * 3
for i in range(len(glc_kwn_acc)):
    plt.scatter(glc_kwn_x[i], glc_kwn_acc[i], color=colors[i], s=60)
for i in range(len(glc_ukn_acc)):
    plt.scatter(glc_ukn_x[i], glc_ukn_acc[i], marker=markers[i], color='black', s=60)

chatgpt_kwn_acc = [0.9813747184861431, 0.6336137807820237, 0.9808324152199759, 0.9326755604603856, 0.914032865933279, 0.8997995901823689]
chatgpt_ukn_acc = [0.4040136296705277, 0.5866601724115863, 0.8804920836512119]
chatgpt_kwn_x = [3] * 6
chatgpt_ukn_x = [4] * 3
for i in range(len(chatgpt_kwn_acc)):
    plt.scatter(chatgpt_kwn_x[i], chatgpt_kwn_acc[i], color=colors[i], s=60)
for i in range(len(chatgpt_ukn_acc)):
    plt.scatter(chatgpt_ukn_x[i], chatgpt_ukn_acc[i], marker=markers[i], color='black', s=60)

llama_kwn_acc = [0.6792452687789252, 0.4621676844359133, 0.4082687233005498, 0.7046070269754193, 0.5366876197759409, 0.5410627757940688]
llama_ukn_acc = [0.6973414990318713, 0.8574938364252129, 0.7607655138389706]
llama_kwn_x = [6] * 6
llama_ukn_x = [7] * 3
for i in range(len(llama_kwn_acc)):
    plt.scatter(llama_kwn_x[i], llama_kwn_acc[i], color=colors[i], s=60)
for i in range(len(llama_ukn_acc)):
    plt.scatter(llama_ukn_x[i], llama_ukn_acc[i], marker=markers[i], color='black', s=60)

qwen_kwn_acc = [0.9392033346078966, 0.6881390522685169, 0.9793281400690403, 0.829268270209532, 0.8742138181506538, 0.7391303990758261]
qwen_ukn_acc = [0.4110429363794901, 0.4815724697402342, 0.46889949909571776]
qwen_kwn_x = [9] * 6
qwen_ukn_x = [10] * 3
for i in range(len(qwen_kwn_acc)):
    plt.scatter(qwen_kwn_x[i], qwen_kwn_acc[i], color=colors[i], s=60)
for i in range(len(qwen_ukn_acc)):
    plt.scatter(qwen_ukn_x[i], qwen_ukn_acc[i], marker=markers[i], color='black', s=60)

plt.yticks(rotation=0, fontsize=20)
plt.title(f'VisDA Known vs Unknown', fontsize=30)
plt.legend(['horse', 'car', 'aeroplane', 'bicycle', 'bus', 'knife', 'truck', 'train', 'skateboard'])
ax.xaxis.set_label_position('bottom')
ax.set_xticks(ticks=np.arange(11), labels=['glc known', 'glc unknown', '', 'chatgpt known', 'chatgpt unknown', '', 'llama known', 'llama unknown', '', 'qwen known', 'qwen unknown'])
plt.xticks(rotation=45, fontsize=10)
ax.xaxis.tick_bottom()
ax.set_ylabel('Accuracy', fontsize=30)
plt.savefig(f"VisDA_Accuracy_{model_name}_dotplot.png")

kwn_sample_size = [477, 978, 387, 369, 477, 207]
ukn_sample_size = [489, 407, 209]

mean_glc_kwn = sum(glc_kwn_acc) / len(glc_kwn_acc)
mean_glc_ukn = sum(glc_ukn_acc) / len(glc_ukn_acc)
print(mean_glc_kwn, mean_glc_ukn)

mean_chatgpt_kwn = sum(chatgpt_kwn_acc) / len(chatgpt_kwn_acc)
mean_chatgpt_ukn = sum(chatgpt_ukn_acc) / len(chatgpt_ukn_acc)
print(mean_chatgpt_kwn, mean_chatgpt_ukn)

mean_llama_kwn = sum(llama_kwn_acc) / len(llama_kwn_acc)
mean_llama_ukn = sum(llama_ukn_acc) / len(llama_ukn_acc)
print(mean_llama_kwn, mean_llama_ukn)

mean_qwen_kwn = sum(qwen_kwn_acc) / len(qwen_kwn_acc)
mean_qwen_ukn = sum(qwen_ukn_acc) / len(qwen_ukn_acc)
print(mean_qwen_kwn, mean_qwen_ukn)
print("------------")
wgt_glc_kwn = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc)])  / sum(kwn_sample_size)
wgt_glc_ukn = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc)])  / sum(ukn_sample_size)
print(wgt_glc_kwn, wgt_glc_ukn)

wgt_chatgpt_kwn = sum([x * y for x, y in zip(kwn_sample_size, chatgpt_kwn_acc)])  / sum(kwn_sample_size)
wgt_chatgpt_ukn = sum([x * y for x, y in zip(ukn_sample_size, chatgpt_ukn_acc)])  / sum(ukn_sample_size)
print(wgt_chatgpt_kwn, wgt_chatgpt_ukn)

wgt_llama_kwn = sum([x * y for x, y in zip(kwn_sample_size, llama_kwn_acc)])  / sum(kwn_sample_size)
wgt_llama_ukn = sum([x * y for x, y in zip(ukn_sample_size, llama_ukn_acc)])  / sum(ukn_sample_size)
print(wgt_llama_kwn, wgt_llama_ukn)

wgt_qwen_kwn = sum([x * y for x, y in zip(kwn_sample_size, qwen_kwn_acc)])  / sum(kwn_sample_size)
wgt_qwen_ukn = sum([x * y for x, y in zip(ukn_sample_size, qwen_ukn_acc)])  / sum(ukn_sample_size)
print(wgt_qwen_kwn, wgt_qwen_ukn)
print("------------")
ratio_mean_glc = mean_glc_kwn / mean_glc_ukn
ratio_mean_chatgpt = mean_chatgpt_kwn / mean_chatgpt_ukn
ratio_mean_llama = mean_llama_kwn / mean_llama_ukn
ratio_mean_qwen = mean_qwen_kwn / mean_qwen_ukn
print(ratio_mean_glc, ratio_mean_chatgpt, ratio_mean_llama, ratio_mean_qwen)

ratio_wgt_mean_glc = wgt_glc_kwn / wgt_glc_ukn
ratio_wgt_mean_chatgpt = wgt_chatgpt_kwn / wgt_chatgpt_ukn
ratio_wgt_mean_llama = wgt_llama_kwn / wgt_llama_ukn
ratio_wgt_mean_qwen = wgt_qwen_kwn / wgt_qwen_ukn
print(ratio_wgt_mean_glc, ratio_wgt_mean_chatgpt, ratio_wgt_mean_llama, ratio_wgt_mean_qwen)
# ############################ DomainNet Painting ##########################################
# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# glc_kwn_acc_r2p = [0.801, 0.521, 0.682,	0.936,	0.780,	0.319,	0.000,	0.323,	0.236,	0.803,	\
# 0.801,	0.963,	0.247,	0.042,	0.552,	0.856,	0.705,	0.005,	0.783,	0.920,	\
# 0.025,	0.588,	0.819,	0.277,	0.716,	0.106,	0.692,	0.258,	0.700,	0.759,	\
# 0.354,	0.388,	0.803,	0.851,	0.345,	0.453,	0.056,	0.012,	0.181,	0.223,	\
# 0.320,	0.857,	0.481,	0.543,	0.253,	0.480,	0.473,	0.629,	0.654,	0.677,	\
# 0.011,	0.000,	0.877,	0.032,	0.004,	0.757,	0.739,	0.718,	0.596,	0.234,	\
# 0.065,	0.443,	0.620,	0.366,	0.496,	0.007,	0.674,	0.614,	0.719,	0.063,	\
# 0.778,	0.775,	0.412,	0.751,	0.413,	0.646,	0.145,	0.590,	0.135,	0.585,	\
# 0.445,	0.000,	0.311,	0.754,	0.427,	0.639,	0.000,	0.766,	0.676,	0.834,	\
# 0.000,	0.050,	0.508,	0.008,	0.522,	0.274,	0.621,	0.714,	0.212,	0.910,	\
# 0.524,	0.861,	0.698,	0.706,	0.224,	0.015,	0.720,	0.244,	0.247,	0.570,	\
# 0.812,	0.010,	0.498,	0.679,	0.431,	0.011,	0.784,	0.730,	0.000,	0.528,	\
# 0.764,	0.800,	0.695,	0.744,	0.481,	0.429,	0.061,	0.507,	0.710,	0.850,	\
# 0.000,	0.058,	0.867,	0.004,	0.000,	0.157,	0.000,	0.859,	0.655,	0.553,	\
# 0.012,	0.814,	0.584,	0.650,	0.784,	0.704,	0.757,	0.502,	0.410,	0.416]
# glc_ukn_acc_r2p = [0.62410329, 0.88397788, 0.9576988,  0.63655913, 0.93112946, 0.74457428,
#  0.60569104, 0.97622192, 0.78035713, 0.92978206, 0.65465463, 0.96933559,
#  0.9648241,  0.9653916,  0.58823528, 0.93469909, 0.96822428, 0.96217492,
#  0.65753423, 0.97647057, 0.97613881, 0.91714284, 0.87719297, 0.9854604,
#  0.83091783, 0.98613516, 0.85518291, 0.50668647, 0.98833332, 0.94701985,
#  0.88918918, 0.98326897, 0.90882352, 0.95931141, 0.94496854, 0.97697367,
#  0.89999999, 0.81249999, 0.9856115,  0.99112425, 0.91457284, 0.98175179,
#  0.95670992, 0.77609426, 0.82310468, 0.97807016, 0.94423075, 0.97081412,
#  0.8854003,  0.96754562, 0.96445495, 0.82901553, 0.83898298, 0.406639,
#  0.97489538, 0.94894365, 0.05592841, 0.88968823, 0.98067631, 0.97069594,
#  0.63934423, 0.95979898, 0.68313457, 0.95485325, 0.87111109, 0.70129867,
#  0.93177736, 0.89057748, 0.90492955, 0.92610835, 0.95132739, 0.9784946,
#  0.97804389, 0.90856312, 0.92424228, 0.97368413, 0.49632351, 0.91525422,
#  0.94999998, 0.2782462,  0.61985018, 0.68575232, 0.88775501, 0.8309858,
#  0.99134198, 0.25212464, 0.99999984, 0.98680737, 0.84360186, 0.89516127,
#  0.93719802, 0.9821428,  0.72964168, 0.97356826, 0.80993519, 0.99389,
#  0.88467873, 0.78240739, 0.83850926, 0.25460122, 0.50259066, 0.98021581,
#  0.93401014, 0.9252971,  0.29155313, 0.54174067, 0.87797146, 0.92992422,
#  0.46764091, 0.73249998, 0.94517542, 0.92711863, 0.46787479, 0.73880596,
#  0.92134829, 0.73927958, 0.91439685, 0.975945,   0.97455967, 0.78068409,
#  0.95754715, 0.94722953, 0.97944198, 0.97761192, 0.93351061, 0.97933882,
#  0.99405645, 0.96675189, 0.70441987, 0.89510486, 0.99321265, 0.90664556,
#  0.25,       0.2360515,  0.96870341, 0.97029701, 0.95827122, 0.67317072,
#  0.87244093, 0.95823093, 0.98520707, 0.38878842, 0.96226413, 0.96778915,
#  0.91844658]

# glc_kwn_acc_s2p = [0.785, 0.717, 0.901, 0.737, 0.769, 0.357, 0.068, 0.014, 0.010, 0.374, 0.151, 0.387, 0.114, 0.525, 0.096, 0.694, 0.579, 0.061, 0.678, 0.434, 0.103, 0.278, 0.409, 0.147, 0.422, 0.014, 0.491, 0.491, 0.087, 0.610, 0.000, 0.024, 0.806, 0.061, 0.293, 0.244, 0.214, 0.569, 0.000, 0.146, 0.454, 0.483, 0.207, 0.652, 0.321, 0.461, 0.780, 0.083, 0.492, 0.897, 0.804, 0.000, 0.705, 0.582, 0.302, 0.500, 0.227, 0.512, 0.615, 0.319, 0.498, 0.448, 0.556, 0.567, 0.867, 0.751, 0.804, 0.712, 0.711, 0.257, 0.715, 0.453, 0.509, 0.634, 0.192, 0.449, 0.278, 0.633, 0.346, 0.705, 0.053, 0.444, 0.462, 0.423, 0.808, 0.810, 0.158, 0.567, 0.395, 0.798, 0.057, 0.529, 0.813, 0.598, 0.276, 0.192, 0.721, 0.657, 0.216, 0.901, 0.048, 0.571, 0.659, 0.693, 0.492, 0.010, 0.718, 0.199, 0.588, 0.767, 0.036, 0.600, 0.000, 0.625, 0.102, 0.000, 0.621, 0.600, 0.666, 0.238, 0.866, 0.012, 0.417, 0.300, 0.089, 0.322, 0.233, 0.536, 0.862, 0.775, 0.268, 0.363, 0.771, 0.155, 0.362, 0.601, 0.386, 0.655, 0.796, 0.239, 0.080, 0.621, 0.302, 0.144, 0.786, 0.665, 0.148, 0.669, 0.542, 0.577]
# glc_ukn_acc_s2p = [0.38040345, 0.44531249, 0.82315788, 0.42105241, 0.92145012, 0.90870486,
#  0.63636306, 0.77419353, 0.60465113, 0.60273971, 0.37232289, 0.29545453,
#  0.7270341, 0.89285706, 0.61428567, 0.21428571, 0.85294093, 0.85365843,
#  0.87946427, 0.64444437, 0.50273221, 0.7494407,  0.84121619, 0.06293706,
#  0.83602148, 0.69938648, 0.70138884, 0.40540539, 0.8346456,  0.4982935,
#  0.25287353, 0.88557212, 0.94444434, 0.60233915, 0.85227263, 0.89655157,
#  0.87254893, 0.83673452, 0.61710035, 0.38554215, 0.76923065, 0.90056816,
#  0.88095228, 0.87931019, 0.72072066, 0.48181816, 0.38333332, 0.84946235,
#  0.7199999, 0.78260867, 0.51863352, 0.64748197, 0.75999995, 0.54469272,
#  0.28787874, 0.8307691,  0.45864658, 0.61643827, 0.96097559, 0.92168669,
#  0.29368029, 0.59281435, 0.3653846,  0.73913039, 0.88392849, 0.55319143,
#  0.81578942, 0.90476186, 0.83798878, 0.94117592, 0.87012976, 0.78816197,
#  0.87294116, 0.94972062, 0.77037035, 0.92674805, 0.29477611, 0.87096746,
#  0.55319145, 0.6233766,  0.49999997, 0.79411741, 0.90972216, 0.87388723,
#  0.4736842, 0.74074047, 0.87755093, 0.91999982, 0.66666611, 0.65317917,
#  0.52941145, 0.91954012, 0.99999938, 0.87358489, 0.90316572, 0.65714276,
#  0.69272726, 0.88235289, 0.89860138, 0.40631163, 0.73856204, 0.8914728,
#  0.84680849, 0.999999,   0.99999917, 0.87499992, 0.69309461, 0.32890364,
#  0.66666658, 0.82352925, 0.77777735, 0.86524817, 0.37677724, 0.80373824,
#  0.99999917, 0.87096746, 0.88990818, 0.81818107, 0.87096746, 0.72654154,
#  0.19125682, 0.89999985, 0.79310343, 0.93345007, 0.50671139, 0.73142853,
#  0.2341772, 0.59016389, 0.85284278, 0.99999917, 0.16666653, 0.83206104,
#  0.24630541, 0.79999947, 0.92926827, 0.99999917, 0.62745096, 0.68421017,
#  0.82115867, 0.71186429, 0.74404757, 0.11111105, 0.9192546,  0.86577178,
#  0.79090902]

# chatgpt_kwn_acc = [0.9999997435898093, 0.8571422448983965, 0.9999995652175804, 0.8571422448983965, 0.3499998250000875, 0.571427755103207, 0.7999992000008, 0.7090907801653127, 0.035087713142506465, 0.6363633471075695, 0.27777762345687584, 0.5238093990930002, 0.11538459319527053, 0.7708331727430889, 0.04166664930556279, 0.9199996320001472, 0.8611108719136467, 0.1818180165290759, 0.865384448964529, 0.9999988888901234, 0.23076905325457442, 0.4210525207756524, 0.914285453061299, 0.7499998295454933, 0.8571416326548106, 0.8599998280000344, 0.8999997750000561, 0.258064432882441, 0.0, 0.8918916508400943, 0.0, 0.10526310249310396, 0.9999994117650519, 0.5384611242606736, 0.749999531250293, 0.37499953125058594, 0.9999900000999989, 0.611110771605127, 0.16666652777789354, 0.7499981250046875, 0.34090901342976965, 0.3846152366864474, 0.5833328472226273, 0.6896549346017466, 0.5483869198751872, 0.6774191363164076, 0.8888879012356653, 0.1428570408163994, 0.571427755103207, 0.9999988888901234, 0.5333329777780148, 0.0, 0.9374997070313414, 0.9999985714306123, 0.1333332444445037, 0.9999900000999989, 0.9999950000249999, 0.8181814462811607, 0.6470584429067983, 0.49999950000050003, 0.7499996875001302, 0.6551721878716593, 0.49999750001249993, 0.5999998500000374, 0.6666644444518518, 0.8571425510205175, 0.9583329340279442, 0.7441858734451456, 0.7499981250046875, 0.18749988281257324, 0.8571422448983965, 0.8333319444467593, 0.9999988888901234, 0.5555552469137517, 0.24999987500006252, 0.5999994000006, 0.3571427295918823, 0.29629618655696793, 0.3461537130178027, 0.9999900000999989, 0.0, 0.7499981250046875, 0.0, 0.6666644444518518, 0.8749994531253418, 0.9130430812856168, 0.3333322222259259, 0.9090900826453795, 0.39999920000160005, 0.9166662847223814, 0.09999998333333611, 0.6666644444518518, 0.4999993750007813, 0.39999973333351113, 0.7586205588585243, 0.6896549346017466, 0.6285712489796431, 0.6052629986150003, 0.9090904958679564, 0.46153810650914884, 0.0, 0.8571425510205175, 0.6964284470663487, 0.8749994531253418, 0.5714282993198575, 0.0, 0.9999998076923446, 0.29629618655696793, 0.0, 0.49999977272737606, 0.0, 0.0, 0.0, 0.4705880968858538, 0.0, 0.0, 0.9999950000249999, 0.0, 0.760869399811, 0.8292680904224169, 0.8787876124886022, 0.06451610822061025, 0.699999650000175, 0.0, 0.7999998222222616, 0.5652171455577628, 0.3999996000004, 0.1999998000002, 0.9523804988664292, 0.4736839612189678, 0.6315786149586237, 0.523809274376536, 0.8749989062513672, 0.0, 0.21212117998163937, 0.6764703892734149, 0.22580637877213589, 0.8095234240364647, 0.9444439197533779, 0.6666655555574075, 0.27272714876038695, 0.40909072314058037, 0.09374997070313415, 0.19999990000005, 0.9999996153847633, 0.7307689497042501, 0.7499981250046875, 0.45454504132268975, 0.13636357438019348, 0.31578930747931183]
# chatgpt_ukn_acc = [0.6578945637119569, 0.40384607618344687, 0.6470586966551575, 0.0, 0.9499997625000594, 0.9583331336805971, 0.9999950000249999, 0.5799998840000231, 0.8888883950620028, 0.8749997265625854, 0.4782608002520579, 0.6799997280001088, 0.9487177054569985, 0.9999990909099173, 0.9999990000010001, 0.33333324786326973, 0.39999920000160005, 0.7999994666670223, 0.982455967990181, 0.3636360330581518, 0.46153810650914884, 0.9230767455621642, 0.9444441820988382, 0.9999987500015626, 0.8333329861112558, 0.933333125925972, 0.7499990625011719, 0.9999997435898093, 0.9285707653065962, 0.9459456902849485, 0.666665925926749, 0.7627117351336041, 0.9999988888901234, 0.6666662222225186, 0.4999991666680556, 0.8749989062513672, 0.9999988888901234, 0.9999980000040001, 0.8518515363512829, 0.9583329340279442, 0.8571416326548106, 0.9166664120371076, 0.571427755103207, 0.9999975000062501, 0.7142852040819971, 0.9565213232515986, 0.9999996296297669, 0.7058822145328991, 0.6249992187509766, 0.9310341617123581, 0.8965514149822708, 0.5999996000002666, 0.666665925926749, 0.5217390170132571, 0.0, 0.9999900000999989, 0.9999992307698226, 0.8749989062513672, 0.9230766863905931, 0.7999994666670223, 0.791666336805693, 0.9714282938776302, 0.9729727100073756, 0.8499995750002125, 0.8571416326548106, 0.6874995703127685, 0.7272720661163036, 0.7777773456792524, 0.9999991666673611, 0.0, 0.7142846938790088, 0.9090906336088989, 0.8918916508400943, 0.6842101662051757, 0.9230766863905931, 0.9560438509841922, 0.8064513527576281, 0.9999966666777778, 0.7777769135812072, 0.7999997714286367, 0.5714282993198575, 0.9999980000040001, 0.9999990909099173, 0.8333331944444675, 0.9499998812500148, 0.9999950000249999, 0.8888879012356653, 0.9999966666777778, 0.0, 0.9230766863905931, 0.0, 0.9999983333361112, 0.0, 0.9464284024234995, 0.8823527681661238, 0.6666655555574075, 0.7857141454081883, 0.6666661111115741, 0.7586205588585243, 0.9729727100073756, 0.8124994921878174, 0.7499993750005208, 0.8536583283760174, 0.9999900000999989, 0.0, 0.8571416326548106, 0.9354835691988487, 0.3333331944445023, 0.6666655555574075, 0.9999975000062501, 0.6666644444518518, 0.8999991000009, 0.8780485663296179, 0.9999987500015626, 0.0, 0.9999900000999989, 0.7272720661163036, 0.9999900000999989, 0.9999987500015626, 0.7777776049383099, 0.8571424489797862, 0.9999975000062501, 0.9574466047985947, 0.933333125925972, 0.8965514149822708, 0.5499997250001375, 0.8181814462811607, 0.2857140816327988, 0.8571425510205175, 0.9999950000249999, 0.0, 0.10714281887756469, 0.4999996875001953, 0.9999900000999989, 0.9999997500000625, 0.0, 0.849999787500053, 0.0, 0.8780485663296179, 0.9999966666777778, 0.5333329777780148, 0.9999900000999989, 0.8666660888892741, 0.9999996774194589, 0.7272720661163036]
# print(len(chatgpt_kwn_acc), len(chatgpt_ukn_acc))

# llama_kwn_acc = [0.868420824099783, 0.7857137244901968, 0.9565213232515986, 0.5714281632655976, 0.5499997250001375, 0.4285708163274053, 0.2999997000003, 0.5925924828532438, 0.05263156971375969, 0.27272714876038695, 0.2222220987655007, 0.7380950623583185, 0.19607839292580528, 0.6170211453146499, 0.0, 0.7199997120001153, 0.8055553317901856, 0.0, 0.7799998440000312, 0.5555549382722909, 0.30769207100609924, 0.05263156509695655, 0.4705880968858538, 0.318181745867785, 0.8571416326548106, 0.6938774094127735, 0.7499998125000468, 0.06451610822061025, 0.0, 0.810810591672813, 0.9999900000999989, 0.2222220987655007, 0.9333327111115259, 0.15384603550304962, 0.6666662222225186, 0.571427755103207, 0.9999900000999989, 0.4444441975310014, 0.08333326388894677, 0.7499981250046875, 0.24999994318183108, 0.1923076183432237, 0.4166663194447338, 0.5925923731139359, 0.19354832466183075, 0.4999998333333889, 0.666665925926749, 0.1428570408163994, 0.4285708163274053, 0.5555549382722909, 0.5714281632655976, 0.0, 0.8749997265625854, 0.7142846938790088, 0.2857140816327988, 0.9999900000999989, 0.0, 0.7272723966943652, 0.7647054325262161, 0.5999994000006, 0.4782606616257993, 0.6071426403061999, 0.49999750001249993, 0.4749998812500297, 0.3333322222259259, 0.9230765680474738, 0.6666663888890046, 0.8095236167800912, 0.7499981250046875, 0.12499992187504883, 0.42857112244919826, 0.7999984000032001, 0.7777769135812072, 0.7222218209878772, 0.1499999250000375, 0.3999996000004, 0.4444442798354519, 0.23076914201186846, 0.599999760000096, 0.9999900000999989, 0.0, 0.0, 0.3333322222259259, 0.3333322222259259, 0.8749994531253418, 0.7272723966943652, 0.3333322222259259, 0.8181810743808415, 0.19999960000080003, 0.8333329861112558, 0.017241376337693733, 0.3333322222259259, 0.4999993750007813, 0.3333331111112593, 0.851851694101538, 0.8275859215220961, 0.8787876124886022, 0.578947216066522, 0.9090904958679564, 0.8461531952667729, 0.0, 0.6153843786983159, 0.7818180396694473, 0.5333329777780148, 0.4285712244898931, 0.0, 0.9807690421597995, 0.2692306656805132, 0.0, 0.6363633471075695, 0.16666638888935187, 0.49999750001249993, 0.0, 0.5757574012856359, 0.3333322222259259, 0.0, 0.9999950000249999, 0.0, 0.5869563941399143, 0.8249997937500515, 0.9393936547291954, 0.06666664444445185, 0.7499996250001875, 0.0, 0.7272725619835085, 0.3333331746032502, 0.1999998000002, 0.1999998000002, 0.9523804988664292, 0.31578930747931183, 0.36842085872586383, 0.28571414965992875, 0.8749989062513672, 0.0, 0.26153842130178134, 0.5588233650519514, 0.16129027055152564, 0.699999650000175, 0.7222218209878772, 0.16666638888935187, 0.38095219954657167, 0.09090904958679565, 0.21874993164064635, 0.3999998000001, 0.9230765680474738, 0.49999980769238167, 0.49999875000312505, 0.09090900826453795, 0.0, 0.05263155124655198]
# llama_ukn_acc = [0.8157892590028265, 0.7692306213018035, 0.6599998680000264, 0.0, 0.9743587245234039, 0.9374998046875407, 0.9999950000249999, 0.899999820000036, 0.882352422145634, 0.9354835691988487, 0.4852940462802873, 0.6399997440001024, 0.9999997368421745, 0.9999990909099173, 0.8999991000009, 0.4615383431952966, 0.9999980000040001, 0.9333327111115259, 0.7368419759926358, 0.9999990909099173, 0.6153841420121985, 0.960784125336446, 0.9166664120371076, 0.8749989062513672, 0.8695648393196351, 0.9318179700413703, 0.7142846938790088, 0.9999997368421745, 0.9999992857147959, 0.9999997297298027, 0.666665925926749, 0.6315788365651164, 0.9999988888901234, 0.8571422448983965, 0.4999991666680556, 0.9999987500015626, 0.9999988888901234, 0.7999984000032001, 0.9629626063101459, 0.9583329340279442, 0.9999985714306123, 0.8571426122449679, 0.9999985714306123, 0.9999975000062501, 0.9285707653065962, 0.9999995238097505, 0.9230765680474738, 0.9215684467512849, 0.8571416326548106, 0.928571096938894, 0.8965514149822708, 0.8666660888892741, 0.7777769135812072, 0.760869399811, 0.24999968750039064, 0.9999900000999989, 0.9999992307698226, 0.8749989062513672, 0.8717946482577824, 0.7999994666670223, 0.772726921487763, 0.9411761937717076, 0.9459456902849485, 0.9999994444447531, 0.8571416326548106, 0.4374997265626709, 0.6363630578517656, 0.9444439197533779, 0.9999990909099173, 0.0, 0.9999985714306123, 0.969696675849492, 0.9189186705625214, 0.6842101662051757, 0.9729727100073756, 0.9333332296296412, 0.9677416233091538, 0.9999966666777778, 0.8888879012356653, 0.9714282938776302, 0.9047614739231077, 0.9999980000040001, 0.9090900826453795, 0.9333331777778037, 0.9374998828125146, 0.9999950000249999, 0.9999988888901234, 0.9999966666777778, 0.0, 0.9999997368421745, 0.9999900000999989, 0.9999983333361112, 0.0, 0.9272725586777165, 0.899999820000036, 0.8333319444467593, 0.9090907438016829, 0.9166659027784144, 0.8571427040816599, 0.9189186705625214, 0.8749994531253418, 0.9999991666673611, 0.8048778524688164, 0.9999900000999989, 0.0, 0.7142846938790088, 0.8387094068679333, 0.4166664930556279, 0.8333319444467593, 0.9999975000062501, 0.9999966666777778, 0.9999990000010001, 0.9749997562500609, 0.9999987500015626, 0.0, 0.0, 0.7272720661163036, 0.9999900000999989, 0.9999987500015626, 0.8666664740741168, 0.9999995238097505, 0.7499981250046875, 0.9565215311909714, 0.933333125925972, 0.8965514149822708, 0.899999550000225, 0.9090904958679564, 0.7142852040819971, 0.7499997321429528, 0.9999950000249999, 0.0, 0.2499999107143176, 0.4999996875001953, 0.9999900000999989, 0.9999997500000625, 0.0, 0.8947366066482614, 0.0, 0.926829042236819, 0.9999966666777778, 0.6428566836737974, 0.9999900000999989, 0.9999993333337778, 0.9677416233091538, 0.8181810743808415]

# qwen_kwn_acc = [0.846153629191377, 0.7857137244901968, 0.8260865973536533, 0.7142852040819971, 0.49999975000012503, 0.571427755103207, 0.49999950000050003, 0.690908965289279, 0.035087713142506465, 0.5454542975207739, 0.16666657407412552, 0.7380950623583185, 0.3269230140532665, 0.6874998567708631, 0.0, 0.9599996160001536, 0.6666664814815328, 0.1818180165290759, 0.8076921523668937, 0.7777769135812072, 0.23076905325457442, 0.2105262603878262, 0.771428351020471, 0.5909089566116007, 0.8571416326548106, 0.8399998320000336, 0.8749997812500546, 0.3870966493236615, 0.49999750001249993, 0.83783761139524, 0.9999900000999989, 0.10526310249310396, 0.7647054325262161, 0.23076905325457442, 0.8749994531253418, 0.4999993750007813, 0.9999900000999989, 0.6666662962965021, 0.4166663194447338, 0.7499981250046875, 0.5454544214876315, 0.30769218934915793, 0.6666661111115741, 0.6551721878716593, 0.4838708116545769, 0.774193298647323, 0.7777769135812072, 0.0714285204081997, 0.571427755103207, 0.7777769135812072, 0.7333328444447704, 0.0, 0.8437497363282073, 0.7142846938790088, 0.5333329777780148, 0.9999900000999989, 0.9999950000249999, 0.772726921487763, 0.8235289273359251, 0.5999994000006, 0.3333331944445023, 0.724137681331834, 0.49999750001249993, 0.6499998375000405, 0.6666644444518518, 0.7499997321429528, 0.8749996354168186, 0.9302323418064321, 0.7499981250046875, 0.5624996484377197, 0.6428566836737974, 0.8333319444467593, 0.9999988888901234, 0.7777773456792524, 0.3999998000001, 0.5999994000006, 0.5357140943878235, 0.40740725651583093, 0.5769228550296711, 0.9999900000999989, 0.0, 0.49999875000312505, 0.3333322222259259, 0.9999966666777778, 0.9999993750003906, 0.8695648393196351, 0.6666644444518518, 0.9090900826453795, 0.19999960000080003, 0.9166662847223814, 0.049999991666668055, 0.6666644444518518, 0.4999993750007813, 0.46666635555576297, 0.9137929458977679, 0.6896549346017466, 0.7428569306123055, 0.76315769390587, 0.9090904958679564, 0.9999992307698226, 0.9999900000999989, 0.7499997321429528, 0.7321427264030845, 0.6249996093752441, 0.4285712244898931, 0.0, 0.9615382766272544, 0.37037023319620993, 0.0, 0.6363633471075695, 0.16666638888935187, 0.9999950000249999, 0.0, 0.8235291695502441, 0.3333322222259259, 0.0, 0.9999950000249999, 0.0, 0.760869399811, 0.926829042236819, 0.9393936547291954, 0.06451610822061025, 0.699999650000175, 0.0, 0.7999998222222616, 0.30434769376187226, 0.49999950000050003, 0.0999999000001, 0.9523804988664292, 0.5263155124655198, 0.6842101662051757, 0.5714282993198575, 0.8749989062513672, 0.0, 0.515151437098267, 0.7647056574395125, 0.3225805411030513, 0.5714282993198575, 0.8888883950620028, 0.4999991666680556, 0.2272726239669891, 0.4545452479339782, 0.4062498730469146, 0.24999987500006252, 0.8846150443788291, 0.9230765680474738, 0.49999875000312505, 0.3636360330581518, 0.2272726239669891, 0.2631577562327599]
# qwen_ukn_acc = [0.7894734764543483, 0.4615383727810821, 0.6862743752403185, 0.0, 0.9499997625000594, 0.9583331336805971, 0.9999950000249999, 0.4399999120000176, 0.8888883950620028, 0.9062497167969634, 0.40579704263810973, 0.3199998720000512, 0.7179485338593502, 0.9090900826453795, 0.7999992000008, 0.025641019066405365, 0.7999984000032001, 0.8666660888892741, 0.7368419759926358, 0.8181810743808415, 0.38461508875762407, 0.7499998557692584, 0.9166664120371076, 0.8749989062513672, 0.7083330381945674, 0.933333125925972, 0.4999993750007813, 0.7435895529257556, 0.9999992857147959, 0.83783761139524, 0.11111098765445816, 0.5593219390979763, 0.666665925926749, 0.7333328444447704, 0.33333277777870374, 0.8749989062513672, 0.9999988888901234, 0.5999988000024, 0.9629626063101459, 0.9583329340279442, 0.9999985714306123, 0.8333331018519161, 0.7142846938790088, 0.49999875000312505, 0.7142852040819971, 0.9565213232515986, 0.8148145130316619, 0.627450857362577, 0.7499990625011719, 0.8965514149822708, 0.8965514149822708, 0.5333329777780148, 0.8888879012356653, 0.6304346455576857, 0.0, 0.9999900000999989, 0.9230762130182977, 0.7499990625011719, 0.846153629191377, 0.7333328444447704, 0.7083330381945674, 0.914285453061299, 0.8648646311176672, 0.8499995750002125, 0.8571416326548106, 0.5624996484377197, 0.7272720661163036, 0.9444439197533779, 0.9999991666673611, 0.0, 0.8571416326548106, 0.7878785491277123, 0.8648646311176672, 0.5789470637120717, 0.9230766863905931, 0.9120878118584822, 0.8387094068679333, 0.9999966666777778, 0.8888879012356653, 0.7999997714286367, 0.8095234240364647, 0.7999984000032001, 0.7272720661163036, 0.7833332027777995, 0.8874998890625139, 0.9999950000249999, 0.666665925926749, 0.9999966666777778, 0.0, 0.8947366066482614, 0.0, 0.9999983333361112, 0.9999900000999989, 0.8928569834183958, 0.8431370895809628, 0.6666655555574075, 0.8749998437500278, 0.7499993750005208, 0.7241378061831368, 0.7837835719503858, 0.749999531250293, 0.8333326388894676, 0.7804876145152159, 0.9999900000999989, 0.9999900000999989, 0.8571416326548106, 0.7419352445370179, 0.16666659722225116, 0.9999983333361112, 0.7499981250046875, 0.9999966666777778, 0.6999993000007, 0.9024388042832184, 0.37499953125058594, 0.0, 0.0, 0.7272720661163036, 0.9999900000999989, 0.8749989062513672, 0.7999998222222616, 0.9999995238097505, 0.7499981250046875, 0.9361700135808481, 0.8222220395062134, 0.8965514149822708, 0.899999550000225, 0.6363633471075695, 0.7142852040819971, 0.7857140051021411, 0.9999950000249999, 0.9999900000999989, 0.10714281887756469, 0.6249996093752441, 0.9999900000999989, 0.9499997625000594, 0.0, 0.6999998250000437, 0.0, 0.8780485663296179, 0.9999966666777778, 0.6666662222225186, 0.0, 0.9999993333337778, 0.9354835691988487, 0.6363630578517656]

# print(len(qwen_kwn_acc), len(qwen_ukn_acc))
# ax.boxplot([glc_kwn_acc_r2p, glc_ukn_acc_r2p, glc_kwn_acc_s2p, glc_ukn_acc_s2p, chatgpt_kwn_acc, chatgpt_ukn_acc, llama_kwn_acc, llama_ukn_acc, qwen_kwn_acc, qwen_ukn_acc], labels=['glc known r2p', 'glc unknown r2p', 'glc known s2p', 'glc unknown s2p', 'chatgpt known', 'chatgpt unknown', 'llama known', 'llama unknown', 'qwen known', 'qwen unknown'])
# plt.yticks(rotation=45, fontsize=20)
# plt.title(f'{dataset} Painting Domain Known vs Unknown', fontsize=30)
# ax.xaxis.set_label_position('bottom')
# # ax.set_xticks(ticks=np.arange(8), labels=['known', 'unknown', '', 'known', 'unknown', '', 'known', 'unknown'])
# plt.xticks(rotation=45, fontsize=10)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"{dataset}_Painting_Accuracy_boxplot.png")

# kwn_sample_size = [39, 14, 23, 14, 20, 7, 10, 55, 57, 22, 18, 42, 52, 48, 24, 25, 36, 11, 52, 9, 13, 38, 35, 44, 7, 50, 40, 31, 2, 37, 1, 19, 17, 13, 16, 8, 1, 18, 12, 4, 44, 26, 12, 29, 31, 31, 9, 14, 7, 9, 15, 11, 32, 7, 15, 1, 2, 22, 17, 10, 24, 29, 2, 40, 3, 28, 24, 43, 4, 16, 14, 6, 9, 18, 20, 10, 28, 27, 26, 1, 1, 4, 3, 3, 16, 23, 3, 11, 5, 24, 60, 3, 8, 15, 58, 29, 35, 38, 22, 13, 1, 28, 56, 16, 21, 11, 52, 27, 0, 22, 6, 2, 1, 34, 3, 3, 2, 0, 46, 41, 33, 31, 20, 0, 45, 23, 10, 10, 21, 19, 19, 21, 8, 14, 66, 34, 31, 21, 18, 6, 22, 22, 32, 20, 26, 26, 4, 11, 22, 19]
# ukn_sample_size = [38, 52, 51, 0, 40, 48, 2, 50, 18, 32, 69, 25, 39, 11, 10, 39, 5, 15, 57, 11, 13, 52, 36, 8, 24, 45, 8, 39, 14, 37, 9, 59, 9, 15, 6, 8, 9, 5, 27, 24, 7, 36, 7, 4, 14, 23, 27, 51, 8, 29, 29, 15, 9, 46, 8, 1, 13, 8, 39, 15, 24, 35, 37, 20, 7, 16, 11, 18, 12, 0, 7, 33, 37, 19, 39, 91, 31, 3, 9, 35, 21, 5, 11, 60, 80, 2, 9, 3, 0, 39, 1, 6, 1, 56, 51, 6, 56, 12, 58, 37, 16, 12, 41, 1, 1, 7, 31, 24, 6, 4, 3, 10, 41, 8, 0, 1, 11, 1, 8, 45, 21, 4, 47, 45, 29, 20, 22, 14, 28, 2, 1, 28, 16, 1, 40, 0, 40, 0, 41, 3, 15, 1, 15, 31, 11]

# mean_glc_kwn_r2p = sum(glc_kwn_acc_r2p) / len(glc_kwn_acc_r2p)
# mean_glc_ukn_r2p = sum(glc_ukn_acc_r2p) / len(glc_ukn_acc_r2p)
# print(mean_glc_kwn_r2p, mean_glc_ukn_r2p)

# mean_glc_kwn_s2p = sum(glc_kwn_acc_s2p) / len(glc_kwn_acc_s2p)
# mean_glc_ukn_s2p = sum(glc_ukn_acc_s2p) / len(glc_ukn_acc_s2p)
# print(mean_glc_kwn_s2p, mean_glc_ukn_s2p)

# mean_chatgpt_kwn = sum(chatgpt_kwn_acc) / len(chatgpt_kwn_acc)
# mean_chatgpt_ukn = sum(chatgpt_ukn_acc) / len(chatgpt_ukn_acc)
# print(mean_chatgpt_kwn, mean_chatgpt_ukn)

# mean_llama_kwn = sum(llama_kwn_acc) / len(llama_kwn_acc)
# mean_llama_ukn = sum(llama_ukn_acc) / len(llama_ukn_acc)
# print(mean_llama_kwn, mean_llama_ukn)

# mean_qwen_kwn = sum(qwen_kwn_acc) / len(qwen_kwn_acc)
# mean_qwen_ukn = sum(qwen_ukn_acc) / len(qwen_ukn_acc)
# print(mean_qwen_kwn, mean_qwen_ukn)
# print("------------")
# wgt_glc_kwn_r2p = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc_r2p)])  / sum(kwn_sample_size)
# wgt_glc_ukn_r2p = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc_r2p)])  / sum(ukn_sample_size)
# print(wgt_glc_kwn_r2p, wgt_glc_ukn_r2p)

# wgt_glc_kwn_s2p = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc_s2p)])  / sum(kwn_sample_size)
# wgt_glc_ukn_s2p = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc_s2p)])  / sum(ukn_sample_size)
# print(wgt_glc_kwn_s2p, wgt_glc_ukn_s2p)

# wgt_chatgpt_kwn = sum([x * y for x, y in zip(kwn_sample_size, chatgpt_kwn_acc)])  / sum(kwn_sample_size)
# wgt_chatgpt_ukn = sum([x * y for x, y in zip(ukn_sample_size, chatgpt_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_chatgpt_kwn, wgt_chatgpt_ukn)

# wgt_llama_kwn = sum([x * y for x, y in zip(kwn_sample_size, llama_kwn_acc)])  / sum(kwn_sample_size)
# wgt_llama_ukn = sum([x * y for x, y in zip(ukn_sample_size, llama_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_llama_kwn, wgt_llama_ukn)

# wgt_qwen_kwn = sum([x * y for x, y in zip(kwn_sample_size, qwen_kwn_acc)])  / sum(kwn_sample_size)
# wgt_qwen_ukn = sum([x * y for x, y in zip(ukn_sample_size, qwen_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_qwen_kwn, wgt_qwen_ukn)
# print("------------")
# ratio_mean_glc_r2p = mean_glc_kwn_r2p / mean_glc_ukn_r2p
# ratio_mean_glc_s2p = mean_glc_kwn_s2p / mean_glc_ukn_s2p
# ratio_mean_chatgpt = mean_chatgpt_kwn / mean_chatgpt_ukn
# ratio_mean_llama = mean_llama_kwn / mean_llama_ukn
# ratio_mean_qwen = mean_qwen_kwn / mean_qwen_ukn
# print(ratio_mean_glc_r2p, ratio_mean_glc_s2p, ratio_mean_chatgpt, ratio_mean_llama, ratio_mean_qwen)

# ratio_wgt_mean_glc_r2p = wgt_glc_kwn_r2p / wgt_glc_ukn_r2p
# ratio_wgt_mean_glc_s2p = wgt_glc_kwn_s2p / wgt_glc_ukn_s2p
# ratio_wgt_mean_chatgpt = wgt_chatgpt_kwn / wgt_chatgpt_ukn
# ratio_wgt_mean_llama = wgt_llama_kwn / wgt_llama_ukn
# ratio_wgt_mean_qwen = wgt_qwen_kwn / wgt_qwen_ukn
# print(ratio_wgt_mean_glc_r2p, ratio_wgt_mean_glc_s2p, ratio_wgt_mean_chatgpt, ratio_wgt_mean_llama, ratio_wgt_mean_qwen)
############################## DomainNet Sketch ##################################################

# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# glc_kwn_acc_p2s = [0.924,	0.527,	0.669,	0.556,	0.607,	0.827,	0.035,	0.224,	0.821,	0.685,	0.220,	0.718,	0.028,	0.718,	0.342,	0.445,	0.760,	0.000,	0.806,	0.729,	0.301,	0.437,	0.381,	0.556,	0.205,	0.139,	0.601,	0.137,	0.229,	0.604,	0.000,	0.293,	0.901,	0.481,	0.601,	0.519,	0.300,	0.047,	0.021,	0.017,	0.230,	0.575,	0.193,	0.837,	0.304,	0.749,	0.558,	0.184,	0.512,	0.578,	0.459,	0.000,	0.771,	0.197,	0.091,	0.841,	0.000,	0.762,	0.642,	0.043,	0.000,	0.662,	0.258,	0.070,	0.531,	0.000,	0.500,	0.292,	0.480,	0.043,	0.719,	0.156,	0.029,	0.400,	0.277,	0.424,	0.091,	0.250,	0.667,	0.429,	0.097,	0.061,	0.000,	0.367,	0.353,	0.724,	0.000,	0.752,	0.608,	0.816,	0.000,	0.368,	0.025,	0.028,	0.424,	0.588,	0.409,	0.413,	0.051,	0.154,	0.465,	0.486,	0.478,	0.542,	0.583,	0.000,	0.771,	0.049,	0.000,	0.695,	0.365,	0.215,	0.000,	0.732,	0.429,	0.000,	0.534,	0.033,	0.293,	0.472,	0.887,	0.011,	0.542,	0.115,	0.417,	0.380,	0.398,	0.392,	0.552,	0.106,	0.020,	0.036,	0.656,	0.000,	0.075,	0.749,	0.208,	0.721,	0.822,	0.183,	0.337,	0.867,	0.078,	0.495,	0.771,	0.530,	0.138,	0.440,	0.221,	0.084,	0.758]
# glc_ukn_acc_p2s = [0.53508767, 0.83495138, 0.75324666, 0.64102559, 0.85234894, 0.88607589,
#  0.6363636,  0.70297026, 0.58823512, 0.94666654, 0.79518067, 0.58227841,
#  0.88235288, 0.94117639, 0.28755364, 0.60902253, 0.75257724, 0.89230762,
#  0.87431689, 0.83950607, 0.96153809, 0.77990427, 0.42016803, 0.72872337,
#  0.6434782, 0.74449336, 0.8956521,  0.67938926, 0.8613861,  0.9079754,
#  0.46218484, 0.8631578,  0.87378632, 0.86324779, 0.7959182,  0.84337339,
#  0.93684201, 0.76315786, 0.48936165, 0.66666665, 0.80606056, 0.87659571,
#  0.86956503, 0.76344078, 0.51063819, 0.69398903, 0.90163927, 0.62162157,
#  0.80327856, 0.97163114, 0.88642657, 0.48484845, 0.93636355, 0.36451612,
#  0.65432097, 0.88329517, 0.54065933, 0.70241285, 0.7637795,  0.89788052,
#  0.57142856, 0.5705263,  0.90697673, 0.82986766, 0.83968253, 0.7564655,
#  0.91169449, 0.86333332, 0.69313303, 0.86971234, 0.85260769, 0.91358022,
#  0.75319147, 0.87909318, 0.82826085, 0.89887639, 0.32891246, 0.88079468,
#  0.77618068, 0.66201549, 0.6280788,  0.93500737, 0.78817055, 0.91402713,
#  0.65038559, 0.85714284, 0.8243902,  0.82773106, 0.59999993, 0.84810123,
#  0.88235268, 0.825688,   0.71003715, 0.66161613, 0.82835818, 0.77941165,
#  0.80676325, 0.90291259, 0.85658911, 0.40254236, 0.46706584, 0.86458324,
#  0.88802081, 0.67567565, 0.44516126, 0.79333331, 0.8654434,  0.52941174,
#  0.78431369, 0.61417318, 0.48514849, 0.84660764, 0.23575129, 0.92134828,
#  0.64102556, 0.7288135,  0.91712702, 0.90212762, 0.92424238, 0.92890991,
#  0.41825093, 0.85039363, 0.86666663, 0.83603602, 0.91089106, 0.8743455,
#  0.88679242, 0.84574464, 0.90909088, 0.87878781, 0.71739125, 0.72727269,
#  0.19704433, 0.26451611, 0.89062493, 0.9130434,  0.62867645, 0.56024093,
#  0.64489793, 0.86131384, 0.9428571,  0.46428569, 0.92430275, 0.28776977,
#  0.93055549]

# glc_kwn_acc_r2s = [0.768,	0.486,	0.559,	0.667,	0.405,	0.515,	0.139,	0.000,	0.723,	0.667,	0.626,	0.204,	0.064,	0.545,	0.406,	0.327,	0.603,	0.000,	0.766,	0.812,	0.316,	0.599,	0.044,	0.306,	0.300,	0.114,	0.388,	0.281,	0.633,	0.715,	0.016,	0.297,	0.359,	0.489,	0.402,	0.631,	0.017,	0.000,	0.123,	0.150,	0.353,	0.492,	0.250,	0.696,	0.257,	0.576,	0.138,	0.252,	0.500,	0.432,	0.361,	0.013,	0.655,	0.082,	0.000,	0.913,	0.407,	0.785,	0.532,	0.058,	0.244,	0.078,	0.419,	0.202,	0.503,	0.000,	0.411,	0.238,	0.360, 0.000,	0.469,	0.323,	0.059,	0.629,	0.178,	0.242,	0.295,	0.220,	0.576,	0.286,	0.129,	0.000,	0.089,	0.533,	0.294,	0.586,	0.000,	0.596,	0.500,	0.728,	0.126,	0.564,	0.350,	0.014,	0.447,	0.306,	0.291,	0.421,	0.000,	0.154,	0.562,	0.435,	0.348,	0.542,	0.472,	0.006,	0.650,	0.213,	0.078,	0.000,	0.406,	0.018,	0.424,	0.631,	0.529,	0.064,	0.716,	0.163,	0.320,	0.314,	0.859,	0.305,	0.517,	0.299,	0.155,	0.547,	0.356,	0.409,	0.458,	0.205,	0.245,	0.083,	0.602,	0.128,	0.089,	0.578,	0.156,	0.590,	0.530,	0.437,	0.277,	0.911,	0.403,	0.569,	0.706,	0.570,	0.210,	0.474,	0.253,	0.319, 0.799]
# glc_ukn_acc_r2s = [0.53508767, 0.92233001, 0.79220769, 0.70940165, 0.95302007, 0.89240501,
#  0.52840906, 0.70297026, 0.70588215, 0.7599999,  0.89156621, 0.62025309,
#  0.87499994, 0.91596631, 0.35622316, 0.84210523, 0.85567001, 0.93846147,
#  0.90163929, 0.8765431,  0.46153828, 0.8516746,  0.56302516, 0.76595741,
#  0.70434776, 0.87665194, 0.79999993, 0.94656481, 0.94059401, 0.74233124,
#  0.512605,   0.7894736,  0.76699022, 0.77777771, 0.7959182,  0.92771073,
#  0.93684201, 0.91228066, 0.42553187, 0.74425285, 0.72121208, 0.86808507,
#  0.80434765, 0.79569884, 0.38297864, 0.85245897, 0.85245895, 0.74774768,
#  0.819672,   0.94326234, 0.90304707, 0.74242419, 0.83636356, 0.48709676,
#  0.75802467, 0.86041188, 0.60659339, 0.69705092, 0.90551178, 0.88246626,
#  0.83646615, 0.83789472, 0.88062014, 0.86389412, 0.83015872, 0.72198274,
#  0.88305487, 0.92499998, 0.76180256, 0.89678509, 0.86848071, 0.90370368,
#  0.89148934, 0.89420653, 0.86739129, 0.88202246, 0.90716178, 0.92052978,
#  0.7392197,  0.72713177, 0.65024629, 0.69423928, 0.74828059, 0.9072398,
#  0.59640101, 0.80952379, 0.79024386, 0.84453778, 0.62222215, 0.8818565,
#  0.88235268, 0.76146782, 0.65799254, 0.86868682, 0.76492534, 0.79411753,
#  0.81642508, 0.91262133, 0.89147283, 0.69915251, 0.91616761, 0.89583324,
#  0.86979164, 0.74774771, 0.93548381, 0.63666665, 0.90214065, 0.79831929,
#  0.78823526, 0.50393697, 0.89108906, 0.84365779, 0.51295335, 0.90636701,
#  0.60256403, 0.84745756, 0.86740327, 0.88510635, 0.85353531, 0.92890991,
#  0.88212924, 0.81102356, 0.82916663, 0.85585584, 0.80198017, 0.61780101,
#  0.82264148, 0.47872338, 0.89225586, 0.93181811, 0.73913038, 0.90909086,
#  0.31527092, 0.38064514, 0.85937493, 0.9130434,  0.90073526, 0.56626503,
#  0.76326527, 0.87591238, 0.78775507, 0.53571426, 0.89641431, 0.66906472,
#  0.83333328]

# chatgpt_kwn_acc = [0.9473679224379356, 0.6428566836737974, 0.749999531250293, 0.9999983333361112, 0.7199997120001153, 0.8333328703706276, 0.9166659027784144, 0.9565213232515986, 0.19999960000080003, 0.9999980000040001, 0.8888879012356653, 0.8999997000001, 0.2631577562327599, 0.9230765680474738, 0.0, 0.9545450206613543, 0.9565213232515986, 0.24999968750039064, 0.8499995750002125, 0.6666644444518518, 0.6499996750001625, 0.882352422145634, 0.9999994444447531, 0.9999994117650519, 0.941175916955343, 0.9999966666777778, 0.9999995833335069, 0.5135133747261149, 0.8799996480001409, 0.9999985714306123, 0.5714281632655976, 0.7999997714286367, 0.9999997058824394, 0.620689441141572, 0.9687496972657195, 0.611110771605127, 0.14285693877580175, 0.3529409688582536, 0.7499993750005208, 0.749999531250293, 0.46153810650914884, 0.5945944338933962, 0.6086953875237445, 0.9047614739231077, 0.7941174134948783, 0.8888886419753772, 0.6923071597637233, 0.3599998560000576, 0.42105240997241583, 0.9166659027784144, 0.8421048199448317, 0.0, 0.9999993750003906, 0.9999980000040001, 0.0, 0.9999983333361112, 0.7999984000032001, 0.9999987500015626, 0.9999988888901234, 0.7499981250046875, 0.39999920000160005, 0.9999985714306123, 0.8749989062513672, 0.49999958333368055, 0.9999993750003906, 0.9999950000249999, 0.9999975000062501, 0.7857137244901968, 0.49999750001249993, 0.19999960000080003, 0.8333319444467593, 0.9999980000040001, 0.3333322222259259, 0.49999875000312505, 0.45454504132268975, 0.9999966666777778, 0.39999920000160005, 0.8333326388894676, 0.3333322222259259, 0.0, 0.571427755103207, 0.0, 0.0, 0.9999983333361112, 0.9999950000249999, 0.9999995652175804, 0.0, 0.7777773456792524, 0.8235289273359251, 0.9333327111115259, 0.15555552098766198, 0.8181810743808415, 0.8571416326548106, 0.6249992187509766, 0.9047616893424548, 0.5999994000006, 0.6999993000007, 0.6041665407986373, 0.7999994666670223, 0.9999900000999989, 0.611110771605127, 0.9599996160001536, 0.7419352445370179, 0.9047614739231077, 0.5999996000002666, 0.0, 0.9999996774194589, 0.9130430812856168, 0.24999937500156252, 0.6666665555555741, 0.7391301134216899, 0.6808509189678895, 0.7499981250046875, 0.5999998000000667, 0.46666635555576297, 0.0, 0.7999992000008, 0.7999984000032001, 0.5714283673470116, 0.9791664626736536, 0.9230762130182977, 0.7999992000008, 0.9999985714306123, 0.5666664777778407, 0.8918916508400943, 0.8181810743808415, 0.8275859215220961, 0.7777773456792524, 0.8695648393196351, 0.45454504132268975, 0.7499990625011719, 0.49999875000312505, 0.9999994117650519, 0.0, 0.2531645249158829, 0.9583329340279442, 0.38461508875762407, 0.9999995238097505, 0.8461531952667729, 0.8333319444467593, 0.9583329340279442, 0.3333322222259259, 0.7499990625011719, 0.9999995454547521, 0.9999990000010001, 0.9523804988664292, 0.7692304733728949, 0.6999993000007, 0.4285708163274053, 0.4999993750007813]
# chatgpt_ukn_acc = [0.8333326388894676, 0.08333326388894677, 0.5454540495872277, 0.7999996000002, 0.9999990909099173, 0.8499995750002125, 0.5294114532873805, 0.21052620498620792, 0.7499981250046875, 0.7999984000032001, 0.8666660888892741, 0.33333277777870374, 0.8999991000009, 0.7857137244901968, 0.8749997265625854, 0.03448274673008733, 0.8749989062513672, 0.9999990909099173, 0.8095234240364647, 0.0, 0.7499981250046875, 0.5199997920000832, 0.9999990909099173, 0.4761902494332146, 0.8333326388894676, 0.9655169084424454, 0.6363630578517656, 0.9999990909099173, 0.9090904958679564, 0.899999550000225, 0.49999950000050003, 0.5999994000006, 0.5384611242606736, 0.4285708163274053, 0.9999975000062501, 0.8749989062513672, 0.6363630578517656, 0.5652171455577628, 0.9999988888901234, 0.9599996160001536, 0.9523804988664292, 0.6363633471075695, 0.6666644444518518, 0.5999996000002666, 0.7499981250046875, 0.9230762130182977, 0.9999988888901234, 0.6666661111115741, 0.9999985714306123, 0.5833328472226273, 0.8437497363282073, 0.38461508875762407, 0.9999980000040001, 0.18181812672177977, 0.2972972169466981, 0.8301885226059391, 0.8545452991735819, 0.8387094068679333, 0.7999996000002, 0.8928569834183958, 0.690908965289279, 0.9361700135808481, 0.9818180033058175, 0.9038459800296191, 0.7297296311176174, 0.8043476512287714, 0.8666664740741168, 0.7777776049383099, 0.8723402399276085, 0.779411650086522, 0.571428435374182, 0.9499997625000594, 0.8749998177083712, 0.8285711918368023, 0.8249997937500515, 0.9436618389208677, 0.6999998250000437, 0.9347824054820857, 0.7407406035665548, 0.8072288184061664, 0.571428435374182, 0.805555443672855, 0.7714284612245055, 0.8085104662743688, 0.9499997625000594, 0.799999840000032, 0.7499996875001302, 0.9090904958679564, 0.9999950000249999, 0.7368417174517277, 0.9999980000040001, 0.7333328444447704, 0.8181814462811607, 0.9999994117650519, 0.8620686682521834, 0.12499984375019532, 0.8124994921878174, 0.30769218934915793, 0.7352939013841466, 0.6551721878716593, 0.7777773456792524, 0.9230762130182977, 0.8958331467014277, 0.7894732686982796, 0.8666660888892741, 0.7058821453287807, 0.8333330555556482, 0.1249999609375122, 0.7499996875001302, 0.8333319444467593, 0.7272723966943652, 0.8857140326531334, 0.76315769390587, 0.8518515363512829, 0.19999960000080003, 0.49999958333368055, 0.9333327111115259, 0.9687496972657195, 0.9523804988664292, 0.59999970000015, 0.9166662847223814, 0.9999991666673611, 0.8095234240364647, 0.9473682548476745, 0.7187497753906951, 0.5263155124655198, 0.5666664777778407, 0.21428556122459913, 0.868420824099783, 0.9999987500015626, 0.6363630578517656, 0.36842085872586383, 0.3749997656251465, 0.7333328444447704, 0.9999990909099173, 0.8999991000009, 0.8214282780613292, 0.6666663492065004, 0.7777773456792524, 0.6470584429067983, 0.5714283673470116, 0.7894732686982796, 0.9999995833335069, 0.9677416233091538, 0.7272720661163036]
# print(len(chatgpt_kwn_acc), len(chatgpt_ukn_acc))

# llama_kwn_acc = [0.8421048199448317, 0.42857112244919826, 0.749999531250293, 0.571427755103207, 0.6399997440001024, 0.27777762345687584, 0.4166663194447338, 0.4583331423611907, 0.0, 0.39999920000160005, 0.666665925926749, 0.5666664777778407, 0.10526310249310396, 0.7307689497042501, 0.13636357438019348, 0.5909088223141716, 0.8333329861112558, 0.0, 0.5909088223141716, 0.9999966666777778, 0.3499998250000875, 0.41176446366796254, 0.5263155124655198, 0.49999972222237654, 0.7058819377165072, 0.9999966666777778, 0.5416664409723163, 0.07894734764543482, 0.6153843786983159, 0.24999968750039064, 0.21428556122459913, 0.37142846530615276, 0.7058821453287807, 0.24137922711061136, 0.6060604224059325, 0.4736839612189678, 0.14285693877580175, 0.058823494809708936, 0.4166663194447338, 0.18749988281257324, 0.07692301775152481, 0.3243242366691252, 0.30434769376187226, 0.6666663492065004, 0.4571427265306495, 0.36111101080249697, 0.3333331111112593, 0.1599999360000256, 0.19999990000005, 0.24999979166684028, 0.59999970000015, 0.035087713142506465, 0.8124994921878174, 0.5999988000024, 0.0, 0.4999991666680556, 0.39999920000160005, 0.8749989062513672, 0.8888879012356653, 0.24999937500156252, 0.39999920000160005, 0.571427755103207, 0.6249992187509766, 0.24999979166684028, 0.6470584429067983, 0.9999950000249999, 0.7499981250046875, 0.5714281632655976, 0.49999750001249993, 0.39999920000160005, 0.33333277777870374, 0.7999984000032001, 0.0, 0.24999937500156252, 0.2727270247936138, 0.6666644444518518, 0.39999920000160005, 0.16666652777789354, 0.3333322222259259, 0.0, 0.14285693877580175, 0.0, 0.0, 0.33333277777870374, 0.0, 0.5652171455577628, 0.03571427295918823, 0.611110771605127, 0.5555552469137517, 0.6666662222225186, 0.04347825141777143, 0.2727270247936138, 0.571427755103207, 0.24999968750039064, 0.8333331349206821, 0.49999950000050003, 0.49999950000050003, 0.2448979092045083, 0.5999996000002666, 0.9999900000999989, 0.4444441975310014, 0.4799998080000768, 0.5806449739854923, 0.28571414965992875, 0.46666635555576297, 0.0, 0.8387094068679333, 0.5416664409723163, 0.0, 0.7213113571620725, 0.4583331423611907, 0.2553190946129586, 0.0, 0.6129030280957974, 0.2666664888890074, 0.0, 0.49999950000050003, 0.39999920000160005, 0.17857136479594116, 0.6530610912120222, 0.6153841420121985, 0.3999996000004, 0.7142846938790088, 0.09999996666667778, 0.5135133747261149, 0.2727270247936138, 0.41379296076104805, 0.49999972222237654, 0.6666663888890046, 0.23076905325457442, 0.37499953125058594, 0.24999937500156252, 0.7222218209878772, 0.0, 0.18518516232281945, 0.4399998240000704, 0.2857140816327988, 0.7142853741498219, 0.46153810650914884, 0.4999991666680556, 0.49999979166675346, 0.6666644444518518, 0.4999993750007813, 0.772726921487763, 0.8181810743808415, 0.5714282993198575, 0.6296293964335569, 0.1999998000002, 0.0, 0.12499984375019532]
# llama_ukn_acc = [0.9166659027784144, 0.9230762130182977, 0.8181810743808415, 0.8571424489797862, 0.9999990909099173, 0.9999995000002501, 0.9999994117650519, 0.8947363711913836, 0.7499981250046875, 0.9999980000040001, 0.8666660888892741, 0.6666655555574075, 0.9999990000010001, 0.9999992857147959, 0.9090906336088989, 0.51724120095131, 0.8749989062513672, 0.9999991666673611, 0.9047614739231077, 0.7499981250046875, 0.9999975000062501, 0.8399996640001344, 0.9999990909099173, 0.8095234240364647, 0.9166659027784144, 0.9677416233091538, 0.9999990909099173, 0.9999990909099173, 0.9999995454547521, 0.9499995250002375, 0.9999990000010001, 0.6999993000007, 0.7692301775152481, 0.571427755103207, 0.9999975000062501, 0.9999987500015626, 0.7499993750005208, 0.8260865973536533, 0.9999988888901234, 0.9999996153847633, 0.9523804988664292, 0.8181814462811607, 0.9999966666777778, 0.9333327111115259, 0.9999975000062501, 0.9230762130182977, 0.9999988888901234, 0.8333326388894676, 0.9999985714306123, 0.8333326388894676, 0.9999996875000976, 0.6923071597637233, 0.9999980000040001, 0.6666664646465258, 0.4324323155588336, 0.9811318903524735, 0.9818180033058175, 0.9062497167969634, 0.9523804988664292, 0.9298243982764213, 0.9107141230867637, 0.9583331336805971, 0.9827584512485429, 0.9622639693841567, 0.9210525103878275, 0.9148934223631016, 0.933333125925972, 0.933333125925972, 0.9574466047985947, 0.869565091367378, 0.7727270971074779, 0.8749997812500546, 0.8958331467014277, 0.8571426122449679, 0.9285712074830458, 0.9583332002314999, 0.9756095181440199, 0.9130432797732, 0.9629627846365213, 0.9879516881985917, 0.8139532990806281, 0.9861109741512536, 0.9577463439793882, 0.9787231960163412, 0.9512192801904193, 0.9811318903524735, 0.9999995833335069, 0.8181814462811607, 0.9999950000249999, 0.9999994736844876, 0.9999980000040001, 0.8666660888892741, 0.99999960000016, 0.9999994444447531, 0.866666377777874, 0.6249992187509766, 0.882352422145634, 0.9615380917161186, 0.9714282938776302, 0.9310341617123581, 0.9999994736844876, 0.9230762130182977, 0.8431370895809628, 0.8947363711913836, 0.8666660888892741, 0.9411761937717076, 0.9677416233091538, 0.39393927456385613, 0.99999960000016, 0.8333319444467593, 0.9999995652175804, 0.8888886419753772, 0.8421050415513048, 0.928571096938894, 0.39999920000160005, 0.7499993750005208, 0.9333327111115259, 0.9999996875000976, 0.9047614739231077, 0.9047614739231077, 0.9999995833335069, 0.9999992307698226, 0.9999995238097505, 0.9655170749108491, 0.9117644377163417, 0.7999996000002, 0.9354835691988487, 0.7857137244901968, 0.9743587245234039, 0.9999987500015626, 0.9090900826453795, 0.5714282993198575, 0.4999996875001953, 0.9999993333337778, 0.9999990909099173, 0.9999990000010001, 0.928571096938894, 0.9047614739231077, 0.8947363711913836, 0.8888883950620028, 0.9642853698980822, 0.6315786149586237, 0.9999995833335069, 0.9374997070313414, 0.9999990909099173]

# qwen_kwn_acc = [0.9473679224379356, 0.7142852040819971, 0.5624996484377197, 0.8333319444467593, 0.9199996320001472, 0.8888883950620028, 0.8333326388894676, 0.9565213232515986, 0.0, 0.7999984000032001, 0.666665925926749, 0.9999996666667778, 0.10526310249310396, 0.9230765680474738, 0.0, 0.8181814462811607, 0.9565213232515986, 0.0, 0.699999650000175, 0.6666644444518518, 0.7499996250001875, 0.7647054325262161, 0.9999994444447531, 0.9999994117650519, 0.941175916955343, 0.6666644444518518, 0.9583329340279442, 0.3783782761139794, 0.8799996480001409, 0.8571416326548106, 0.7142852040819971, 0.7142855102041399, 0.9999997058824394, 0.5862066944114847, 0.9687496972657195, 0.7222218209878772, 0.14285693877580175, 0.23529397923883574, 0.8333326388894676, 0.5624996484377197, 0.46153810650914884, 0.5675674141709691, 0.5217389035917811, 0.9523804988664292, 0.7941174134948783, 0.6666664814815328, 0.6153841420121985, 0.2799998880000448, 0.5263155124655198, 0.8333326388894676, 0.9473679224379356, 0.035087713142506465, 0.9374994140628662, 0.9999980000040001, 0.1818180165290759, 0.9999983333361112, 0.7999984000032001, 0.8749989062513672, 0.8888879012356653, 0.49999875000312505, 0.5999988000024, 0.8571416326548106, 0.6249992187509766, 0.49999958333368055, 0.9999993750003906, 0.9999950000249999, 0.7499981250046875, 0.7142852040819971, 0.0, 0.5999988000024, 0.8333319444467593, 0.9999980000040001, 0.3333322222259259, 0.7499981250046875, 0.3636360330581518, 0.3333322222259259, 0.7999984000032001, 0.5833328472226273, 0.9999966666777778, 0.0, 0.4285708163274053, 0.0, 0.12499984375019532, 0.4999991666680556, 0.9999950000249999, 0.9565213232515986, 0.10714281887756469, 0.7777773456792524, 0.8235289273359251, 0.9333327111115259, 0.0, 0.7272720661163036, 0.8571416326548106, 0.4999993750007813, 0.880952171201864, 0.9999990000010001, 0.5999994000006, 0.5833332118055808, 0.7333328444447704, 0.9999900000999989, 0.8333328703706276, 0.9199996320001472, 0.6451610822061026, 0.8095234240364647, 0.46666635555576297, 0.0, 0.9677416233091538, 0.8695648393196351, 0.24999937500156252, 0.8166665305555783, 0.7826083553876716, 0.8085104662743688, 0.7499981250046875, 0.7666664111111963, 0.5999996000002666, 0.0, 0.6999993000007, 0.5999988000024, 0.464285548469447, 0.9791664626736536, 0.9230762130182977, 0.6999993000007, 0.9999985714306123, 0.5333331555556148, 0.9459456902849485, 0.45454504132268975, 0.7586204280619213, 0.611110771605127, 0.7826083553876716, 0.45454504132268975, 0.6249992187509766, 0.7499981250046875, 0.941175916955343, 0.0, 0.3037974298990595, 0.9583329340279442, 0.5384611242606736, 0.9047614739231077, 0.7692301775152481, 0.8333319444467593, 0.9999995833335069, 0.9999966666777778, 0.8749989062513672, 0.9090904958679564, 0.8999991000009, 0.8571424489797862, 0.8076919970415396, 0.3999996000004, 0.2857138775516035, 0.24999968750039064]
# qwen_ukn_acc = [0.7499993750005208, 0.7499993750005208, 0.45454504132268975, 0.699999650000175, 0.9999990909099173, 0.9999995000002501, 0.41176446366796254, 0.42105240997241583, 0.7499981250046875, 0.9999980000040001, 0.9333327111115259, 0.16666638888935187, 0.9999990000010001, 0.9285707653065962, 0.8437497363282073, 0.13793098692034933, 0.6249992187509766, 0.9999990909099173, 0.7619043990931433, 0.9999975000062501, 0.7499981250046875, 0.3199998720000512, 0.7272720661163036, 0.19047609977328583, 0.8333326388894676, 0.9310341617123581, 0.45454504132268975, 0.8181810743808415, 0.9999995454547521, 0.9499995250002375, 0.0999999000001, 0.5999994000006, 0.46153810650914884, 0.2857138775516035, 0.9999975000062501, 0.8749989062513672, 0.7272720661163036, 0.6521736294897263, 0.9999988888901234, 0.799999680000128, 0.8095234240364647, 0.7272723966943652, 0.3333322222259259, 0.6666662222225186, 0.9999975000062501, 0.9999992307698226, 0.9999988888901234, 0.7499993750005208, 0.9999985714306123, 0.8333326388894676, 0.7187497753906951, 0.7692301775152481, 0.7999984000032001, 0.30303021120296625, 0.2972972169466981, 0.8679243645425727, 0.8363634842975483, 0.7419352445370179, 0.8499995750002125, 0.8571427040816599, 0.8909089289256492, 0.8085104662743688, 0.8545452991735819, 0.8461536834319839, 0.7432431428049807, 0.7391302741021143, 0.8888886913580685, 0.7999998222222616, 0.8936168311453551, 0.6911763689446516, 0.47619036281181837, 0.9249997687500577, 0.8958331467014277, 0.771428351020471, 0.8749997812500546, 0.9295773338623473, 0.7499998125000468, 0.760869399811, 0.6111109979424078, 0.8795179663231365, 0.6428569897959547, 0.6527776871142101, 0.8285713102040985, 0.7659572838388757, 0.9249997687500577, 0.8399998320000336, 0.8749996354168186, 0.9545450206613543, 0.9999950000249999, 0.8421048199448317, 0.7999984000032001, 0.5333329777780148, 0.8181814462811607, 0.9999994117650519, 0.724137681331834, 0.7499990625011719, 0.8124994921878174, 0.6538459023669606, 0.5882351211073172, 0.37931021403096066, 0.4444441975310014, 0.8461531952667729, 0.6666665277778067, 0.7894732686982796, 0.8666660888892741, 0.6176468771626832, 0.5333331555556148, 0.1249999609375122, 0.8749996354168186, 0.6666655555574075, 0.9999995454547521, 0.7142855102041399, 0.6315787811634785, 0.7777774897120409, 0.19999960000080003, 0.8333326388894676, 0.9999993333337778, 0.8749997265625854, 0.9047614739231077, 0.6499996750001625, 0.9166662847223814, 0.9999991666673611, 0.9047614739231077, 0.912280541705168, 0.7499997656250732, 0.8421048199448317, 0.6666664444445185, 0.7142852040819971, 0.76315769390587, 0.9999987500015626, 0.45454504132268975, 0.31578930747931183, 0.18749988281257324, 0.9333327111115259, 0.9999990909099173, 0.5999994000006, 0.4999998214286352, 0.6666663492065004, 0.9444439197533779, 0.8235289273359251, 0.8214282780613292, 0.10526310249310396, 0.8749996354168186, 0.9677416233091538, 0.7272720661163036]

# print(len(qwen_kwn_acc), len(qwen_ukn_acc))
# ax.boxplot([glc_kwn_acc_p2s, glc_ukn_acc_p2s, glc_kwn_acc_r2s, glc_ukn_acc_r2s, chatgpt_kwn_acc, chatgpt_ukn_acc, llama_kwn_acc, llama_ukn_acc, qwen_kwn_acc, qwen_ukn_acc], labels=['glc known p2s', 'glc unknown p2s', 'glc known r2s', 'glc unknown r2s', 'chatgpt known', 'chatgpt unknown', 'llama known', 'llama unknown', 'qwen known', 'qwen unknown'])
# plt.yticks(rotation=45, fontsize=20)
# plt.title(f'{dataset} Sketch Domain Known vs Unknown', fontsize=30)
# ax.xaxis.set_label_position('bottom')
# # ax.set_xticks(ticks=np.arange(8), labels=['known', 'unknown', '', 'known', 'unknown', '', 'known', 'unknown'])
# plt.xticks(rotation=45, fontsize=10)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"{dataset}_Sketch_Accuracy_boxplot.png")

# kwn_sample_size = [19, 14, 16, 6, 25, 18, 12, 23, 5, 5, 9, 30, 19, 26, 22, 22, 23, 8, 20, 3, 20, 17, 18, 17, 17, 3, 24, 37, 25, 7, 14, 35, 34, 29, 32, 18, 7, 17, 12, 16, 13, 37, 23, 21, 34, 36, 13, 25, 19, 12, 19, 57, 16, 5, 11, 6, 5, 8, 9, 4, 5, 7, 8, 12, 16, 2, 4, 14, 2, 5, 6, 5, 3, 4, 11, 3, 5, 12, 3, 0, 7, 0, 8, 6, 2, 23, 28, 18, 17, 15, 45, 11, 7, 8, 42, 10, 10, 48, 15, 1, 18, 25, 31, 21, 15, 12, 31, 23, 4, 60, 23, 47, 4, 30, 15, 30, 10, 5, 28, 48, 13, 10, 7, 30, 37, 11, 29, 18, 23, 11, 8, 4, 17, 20, 79, 24, 13, 21, 13, 6, 24, 3, 8, 22, 10, 21, 26, 10, 7, 8]
# ukn_sample_size = [12, 12, 11, 20, 11, 20, 17, 19, 4, 5, 15, 6, 10, 14, 32, 29, 8, 11, 21, 4, 4, 25, 11, 21, 12, 29, 11, 11, 22, 20, 10, 10, 13, 7, 4, 8, 11, 23, 9, 25, 21, 22, 3, 15, 4, 13, 9, 12, 7, 12, 32, 13, 5, 33, 37, 53, 55, 31, 20, 56, 55, 47, 55, 52, 74, 46, 45, 45, 47, 68, 42, 40, 48, 35, 40, 71, 40, 46, 54, 83, 42, 72, 70, 47, 40, 50, 24, 22, 2, 19, 5, 15, 22, 17, 29, 8, 16, 26, 34, 29, 18, 13, 48, 19, 15, 34, 30, 32, 24, 6, 22, 35, 38, 27, 5, 12, 15, 32, 21, 20, 24, 12, 21, 57, 32, 19, 30, 14, 38, 8, 11, 19, 16, 15, 11, 10, 28, 21, 18, 17, 28, 19, 24, 31, 11]

# mean_glc_kwn_p2s = sum(glc_kwn_acc_p2s) / len(glc_kwn_acc_p2s)
# mean_glc_ukn_p2s = sum(glc_ukn_acc_p2s) / len(glc_ukn_acc_p2s)
# print(mean_glc_kwn_p2s, mean_glc_ukn_p2s)

# mean_glc_kwn_r2s = sum(glc_kwn_acc_r2s) / len(glc_kwn_acc_r2s)
# mean_glc_ukn_r2s = sum(glc_ukn_acc_r2s) / len(glc_ukn_acc_r2s)
# print(mean_glc_kwn_r2s, mean_glc_ukn_r2s)

# mean_chatgpt_kwn = sum(chatgpt_kwn_acc) / len(chatgpt_kwn_acc)
# mean_chatgpt_ukn = sum(chatgpt_ukn_acc) / len(chatgpt_ukn_acc)
# print(mean_chatgpt_kwn, mean_chatgpt_ukn)

# mean_llama_kwn = sum(llama_kwn_acc) / len(llama_kwn_acc)
# mean_llama_ukn = sum(llama_ukn_acc) / len(llama_ukn_acc)
# print(mean_llama_kwn, mean_llama_ukn)

# mean_qwen_kwn = sum(qwen_kwn_acc) / len(qwen_kwn_acc)
# mean_qwen_ukn = sum(qwen_ukn_acc) / len(qwen_ukn_acc)
# print(mean_qwen_kwn, mean_qwen_ukn)
# print("------------")
# wgt_glc_kwn_p2s = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc_p2s)])  / sum(kwn_sample_size)
# wgt_glc_ukn_p2s = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc_p2s)])  / sum(ukn_sample_size)
# print(wgt_glc_kwn_p2s, wgt_glc_ukn_p2s)

# wgt_glc_kwn_r2s = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc_r2s)])  / sum(kwn_sample_size)
# wgt_glc_ukn_r2s = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc_r2s)])  / sum(ukn_sample_size)
# print(wgt_glc_kwn_r2s, wgt_glc_ukn_r2s)

# wgt_chatgpt_kwn = sum([x * y for x, y in zip(kwn_sample_size, chatgpt_kwn_acc)])  / sum(kwn_sample_size)
# wgt_chatgpt_ukn = sum([x * y for x, y in zip(ukn_sample_size, chatgpt_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_chatgpt_kwn, wgt_chatgpt_ukn)

# wgt_llama_kwn = sum([x * y for x, y in zip(kwn_sample_size, llama_kwn_acc)])  / sum(kwn_sample_size)
# wgt_llama_ukn = sum([x * y for x, y in zip(ukn_sample_size, llama_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_llama_kwn, wgt_llama_ukn)

# wgt_qwen_kwn = sum([x * y for x, y in zip(kwn_sample_size, qwen_kwn_acc)])  / sum(kwn_sample_size)
# wgt_qwen_ukn = sum([x * y for x, y in zip(ukn_sample_size, qwen_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_qwen_kwn, wgt_qwen_ukn)
# print("------------")

# ratio_mean_glc_p2s = mean_glc_kwn_p2s / mean_glc_ukn_p2s
# ratio_mean_glc_r2s = mean_glc_kwn_r2s / mean_glc_ukn_r2s
# ratio_mean_chatgpt = mean_chatgpt_kwn / mean_chatgpt_ukn
# ratio_mean_llama = mean_llama_kwn / mean_llama_ukn
# ratio_mean_qwen = mean_qwen_kwn / mean_qwen_ukn
# print(ratio_mean_glc_p2s, ratio_mean_glc_r2s, ratio_mean_chatgpt, ratio_mean_llama, ratio_mean_qwen)

# ratio_wgt_mean_glc_p2s = wgt_glc_kwn_p2s / wgt_glc_ukn_p2s
# ratio_wgt_mean_glc_r2s = wgt_glc_kwn_r2s / wgt_glc_ukn_r2s
# ratio_wgt_mean_chatgpt = wgt_chatgpt_kwn / wgt_chatgpt_ukn
# ratio_wgt_mean_llama = wgt_llama_kwn / wgt_llama_ukn
# ratio_wgt_mean_qwen = wgt_qwen_kwn / wgt_qwen_ukn
# print(ratio_wgt_mean_glc_p2s, ratio_wgt_mean_glc_r2s, ratio_wgt_mean_chatgpt, ratio_wgt_mean_llama, ratio_wgt_mean_qwen)

# #################### DomainNet real domain ######################################
# fig = plt.figure(figsize=(20, 14))
# ax = plt.subplot(111)
# glc_kwn_acc_p2r = [0.801,	0.521,	0.682,	0.936,	0.780,	0.319,	0.000,	0.323,	0.236,	0.803,	0.801,	0.963,	0.247,	0.042,	0.552,	0.856,	0.705,	0.005,	0.783,	0.920,	0.025,	0.588,	0.819,	0.277,	0.716,	0.106,	0.692,	0.258,	0.700,	0.759,	0.354,	0.388,	0.803,	0.851,	0.345,	0.453,	0.056,	0.012,	0.181,	0.223,	0.320,	0.857,	0.481,	0.543,	0.253,	0.480,	0.473,	0.629,	0.654,	0.677,	0.011,	0.000,	0.877,	0.032,	0.004,	0.757,	0.739,	0.718,	0.596,	0.234,	0.065,	0.443,	0.620,	0.366,	0.496,	0.007,	0.674,	0.614,	0.719,	0.063,	0.778,	0.775,	0.412,	0.751,	0.413,	0.646,	0.145,	0.590,	0.135,	0.585,	0.445,	0.000,	0.311,	0.754,	0.427,	0.639,	0.000,	0.766,	0.676,	0.834,	0.000,	0.050,	0.508,	0.008,	0.522,	0.274,	0.621,	0.714,	0.212,	0.910,	0.524,	0.861,	0.698,	0.706,	0.224,	0.015,	0.720,	0.244,	0.247,	0.570,	0.812,	0.010,	0.498,	0.679,	0.431,	0.011,	0.784,	0.730,	0.000,	0.528,	0.764,	0.800,	0.695,	0.744,	0.481,	0.429,	0.061,	0.507,	0.710,	0.850,	0.000,	0.058,	0.867,	0.004,	0.000,	0.157,	0.000,	0.859,	0.655,	0.553,	0.012,	0.814,	0.584,	0.650,	0.784,	0.704,	0.757,	0.502,	0.410,	0.416]
# glc_ukn_acc_p2r = [0.62410329, 0.88397788, 0.9576988,  0.63655913, 0.93112946, 0.74457428,
#  0.60569104, 0.97622192, 0.78035713, 0.92978206, 0.65465463, 0.96933559,
#  0.9648241,  0.9653916,  0.58823528, 0.93469909, 0.96822428, 0.96217492,
#  0.65753423, 0.97647057, 0.97613881, 0.91714284, 0.87719297, 0.9854604,
#  0.83091783, 0.98613516, 0.85518291, 0.50668647, 0.98833332, 0.94701985,
#  0.88918918, 0.98326897, 0.90882352, 0.95931141, 0.94496854, 0.97697367,
#  0.89999999, 0.81249999, 0.9856115,  0.99112425, 0.91457284, 0.98175179,
#  0.95670992, 0.77609426, 0.82310468, 0.97807016, 0.94423075, 0.97081412,
#  0.8854003,  0.96754562, 0.96445495, 0.82901553, 0.83898298, 0.406639,
#  0.97489538, 0.94894365, 0.05592841, 0.88968823, 0.98067631, 0.97069594,
#  0.63934423, 0.95979898, 0.68313457, 0.95485325, 0.87111109, 0.70129867,
#  0.93177736, 0.89057748, 0.90492955, 0.92610835, 0.95132739, 0.9784946,
#  0.97804389, 0.90856312, 0.92424228, 0.97368413, 0.49632351, 0.91525422,
#  0.94999998, 0.2782462,  0.61985018, 0.68575232, 0.88775501, 0.8309858,
#  0.99134198, 0.25212464, 0.99999984, 0.98680737, 0.84360186, 0.89516127,
#  0.93719802, 0.9821428,  0.72964168, 0.97356826, 0.80993519, 0.99389,
#  0.88467873, 0.78240739, 0.83850926, 0.25460122, 0.50259066, 0.98021581,
#  0.93401014, 0.9252971,  0.29155313, 0.54174067, 0.87797146, 0.92992422,
#  0.46764091, 0.73249998, 0.94517542, 0.92711863, 0.46787479, 0.73880596,
#  0.92134829, 0.73927958, 0.91439685, 0.975945,   0.97455967, 0.78068409,
#  0.95754715, 0.94722953, 0.97944198, 0.97761192, 0.93351061, 0.97933882,
#  0.99405645, 0.96675189, 0.70441987, 0.89510486, 0.99321265, 0.90664556,
#  0.25,       0.2360515,  0.96870341, 0.97029701, 0.95827122, 0.67317072,
#  0.87244093, 0.95823093, 0.98520707, 0.38878842, 0.96226413, 0.96778915,
#  0.91844658]

# glc_kwn_acc_s2r = [0.826, 0.619,	0.702,	0.936,	0.803,	0.336,	0.403,	0.419,	0.329,	0.803,	0.819,	0.981,	0.383,	0.357,	0.615,	0.902,	0.736,	0.003,	0.840,	0.931,	0.042,	0.597,	0.848,	0.319,	0.841,	0.135,	0.797,	0.459,	0.790,	0.750,	0.383,	0.488,	0.868,	0.843,	0.392,	0.508,	0.060,	0.065,	0.192,	0.225,	0.328,	0.863,	0.582,	0.569,	0.407,	0.545,	0.613,	0.681,	0.624,	0.773,	0.448,	0.000,	0.913,	0.027,	0.008,	0.813,	0.750,	0.718,	0.656,	0.274,	0.221,	0.469,	0.630,	0.801,	0.530,	0.005,	0.735,	0.817,	0.724,	0.100,	0.853,	0.759,	0.382,	0.795,	0.475,	0.673,	0.231,	0.698,	0.162,	0.588,	0.528,	0.003,	0.335,	0.757,	0.508,	0.773,	0.000,	0.833,	0.641,	0.848,	0.020,	0.055,	0.528,	0.029,	0.678,	0.496,	0.643,	0.739,	0.338,	0.906,	0.506,	0.889,	0.728,	0.747,	0.307,	0.023,	0.817,	0.245,	0.267,	0.594,	0.850,	0.447,	0.500,	0.679,	0.655,	0.019,	0.789,	0.784,	0.002,	0.547,	0.777,	0.781,	0.773,	0.756,	0.525,	0.424,	0.213,	0.558,	0.855,	0.842,	0.013,	0.157,	0.886,	0.009,	0.500,	0.525,	0.000,	0.900,	0.710,	0.539,	0.057,	0.821,	0.563,	0.606,	0.890,	0.739,	0.799,	0.645,	0.377,	0.444]
# glc_ukn_acc_s2r = [0.46628407, 0.69889501, 0.91032147, 0.47526881, 0.85812671, 0.6093489,
#  0.45731706, 0.94187581, 0.79464284, 0.82808715, 0.63963962, 0.94037477,
#  0.95226128, 0.94535517, 0.44038155, 0.88860434, 0.94953269, 0.86761227,
#  0.49315067, 0.87205881, 0.91106289, 0.87999999, 0.70350876, 0.73021,
#  0.70048306, 0.96360484, 0.80182926, 0.49628528, 0.97999998, 0.90507724,
#  0.66621621, 0.96525095, 0.82205881, 0.95931141, 0.92295596, 0.93585525,
#  0.79516128, 0.70220587, 0.94244603, 0.95857987, 0.85427134, 0.95255471,
#  0.91341987, 0.70202019, 0.77617327, 0.94152045, 0.94230767, 0.92165897,
#  0.78806906, 0.93711966, 0.89099524, 0.77202071, 0.80508468, 0.313278,
#  0.44351464, 0.89612674, 0.03355705, 0.88968823, 0.95491142, 0.91575088,
#  0.25136611, 0.91080401, 0.52470186, 0.94582391, 0.77777776, 0.4155844,
#  0.89766605, 0.84498478, 0.86619715, 0.8965517,  0.91150438, 0.9612903,
#  0.94011974, 0.87663279, 0.89393926, 0.95614027, 0.40073528, 0.73446326,
#  0.89354837, 0.18043845, 0.53370786, 0.61784287, 0.82653053, 0.77464778,
#  0.97258296, 0.23512747, 0.95081952, 0.96701846, 0.78199048, 0.85080643,
#  0.80676325, 0.93452375, 0.66123777, 0.94273126, 0.77969761, 0.96741342,
#  0.82207577, 0.69675924, 0.73291921, 0.17484662, 0.46977547, 0.94784171,
#  0.81725887, 0.92359931, 0.28065394, 0.39964475, 0.90808239, 0.89204544,
#  0.41544884, 0.64499998, 0.85087717, 0.73220338, 0.09884679, 0.68283581,
#  0.82584267, 0.71698112, 0.85992215, 0.93986253, 0.95303325, 0.46478872,
#  0.92610061, 0.93403691, 0.94273126, 0.93097013, 0.90957444, 0.97520659,
#  0.961367,   0.94373399, 0.4116022,  0.86363633, 0.95701355, 0.72784809,
#  0.14648437, 0.19742489, 0.92995528, 0.93729371, 0.82414306, 0.60731706,
#  0.8062992,  0.92874691, 0.98224849, 0.29656419, 0.85444742, 0.14641288,
#  0.87184464]

# lead_kwn_acc_p2r = [0.770,	0.789,	0.668,	0.938,	0.807,	0.288,	0.867,	0.387,	0.626,	0.795,	0.726,	0.926,	0.102,	0.561,	0.660,	0.781,	0.760,	0.115,	0.645,	0.874,	0.483,	0.320,	0.814,	0.388,	0.089,	0.754,	0.872,	0.250,	0.075,	0.690,	0.183,	0.140,	0.810,	0.796,	0.474,	0.127,	0.012,	0.307,	0.023,	0.073,	0.284,	0.610,	0.350,	0.457,	0.705,	0.495,	0.675,	0.527,	0.597,	0.894,	0.827,	0.000,	0.783,	0.242,	0.392,	0.741,	0.227,	0.742,	0.621,	0.500,	0.695,	0.462,	0.677,	0.781,	0.775,	0.425,	0.732,	0.847,	0.793,	0.250,	0.851,	0.375,	0.211,	0.677,	0.467,	0.727,	0.570,	0.648,	0.065,	0.665,	0.061,	0.129,	0.017,	0.577,	0.739,	0.587,	0.029,	0.736,	0.359,	0.881,	0.224,	0.194,	0.500,	0.523,	0.785,	0.842,	0.603,	0.377,	0.421,	0.799,	0.449,	0.865,	0.775,	0.726,	0.445,	0.050,	0.843,	0.475,	0.331,	0.518,	0.696,	0.037,	0.000,	0.592,	0.462,	0.000,	0.449,	0.704,	0.758,	0.572,	0.731,	0.193,	0.537,	0.508,	0.575,	0.351,	0.109,	0.513,	0.837,	0.882,	0.472,	0.406,	0.840,	0.036,	0.861,	0.290,	0.214,	0.853,	0.685,	0.412,	0.153,	0.772,	0.437,	0.468,	0.882,	0.777,	0.447,	0.437,	0.655,	0.612]
# lead_ukn_acc_p2r = [0.73170731, 0.71823202, 0.607445,   0.59784945, 0.8567493, 0.81803004,
#  0.39837398, 0.85601056, 0.85535713, 0.86682807, 0.7207207,  0.8773424,
#  0.68341707, 0.91803277, 0.56915738, 0.49551856, 0.92336447, 0.73758864,
#  0.63698629, 0.85882352, 0.86550974, 0.74857142, 0.3245614,  0.59612277,
#  0.95169078, 0.932409,   0.88719511, 0.90044575, 0.66666666, 0.86313464,
#  0.34189189, 0.72844272, 0.4117647,  0.85133019, 0.74999999, 0.80756578,
#  0.80161289, 0.74264705, 0.47913668, 0.9245562,  0.83165827, 0.85401457,
#  0.75757572, 0.64141413, 0.65342959, 0.76315788, 0.71730768, 0.69585252,
#  0.74411302, 0.93509126, 0.75829382, 0.45077719, 0.74576265, 0.3692946,
#  0.07112971, 0.77640844, 0.56599551, 0.57553955, 0.84219,    0.86080583,
#  0.33333332, 0.85929647, 0.66098806, 0.70654626, 0.52444443, 0.6233766,
#  0.86355474, 0.96656532, 0.89436617, 0.91625614, 0.6814159,  0.96989245,
#  0.94011974, 0.66618286, 0.9393938,  0.91228062, 0.58823527, 0.68173257,
#  0.71129031, 0.52107925, 0.47565542, 0.6484687,  0.91836725, 0.91549283,
#  0.37806637, 0.91501414, 0.80327856, 0.66490764, 0.78672982, 0.40524193,
#  0.77777774, 0.87499995, 0.37785016, 0.89867839, 0.71274297, 0.82688389,
#  0.51235584, 0.86805554, 0.63975151, 0.35276073, 0.71675301, 0.78597121,
#  0.70558374, 0.68251272, 0.87193458, 0.82770869, 0.35023771, 0.4090909,
#  0.49478078, 0.82499998, 0.93640349, 0.92711863, 0.40691927, 0.79850745,
#  0.60955054, 0.70840479, 0.92217895, 0.8436426,  0.93737767, 0.32193158,
#  0.4481132,  0.94195248, 0.72099852, 0.79291043, 0.92021274, 0.80785122,
#  0.52897473, 0.86445011, 0.83425412, 0.91958039, 0.34389139, 0.86392404,
#  0.17382812, 0.30686695, 0.92399402, 0.81023101, 0.4262295,  0.80975608,
#  0.84409447, 0.96068794, 0.95562127, 0.28209764, 0.71698111, 0.96046851,
#  0.92233008]

# lead_kwn_acc_s2r = [0.673,	0.168,	0.599,	0.851,	0.789,	0.349,	0.711,	0.516,	0.205,	0.619,	0.720,	0.870,	0.230,	0.504,	0.597,	0.745,	0.702,	0.035,	0.639,	0.736,	0.712,	0.365,	0.679,	0.255,	0.776,	0.397,	0.774,	0.674,	0.776,	0.677,	0.416,	0.426,	0.770,	0.828,	0.539,	0.515,	0.018,	0.087,	0.216,	0.212,	0.269,	0.728,	0.571,	0.372,	0.467,	0.415,	0.134,	0.562,	0.567,	0.750,	0.505,	0.097,	0.745,	0.033,	0.015,	0.727,	0.420,	0.544,	0.562,	0.379,	0.078,	0.224,	0.677,	0.432,	0.640,	0.021,	0.304,	0.698,	0.235,	0.087,	0.847,	0.631,	0.232,	0.521,	0.278,	0.661,	0.365,	0.290,	0.003,	0.533,	0.395,	0.016,	0.312,	0.662,	0.351,	0.467,	0.004,	0.717,	0.500,	0.834,	0.414,	0.075,	0.449,	0.111,	0.630,	0.337,	0.411,	0.305,	0.227,	0.081,	0.459,	0.857,	0.537,	0.546,	0.175,	0.070,	0.739,	0.170,	0.270,	0.471,	0.813,	0.180,	0.507,	0.566,	0.216,	0.088,	0.473,	0.424,	0.630,	0.534,	0.502,	0.690,	0.686,	0.715,	0.056,	0.384,	0.276,	0.538,	0.558,	0.789,	0.099,	0.112,	0.685,	0.036,	0.310,	0.131,	0.011,	0.828,	0.342,	0.542,	0.227,	0.638,	0.386,	0.593,	0.813,	0.534,	0.582,	0.461,	0.327,	0.454]
# lead_ukn_acc_s2r = [0.23098995, 0.38674032, 0.48054145, 0.43010752, 0.59090908, 0.58263772,
#  0.21138211, 0.23249669, 0.51785713, 0.62469732, 0.60660659, 0.28449744,
#  0.1281407,  0.64298724, 0.36248012, 0.11011524, 0.35327102, 0.73758864,
#  0.52511414, 0.9735294,  0.44902385, 0.33142857, 0.62631578, 0.05008077,
#  0.4927536,  0.33968804, 0.76676828, 0.49182763, 0.94666665, 0.22737306,
#  0.06486486, 0.72844272, 0.30735294, 0.40532081, 0.54874213, 0.82236841,
#  0.4516129,  0.57536764, 0.25467626, 0.53846153, 0.48743717, 0.87226274,
#  0.77489174, 0.58249157, 0.29602888, 0.12865497, 0.11346154, 0.78648232,
#  0.26530612, 0.28803245, 0.13744076, 0.5060449,  0.57627114, 0.12448133,
#  0.07949791, 0.31161971, 0.08948546, 0.54676258, 0.13365539, 0.66300364,
#  0.0273224,  0.22236181, 0.1056218,  0.15801354, 0.36444444, 0.26839826,
#  0.64452423, 0.32826747, 0.83450701, 0.58866994, 0.61504422, 0.18279569,
#  0.88223551, 0.25399129, 0.96969682, 0.8157894,  0.83455879, 0.259887,
#  0.42741935, 0.26812816, 0.32771535, 0.38082556, 0.40816322, 0.63380273,
#  0.13419913, 0.29461756, 0.83606544, 0.83509234, 0.46919429, 0.17741935,
#  0.42995167, 0.80357138, 0.27198697, 0.77533038, 0.67386608, 0.96741342,
#  0.28995057, 0.4074074,  0.44099376, 0.18711656, 0.19689119, 0.48741006,
#  0.35702199, 0.1426146,  0.07629428, 0.36589697, 0.14263074, 0.12121212,
#  0.24425887, 0.47499999, 0.21271929, 0.37457626, 0.08896211, 0.16044776,
#  0.25842696, 0.23499142, 0.73540853, 0.53092783, 0.33072406, 0.17907444,
#  0.05503145, 0.91292874, 0.62995594, 0.92350745, 0.47872339, 0.22314049,
#  0.09658247, 0.12531969, 0.25138121, 0.73076921, 0.04072398, 0.61550632,
#  0.109375,   0.0944206,  0.83010431, 0.68316831, 0.30849478, 0.54878047,
#  0.69291337, 0.54054053, 0.2337278,  0.12477396, 0.28301886, 0.13762811,
#  0.8660194 ]

# chatgpt_kwn_acc = [0.9999997872340878, 0.9166664756944841, 0.7575755280074157, 0.9761902437642276, 0.8333319444467593, 0.7999997714286367, 0.8823528114187041, 0.9999900000999989, 0.5094338661445535, 0.8787876124886022, 0.9599996160001536, 0.7142846938790088, 0.4583331423611907, 0.9464284024234995, 0.0, 0.9999997368421745, 0.9999996153847633, 0.731707138608015, 0.8823526816609759, 0.9999991666673611, 0.8749989062513672, 0.9473681717452179, 0.9999995238097505, 0.9687496972657195, 0.9599998080000384, 0.9836063961300989, 0.8823527681661238, 0.43749993164063566, 0.917808093450946, 0.99999979166671, 0.3399999320000136, 0.5961537315088977, 0.865384448964529, 0.9444441820988382, 0.776119287146375, 0.536585234979211, 0.5384614349112625, 0.5606059756657612, 0.6578946502770197, 0.4761904006046983, 0.39682533383724855, 0.8888886913580685, 0.951612749739879, 0.6760562428089798, 0.7424241299357378, 0.7808218108463273, 0.9342104033933679, 0.7580643938605816, 0.8076919970415396, 0.9857141448979793, 0.8529410510380807, 0.0, 0.9836063961300989, 0.851063648709862, 0.03260869210775086, 0.9230766863905931, 0.9999993750003906, 0.933333125925972, 0.8749997812500546, 0.33333314814825105, 0.8478259026465429, 0.8870966311134465, 0.8799996480001409, 0.8983049324906893, 0.8039214109958017, 0.8666665222222463, 0.8656715125863413, 0.7560974687686013, 0.9999996296297669, 0.722222088477391, 0.8857140326531334, 0.9629626063101459, 0.9230766863905931, 0.9032256607700546, 0.6799997280001088, 0.933333125925972, 0.2499999652777826, 0.7499996875001302, 0.19672127922601979, 0.8846150443788291, 0.818181570248009, 0.706666572444457, 0.1851851508916387, 0.8113206016376223, 0.9259257544581936, 0.8461537159763514, 0.7021275101856361, 0.917808093450946, 0.749999531250293, 0.9807690421597995, 0.29032248699274615, 0.7441858734451456, 0.9411762860438654, 0.5781249096679828, 0.5454544746163019, 0.9199998160000368, 0.6249998697916938, 0.6052629986150003, 0.7567565522279588, 0.8399996640001344, 0.6111109979424078, 0.9850744798396298, 0.8918916508400943, 0.9852939727508863, 0.6999997666667445, 0.0, 0.9852939727508863, 0.8928569834183958, 0.5151513590450426, 0.49999991666668053, 0.9999998113207903, 0.5322579786680679, 0.6744184478096632, 0.7380950623583185, 0.8076921523668937, 0.0, 0.9454543735537502, 0.7586205588585243, 0.841269707734967, 0.6666665432098994, 0.8793101932223805, 0.9749997562500609, 0.9615382766272544, 0.9999995238097505, 0.8536583283760174, 0.7999998222222616, 0.6206895481569744, 0.8235291695502441, 0.9473682963989083, 0.9565215311909714, 0.8799998826666823, 0.8749998632812713, 0.9272725586777165, 0.10256407626562146, 0.32727266776860586, 0.8749998632812713, 0.7105261288089134, 0.9999998305085033, 0.7499998557692584, 0.9117644377163417, 0.5555554526749161, 0.8032785568395808, 0.8367345231154034, 0.9827584512485429, 0.9999998305085033, 0.9531248510742419, 0.839285564413292, 0.699999860000028, 0.6521737712665714, 0.8372091076257888]
# chatgpt_ukn_acc = [0.49999993421053496, 0.2399999040000384, 0.7115383247041683, 0.5416665538194679, 0.9636361884297838, 0.8076921523668937, 0.6170211453146499, 0.5844155085174664, 0.8947366066482614, 0.9428568734694647, 0.7499997916667245, 0.7666665388889101, 0.9215684467512849, 0.9508195162590957, 0.8775508413161548, 0.14432988202784722, 0.6491226931363696, 0.7249998187500453, 0.8979590004165304, 0.527272631404976, 0.777777561728455, 0.8714284469387933, 0.9615382766272544, 0.581818076033077, 0.9999994736844876, 0.9411762860438654, 0.27272722314050485, 0.9242422842057144, 0.8983049324906893, 0.9999997560976204, 0.5999999076923218, 0.2820512458908659, 0.8313252010451565, 0.8153844899408477, 0.7966100344728755, 0.912280541705168, 0.9636361884297838, 0.6964284470663487, 0.9999998630137173, 0.9999998461538697, 0.7096771904267127, 0.772726921487763, 0.6842101662051757, 0.8095236810279871, 0.851851694101538, 0.9117645717993277, 0.99999980000004, 0.5409835178715544, 0.9999998387097034, 0.9803919646290264, 0.914285453061299, 0.44444434567903424, 0.33333277777870374, 0.16326527280300554, 0.021276591217746547, 0.9259257544581936, 0.9803919646290264, 0.9756095181440199, 0.9076921680473587, 0.5769228550296711, 0.8333326388894676, 0.9672129561945972, 0.9649121114189277, 0.9803919646290264, 0.9761902437642276, 0.6086953875237445, 0.9787231960163412, 0.9230765680474738, 0.9655169084424454, 0.6764703892734149, 0.772726921487763, 0.9736839542936961, 0.99999980000004, 0.8684209383656659, 0.5999988000024, 0.9999987500015626, 0.7666664111111963, 0.9999997826087429, 0.7818180396694473, 0.6181817057851444, 0.49206341395818826, 0.7499998828125183, 0.7999992000008, 0.8749989062513672, 0.9710143520269054, 0.9090906336088989, 0.5999988000024, 0.9117645717993277, 0.6666664444445185, 0.9459456902849485, 0.8421048199448317, 0.9999990000010001, 0.7096773048907572, 0.9387753186172818, 0.9230766863905931, 0.199999960000008, 0.7419353642039734, 0.2857142274052597, 0.8749994531253418, 0.6428569132653881, 0.7377047970975742, 0.7142855685131493, 0.8548385718002303, 0.999999803921607, 0.9024388042832184, 0.7619045804989093, 0.8524588766460858, 0.17021272974197238, 0.5128203813281073, 0.8181816322314471, 0.9230766863905931, 0.9137929458977679, 0.8823527681661238, 0.868420824099783, 0.3225805411030513, 0.6999998833333527, 0.5652171455577628, 0.99999980000004, 0.9999998076923446, 0.8333331018519161, 0.805555443672855, 0.9374997070313414, 0.8070174022776487, 0.8043476512287714, 0.9393936547291954, 0.4285713265306365, 0.7671231825858653, 0.2857141836735058, 0.933333125925972, 0.9999995238097505, 0.6315787811634785, 0.753623079185061, 0.6904760260771367, 0.5098038216070938, 0.9508195162590957, 0.8979590004165304, 0.7833332027777995, 0.5510202957101437, 0.8360654367105841, 0.9117644377163417, 0.656249794921939, 0.7391302741021143, 0.9310341617123581, 0.9636361884297838, 0.8999997750000561]
# print(len(chatgpt_kwn_acc), len(chatgpt_ukn_acc))

# llama_kwn_acc = [0.7499997916667245, 0.7962961488340464, 0.8275859215220961, 0.8292680904224169, 0.40909072314058037, 0.3555554765432274, 0.8378377246165236, 0.9999950000249999, 0.35294110726644956, 0.466666511111163, 0.799999680000128, 0.9999966666777778, 0.15384611439843218, 0.7213113571620725, 0.0, 0.9310341617123581, 0.7083330381945674, 0.5952379535147729, 0.6956518714557081, 0.4999991666680556, 0.2999997000003, 0.7441858734451456, 0.799999680000128, 0.31428562448982156, 0.7142855685131493, 0.7968748754883007, 0.42105255771007755, 0.38095232048375866, 0.5735293274221577, 0.8124997460938292, 0.2941176038062347, 0.38235288494810515, 0.8285713102040985, 0.7586204280619213, 0.48611104359568835, 0.8260865973536533, 0.26086950850662854, 0.3768115395925305, 0.30769226035503683, 0.2962962414266219, 0.36538454511835666, 0.6136362241735853, 0.7236841153047217, 0.46969689853077295, 0.39534879123851263, 0.47368414819945415, 0.6268655780797644, 0.3676470047577934, 0.4705880968858538, 0.34848479568412183, 0.6562498974609535, 0.0, 0.838235170847769, 0.5312499169922005, 0.07142856292517108, 0.49999987500003124, 0.35714260204099857, 0.7297295325055317, 0.6458331987847502, 0.6428566836737974, 0.627450857362577, 0.7966100344728755, 0.6086953875237445, 0.7462685453330529, 0.399999920000016, 0.6739128969754571, 0.49090900165290874, 0.8356163238881748, 0.8333328703706276, 0.399999920000016, 0.8863634349174011, 0.583333090277879, 0.5757574012856359, 0.8113206016376223, 0.3199998720000512, 0.7428569306123055, 0.6119402071731034, 0.5483869198751872, 0.4799999040000192, 0.6799997280001088, 0.310344720570786, 0.47368414819945415, 0.2068965160523248, 0.6617646085640281, 0.6406248999023594, 0.6406248999023594, 0.7111109530864548, 0.7719296891351423, 0.6842101662051757, 0.9014083237453064, 0.1333332888889037, 0.5882351787774159, 0.7058822145328991, 0.36363628099175427, 0.7631577943213428, 0.8461536834319839, 0.9074072393690297, 0.6428569132653881, 0.6666665497076228, 0.8387094068679333, 0.3965516557669559, 0.8333332070707261, 0.5999998285714775, 0.5416665538194679, 0.8292680904224169, 0.0, 0.9374998828125146, 0.5238094406651681, 0.3437498925781585, 0.4177214661112068, 0.8490564435742559, 0.31818176997245906, 0.4399999120000176, 0.6382977365323964, 0.6329113122897073, 0.018181814876033656, 0.7343748852539241, 0.8333331944444675, 0.7096773048907572, 0.4722220910494191, 0.8199998360000328, 0.7499998660714524, 0.8043476512287714, 0.6818178719009673, 0.774193298647323, 0.34782601134217145, 0.3684209879963179, 0.36111101080249697, 0.6486485609934377, 0.5454543801653393, 0.26470584342561126, 0.5588233650519514, 0.9245281274475231, 0.0, 0.17391300567108572, 0.38571423061225274, 0.4827584542212227, 0.8571427210884569, 0.5423727894283408, 0.282051209730459, 0.3043477599244, 0.6935482752341491, 0.6842104062788761, 0.8124998307292018, 0.9493669684345609, 0.8260868367990091, 0.799999885714302, 0.2711863947141704, 0.13793098692034933, 0.5263156509695655]
# llama_ukn_acc = [0.8243242129291604, 0.8124997460938292, 0.7343748852539241, 0.8780485663296179, 0.9523808012093966, 0.9761902437642276, 0.9999997619048185, 0.7083332349537174, 0.9482756985731554, 0.8999997750000561, 0.8571425510205175, 0.6603772338910879, 0.9999997058824394, 0.9387753186172818, 0.8985505944129573, 0.4615383905325553, 0.9782606568998571, 0.9285712074830458, 0.8636361673554165, 0.949999841666693, 0.9230766863905931, 0.9594593298027932, 0.884615214497074, 0.9166664756944841, 0.9999993333337778, 0.9649121114189277, 0.825396694381477, 0.971428432653081, 0.9818180033058175, 0.9999997222222993, 0.9076921680473587, 0.807692204142025, 0.9538460071006143, 0.7968748754883007, 0.8983049324906893, 0.9512192801904193, 0.99999979166671, 0.8928569834183958, 0.9384613940828624, 0.9999998214286032, 0.9499997625000594, 0.9999994736844876, 0.9999995454547521, 0.9827584512485429, 0.9069765332612713, 0.9583332002314999, 0.9999997872340878, 0.9166665393518695, 0.9701491089329688, 0.9999997777778271, 0.9302323418064321, 0.949999841666693, 0.9999993750003906, 0.5333332148148411, 0.27083327690973397, 0.9999997674419144, 0.9791664626736536, 0.969696675849492, 0.9104476253063245, 0.8518515363512829, 0.9090904958679564, 0.9571427204081827, 0.9259257544581936, 0.9787231960163412, 0.9999997500000625, 0.8124994921878174, 0.9999997619048185, 0.9999997560976204, 0.9565213232515986, 0.8292680904224169, 0.941175916955343, 0.9565215311909714, 0.9791664626736536, 0.7894735457063955, 0.9999975000062501, 0.9166659027784144, 0.9523804988664292, 0.9074072393690297, 0.9843748461914302, 0.9399998120000376, 0.9111109086420203, 0.9999998412698664, 0.8749989062513672, 0.9090900826453795, 0.9999998412698664, 0.9999996551725328, 0.9999975000062501, 0.9620251946803551, 0.899999550000225, 0.9818180033058175, 0.9374997070313414, 0.9999994444447531, 0.8070174022776487, 0.9210523891967396, 0.99999980000004, 0.6956520226843429, 0.8823527681661238, 0.9487177054569985, 0.9999995454547521, 0.8857140326531334, 0.9275360974585366, 0.9803919646290264, 0.7857141454081883, 0.9795916368180332, 0.9117644377163417, 0.9814812997256852, 0.9661015311692319, 0.26923071745563126, 0.851063648709862, 0.9090906336088989, 0.9512192801904193, 0.999999803921607, 0.9814812997256852, 0.9591834777176576, 0.6458331987847502, 0.9285712627551316, 0.9999996296297669, 0.999999803921607, 0.9999997500000625, 0.8888886913580685, 0.9833331694444717, 0.9999997619048185, 0.9642855420918675, 0.9821426817602353, 0.9615380917161186, 0.9787231960163412, 0.9298243982764213, 0.810810591672813, 0.9189186705625214, 0.9999996551725328, 0.9705879498270735, 0.8593748657226772, 0.6206895481569744, 0.9999997560976204, 0.999999803921607, 0.9464284024234995, 0.9016391964525907, 0.8333331018519161, 0.9999997777778271, 0.9117644377163417, 0.9999996666667778, 0.8214284247449241, 0.8571426122449679, 0.9677417793964871, 0.999999814814849]

# print(len(llama_kwn_acc), len(llama_ukn_acc))

# qwen_kwn_acc = [0.851063648709862, 0.9166664756944841, 0.6666664646465258, 0.9761902437642276, 0.9999983333361112, 0.7142855102041399, 0.8235292906574572, 0.9999900000999989, 0.43396218227128636, 0.8787876124886022, 0.8799996480001409, 0.9999985714306123, 0.4583331423611907, 0.8571427040816599, 0.0, 0.9736839542936961, 0.8846150443788291, 0.731707138608015, 0.8823526816609759, 0.9166659027784144, 0.9999987500015626, 0.9999997368421745, 0.9999995238097505, 0.9999996875000976, 0.8799998240000352, 0.9836063961300989, 0.9019606074587043, 0.5781249096679828, 0.9452053499718698, 0.8541664887153148, 0.49999990000002, 0.7307690902367133, 0.8269229178994388, 0.9444441820988382, 0.761193916239714, 0.7560973765616155, 0.7307690902367133, 0.590909001377424, 0.7631577943213428, 0.4761904006046983, 0.42857136054422845, 0.8444442567901651, 0.9193546904266627, 0.7605632731601023, 0.8333332070707261, 0.7671231825858653, 0.8157893663435044, 0.6935482752341491, 0.8461535207101843, 0.971428432653081, 0.9264704519896394, 0.0, 0.9672129561945972, 0.8297870574921153, 0.26086953686200687, 0.846153629191377, 0.9999993750003906, 0.9555553432099236, 0.8749997812500546, 0.4444441975310014, 0.7826085255198857, 0.8064514828304059, 0.8799996480001409, 0.9322032318299607, 0.8039214109958017, 0.8999998500000249, 0.6865670617064087, 0.9024389143367177, 0.9999996296297669, 0.722222088477391, 0.914285453061299, 0.9629626063101459, 0.9487177054569985, 0.9354837200832709, 0.7199997120001153, 0.933333125925972, 0.7777776697531014, 0.8333329861112558, 0.45901631819404615, 0.8846150443788291, 0.818181570248009, 0.7333332355555685, 0.31481475651578583, 0.9056602064792063, 0.851851694101538, 0.9230767810651106, 0.7446806926211291, 0.8904108369300223, 0.749999531250293, 0.9423075110947093, 0.19354832466183075, 0.7906974905354672, 0.9411762860438654, 0.6406248999023594, 0.7012986102209596, 0.9599998080000384, 0.8333331597222583, 0.7894734764543483, 0.83783761139524, 0.9599996160001536, 0.6851850582990633, 0.9850744798396298, 0.8918916508400943, 0.8823528114187041, 0.7333330888889704, 0.02499999375000156, 0.9705880925605745, 0.8928569834183958, 0.5454543801653393, 0.5333332444444592, 0.9811318903524735, 0.6612902159209328, 0.3953487452677336, 0.8095236167800912, 0.8076921523668937, 0.0, 0.9454543735537502, 0.7586205588585243, 0.8730157344419469, 0.9074072393690297, 0.9655170749108491, 0.9749997562500609, 0.9807690421597995, 0.9523804988664292, 0.8780485663296179, 0.6666665185185514, 0.6379309244946682, 0.7941174134948783, 0.9210525103878275, 0.9782606568998571, 0.7466665671111243, 0.843749868164083, 0.9272725586777165, 0.05128203813281073, 0.9090907438016829, 0.8593748657226772, 0.868420824099783, 0.9830506808388676, 0.8076921523668937, 0.8823526816609759, 0.6111109979424078, 0.885245756517089, 0.857142682215779, 0.9827584512485429, 0.9999998305085033, 0.9218748559570538, 0.8928569834183958, 0.7799998440000312, 0.7391302741021143, 0.7674416819903065]
# qwen_ukn_acc = [0.6842104362881004, 0.6399997440001024, 0.7115383247041683, 0.5624998828125244, 0.9636361884297838, 0.7692306213018035, 0.5531913716614103, 0.41558436161242057, 0.9473681717452179, 0.9428568734694647, 0.7222220216049939, 0.46666658888890183, 0.7843135717032211, 0.8360654367105841, 0.7346937276135249, 0.09278349558933034, 0.8771928285626616, 0.7249998187500453, 0.7346937276135249, 0.8545452991735819, 0.6111109413580718, 0.8142855979592003, 0.7115383247041683, 0.3454544826446395, 0.8947363711913836, 0.9215684467512849, 0.23636359338843754, 0.7575756427915692, 0.9999998305085033, 0.9999997560976204, 0.19999996923077396, 0.3205127794214385, 0.7469878618086914, 0.5230768426035627, 0.7118642861246972, 0.8070174022776487, 0.8909089289256492, 0.5714284693877733, 0.9315067217114079, 0.9384613940828624, 0.6774191363164076, 0.8636359710745586, 0.7368417174517277, 0.8095236810279871, 0.9074072393690297, 0.9705880925605745, 0.9599998080000384, 0.5737703977425577, 0.9677417793964871, 0.999999803921607, 0.771428351020471, 0.5555554320987929, 0.6666655555574075, 0.14285711370262985, 0.021276591217746547, 0.999999814814849, 0.9803919646290264, 0.9999997560976204, 0.7999998769230958, 0.8846150443788291, 0.7499993750005208, 0.8688523165815873, 0.6842104062788761, 0.960784125336446, 0.9761902437642276, 0.6521736294897263, 0.9574466047985947, 0.9999996153847633, 0.9310341617123581, 0.8235291695502441, 0.5454542975207739, 0.9736839542936961, 0.99999980000004, 0.7894735803324237, 0.9999980000040001, 0.8749989062513672, 0.7333330888889704, 0.6956520226843429, 0.74545440991738, 0.581818076033077, 0.6666665608465776, 0.593749907226577, 0.8999991000009, 0.6249992187509766, 0.9855071035496951, 0.9999996969697887, 0.9999980000040001, 0.838235170847769, 0.6999997666667445, 0.8918916508400943, 0.7368417174517277, 0.9999990000010001, 0.6935482752341491, 0.8775508413161548, 0.8717946482577824, 0.1799999640000072, 0.9677417793964871, 0.6326529321116464, 0.9374994140628662, 0.6785711862245764, 0.5573769578070561, 0.7142855685131493, 0.6290321566077166, 0.9803919646290264, 0.8536583283760174, 0.6666665079365457, 0.885245756517089, 0.10638295608873274, 0.5897434385273234, 0.8409088997934318, 0.9743587245234039, 0.8793101932223805, 0.9411762860438654, 0.5526314335180438, 0.22580637877213589, 0.8333331944444675, 0.43478241965981757, 0.9599998080000384, 0.9999998076923446, 0.6666664814815328, 0.9444443132716231, 0.8437497363282073, 0.7894735457063955, 0.8260867769376571, 0.9393936547291954, 0.9047616893424548, 0.7671231825858653, 0.8214282780613292, 0.933333125925972, 0.9523804988664292, 0.47368408587260896, 0.7391303276622713, 0.7142855442177275, 0.8823527681661238, 0.885245756517089, 0.5510202957101437, 0.6333332277777953, 0.6734692503123979, 0.9180326363880923, 0.9117644377163417, 0.8749997265625854, 0.34782601134217145, 0.724137681331834, 0.9272725586777165, 0.5249998687500328]

# print(len(qwen_kwn_acc), len(qwen_ukn_acc))
# ax.boxplot([glc_kwn_acc_p2r, glc_ukn_acc_p2r, glc_kwn_acc_s2r, glc_ukn_acc_s2r, chatgpt_kwn_acc, chatgpt_ukn_acc, llama_kwn_acc, llama_ukn_acc, qwen_kwn_acc, qwen_ukn_acc], labels=['glc known p2r', 'glc unknown p2r', 'glc known s2r', 'glc unknown s2r', 'chatgpt known', 'chatgpt unknown', 'llama known', 'llama unknown', 'qwen known', 'qwen unknown'])
# plt.yticks(rotation=45, fontsize=20)
# plt.title(f'{dataset} Real Domain Known vs Unknown', fontsize=30)
# ax.xaxis.set_label_position('bottom')
# # ax.set_xticks(ticks=np.arange(8), labels=['known', 'unknown', '', 'known', 'unknown', '', 'known', 'unknown'])
# plt.xticks(rotation=45, fontsize=10)
# ax.xaxis.tick_bottom()
# ax.set_ylabel('Accuracy', fontsize=30)
# plt.savefig(f"{dataset}_Real_Accuracy_boxplot.png")

# kwn_sample_size = [47, 48, 33, 42, 6, 35, 68, 1, 53, 33, 25, 7, 24, 56, 41, 38, 26, 41, 34, 12, 8, 38, 21, 32, 50, 61, 51, 64, 73, 48, 50, 52, 52, 36, 67, 41, 52, 66, 76, 63, 63, 45, 62, 71, 66, 73, 76, 62, 26, 70, 68, 2, 61, 47, 92, 39, 16, 45, 40, 18, 46, 62, 25, 59, 51, 60, 67, 82, 27, 54, 35, 27, 39, 62, 25, 45, 72, 24, 61, 26, 33, 75, 54, 53, 54, 65, 47, 73, 16, 52, 31, 43, 51, 64, 77, 50, 48, 38, 37, 25, 54, 67, 37, 68, 30, 40, 68, 56, 33, 60, 53, 62, 43, 42, 52, 53, 55, 58, 63, 54, 58, 40, 52, 21, 41, 45, 58, 34, 76, 46, 75, 64, 55, 39, 55, 64, 38, 59, 52, 34, 54, 61, 49, 58, 59, 64, 56, 50, 46, 43]
# ukn_sample_size = [76, 25, 52, 48, 55, 52, 47, 77, 38, 35, 36, 60, 51, 61, 49, 97, 57, 40, 49, 55, 36, 70, 52, 55, 19, 51, 55, 66, 59, 41, 65, 78, 83, 65, 59, 57, 55, 56, 73, 65, 31, 22, 19, 63, 54, 68, 50, 61, 62, 51, 35, 45, 6, 49, 47, 54, 51, 41, 65, 26, 12, 61, 57, 51, 42, 23, 47, 26, 29, 34, 22, 38, 50, 76, 5, 8, 30, 46, 55, 55, 63, 64, 10, 8, 69, 33, 5, 68, 30, 37, 19, 10, 62, 49, 39, 50, 62, 49, 16, 28, 61, 49, 62, 51, 41, 42, 61, 47, 39, 44, 39, 58, 51, 38, 31, 60, 23, 50, 52, 36, 72, 32, 57, 46, 33, 42, 73, 28, 45, 21, 38, 69, 42, 51, 61, 49, 60, 49, 61, 34, 32, 46, 29, 55, 40]

# mean_glc_kwn_p2s = sum(glc_kwn_acc_p2r) / len(glc_kwn_acc_p2r)
# mean_glc_ukn_p2s = sum(glc_ukn_acc_p2r) / len(glc_ukn_acc_p2r)
# print(mean_glc_kwn_p2s, mean_glc_ukn_p2s)

# mean_glc_kwn_r2s = sum(glc_kwn_acc_s2r) / len(glc_kwn_acc_s2r)
# mean_glc_ukn_r2s = sum(glc_ukn_acc_s2r) / len(glc_ukn_acc_s2r)
# print(mean_glc_kwn_r2s, mean_glc_ukn_r2s)

# mean_chatgpt_kwn = sum(chatgpt_kwn_acc) / len(chatgpt_kwn_acc)
# mean_chatgpt_ukn = sum(chatgpt_ukn_acc) / len(chatgpt_ukn_acc)
# print(mean_chatgpt_kwn, mean_chatgpt_ukn)

# mean_llama_kwn = sum(llama_kwn_acc) / len(llama_kwn_acc)
# mean_llama_ukn = sum(llama_ukn_acc) / len(llama_ukn_acc)
# print(mean_llama_kwn, mean_llama_ukn)

# mean_qwen_kwn = sum(qwen_kwn_acc) / len(qwen_kwn_acc)
# mean_qwen_ukn = sum(qwen_ukn_acc) / len(qwen_ukn_acc)
# print(mean_qwen_kwn, mean_qwen_ukn)
# print("------------")
# wgt_glc_kwn_p2s = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc_p2r)])  / sum(kwn_sample_size)
# wgt_glc_ukn_p2s = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc_p2r)])  / sum(ukn_sample_size)
# print(wgt_glc_kwn_p2s, wgt_glc_ukn_p2s)

# wgt_glc_kwn_r2s = sum([x * y for x, y in zip(kwn_sample_size, glc_kwn_acc_s2r)])  / sum(kwn_sample_size)
# wgt_glc_ukn_r2s = sum([x * y for x, y in zip(ukn_sample_size, glc_ukn_acc_s2r)])  / sum(ukn_sample_size)
# print(wgt_glc_kwn_r2s, wgt_glc_ukn_r2s)

# wgt_chatgpt_kwn = sum([x * y for x, y in zip(kwn_sample_size, chatgpt_kwn_acc)])  / sum(kwn_sample_size)
# wgt_chatgpt_ukn = sum([x * y for x, y in zip(ukn_sample_size, chatgpt_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_chatgpt_kwn, wgt_chatgpt_ukn)

# wgt_llama_kwn = sum([x * y for x, y in zip(kwn_sample_size, llama_kwn_acc)])  / sum(kwn_sample_size)
# wgt_llama_ukn = sum([x * y for x, y in zip(ukn_sample_size, llama_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_llama_kwn, wgt_llama_ukn)

# wgt_qwen_kwn = sum([x * y for x, y in zip(kwn_sample_size, qwen_kwn_acc)])  / sum(kwn_sample_size)
# wgt_qwen_ukn = sum([x * y for x, y in zip(ukn_sample_size, qwen_ukn_acc)])  / sum(ukn_sample_size)
# print(wgt_qwen_kwn, wgt_qwen_ukn)
# print("------------")
# ratio_mean_glc_p2s = mean_glc_kwn_p2s / mean_glc_ukn_p2s
# ratio_mean_glc_r2s = mean_glc_kwn_r2s / mean_glc_ukn_r2s
# ratio_mean_chatgpt = mean_chatgpt_kwn / mean_chatgpt_ukn
# ratio_mean_llama = mean_llama_kwn / mean_llama_ukn
# ratio_mean_qwen = mean_qwen_kwn / mean_qwen_ukn
# print(ratio_mean_glc_p2s, ratio_mean_glc_r2s, ratio_mean_chatgpt, ratio_mean_llama, ratio_mean_qwen)

# ratio_wgt_mean_glc_p2s = wgt_glc_kwn_p2s / wgt_glc_ukn_p2s
# ratio_wgt_mean_glc_r2s = wgt_glc_kwn_r2s / wgt_glc_ukn_r2s
# ratio_wgt_mean_chatgpt = wgt_chatgpt_kwn / wgt_chatgpt_ukn
# ratio_wgt_mean_llama = wgt_llama_kwn / wgt_llama_ukn
# ratio_wgt_mean_qwen = wgt_qwen_kwn / wgt_qwen_ukn
# print(ratio_wgt_mean_glc_p2s, ratio_wgt_mean_glc_r2s, ratio_wgt_mean_chatgpt, ratio_wgt_mean_llama, ratio_wgt_mean_qwen)
