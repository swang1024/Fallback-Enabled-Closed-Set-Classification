import os
import base64
import re
from openai import OpenAI
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
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPModel
from transformers import MllamaTextModel, AutoProcessor
import torch
import seaborn as sns
from collections import Counter
from scipy import stats

set_random_seed(2025)

# Load llm generated image summary
dataset = "INaturalist"
model_name = "CLIP"
second_opinion_model = "llama"
version = 'v10'
load_scores = False
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
    df = pd.read_csv(f'{dataset}_target_domain{domain}_{model_name}_summary_{version}.csv')[:12000]
    # df = df.reset_index(drop=True)
elif dataset == "VisDA":
    domain = 1
    num_samples = 40000
    class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
    df = pd.read_csv(f'{dataset}_target_domain{domain}_{model_name}_summary_v10.csv', index_col=False)[:num_samples]
    df_direct = pd.read_csv(f"{dataset}_target_domain{domain}_{model_name}_v8.csv", index_col=False)[:num_samples]
    # df = df.sort_values(by='idx')
    # df_direct = df_direct.sort_values(by='idx')
elif dataset == "INaturalist":
    domain = 1
    num_samples = 40000

    random_seed = 0

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    # Load the validation dataset 
    dataset_ = INaturalist(
        root='/work/sw361/',
        version='2021_valid',
        target_type='phylum',
        transform=transform,
        download=True,
    )
    # Find out how many phylum categories there are by inspecting one sample
    # We'll scan through increasing IDs until category_name errors out or repeats
    phylum_names_list = []
    phylum_ids_list = []
    i = 0
    while True:
        try:
            name = dataset_.category_name('phylum', i)
            phylum_names_list.append(name)
            phylum_ids_list.append(i)
            i += 1
        except (IndexError, ValueError):
            break

    print(f"Total phylum categories: {len(phylum_names_list)}")
    print("phylum categories:", phylum_names_list, phylum_ids_list)

    # randomly split the phylum_ids_list into two
    print("seed", random_seed)
    random.Random(random_seed).shuffle(phylum_ids_list)
    shared_class_num = 5
    source_private_class_num = 4
    target_private_class_num = len(phylum_ids_list) - shared_class_num - source_private_class_num
    shared_class_ids_list = phylum_ids_list[:shared_class_num]
    source_private_class_ids_list = phylum_ids_list[shared_class_num:shared_class_num+source_private_class_num]
    target_private_class_ids_list = phylum_ids_list[shared_class_num+source_private_class_num:]
    source_classes = shared_class_ids_list + source_private_class_ids_list
    target_classes = shared_class_ids_list + target_private_class_ids_list
    shared_class_list = [phylum_names_list[i] for i in shared_class_ids_list]
    src_priv_class_list = [phylum_names_list[i] for i in source_private_class_ids_list]
    tgt_priv_class_list = [phylum_names_list[i] for i in target_private_class_ids_list]
    src_class_list = [phylum_names_list[i] for i in source_classes]
    # print("known class", src_class_list)
    src_class_list_new = src_class_list
    # src_class_list_new = []
    # for cls in src_class_list:
    #     if cls == 'Tracheophyta':
    #         src_class_list_new.append("Vascular plant")
    #     # elif cls == 'Arthropoda':
    #     #     src_class_list_new.append("Invertebrate animals, such as insects, arachnids, and crustaceans")
    #     # elif cls == 'Rhodophyta':
    #     #     src_class_list_new.append("Red algae")
    #     else:
    #         src_class_list_new.append(cls)
    print("known class", src_class_list_new)
    
    tgt_class_list = [phylum_names_list[i] for i in target_classes]
    
    class_dict = {cls: idx for idx, cls in enumerate(shared_class_list + src_priv_class_list + tgt_priv_class_list)} 

    df = pd.read_csv(f'llm_data/{dataset}_target_domain{domain}_summary_v10_randomseed{random_seed}_300tokens.csv', index_col=False)[:num_samples]
    df_direct = pd.read_csv(f"llm_data/{dataset}_target_domain{domain}_{model_name}_v8_randomseed{random_seed}.csv", index_col=False)[:num_samples]

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

# Add context to the labels
src_class_list_w_context = ["An image of " + cls for cls in src_class_list_new]

def plot_tsne(embeddings1, embeddings2, labels=('Set 1', 'Set 2'), title='t-SNE Visualization'):
    # Combine embeddings for joint t-SNE computation
    embeddings = np.concatenate([embeddings1.cpu(), embeddings2.cpu()], axis=0)

    # Compute t-SNE projection
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='random')
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Split back into two sets
    tsne_1 = embeddings_tsne[:len(embeddings1)]
    tsne_2 = embeddings_tsne[len(embeddings1):]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_1[:, 0], tsne_1[:, 1], c='blue', alpha=0.6, label=labels[0])
    plt.scatter(tsne_2[:, 0], tsne_2[:, 1], c='red', alpha=0.6, label=labels[1])

    plt.legend(fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig("tsne.png")

def process_preds(row):
    # return re.sub('\'.', '', row['predicted class name'])
    return row['predicted class name'].replace('\'', '')
df_direct['predicted class name'] = df_direct.apply(process_preds, axis=1)
llm_preds = list(df_direct['predicted class name'].values)
# print("llm preds", llm_preds)
llm_preds_w_context = ["An image of " + cls for cls in llm_preds]

# def process_summary(row):
#     if "main object of the " in row['summary']:
#         return row['summary'].split("main object of the ")[1]
#     else:
#         return row['summary']
# df['summary'] = df.apply(process_summary, axis=1)
pred_summary = list(df['summary'].values)

if not load_scores:
    if second_opinion_model == "CLIP":
        # Load the tokenizer and model for the CLIP text encoder
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Tokenize the sentence and convert it to tensor format
        summary_tokens = tokenizer(pred_summary, return_tensors="pt", truncation=True, padding=True)
        src_class_list_w_context_tokens = tokenizer(src_class_list_w_context, return_tensors="pt", truncation=True, padding=True)
        llm_preds_w_context_tokens = tokenizer(llm_preds_w_context, return_tensors="pt", truncation=True, padding=True)

        # Compute the embeddings using the text encoder
        with torch.no_grad():
            # Get the text embeddings using the CLIP model's text encoder
            sentence_embedding = model.get_text_features(**summary_tokens)
            class_list_w_context_embeddings = model.get_text_features(**src_class_list_w_context_tokens)
            llm_preds_w_context_embedding = model.get_text_features(**llm_preds_w_context_tokens)
    elif second_opinion_model == "qwen":
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_name)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    elif second_opinion_model == "llama":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = MllamaTextModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",              # will shard across GPUs if you have more than one
            token=os.environ["HF_TOKEN"]
        )
        # 1) Tokenize your list of sentences
        processor = AutoProcessor.from_pretrained(model_id)

        sentence_embedding = []
        for i in range(len(pred_summary)):
            enc_summary = processor(
                text=pred_summary[i],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            enc_summary = {k: v.to(model.device) for k, v in enc_summary.items()}

            # 2) Forward‐pass to get hidden states
            with torch.no_grad():
                outputs = model(**enc_summary)
                last_hidden_summary = outputs.last_hidden_state

            # 3) Mean‐pool over non‐padded tokens
            mask = enc_summary["attention_mask"].unsqueeze(-1)     # [batch, seq_len, 1]
            sum_hidden = (last_hidden_summary * mask).sum(dim=1)   # [batch, dim]
            lengths = mask.sum(dim=1)                    # [batch, 1]
            sentence_embedding.append(sum_hidden / lengths)             # [batch, dim]
        sentence_embedding = torch.cat(sentence_embedding, dim=0).to(torch.float32)
        print(sentence_embedding.shape)
        print("sentence_embedding done", flush=True)

        torch.cuda.empty_cache()
        enc_src_class_list_w_context = processor(
            text=src_class_list_w_context,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ) 
        enc_src_class_list_w_context = {k: v.to(model.device) for k, v in enc_src_class_list_w_context.items()}
        # 2) Forward‐pass to get hidden states
        with torch.no_grad():
            outputs = model(**enc_src_class_list_w_context)
            last_hidden_src_class_list_w_context = outputs.last_hidden_state

        # 3) Mean‐pool over non‐padded tokens
        mask = enc_src_class_list_w_context["attention_mask"].unsqueeze(-1)     # [batch, seq_len, 1]
        sum_hidden = (last_hidden_src_class_list_w_context * mask).sum(dim=1)   # [batch, dim]
        lengths   = mask.sum(dim=1)                    # [batch, 1]
        class_list_w_context_embeddings = sum_hidden / lengths              # [batch, dim]
        class_list_w_context_embeddings = class_list_w_context_embeddings.to(torch.float32)
        print("class_list_w_context_embeddings done", flush=True)    

        torch.cuda.empty_cache()
        llm_preds_w_context_embedding = []
        for i in range(len(llm_preds_w_context)):
            enc_llm_preds_w_context = processor(
                text=llm_preds_w_context[i],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            enc_llm_preds_w_context = {k: v.to(model.device) for k, v in enc_llm_preds_w_context.items()}
        
            # 2) Forward‐pass to get hidden states
            with torch.no_grad():
                outputs = model(**enc_llm_preds_w_context)
                last_hidden_llm_preds_w_context = outputs.last_hidden_state

            # 3) Mean‐pool over non‐padded tokens
            mask = enc_llm_preds_w_context["attention_mask"].unsqueeze(-1)     # [batch, seq_len, 1]
            sum_hidden = (last_hidden_llm_preds_w_context * mask).sum(dim=1)   # [batch, dim]
            lengths   = mask.sum(dim=1)                    # [batch, 1]
            llm_preds_w_context_embedding.append(sum_hidden / lengths)              # [batch, dim]
        llm_preds_w_context_embedding = torch.cat(llm_preds_w_context_embedding, dim=0).to(torch.float32)
        print("llm_preds_w_context_embedding done", flush=True)    

    # Normalize the embeddings for cosine similarity
    sentence_embedding = sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)
    class_list_w_context_embeddings = class_list_w_context_embeddings / class_list_w_context_embeddings.norm(dim=-1, keepdim=True)
    llm_preds_w_context_embedding = llm_preds_w_context_embedding / llm_preds_w_context_embedding.norm(dim=-1, keepdim=True)

    plot_tsne(sentence_embedding, class_list_w_context_embeddings)
    # Compute cosine similarity: dot product of the normalized vectors
    similarity_scores_w_all_lbls = (sentence_embedding @ class_list_w_context_embeddings.T).squeeze(0).cpu().numpy()

    cos = torch.nn.CosineSimilarity(dim=1) 
    similarity_scores_w_llm_preds = cos(sentence_embedding, llm_preds_w_context_embedding).cpu().numpy()

    cos_sim_df = pd.DataFrame(similarity_scores_w_all_lbls)
    cos_sim_w_llm_df = pd.DataFrame(list(zip(llm_preds, similarity_scores_w_llm_preds)), columns=['llm preds', 'scores'])

    cos_sim_df.to_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{model_name}_{second_opinion_model}_{version}_randomseed{random_seed}.csv", header=False, index=False)
    cos_sim_w_llm_df.to_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{model_name}_{second_opinion_model}_{version}_randomseed{random_seed}.csv", header=True, index=False)
    similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{model_name}_{second_opinion_model}_{version}_randomseed{random_seed}.csv", header=None, index_col=False).to_numpy()
    similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{model_name}_{second_opinion_model}_{version}_randomseed{random_seed}.csv", index_col=False)
else:
    similarity_scores_w_all_lbls = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_{model_name}_{second_opinion_model}_{version}_randomseed{random_seed}.csv", header=None, index_col=False).to_numpy()
    similarity_scores_w_llm_preds = pd.read_csv(f"{dataset}_Domain_{domain} similarity_score_add_context_to_labels_w_llm_preds_{model_name}_{second_opinion_model}_{version}_randomseed{random_seed}.csv", index_col=False)

preds_all = np.argmax(similarity_scores_w_all_lbls, axis=1)
# print("argmax preds", [src_class_list[i] for i in preds_all])
gt = [class_dict[lbl] for lbl in gt_labels]

# ratio of highest cos score / sum of all cos scores
# for known
df.reset_index(drop=True, inplace=True)
similarity_scores_w_llm_preds.reset_index(drop=True, inplace=True)
df_concat = pd.concat([df, similarity_scores_w_llm_preds], axis=1)

known_idx = df_concat.index[df_concat['private'] == False].tolist()

known_outside = df_concat[(df_concat['private'] == False) & (~df_concat['llm preds'].isin(src_class_list_new))]
print("known outside", len(known_outside))
known_in_idx = df_concat.index[(df_concat['private'] == False) & (df_concat['llm preds'].isin(src_class_list_new))]
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
unknown_outside = df_concat[(df_concat['private'] == True) & (~df_concat['llm preds'].isin(src_class_list_new))]
print("unknown outside", len(unknown_outside))
unknown_in_idx = df_concat.index[(df_concat['private'] == True) & (df_concat['llm preds'].isin(src_class_list_new))]
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
        print(label)
        label_idx = df_concat.index[df_concat['ground truth'] == label]
        known_in_idx = df_concat.index[(df_concat['ground truth'] == label) & (df_concat['llm preds'].isin(src_class_list_new))]
        preds_known_cls = preds_all[known_in_idx]
        # corr = [idx for idx, i in zip(known_in_idx, preds_known_cls) if list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx]]
        # print(len(corr))
        ratio = [round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) for idx, i in zip(known_in_idx, preds_known_cls) if list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx]]
        print(ratio)
        most_common = Counter(preds_known_cls).most_common()
        print("most common", most_common, [src_class_list_new[i] for i in list(list(zip(*most_common))[0])])
        # if label == "Tracheophyta":
        #     # corr = [idx for idx, i in zip(known_in_idx, preds_known_cls) if list(df_concat['llm preds'].values)[idx] == "Tracheophyta (Vascular plant)"]
        #     corr = [idx for idx, i in zip(known_in_idx, preds_known_cls) if list(df_concat['llm preds'].values)[idx] == "Vascular plant"]
        #     print(len(corr))
        #     correct_idx = [idx for idx, i in zip(known_in_idx, preds_known_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1 and list(df_concat['llm preds'].values)[idx] == "Vascular plant"]
        # elif label == "Arthropoda":
        #     corr = [idx for idx, i in zip(known_in_idx, preds_known_cls) if list(df_concat['llm preds'].values)[idx] == "Invertebrate animals, such as insects, arachnids, and crustaceans"]
        #     print(len(corr))
        #     correct_idx = [idx for idx, i in zip(known_in_idx, preds_known_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1 and list(df_concat['llm preds'].values)[idx] == "Invertebrate animals, such as insects, arachnids, and crustaceans"]
        # elif label == "Rhodophyta":
        #     corr = [idx for idx, i in zip(known_in_idx, preds_known_cls) if list(df_concat['llm preds'].values)[idx] == "Red algae"]
        #     print(len(corr))
        #     correct_idx = [idx for idx, i in zip(known_in_idx, preds_known_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1 and list(df_concat['llm preds'].values)[idx] == "Red algae"]
        # else:
        correct_idx = [idx for idx, i in zip(known_in_idx, preds_known_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1 and list(df_concat['llm preds'].values)[idx] == list(df_concat['ground truth'].values)[idx]]
        num_corr = len(correct_idx)
        kwn_per_class_num.append(len(label_idx))
        kwn_per_class_correct.append(len(correct_idx))
        kwn_label_list.append(label)
    else:
        # print(label)
        label_idx = df_concat.index[df_concat['ground truth'] == label]
        unknown_in_idx = df_concat.index[(df_concat['ground truth'] == label) & (df_concat['llm preds'].isin(src_class_list_new))]
        preds_unknown_cls = preds_all[unknown_in_idx]
        # most_common = Counter(preds_unknown_cls).most_common()
        # print("most common", most_common, [src_class_list_new[i] for i in list(list(zip(*most_common))[0])])
        incorrect_idx = [idx for idx, i in zip(unknown_in_idx, preds_unknown_cls) if round(list(similarity_scores_w_llm_preds['scores'].values)[idx] / similarity_scores_w_all_lbls[idx, i], 5) == 1]
        num_corr = len(label_idx) - len(incorrect_idx)
        ukn_per_class_num.append(len(label_idx))
        ukn_per_class_correct.append(len(label_idx) - len(incorrect_idx))
        ukn_label_list.append(label)
    cls_lst_cls_num.append(len(label_idx))
    per_cls_acc.append(num_corr / len(label_idx))
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

fig = plt.figure(figsize=(20, 20))
ax = plt.subplot(111)
# cls_lst_cls_num = [12388+15132, 27+43, 23+17, 753+937, 37+33, 94+116, 22+18, 18947+23233, 369+471] 
# cls_lst_cls_num = [150, 17483, 13, 30, 682, 96, 11420, 14, 10112]
cls_lst_cls_num = [17713, 28, 11386, 366, 23, 10210, 151, 15, 108]
ax.bar(np.arange(len(cls_lst_cls_num)), cls_lst_cls_num, label=cls_lst)
plt.yticks(rotation=0, fontsize=10)
plt.title(f'{dataset}', fontsize=30)
plt.xticks(rotation=45, fontsize=20)
ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
ax.xaxis.tick_bottom()
ax.set_ylabel('Sample Size', fontsize=30)
plt.savefig(f"{dataset}_Sample_Size_{model_name}_randomseed{random_seed}.png")

fig = plt.figure(figsize=(20, 14))
ax = plt.subplot(111)
ax.bar(np.arange(len(per_cls_acc)), per_cls_acc, label=cls_lst)
plt.yticks(rotation=0, fontsize=20)
plt.title(f'{dataset} {model_name}', fontsize=30)
ax.set_xticks(ticks=np.arange(len(cls_lst_cls_num)), labels=cls_lst)
plt.xticks(rotation=45, fontsize=30)
ax.xaxis.tick_bottom()
ax.set_ylabel('Accuracy', fontsize=30)
plt.savefig(f"{dataset}_Accuracy_{model_name}_barplot_randomseed{random_seed}.png")

fig = plt.figure(figsize=(20, 14))
ax = plt.subplot(111)
# qwen_kwn_acc = [0.8602960607584615, 0.0232558085451608, 0.0, 0.7225186689165564, 0.09090906336088989]
# qwen_ukn_acc = [0.7586206242568427, 0.5555552469137517, 0.9816640119736306, 0.6008492441433282]
# qwen_kwn_acc = [0.03333333111111126, 0.01418520848013201, 0.0, 0.0, 0.7961876716101515]
# qwen_ukn_acc = [0.8124999153645921, 0.9821366015918244, 0.9285707653065962, 0.9839794294066659]
# qwen_kwn_acc =[0.0038954440219638436, 0.0, 0.8376954146867246, 0.005464480725014188, 0.0]
# qwen_ukn_acc =[0.9087169432823536, 0.9933774176571246, 0.7333328444447704, 0.8333332561728466]
# qwen_kwn_acc = [0.9192118777625405, 0.0, 0.6162831542101852, 0.005464480725014188, 0.0]
# qwen_ukn_acc = [0.8543584712494041, 0.0728476772948558, 0.5999996000002666, 0.7870369641632441]
qwen_kwn_acc = kwn_per_class_acc
qwen_ukn_acc = ukn_per_class_acc
print(len(qwen_kwn_acc), len(qwen_ukn_acc))
ax.boxplot([qwen_kwn_acc, qwen_ukn_acc], labels=['qwen known', 'qwen unknown'])
plt.yticks(rotation=45, fontsize=20)
plt.title(f'{dataset} Known vs Unknown', fontsize=30)
ax.xaxis.set_label_position('bottom')
# ax.set_xticks(ticks=np.arange(2), labels=['known', 'unknown'])
plt.xticks(rotation=45, fontsize=10)
ax.xaxis.tick_bottom()
ax.set_ylabel('Accuracy', fontsize=30)
plt.savefig(f"{dataset}_Accuracy_boxplot_randomseed{random_seed}.png")

# kwn_sample_size = [12388+15132, 27+43, 23+17, 753+937, 37+33]
# ukn_sample_size = [94+116, 22+18, 18947+23233, 369+471]
# kwn_sample_size = [150, 17483, 13, 30, 682] 
# ukn_sample_size = [96, 11420, 14, 10112]
kwn_sample_size = [17713, 28, 11386, 366, 23]
ukn_sample_size = [10210, 151, 15, 108]
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

##################################################################

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