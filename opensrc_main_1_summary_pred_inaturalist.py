import os
import base64
from dataset.dataset import SFUniDADataset_BLIP, SFUniDADataset, INaturalist_UniDA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import INaturalist
from torchvision import transforms
from config.model_config import build_args
from net_utils import set_random_seed
import numpy as np
from pathlib import Path
import pandas as pd
import json
from PIL import Image
import tqdm
import time
from datetime import datetime
from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import random
# from open_flamingo import create_model_and_transforms 
# from open_flamingo.train.any_res_data_utils import process_images

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
args = build_args()

dataset = "INaturalist"
args.dataset = "INaturalist"

set_random_seed(2025)

columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

data = []

model_name = "qwen"
print("model_name", model_name, flush=True)

if model_name == "llama":
    device = "cuda"
    version = 'v13'
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",              # will shard across GPUs if you have more than one
        token=os.environ["HF_TOKEN"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    target_type = 'class'
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
    random_seed = 0
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
    unknown_class_list = [label_names_list[i] for i in target_classes]
    print("known class list", known_class_list)

    target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data/', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))

    df_summary = pd.read_csv(f'llm_data/{target_type}_{dataset}_target_domain1_{model_name}_summary_{version}_randomseed{random_seed}.csv', index_col=False)

    system_promptv8 = "You are an AI that classifies images based on a summary of the image. \
    If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
    If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."
    user_promptv8 = "Does this image belong to one of the categories in the following list \
    {} based on the following summary: {}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1: \
    Image: (picture of a aeroplane) \
    Response: unknown: False, class_name: 'aeroplane' \
    Example 2: \
    Image: (picture of a donkey) \
    Response: unknown: True, class_name: 'horse' \
    "
    
    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 
    columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

    for idx, (imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        # if idx < 0.4 * len(target_train_dataloader): 
        #     continue
        # if idx > 5: 
        #     break
        summary = df_summary[df_summary['idx'] == imgs_idx[0].cpu().numpy()]['summary'].values[0]
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_promptv8}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_promptv8.format(known_class_list, summary)}
                ]
            }
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=input_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        output = model.generate(**inputs, max_new_tokens=300)
        answer = processor.decode(output[0], skip_special_tokens=True).split('assistant')[1]
        if "unknown: " in answer and ", class_name: " in answer:
            data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], answer.split(', class_name: ')[1], private.cpu().numpy()[0], answer.split(', class_name: ')[0].split('unknown: ')[1], list(imgs_train)[0]])
        else:
            unknown = True
            class_name = "not follow instruction"
            data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], class_name, private.cpu().numpy()[0], unknown, imgs_train[0]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_llama_{}_summary_pred_randomseed{}.csv".format(target_type, dataset, args.t_idx, version, random_seed)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

elif model_name == "qwen":
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    version = 'v13'

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    target_type = 'phylum'
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
    # known_class_list_new = []
    # for cls in known_class_list:
    #     if cls == 'Tracheophyta':
    #         known_class_list_new.append("Vascular plant")
    #     elif cls == 'Rhodophyta':
    #         known_class_list_new.append("Red algae")
    #     elif cls == 'Arthropoda':
    #         known_class_list_new.append("Invertebrate animals, such as insects, arachnids, and crustaceans")
    #     else:
    #         known_class_list_new.append(cls)
    # print(known_class_list_new)
    unknown_class_list = [label_names_list[i] for i in target_classes]

    target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data/', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))

    df_summary = pd.read_csv(f'llm_data/{target_type}_{dataset}_target_domain1_{model_name}_summary_{version}_randomseed{random_seed}.csv', index_col=False) 

    system_promptv8 = "You are an AI that classifies images based on a summary of the image. \
    If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
    If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."

    user_promptv8 = "Does this image belong to one of the categories in the following list \
    {} based on the following summary: {}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1: \
    Image: (picture of a aeroplane) \
    Response: unknown: False, class_name: 'aeroplane' \
    Example 2: \
    Image: (picture of a donkey) \
    Response: unknown: True, class_name: 'horse' \
    "
    print("known_class_list", known_class_list, flush=True)

    start_time = time.time()
    start_datetime = datetime.now()
    data = []
    columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

    for idx, (imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        # print(imgs_train, imgs_idx[0].cpu().numpy())
        summary = df_summary[df_summary['idx'] == imgs_idx[0].cpu().numpy()]['summary'].values[0]
        # print(summary)
        # if idx <= 15000: 
        #     continue
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_promptv8}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_promptv8.format(known_class_list, summary)}
                ]
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=300)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if "unknown: " in answer and ", class_name: " in answer:
            data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], answer.split(', class_name: ')[1], private.cpu().numpy()[0], answer.split(', class_name: ')[0].split('unknown: ')[1], list(imgs_train)[0]])
        else:
            unknown = True
            class_name = "not follow instruction"
            data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], class_name, private.cpu().numpy()[0], unknown, imgs_train[0]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_qwen_{}_randomseed{}_summary_pred.csv".format(target_type, dataset, args.t_idx, version, random_seed)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")