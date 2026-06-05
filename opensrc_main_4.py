import os
import base64
from dataset.dataset import SFUniDADataset_BLIP, SFUniDADataset, INaturalist_UniDA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import INaturalist
from torchvision import transforms
from config.model_config import build_args
from net_utils import set_random_seed
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

args = build_args()
set_random_seed(2025)

dataset = "VisDA"
random_seed = 0
if dataset == "DomainNet":
    args.dataset = "DomainNet"
    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset_BLIP(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))
elif dataset == "VisDA":
    args.dataset = "VisDA"
    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset_BLIP(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))
elif dataset == "INaturalist":
    args.dataset = "INaturalist"

columns = ["idx", "ground truth", "predicted class name", "private", "description", "img url"]

model_name = "qwen"

if model_name == "LLaMa":
    device = "cuda"
    version = 'v10'
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",              # will shard across GPUs if you have more than one
        token=os.environ["HF_TOKEN"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    user_promptv16 = "What are useful visual features for predicting the image to be {}?​  Describe the main visual features using a bulleted list, start each line with a dash. Each bullet should mention only one specific, observable visual detail."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": user_promptv16.format(pred_lbl)}
            ]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    def collate_fn(batch):
        images = [Image.open(p[0]).convert("RGB") for p in batch]
        imgs_label = [p[1] for p in batch]
        imgs_train = [p[0] for p in batch]
        imgs_idx = [p[2] for p in batch]
        ground_truth = [p[3] for p in batch]
        private = [p[4] for p in batch]
        return processor(images=images, text=[input_text for p in batch], return_tensors="pt", add_special_tokens=False), imgs_train, imgs_label, imgs_idx, ground_truth, private

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 

    for idx, (batch, imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        batch = {k: v.to("cuda") for k, v in batch.items()}
        generated_ids = model.generate(**batch, max_new_tokens=1000)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).split('assistant')[1].replace("\n", "").replace("\r", "")
        data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], private.cpu().numpy()[0], answer, list(imgs_train)[0]])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/target_domain{}_llama_pred_feats_{}_randomseed{}.csv".format(dataset, args.t_idx, version, random_seed)), index=False)
    
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
    version = 'v16'

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    user_promptv16 = "What are useful visual features for predicting the image to be {}?​  Describe the main visual features using a bulleted list, start each line with a dash. Each bullet should mention only one specific, observable visual detail."

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 

    df_pred = pd.read_csv(f'llm_data/{dataset}_target_domain1_{model_name}_v8.csv', index_col=False) 
    def process_preds(row):
        return row['predicted class name'].replace('\'', '')

    df_pred['predicted class name'] = df_pred.apply(process_preds, axis=1)

    for idx, (imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        # if idx < 31000: 
        #     continue
        # if idx >= 5: 
        #     break
        pred_lbl = df_pred[df_pred['idx'] == imgs_idx.cpu().numpy()[0]]['predicted class name'].values[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                    "image": list(imgs_train)[0],
                    }, 
                    {"type": "text", "text": user_promptv16.format(pred_lbl)}
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
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], pred_lbl, private.cpu().numpy()[0], answer, list(imgs_train)[0]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/pred_feats_{}_target_domain{}_qwen_{}_randomseed{}.csv".format(dataset, args.t_idx, version, random_seed)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")