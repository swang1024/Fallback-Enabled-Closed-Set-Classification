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

model_name = "LLaMa"

user_promptv10 = "Question: Can you give a summary of the image? Answer:"

if model_name == "LLaMa":
    device = "cuda"
    version = 'v8'
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",              # will shard across GPUs if you have more than one
        token=os.environ["HF_TOKEN"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    system_promptv8 = "You are an AI that classifies images based on a predefined list of categories. \
    If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
    If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."
    if dataset == 'DomainNet':
        user_promptv8 = "Does this image belong to one of the categories in the following list \
        ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', \
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
        'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail']? \
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1:\
        Image: (picture of a cat)\
        Response: unknown: False, class_name: 'cat'\
        Example 2:\
        Image: (picture of a trumpet mouthpiece)\
        Response: unknown: True, class_name: 'clarinet'\
        "
    else:
        user_promptv8 = "Does this image belong to one of the categories in the following list \
        ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']? \
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane'\
        Example 2:\
        Image: (picture of a donkey)\
        Response: unknown: True, class_name: 'horse'\
        "
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
                {"type": "image"}, 
                {"type": "text", "text": user_promptv8}
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

    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset_BLIP(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
    print(len(target_train_dataloader))

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 
    columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

    for idx, (batch, imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        # if idx < 0.4 * len(target_train_dataloader): 
        #     continue
        if idx < 34000: 
            continue
        batch = {k: v.to("cuda") for k, v in batch.items()}
        generated_ids = model.generate(**batch)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).split('assistant')[1]
        if "unknown: " in answer and ", class_name: " in answer:
            data.append([imgs_idx[0], list(ground_truth)[0], answer.split(', class_name: ')[1], private[0], answer.split(', class_name: ')[0].split('unknown: ')[1], list(imgs_train)[0]])
        else:
            unknown = True
            class_name = "not follow instruction"
            data.append([imgs_idx[0], list(ground_truth)[0], class_name, private[0], unknown, imgs_train[0]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/{}_target_domain{}_llama_{}_2.csv".format(dataset, args.t_idx, version)), index=False)

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
    version = 'v8'

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
    random_seed = 0
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
    known_class_list = [phylum_names_list[i] for i in source_classes]
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
    unknown_class_list = [phylum_names_list[i] for i in target_classes]

    target_dataset = INaturalist_UniDA(root='/work/sw361/', version='2021_valid', target_type='phylum', transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, phylum_names_list=phylum_names_list)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))

    df_summary = pd.read_csv(f'llm_data/{dataset}_target_domain2_{model_name}_summary_v10_randomseed{random_seed}_300tokens.csv', index_col=False) 

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
        if idx <= 31000: 
            continue
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
        df.to_csv(os.path.join("llm_data/{}_target_domain{}_qwen_{}_randomseed{}_summary_pred_part2.csv".format(dataset, args.t_idx, version, random_seed)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")