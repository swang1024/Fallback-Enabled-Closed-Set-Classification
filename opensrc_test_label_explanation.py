import os
import base64
from dataset.dataset import SFUniDADataset_BLIP, SFUniDADataset, INaturalist_UniDA
from torchvision.datasets import INaturalist
from torchvision import transforms
import random
import time
from datetime import datetime
import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
from config.model_config import build_args
from net_utils import set_random_seed
from pathlib import Path
import pandas as pd
import json
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

args = build_args()
set_random_seed(2025)

dataset = "INaturalist"
random_seed = 0
if dataset == "DomainNet":
    args.dataset = "DomainNet"
elif dataset == "VisDA":
    args.dataset = "VisDA"
elif dataset == "INaturalist":
    args.dataset = "INaturalist"

if args.dataset == "DomainNet":
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
elif args.dataset == "VisDA":
    class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']
elif args.dataset == "INaturalist": 
    target_type = 'class'
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    # Load the validation dataset 
    dataset_ = INaturalist(
        root='/data/henao/swang1/data',
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
    class_list = known_class_list
    print("known_class_list", class_list, flush=True)

user_promptv15 = "What are useful visual features for distinguishing a {} in a photo?​  Describe the main visual features using a bulleted list, start each line with a dash. Each bullet should mention only one specific, observable visual detail."
user_promptv17 = "What are defining visual features for distinguishing a {} in a photo?​  Describe the main visual features using a bulleted list, start each line with a dash. Each bullet should mention only one specific, observable visual detail."

columns = ["class label", "description"]
 
data = []

model_name = "llama"

if model_name == "llama":
    device = "cuda"
    version = 'v17'
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",              # will shard across GPUs if you have more than one
        token=os.environ["HF_TOKEN"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 

    for idx, class_label in enumerate(class_list):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_promptv17.format(class_label)}
                ]
            }
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=input_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).split('assistant')[1].replace("\n", "").replace("\r", "")
        data.append([class_label, answer])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/label_{}_target_domain{}_llama_label_{}_randomseed{}.csv".format(dataset, args.t_idx, version, random_seed)), index=False)
    
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
        device_map="auto",
    )
    version = 'v17'

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 

    for idx, class_label in enumerate(class_list):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_promptv17.format(class_label)}
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

        data.append([class_label, answer])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/label_{}_target_domain{}_qwen_{}_randomseed{}.csv".format(dataset, args.t_idx, version, random_seed)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

# df = pd.DataFrame(data, columns=columns)
# df.to_csv(os.path.join("target_domain{}.csv".format(args.t_idx)), index=False)

#  confusion matrix
# calculate the percentage of true/true, false/false (ground truth unknown/prediction unknown)
# df[private]
# for false/false, are the predicted labels same as the ground truths? what is the incorrect rate? 
# among the incorrect predictions, are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
# for true/true, do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?

# for true/false, are the predicted labels same as the ground truths? what is the incorrect rate? do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?
# for false/true, do the predicted labels make sense? what kind of logic is followed for the prediction? Are these predicted as a wrong label in the list, or are they predicted as a label outside of the list?