import os
import base64
from dataset.dataset import SFUniDADataset_BLIP, SFUniDADataset
from torch.utils.data.dataloader import DataLoader
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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
args = build_args()

dataset = "VisDA"
if dataset == "DomainNet":
    args.dataset = "DomainNet"
elif dataset == "VisDA":
    args.dataset = "VisDA"

set_random_seed(2025)

columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

data = []

model_name = "Qwen"

if model_name == "BLIP2":  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configuration = Blip2Config()
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        device_map={"": 0},   # or set manually
        torch_dtype=torch.float16,  # or torch.float16
    )
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    def collate_fn(batch):
        images = [Image.open(p[0]).convert("RGB") for p in batch]
        labels = [p[0] for p in batch]
        print(labels)
        return processor(images=images, text=[prompt] * len(batch), return_tensors="pt", padding=True)

    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset_BLIP(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=collate_fn)
    print(len(target_train_dataloader))

    # for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):  
    for idx, batch in enumerate(target_train_dataloader):    
        if idx >= 100:
            break
        batch = {k: v.to("cuda") for k, v in batch.items()}
        generated_ids = model.generate(**batch)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        answer = generated_text.split("Answer:")[1]
        print(answer)

elif model_name == "BLIP3":

    prompt = "Question: You are an AI that classifies images based on a predefined list of categories. \
    If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
    If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`. \
    Does this image belong to one of the categories in the following list \
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
    Response: unknown: True, class_name: 'clarinet'.\
    Answer:\
    "
    # define the prompt template
    def apply_prompt_template(prompt):
        s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
            )
        return s 
    class EosListStoppingCriteria(StoppingCriteria):
        def __init__(self, eos_sequence = [32007]):
            self.eos_sequence = eos_sequence

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
            return self.eos_sequence in last_ids    
    # load models
    # model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    # model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True,
    #                                                 device_map="auto",            # <--- important!
    #                                                 torch_dtype="auto",            # <--- optional but good
    #                                                 low_cpu_mem_usage=True         # <--- critical for big models!
    # )
    # model = model.to('cuda')
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
    # image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    # tokenizer = model.update_special_tokens(tokenizer)
    # tokenizer.padding_side = "left"

    model_ckpt="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    cfg = dict(
        model_family = 'xgenmm_v1',
        lm_path = 'microsoft/Phi-3-mini-4k-instruct',
        vision_encoder_path = 'google/siglip-so400m-patch14-384',
        vision_encoder_pretrained = 'google',
        num_vision_tokens = 128,
        image_aspect_ratio = 'anyres',
        anyres_patch_sampling = True,
        anyres_grids = [(1,2),(2,1),(2,2),(3,1),(1,3)],
        ckpt_pth = model_ckpt,
    )
    cfg = OmegaConf.create(cfg)

    additional_kwargs = {
        "num_vision_tokens": cfg.num_vision_tokens,
        "image_aspect_ratio": cfg.image_aspect_ratio,
        "anyres_patch_sampling": cfg.anyres_patch_sampling,
    }

    # Initialize the model.
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=cfg.vision_encoder_path,
        clip_vision_encoder_pretrained=cfg.vision_encoder_pretrained,
        lang_model_path=cfg.lm_path,
        tokenizer_path=cfg.lm_path,
        model_family=cfg.model_family,
        **additional_kwargs)
    ckpt = torch.load(cfg.ckpt_pth)["model_state_dict"]
    model.load_state_dict(ckpt, strict=True)
    torch.cuda.empty_cache()
    model = model.eval().cuda()

    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=1)
    print(len(target_train_dataloader))

    for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):  
        if idx >= 2:
            break
        img = Image.open(imgs_train).convert("RGB") 
        inputs = image_processor([img], return_tensors="pt", image_aspect_ratio='anyres')
        prompt = apply_prompt_template(user_promptv10)
        language_inputs = tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generated_text = model.generate(**inputs, image_size=[img.size],
                                        pad_token_id=tokenizer.pad_token_id,
                                        do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,
                                        stopping_criteria = [EosListStoppingCriteria()],
                                        )
        prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        print("Assistant: ", textwrap.fill(prediction, width=100))

elif model_name == "LLaMa":
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

elif model_name == "Qwen":
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    version = 'v8'

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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
    
     # Preparation for inference

    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset_BLIP(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))
    
    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 
    columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]

    for idx, (imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        if idx < 0.1 * len(target_train_dataloader): 
            continue
        # if idx >= 5: 
        #     break

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
                    {"type": "image",
                    "image": list(imgs_train)[0],
                    }, 
                    {"type": "text", "text": user_promptv8}
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
        generated_ids = model.generate(**inputs)
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
        df.to_csv(os.path.join("llm_data/{}_target_domain{}_qwen_{}_2.csv".format(dataset, args.t_idx, version)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")