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

dataset = "INaturalist"
if dataset == "DomainNet":
    args.dataset = "DomainNet"
elif dataset == "VisDA":
    args.dataset = "VisDA"
elif dataset == "INaturalist":
    args.dataset = "INaturalist"

columns = ["idx", "ground truth", "private", "summary", "img url"]

model_name = "LLaMa"
print(dataset, model_name)

user_promptv10 = "Question: Can you give a summary of the image? Answer:"
user_promptv13 = "Identify the primary object in the image, excluding any background elements."

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
    version = 'v13'
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",              # will shard across GPUs if you have more than one
        token=os.environ["HF_TOKEN"]
    )
    processor = AutoProcessor.from_pretrained(model_id)

    user_promptv10 = "Can you give a summary of the image?"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": user_promptv13}
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
    
    target_type = 'phylum'
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])
    # Load the validation dataset 
    dataset_ = INaturalist(
        root='/hpc/group/carin/sw361/data',
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
    unknown_class_list = [label_names_list[i] for i in target_classes]
    
    target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
    print(len(target_train_dataloader))
    print("known_class_list", known_class_list, flush=True)

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 

    for idx, (batch, imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        batch = {k: v.to("cuda") for k, v in batch.items()}
        generated_ids = model.generate(**batch, max_new_tokens=300)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).split('assistant')[1].replace("\n", "").replace("\r", "")
        data.append([imgs_idx[0], list(ground_truth)[0], private[0], answer, list(imgs_train)[0]])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_llama_summary_{}_randomseed{}.csv".format(target_type, dataset, args.t_idx, version, random_seed)), index=False)
    
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
    version = 'v13'

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    user_promptv10 = "Can you give a summary of the image?"

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to torch.Tensor!
    ])

    target_type = 'phylum'
    # Load the validation dataset 
    dataset_ = INaturalist(
        root='/hpc/group/carin/sw361/data',
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
    unknown_class_list = [label_names_list[i] for i in target_classes]
    
    target_dataset = INaturalist_UniDA(root='/hpc/group/carin/sw361/data', version='2021_valid', target_type=target_type, transform=transform, download=True, shared_classes=shared_class_ids_list, source_private_classes=source_private_class_ids_list, target_private_classes=target_private_class_ids_list, label_names_list=label_names_list)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    print(len(target_train_dataloader))
    print("known_class_list", known_class_list, flush=True)

    start_time = time.time()
    start_datetime = datetime.now()
    data = [] 

    for idx, (imgs_train, imgs_label, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader): 
        # if idx < 31000: 
        #     continue
        # if idx >= 5: 
        #     break

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                    "image": list(imgs_train)[0],
                    }, 
                    {"type": "text", "text": user_promptv13}
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

        data.append([imgs_idx.cpu().numpy()[0], list(ground_truth)[0], private.cpu().numpy()[0], answer, list(imgs_train)[0]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join("llm_data/{}_{}_target_domain{}_qwen_summary_{}_randomseed{}.csv".format(target_type, dataset, args.t_idx, version, random_seed)), index=False)

    end_time = time.time()
    end_datetime = datetime.now()

    elapsed_time = end_time - start_time

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")