import os
import re
import time
import base64
import random
from collections import Counter

import certifi
import pandas as pd
import torch.multiprocessing as mp
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import INaturalist

from config.model_config import build_args
from dataset.dataset import INaturalist_UniDA, SFUniDADataset
from net_utils import set_random_seed

mp.set_sharing_strategy("file_system")

# Nuke bad values (common on clusters)
for k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(k, None)

# Force Python/httpx/OpenSSL to a known-good CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()


MODEL_ALIAS = {
    "llama3.2-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen-2.5-7B-VL": "Qwen/Qwen2.5-VL-7B-Instruct",
    "gpt 4o-mini": "gpt-4o-mini",
}


def canonicalize_model_name(model_name):
    return MODEL_ALIAS.get(model_name, model_name)


def detect_backend(model_name):
    model_l = model_name.lower()
    if "llama-3.2" in model_l:
        return "llama_local"
    if "qwen2.5-vl" in model_l:
        return "qwen_local"
    if model_l.startswith("gemini"):
        return "gemini"
    if model_l.startswith("gpt-4o"):
        return "openai"
    raise ValueError(f"Unsupported model backend for model={model_name}")


def init_gemini_client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def init_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)


def resolve_hf_token():
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        return token

    try:
        from huggingface_hub import HfFolder
    except ImportError:
        HfFolder = None

    if HfFolder is not None:
        token = HfFolder.get_token()
        if token:
            return token

    raise ValueError(
        "Missing Hugging Face token for gated model access. "
        "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN), or run `huggingface-cli login`, "
        "and ensure your account has access to "
        "https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct"
    )


def init_llama_local(model_name):
    import torch
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    model_id = model_name
    token = resolve_hf_token()
    try:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        processor = AutoProcessor.from_pretrained(model_id, token=token)
    except OSError as exc:
        message = str(exc)
        if "gated repo" in message.lower() or "401" in message:
            raise RuntimeError(
                f"Access denied for gated model `{model_id}`. "
                "Verify your HF token and request access on the model page."
            ) from exc
        raise
    return {"model": model, "processor": processor}


def init_qwen_local(model_name):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return {"model": model, "processor": processor}


def parse_prediction(response_text):
    unknown_match = re.search(r"unknown\s*:\s*(True|False)", response_text, re.IGNORECASE)
    class_match = re.search(r"class_name\s*:\s*['\"]?([^,'\"\n]+)['\"]?", response_text, re.IGNORECASE)

    unknown = ""
    if unknown_match:
        unknown = "True" if unknown_match.group(1).lower() == "true" else "False"

    class_name = ""
    if class_match:
        class_name = class_match.group(1).strip()

    return unknown, class_name


def aggregate_votes(predictions):
    valid_predictions = [pred for pred in predictions if pred[0] or pred[1]]
    if not valid_predictions:
        return "", "", 0

    counts = Counter(valid_predictions)
    max_count = max(counts.values())
    tied = {pred for pred, cnt in counts.items() if cnt == max_count}

    # Stable tiebreak: first max-vote prediction observed
    for pred in valid_predictions:
        if pred in tied:
            return pred[0], pred[1], max_count

    return "", "", 0


def safe_generate_gemini(client, model_name, image_bytes, system_prompt, user_prompt):
    from google.genai import errors as genai_errors
    from google.genai import types

    retries = 5
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=1.0,
                ),
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    user_prompt,
                ],
            )
            return response.text or ""
        except genai_errors.ServerError:
            if i == retries - 1:
                raise
            sleep_time = 2**i
            print(f"Model overloaded, retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def safe_generate_openai_vision(client, model_name, image_b64, system_prompt, user_prompt):
    retries = 5
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                temperature=1.0,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
            if i == retries - 1:
                raise
            sleep_time = 2**i
            print(f"Model overloaded, retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def safe_generate_llama_local(bundle, image_path, system_prompt, user_prompt):
    from PIL import Image

    model = bundle["model"]
    processor = bundle["processor"]
    retries = 3
    for i in range(retries):
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                images=[image],
                text=[input_text],
                return_tensors="pt",
                add_special_tokens=False,
            )
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                max_new_tokens=256,
            )
            prompt_len = inputs["input_ids"].shape[-1]
            trimmed = generated_ids[:, prompt_len:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            return decoded[0] if decoded else ""
        except Exception:
            if i == retries - 1:
                raise
            sleep_time = 2**i
            print(f"Llama generation retry in {sleep_time}s...")
            time.sleep(sleep_time)


def safe_generate_qwen_local(bundle, image_path, system_prompt, user_prompt):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError("qwen_vl_utils is required for Qwen2.5-VL. Install with `pip install qwen-vl-utils`.") from e

    model = bundle["model"]
    processor = bundle["processor"]
    retries = 3
    for i in range(retries):
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                max_new_tokens=300,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text[0] if output_text else ""
        except Exception:
            if i == retries - 1:
                raise
            sleep_time = 2**i
            print(f"Qwen generation retry in {sleep_time}s...")
            time.sleep(sleep_time)


def sanitize_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)


args = build_args()
dataset = "VisDA"
num_consistency_samples = 3
inaturalist_random_seed = 2
inaturalist_download = os.getenv("INAT_DOWNLOAD", "false").strip().lower() == "true"
set_random_seed(2025)

# You can keep aliases ("llama3.2-vision", "qwen-2.5-7B-VL") or full model IDs.
# Override with MODEL_NAMES env var, e.g., MODEL_NAMES="llama3.2-vision"
model_names_env = os.getenv("MODEL_NAMES", "").strip()
if model_names_env:
    model_names = [m.strip() for m in model_names_env.split(",") if m.strip()]
else:
    model_names = [
        # "gemini-2.0-flash",
        # "gpt-4o-mini",
        "llama3.2-vision",
        # "qwen-2.5-7B-VL",
    ]

if dataset == "DomainNet":
    args.dataset = "DomainNet"
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
    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
elif dataset == "VisDA":
    args.dataset = "VisDA"
    args.target_data_dir = "/hpc/group/carin/sw361/data/VisDA/validation/"
    user_promptv8 = "Does this image belong to one of the categories in the following list \
    ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant']? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1:\
    Image: (picture of a aeroplane)\
    Response: unknown: False, class_name: 'aeroplane'\
    Example 2:\
    Image: (picture of a donkey)\
    Response: unknown: True, class_name: 'horse'\
    "
    print(args.target_data_dir)
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
elif dataset == "INaturalist":
    version = "v8_sc"
    target_type = "phylum"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_ = INaturalist(
        root="/hpc/group/carin/sw361/data/",
        version="2021_valid",
        target_type=target_type,
        transform=transform,
        download=inaturalist_download,
    )

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

    print("seed", inaturalist_random_seed)
    random.Random(inaturalist_random_seed).shuffle(label_ids_list)
    if target_type == "phylum":
        shared_class_num = 5
        source_private_class_num = 4
    elif target_type == "class":
        shared_class_num = 20
        source_private_class_num = 15
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    target_private_class_num = len(label_ids_list) - shared_class_num - source_private_class_num
    shared_class_ids_list = label_ids_list[:shared_class_num]
    source_private_class_ids_list = label_ids_list[shared_class_num : shared_class_num + source_private_class_num]
    target_private_class_ids_list = label_ids_list[shared_class_num + source_private_class_num :]
    source_classes = shared_class_ids_list + source_private_class_ids_list
    known_class_list = [label_names_list[i] for i in source_classes]

    user_promptv8 = f"Does this image belong to one of the categories in the following list \
    {known_class_list}? \
    Please format the answer csv format with keys unknown and class_name separated by ',' \
    Example 1: \
    Image: (picture of a aeroplane) \
    Response: unknown: False, class_name: 'aeroplane' \
    Example 2: \
    Image: (picture of a donkey) \
    Response: unknown: True, class_name: 'horse' \
    # "
    print("known_class_list", known_class_list, flush=True)

    target_dataset = INaturalist_UniDA(
        root="/hpc/group/carin/sw361/data/",
        version="2021_valid",
        target_type=target_type,
        transform=transform,
        download=inaturalist_download,
        shared_classes=shared_class_ids_list,
        source_private_classes=source_private_class_ids_list,
        target_private_classes=target_private_class_ids_list,
        label_names_list=label_names_list,
    )
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

print(len(target_train_dataloader))

if dataset != "INaturalist":
    version = "v8_sc"

system_promptv8 = "You are an AI that classifies images based on a predefined list of categories. \
If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."

columns = ["idx", "ground truth", "predicted class name", "private", "unknown", "img url"]
os.makedirs("llm_data", exist_ok=True)

gemini_client = None
openai_client = None
llama_local_bundle = None
qwen_local_bundle = None

for configured_model_name in model_names:
    model_name = canonicalize_model_name(configured_model_name)
    backend = detect_backend(model_name)
    print(f"Running self-consistency for model={configured_model_name} (resolved={model_name}, backend={backend})")

    if backend == "gemini":
        if gemini_client is None:
            gemini_client = init_gemini_client()
    elif backend == "openai":
        if openai_client is None:
            openai_client = init_openai_client()
    elif backend == "llama_local":
        if llama_local_bundle is None:
            llama_local_bundle = init_llama_local(model_name)
    elif backend == "qwen_local":
        if qwen_local_bundle is None:
            qwen_local_bundle = init_qwen_local(model_name)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    data = []
    for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(target_train_dataloader):
        image_path = list(imgs_train)[0]
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        sample_predictions = []
        for _ in range(num_consistency_samples):
            if backend == "gemini":
                response_text = safe_generate_gemini(gemini_client, model_name, image_bytes, system_promptv8, user_promptv8)
            elif backend == "openai":
                response_text = safe_generate_openai_vision(
                    openai_client, model_name, image_b64, system_promptv8, user_promptv8
                )
            elif backend == "llama_local":
                response_text = safe_generate_llama_local(
                    llama_local_bundle, image_path, system_promptv8, user_promptv8
                )
            elif backend == "qwen_local":
                response_text = safe_generate_qwen_local(
                    qwen_local_bundle, image_path, system_promptv8, user_promptv8
                )
            else:
                raise ValueError(f"Unsupported backend in generation: {backend}")
            sample_predictions.append(parse_prediction(response_text))

        final_unknown, final_class_name, _ = aggregate_votes(sample_predictions)

        data.append(
            [
                imgs_idx.cpu().numpy()[0],
                list(ground_truth)[0],
                final_class_name,
                private.cpu().numpy()[0],
                final_unknown,
                image_path,
            ]
        )
        df = pd.DataFrame(data, columns=columns)

        safe_name = sanitize_model_name(configured_model_name)
        if dataset == "DomainNet" or dataset == "VisDA":
            output_path = os.path.join(
                "/hpc/group/carin/sw361/ChatGPT_exp/llm_data",
                "{}_target_domain{}_{}_{}_k{}.csv".format(
                    dataset,
                    args.t_idx,
                    safe_name,
                    f"{version}_self-consistency",
                    num_consistency_samples,
                ),
            )
        else:
            output_path = os.path.join(
                "/hpc/group/carin/sw361/ChatGPT_exp/llm_data",
                "{}_{}_target_domain{}_{}_{}_k{}.csv".format(
                    target_type,
                    dataset,
                    args.t_idx,
                    safe_name,
                    f"{version}_self-consistency",
                    num_consistency_samples,
                ),
            )
        # print(f"Saving results to {output_path}...", flush=True)
        df.to_csv(output_path, index=False)
