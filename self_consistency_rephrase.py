import base64
import json
import os
import random
import re
import time
from collections import Counter

import certifi
import pandas as pd
import torch.multiprocessing as mp
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import INaturalist
from tqdm import tqdm

from config.model_config import build_args
from dataset.dataset import INaturalist_UniDA, SFUniDADataset
from net_utils import set_random_seed

mp.set_sharing_strategy("file_system")

for k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(k, None)
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


def sanitize_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)


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

    token = resolve_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    processor = AutoProcessor.from_pretrained(model_name, token=token)
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
                    temperature=0.0,
                ),
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    user_prompt,
                ],
            )
            return response.text or ""
        except genai_errors.ServerError:
            if i == retries - 1:
                raise
            time.sleep(2**i)


def safe_generate_openai_vision(client, model_name, image_b64, system_prompt, user_prompt):
    retries = 5
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"},
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
            if i == retries - 1:
                raise
            time.sleep(2**i)


def safe_generate_llama_local(bundle, image_path, system_prompt, user_prompt):
    from PIL import Image

    model = bundle["model"]
    processor = bundle["processor"]
    retries = 3
    for i in range(retries):
        try:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
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
                do_sample=False,
                max_new_tokens=256,
            )
            prompt_len = inputs["input_ids"].shape[-1]
            trimmed = generated_ids[:, prompt_len:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            return decoded[0] if decoded else ""
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2**i)


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
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image_path}, {"type": "text", "text": user_prompt}],
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
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
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
            time.sleep(2**i)


def parse_prediction(response_text):
    unknown_match = re.search(
        r'["\']?unknown["\']?\s*[:=]\s*["\']?(true|false)["\']?',
        response_text,
        re.IGNORECASE,
    )
    class_match = re.search(
        r'["\']?class_name["\']?\s*[:=]\s*["\']?([^,"\'}\n]+)',
        response_text,
        re.IGNORECASE,
    )

    unknown = ""
    if unknown_match:
        unknown = "True" if unknown_match.group(1).lower() == "true" else "False"

    class_name = ""
    if class_match:
        class_name = class_match.group(1).strip().strip("'\"")

    return unknown, class_name


def normalize_label(label):
    label = str(label or "").strip().strip("'\"")
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"^(an image of|a photo of|photo of)\s+", "", label, flags=re.IGNORECASE)
    return label.lower().strip()


def canonicalize_predicted_class(raw_class_name, known_lookup):
    cleaned = str(raw_class_name or "").strip().strip("'\"")
    if not cleaned:
        return ""
    normalized = normalize_label(cleaned)
    if normalized in known_lookup:
        return known_lookup[normalized]
    return cleaned


def mode_prediction(predictions):
    valid_predictions = [pred for pred in predictions if pred[0] or pred[1]]
    if not valid_predictions:
        return "", "", 0, 0
    counts = Counter(valid_predictions)
    top_prediction, top_count = counts.most_common(1)[0]
    return top_prediction[0], top_prediction[1], top_count, len(valid_predictions)


def pairwise_agreement(predictions):
    valid_predictions = [pred for pred in predictions if pred[0] or pred[1]]
    n = len(valid_predictions)
    if n <= 1:
        return 1.0 if n == 1 else 0.0
    total_pairs = n * (n - 1) // 2
    matched_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if valid_predictions[i] == valid_predictions[j]:
                matched_pairs += 1
    return matched_pairs / total_pairs


def build_rephrased_prompts(known_class_list, num_rephrases):
    class_list_literal = "[" + ", ".join([f"'{cls}'" for cls in known_class_list]) + "]"
    base_templates = [
        "Check if this image belongs to one category from this list: {class_list}. \
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "Given this image and allowed labels {class_list}, decide if it is in-list or out-of-list.,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "Classify this image using ONLY these categories: {class_list}.,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "Is this image an instance of any class in {class_list}?,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "From the candidate labels {class_list}, pick the best class and report if none is exact.,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "Determine whether this image matches one of the following categories: {class_list}.,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "Use this closed-set label list {class_list} to predict the closest category for this image.,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
        "Analyze the image against this category vocabulary {class_list} and return the best label.,\
        Please format the answer csv format with keys unknown and class_name separated by ',' \
        Example 1: \
        Image: (picture of a aeroplane) \
        Response: unknown: False, class_name: 'aeroplane' \
        Example 2: \
        Image: (picture of a donkey) \
        Response: unknown: True, class_name: 'horse' \
        ",
    ]

    # output_constraint = (
    #     " Return exactly one line in this format: unknown: <True|False>, class_name: '<label-from-list>'. "
    #     "If the image is not in the list, still output the closest label from the list with unknown: True. "
    #     "Do not add explanation."
    # )

    prompts = []
    for i in range(num_rephrases):
        template = base_templates[i % len(base_templates)]
        prompts.append(template.format(class_list=class_list_literal))
    return prompts


def prepare_domainnet_or_visda(args, dataset_name):
    if dataset_name == "VisDA":
        args.dataset = "VisDA"
        args.target_data_dir = "/hpc/group/carin/sw361/data/VisDA/validation/"
    elif dataset_name == "DomainNet":
        args.dataset = "DomainNet"
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    known_class_list = list(target_dataset.src_labels)
    return target_train_dataloader, known_class_list


def prepare_inaturalist(inaturalist_random_seed, inaturalist_download):
    target_type = os.getenv("INAT_TARGET_TYPE", "class")
    transform = transforms.Compose([transforms.ToTensor()])

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

    random.Random(inaturalist_random_seed).shuffle(label_ids_list)
    if target_type == "phylum":
        shared_class_num = 5
        source_private_class_num = 4
    elif target_type == "class":
        shared_class_num = 20
        source_private_class_num = 15
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    shared_class_ids_list = label_ids_list[:shared_class_num]
    source_private_class_ids_list = label_ids_list[shared_class_num : shared_class_num + source_private_class_num]
    target_private_class_ids_list = label_ids_list[shared_class_num + source_private_class_num :]
    source_classes = shared_class_ids_list + source_private_class_ids_list
    known_class_list = [label_names_list[i] for i in source_classes]

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
    return target_train_dataloader, known_class_list, target_type


def build_output_path(dataset_name, target_type, args, configured_model_name, version, num_rephrases):
    safe_name = sanitize_model_name(configured_model_name)
    if dataset_name in ("DomainNet", "VisDA"):
        file_name = "{}_target_domain{}_{}_{}_rephrase-k{}.csv".format(
            dataset_name,
            args.t_idx,
            safe_name,
            version,
            num_rephrases,
        )
    else:
        file_name = "{}_{}_target_domain{}_{}_{}_rephrase-k{}.csv".format(
            target_type,
            dataset_name,
            args.t_idx,
            safe_name,
            version,
            num_rephrases,
        )
    return os.path.join("/hpc/group/carin/sw361/ChatGPT_exp/llm_data", file_name)


def maybe_tensor_first_value(value):
    if hasattr(value, "cpu"):
        return value.cpu().numpy()[0]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def main():
    set_random_seed(2025)
    args = build_args()

    dataset_name = os.getenv("DATASET", "INaturalist")
    version = os.getenv("VERSION", "v8_sc_rephrase")
    num_rephrases = int(os.getenv("NUM_REPHRASE_PROMPTS", "3"))
    agreement_threshold = float(os.getenv("AGREEMENT_THRESHOLD", "0.6"))
    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    inaturalist_random_seed = int(os.getenv("INAT_SPLIT_SEED", "0"))
    inaturalist_download = os.getenv("INAT_DOWNLOAD", "false").strip().lower() == "true"

    model_names_env = os.getenv("MODEL_NAMES", "").strip()
    if model_names_env:
        model_names = [m.strip() for m in model_names_env.split(",") if m.strip()]
    else:
        model_names = [
            # "llama3.2-vision",
            "qwen-2.5-7B-VL",
        ]

    if dataset_name in ("DomainNet", "VisDA"):
        target_train_dataloader, known_class_list = prepare_domainnet_or_visda(args, dataset_name)
        target_type = ""
    elif dataset_name == "INaturalist":
        target_train_dataloader, known_class_list, target_type = prepare_inaturalist(
            inaturalist_random_seed=inaturalist_random_seed,
            inaturalist_download=inaturalist_download,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not known_class_list:
        raise RuntimeError("Known class list is empty. Cannot build rephrased prompts.")

    print(f"dataset={dataset_name} num_samples={len(target_train_dataloader)} known_classes={len(known_class_list)}")
    print(f"num_rephrases={num_rephrases} agreement_threshold={agreement_threshold}")

    known_lookup = {normalize_label(cls): cls for cls in known_class_list}
    rephrased_prompts = build_rephrased_prompts(known_class_list, num_rephrases)
    system_prompt = (
        "You are an AI that classifies images based on a predefined list of categories. "
        "If the image belongs to a category in the GIVEN list, return that exact category and unknown: False. "
        "If the image does not belong to any category in the GIVEN list, return the closest category from the GIVEN list and unknown: True."
    )

    columns = [
        "idx",
        "ground truth",
        "predicted class name",
        "private",
        "unknown",
        "img url",
        "agreement_score",
        "pairwise_agreement",
        "uncertainty",
        "max_vote_count",
        "valid_vote_count",
        "num_rephrase_prompts",
        "agreement_pass",
        "rephrased_predictions",
    ]
    os.makedirs("/hpc/group/carin/sw361/ChatGPT_exp/llm_data", exist_ok=True)

    gemini_client = None
    openai_client = None
    llama_local_bundle = None
    qwen_local_bundle = None

    for configured_model_name in model_names:
        model_name = canonicalize_model_name(configured_model_name)
        backend = detect_backend(model_name)
        print(f"Running rephrase self-consistency for model={configured_model_name} (resolved={model_name}, backend={backend})")

        if backend == "gemini" and gemini_client is None:
            gemini_client = init_gemini_client()
        elif backend == "openai" and openai_client is None:
            openai_client = init_openai_client()
        elif backend == "llama_local" and llama_local_bundle is None:
            llama_local_bundle = init_llama_local(model_name)
        elif backend == "qwen_local" and qwen_local_bundle is None:
            qwen_local_bundle = init_qwen_local(model_name)

        output_path = build_output_path(
            dataset_name=dataset_name,
            target_type=target_type,
            args=args,
            configured_model_name=configured_model_name,
            version=version,
            num_rephrases=num_rephrases,
        )

        rows = []
        for idx, (imgs_train, img_labels, imgs_idx, ground_truth, private) in enumerate(
            tqdm(target_train_dataloader, desc=f"{configured_model_name}")
        ):
            if max_samples > 0 and idx >= max_samples:
                break

            image_path = list(imgs_train)[0]
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            parsed_predictions = []
            parsed_records = []
            for prompt_id, user_prompt in enumerate(rephrased_prompts):
                if backend == "gemini":
                    response_text = safe_generate_gemini(
                        gemini_client, model_name, image_bytes, system_prompt, user_prompt
                    )
                elif backend == "openai":
                    response_text = safe_generate_openai_vision(
                        openai_client, model_name, image_b64, system_prompt, user_prompt
                    )
                elif backend == "llama_local":
                    response_text = safe_generate_llama_local(
                        llama_local_bundle, image_path, system_prompt, user_prompt
                    )
                elif backend == "qwen_local":
                    response_text = safe_generate_qwen_local(
                        qwen_local_bundle, image_path, system_prompt, user_prompt
                    )
                else:
                    raise ValueError(f"Unsupported backend in generation: {backend}")

                unknown, class_name = parse_prediction(response_text)
                class_name = canonicalize_predicted_class(class_name, known_lookup)
                prediction = (unknown, class_name)
                parsed_predictions.append(prediction)
                parsed_records.append(
                    {
                        "prompt_id": prompt_id,
                        "unknown": unknown,
                        "class_name": class_name,
                        "raw_response": response_text,
                    }
                )

            final_unknown, final_class_name, max_vote_count, valid_vote_count = mode_prediction(parsed_predictions)
            agreement_score = max_vote_count / valid_vote_count if valid_vote_count > 0 else 0.0
            pair_agreement = pairwise_agreement(parsed_predictions)
            uncertainty = 1.0 - agreement_score
            agreement_pass = agreement_score >= agreement_threshold

            rows.append(
                [
                    maybe_tensor_first_value(imgs_idx),
                    list(ground_truth)[0],
                    final_class_name,
                    maybe_tensor_first_value(private),
                    final_unknown,
                    image_path,
                    agreement_score,
                    pair_agreement,
                    uncertainty,
                    max_vote_count,
                    valid_vote_count,
                    num_rephrases,
                    agreement_pass,
                    json.dumps(parsed_records, ensure_ascii=False),
                ]
            )

            if len(rows) % 20 == 0:
                pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)

        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
