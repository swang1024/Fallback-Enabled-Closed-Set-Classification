import argparse
import os
import random
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torchvision import transforms
from torchvision.datasets import INaturalist

from dataset.dataset import INaturalist_UniDA, SFUniDADataset_BLIP
from net_utils import set_random_seed
from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = (
    "You are an AI that classifies images based on a predefined list of categories. \
If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), then provide class_name with the correct category name from the given list and respond with `unknown: False`; \
If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list (DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."

)

DIRECT_PROMPT_TEMPLATE = (
    "Does this image belong to one of the categories in the following list {label_list}? "
    "Please answer with keys unknown and class_name."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Does this image belong to one of the categories in the following list {label_list} "
    "based on the following image summary: {summary}? "
    "Please answer with keys unknown and class_name."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Paradox analysis with Qwen hidden-state similarity")
    parser.add_argument("--dataset", type=str, default="INaturalist", choices=["VisDA", "DomainNet", "INaturalist"])
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=1)
    parser.add_argument("--target_label_type", type=str, default="OPDA")
    parser.add_argument("--target_data_dir", type=str, default=None)
    parser.add_argument("--image_list_filename", type=str, default="image_unida_list.txt")
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--summary_version", type=str, default="v13")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--idx_col", type=str, default="idx")
    parser.add_argument("--summary_col", type=str, default="summary")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--inaturalist_root", type=str, default="/hpc/group/carin/sw361/data/")
    parser.add_argument("--inaturalist_version", type=str, default="2021_valid")
    parser.add_argument("--inaturalist_target_type", type=str, default="class", choices=["phylum", "class"])
    parser.add_argument("--inaturalist_split_seed", type=int, default=0)
    parser.add_argument("--inaturalist_download", action="store_true")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def configure_dataset(args: argparse.Namespace) -> None:
    if args.dataset == "VisDA":
        if args.target_data_dir is None:
            args.target_data_dir = "/hpc/group/carin/sw361/data/VisDA/validation/"
        if args.target_label_type == "PDA":
            args.shared_class_num, args.source_private_class_num, args.target_private_class_num = 6, 6, 0
        elif args.target_label_type == "OSDA":
            args.shared_class_num, args.source_private_class_num, args.target_private_class_num = 6, 0, 6
        elif args.target_label_type == "OPDA":
            args.shared_class_num, args.source_private_class_num, args.target_private_class_num = 6, 3, 3
        elif args.target_label_type == "CLDA":
            args.shared_class_num, args.source_private_class_num, args.target_private_class_num = 12, 0, 0
        else:
            raise ValueError(f"Unsupported target_label_type for VisDA: {args.target_label_type}")
    elif args.dataset == "DomainNet":
        if args.target_data_dir is None:
            domain_list = ["Painting", "Real", "Sketch"]
            args.target_data_dir = os.path.join("/hpc/group/carin/sw361/data/DomainNet", domain_list[args.t_idx])
        if args.target_label_type != "OPDA":
            raise ValueError("DomainNet supports only OPDA in this script.")
        args.shared_class_num, args.source_private_class_num, args.target_private_class_num = 150, 50, 145
    elif args.dataset == "INaturalist":
        if args.inaturalist_target_type == "phylum":
            args.shared_class_num, args.source_private_class_num = 5, 4
        else:
            args.shared_class_num, args.source_private_class_num = 20, 15
        # Filled in later after loading INaturalist taxonomy size.
        args.target_private_class_num = 0
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def dtype_from_str(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def build_summary_map(summary_csv: str, idx_col: str, summary_col: str) -> Dict[int, str]:
    df = pd.read_csv(summary_csv, index_col=False)
    if idx_col not in df.columns:
        raise ValueError(f"Missing idx column '{idx_col}' in {summary_csv}")
    if summary_col not in df.columns:
        raise ValueError(f"Missing summary column '{summary_col}' in {summary_csv}")

    idx_series = pd.to_numeric(df[idx_col], errors="coerce")
    summary_map: Dict[int, str] = {}
    for raw_idx, summary in zip(idx_series, df[summary_col]):
        if pd.isna(raw_idx):
            continue
        summary_map[int(raw_idx)] = "" if pd.isna(summary) else str(summary)
    if not summary_map:
        raise ValueError(f"No valid (idx, summary) entries found in {summary_csv}")
    return summary_map


def build_inaturalist_dataset(args: argparse.Namespace) -> INaturalist_UniDA:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_ = INaturalist(
        root=args.inaturalist_root,
        version=args.inaturalist_version,
        target_type=args.inaturalist_target_type,
        transform=transform,
        download=args.inaturalist_download,
    )

    label_names_list: List[str] = []
    label_ids_list: List[int] = []
    i = 0
    while True:
        try:
            label_names_list.append(dataset_.category_name(args.inaturalist_target_type, i))
            label_ids_list.append(i)
            i += 1
        except (IndexError, ValueError):
            break
    if not label_ids_list:
        raise ValueError("No INaturalist labels discovered for the selected target type.")

    random.Random(args.inaturalist_split_seed).shuffle(label_ids_list)

    shared_class_ids_list = label_ids_list[: args.shared_class_num]
    source_private_class_ids_list = label_ids_list[
        args.shared_class_num : args.shared_class_num + args.source_private_class_num
    ]
    target_private_class_ids_list = label_ids_list[args.shared_class_num + args.source_private_class_num :]
    if not target_private_class_ids_list:
        raise ValueError("INaturalist split produced no target-private classes.")

    args.target_private_class_num = len(target_private_class_ids_list)
    return INaturalist_UniDA(
        root=args.inaturalist_root,
        version=args.inaturalist_version,
        target_type=args.inaturalist_target_type,
        transform=transform,
        download=args.inaturalist_download,
        shared_classes=shared_class_ids_list,
        source_private_classes=source_private_class_ids_list,
        target_private_classes=target_private_class_ids_list,
        label_names_list=label_names_list,
    )


def get_last_token_hidden(model: Qwen2_5_VLForConditionalGeneration, model_inputs) -> torch.Tensor:
    outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[-1][:, -1, :]


def make_visual_inputs(processor: AutoProcessor, image_path: str, direct_prompt: str):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": direct_prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def make_text_inputs(processor: AutoProcessor, prompt_text: str):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )


def make_label_inputs(processor: AutoProcessor, label_text: str):
    return processor(
        text=[label_text],
        padding=True,
        return_tensors="pt",
    )


def compute_label_logprob(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    context_inputs,
    label_text: str,
) -> float:
    context_input_ids = context_inputs["input_ids"]
    context_attention_mask = context_inputs["attention_mask"]
    device = context_input_ids.device
    target_ids = processor.tokenizer(
        label_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if target_ids.shape[1] == 0:
        raise ValueError(f"Ground-truth label tokenized to empty sequence: '{label_text}'")

    target_attention_mask = torch.ones(
        target_ids.shape,
        dtype=context_attention_mask.dtype,
        device=device,
    )
    # Drop sequence-aligned metadata from the original context. After appending
    # target_ids, those tensors become stale and can mismatch attention_mask.
    stale_sequence_keys = {
        "input_ids",
        "attention_mask",
        "input_token_type",
        "token_type_ids",
        "position_ids",
        "cache_position",
    }
    model_inputs = {k: v for k, v in context_inputs.items() if k not in stale_sequence_keys}
    model_inputs["input_ids"] = torch.cat([context_input_ids, target_ids], dim=1)
    model_inputs["attention_mask"] = torch.cat([context_attention_mask, target_attention_mask], dim=1)

    outputs = model(**model_inputs, return_dict=True)
    context_len = context_input_ids.shape[1]
    target_len = target_ids.shape[1]
    target_logits = outputs.logits[:, context_len - 1 : context_len + target_len - 1, :].float()
    token_log_probs = F.log_softmax(target_logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def main() -> None:
    args = parse_args()
    configure_dataset(args)
    set_random_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-VL-7B-Instruct in this analysis script.")

    if args.dataset == "INaturalist":
        summary_csv = args.summary_csv or os.path.join(
            "llm_data",
            f"{args.inaturalist_target_type}_{args.dataset}_target_domain{args.t_idx}_qwen_summary_"
            f"{args.summary_version}_randomseed{args.inaturalist_split_seed}.csv",
        )
        output_csv = args.output_csv or os.path.join(
            "llm_data",
            f"{args.inaturalist_target_type}_{args.dataset}_target_domain{args.t_idx}_qwen_paradox_analysis_"
            f"randomseed{args.inaturalist_split_seed}.csv",
        )
    else:
        summary_csv = args.summary_csv or os.path.join(
            "llm_data", f"{args.dataset}_target_domain{args.t_idx}_qwen_summary_{args.summary_version}.csv"
        )
        output_csv = args.output_csv or os.path.join(
            "llm_data", f"{args.dataset}_target_domain{args.t_idx}_qwen_paradox_analysis.csv"
        )
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary csv not found: {summary_csv}")

    if args.dataset == "INaturalist":
        target_dataset = build_inaturalist_dataset(args)
    else:
        image_list_path = os.path.join(args.target_data_dir, args.image_list_filename)
        if not os.path.exists(image_list_path):
            raise FileNotFoundError(f"Image list file not found: {image_list_path}")

        with open(image_list_path, "r") as f:
            target_data_list = f.readlines()

        target_dataset = SFUniDADataset_BLIP(
            args,
            args.target_data_dir,
            target_data_list,
            d_type="target",
            preload_flg=True,
        )
    dataloader = DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    if not target_dataset.src_labels:
        raise ValueError("No source labels were found from dataset metadata.")

    summary_map = build_summary_map(summary_csv, args.idx_col, args.summary_col)
    label_list_text = str(target_dataset.src_labels)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype_from_str(args.dtype),
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    rows: List[Dict[str, object]] = []
    skipped_missing_summary = 0
    device = "cuda"

    iterator = tqdm(dataloader, ncols=80, desc="Paradox analysis")
    for i, (imgs_train, _, imgs_idx, ground_truth, private) in enumerate(iterator):
        if args.max_samples is not None and i >= args.max_samples:
            break

        idx_val = int(imgs_idx[0].item())
        image_path = list(imgs_train)[0]
        gt_label = str(list(ground_truth)[0])
        private_flag = bool(private[0].item())

        if idx_val not in summary_map:
            skipped_missing_summary += 1
            continue
        summary_text = summary_map[idx_val]

        direct_prompt = DIRECT_PROMPT_TEMPLATE.format(label_list=label_list_text)
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(label_list=label_list_text, summary=summary_text)

        visual_inputs = make_visual_inputs(processor, image_path, direct_prompt).to(device)
        textual_inputs = make_text_inputs(processor, summary_prompt).to(device)
        label_inputs = make_label_inputs(processor, gt_label).to(device)

        with torch.no_grad():
            # h_visual/h_textual/h_label are the last hidden states at the final input token.
            h_visual = get_last_token_hidden(model, visual_inputs).float()
            h_textual = get_last_token_hidden(model, textual_inputs).float()
            h_label = get_last_token_hidden(model, label_inputs).float()

            sim_visual = F.cosine_similarity(h_visual, h_label, dim=-1).item()
            sim_textual = F.cosine_similarity(h_textual, h_label, dim=-1).item()
            gap = sim_textual - sim_visual
            logp_visual = compute_label_logprob(model, processor, visual_inputs, gt_label)
            logp_textual = compute_label_logprob(model, processor, textual_inputs, gt_label)
            gap_logp = logp_textual - logp_visual

        rows.append(
            {
                "idx": idx_val,
                "ground_truth": gt_label,
                "private": private_flag,
                "image_path": image_path,
                "summary": summary_text,
                "sim_visual": sim_visual,
                "sim_textual": sim_textual,
                "gap": gap,
                "logp_visual": logp_visual,
                "logp_textual": logp_textual,
                "gap_logp": gap_logp,
            }
        )

    if not rows:
        raise ValueError("No rows were processed. Check --max_samples and dataset inputs.")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(f"Rows: {len(result_df)}")
    print(f"Skipped missing summaries: {skipped_missing_summary}")
    print(f"Mean sim_visual: {result_df['sim_visual'].mean():.6f}")
    print(f"Mean sim_textual: {result_df['sim_textual'].mean():.6f}")
    print(f"Mean gap (sim_textual - sim_visual): {result_df['gap'].mean():.6f}")
    print(f"Mean logp_visual: {result_df['logp_visual'].mean():.6f}")
    print(f"Mean logp_textual: {result_df['logp_textual'].mean():.6f}")
    print(f"Mean gap_logp (logp_textual - logp_visual): {result_df['gap_logp'].mean():.6f}")


if __name__ == "__main__":
    main()
