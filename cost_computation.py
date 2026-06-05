#!/usr/bin/env python3
"""Run direct/summary/summary_pred API calls and compute true cost metrics.

This script executes the 3-call pipeline (matching your direct/summary/summary_pred flow):
- direct classification (V)
- image summary generation
- summary-based classification

Then it reports, per Experimental_Plan.md:
1) calls per image
2) latency mean/std + median/IQR
3) throughput (images/sec, sec/100 images)
4) true cost per image and per 1,000 images
5) relative overhead of (V+T) vs (V)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from dataset.dataset import SFUniDADataset
from net_utils import set_random_seed


DOMAINNET_CLASSES = [
    "The Eiffel Tower", "The Great Wall of China", "The Mona Lisa", "aircraft carrier", "airplane", "alarm clock",
    "ambulance", "angel", "animal migration", "ant", "anvil", "apple", "arm", "asparagus", "axe", "backpack",
    "banana", "bandage", "barn", "baseball", "baseball bat", "basket", "basketball", "bat", "bathtub",
    "beach", "bear", "beard", "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird", "birthday cake",
    "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie", "bracelet", "brain", "bread", "bridge",
    "broccoli", "broom", "bucket", "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator",
    "calendar", "camel", "camera", "camouflage", "campfire", "candle", "cannon", "canoe", "car", "carrot",
    "castle", "cat", "ceiling fan", "cell phone", "cello", "chair", "chandelier", "church", "circle",
    "clarinet", "clock", "cloud", "coffee cup", "compass", "computer", "cookie", "cooler", "couch",
    "cow", "crab", "crayon", "crocodile", "crown", "cruise ship", "cup", "diamond", "dishwasher",
    "diving board", "dog", "dolphin", "donut", "door", "dragon", "dresser", "drill", "drums",
    "duck", "dumbbell", "ear", "elbow", "elephant", "envelope", "eraser", "eye", "eyeglasses",
    "face", "fan", "feather", "fence", "finger", "fire hydrant", "fireplace", "firetruck",
    "fish", "flamingo", "flashlight", "flip flops", "floor lamp", "flower", "flying saucer",
    "foot", "fork", "frog", "frying pan", "garden", "garden hose", "giraffe", "goatee",
    "golf club", "grapes", "grass", "guitar", "hamburger", "hammer", "hand", "harp", "hat", "headphones",
    "hedgehog", "helicopter", "helmet", "hexagon", "hockey puck", "hockey stick", "horse", "hospital",
    "hot air balloon", "hot dog", "hot tub", "hourglass", "house", "house plant", "hurricane", "ice cream",
    "jacket", "jail", "kangaroo", "key", "keyboard", "knee", "knife", "ladder", "lantern", "laptop", "leaf",
    "leg", "light bulb", "lighter", "lighthouse", "lightning", "line", "lion", "lipstick", "lobster", "lollipop",
    "mailbox", "map", "marker", "matches", "megaphone", "mermaid", "microphone", "microwave", "monkey", "moon",
    "mosquito", "motorbike", "mountain", "mouse", "moustache", "mouth", "mug", "mushroom", "nail",
]

VISDA_CLASSES = ["aeroplane", "bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant"]


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_image_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


def parse_response_text(text: str) -> Dict[str, str]:
    unknown = ""
    class_name = ""
    if not text:
        return {"unknown": unknown, "class_name": class_name}
    m_unk = re.search(r"unknown\s*:\s*([^,\n]+)", text, flags=re.IGNORECASE)
    m_cls = re.search(r"class_name\s*:\s*([^,\n]+)", text, flags=re.IGNORECASE)
    if m_unk:
        unknown = m_unk.group(1).strip().strip("'\"")
    if m_cls:
        class_name = m_cls.group(1).strip().strip("'\"")
    return {"unknown": unknown, "class_name": class_name}


def usage_to_dict(resp) -> Dict[str, float]:
    usage = getattr(resp, "usage", None)
    d = {"prompt_tokens": np.nan, "completion_tokens": np.nan, "total_tokens": np.nan}
    if usage is None:
        return d
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    d["prompt_tokens"] = float(pt) if pt is not None else np.nan
    d["completion_tokens"] = float(ct) if ct is not None else np.nan
    d["total_tokens"] = float(tt) if tt is not None else np.nan
    return d


def gemini_usage_to_dict(resp) -> Dict[str, float]:
    usage = getattr(resp, "usage_metadata", None)
    d = {"prompt_tokens": np.nan, "completion_tokens": np.nan, "total_tokens": np.nan}
    if usage is None:
        return d
    pt = getattr(usage, "prompt_token_count", None)
    ct = getattr(usage, "candidates_token_count", None)
    if ct is None:
        ct = getattr(usage, "output_token_count", None)
    tt = getattr(usage, "total_token_count", None)
    d["prompt_tokens"] = float(pt) if pt is not None else np.nan
    d["completion_tokens"] = float(ct) if ct is not None else np.nan
    d["total_tokens"] = float(tt) if tt is not None else np.nan
    return d


def get_prices(model: str) -> Dict[str, Optional[float]]:
    # Per-1M-token prices (USD). Override with CLI if needed.
    table = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
        "gpt-4o": {"input": 5.0, "output": 15.0, "cached_input": 2.5},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "cached_input": None},
    }
    return table.get(model, {"input": None, "output": None, "cached_input": None})


def per_request_cost(usage: Dict[str, float], input_price: float, output_price: float) -> float:
    pt = usage["prompt_tokens"] if np.isfinite(usage["prompt_tokens"]) else 0.0
    ct = usage["completion_tokens"] if np.isfinite(usage["completion_tokens"]) else 0.0
    return (pt * input_price + ct * output_price) / 1_000_000.0


def latency_stats(lat_s: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(lat_s, errors="coerce").dropna()
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    median = float(vals.median())
    iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
    return {
        "latency_mean_sec": mean,
        "latency_std_sec": std,
        "latency_median_sec": median,
        "latency_iqr_sec": iqr,
        "throughput_images_per_sec": (1.0 / mean) if mean > 0 else np.nan,
        "sec_per_100_images": 100.0 * mean if np.isfinite(mean) else np.nan,
    }


def aggregate_variant(df: pd.DataFrame, model: str, variant: str) -> Dict[str, float]:
    if variant == "V":
        calls = ["direct"]
    elif variant == "T":
        calls = ["summary", "summary_pred"]
    else:
        calls = ["direct", "summary", "summary_pred"]

    sub = df.copy()
    sub["calls_per_image"] = len(calls)
    sub["latency_total_sec"] = sum(sub[f"{c}_latency_sec"] for c in calls)
    sub["cost_total_usd"] = sum(sub[f"{c}_cost_usd"] for c in calls)
    sub["prompt_tokens_total"] = sum(sub[f"{c}_prompt_tokens"] for c in calls)
    sub["completion_tokens_total"] = sum(sub[f"{c}_completion_tokens"] for c in calls)
    sub["tokens_total"] = sum(sub[f"{c}_total_tokens"] for c in calls)

    lat = latency_stats(sub["latency_total_sec"])
    cpi = float(pd.to_numeric(sub["cost_total_usd"], errors="coerce").mean())
    return {
        "model": model,
        "variant": variant,
        "images_evaluated": int(len(sub)),
        "calls_per_image": float(sub["calls_per_image"].mean()),
        "total_calls": int(sub["calls_per_image"].sum()),
        "prompt_tokens_per_image": float(sub["prompt_tokens_total"].mean()),
        "completion_tokens_per_image": float(sub["completion_tokens_total"].mean()),
        "total_tokens_per_image": float(sub["tokens_total"].mean()),
        **lat,
        "cost_per_image_usd": cpi,
        "cost_per_1000_images_usd": cpi * 1000.0,
        "overhead_calls_vplusT_vs_V": np.nan,
        "overhead_latency_vplusT_vs_V": np.nan,
        "overhead_cost_vplusT_vs_V": np.nan,
    }


def build_prompts(dataset: str):
    system_prompt = (
        "You are an AI that classifies images based on a predefined list of categories. "
        "If the image belongs to a category in the GIVEN list (ONLY from the GIVEN list), "
        "then provide class_name with the correct category name from the given list and respond with `unknown: False`; "
        "If the image does not belong to any category in the GIVEN list, then select the closest possible match from the GIVEN list "
        "(DO NOT reply with labels outside of the list) as class_name and respond with `unknown: True`."
    )
    summary_prompt = "Identify the primary object in the image, excluding any background elements."

    if dataset == "DomainNet":
        cls = DOMAINNET_CLASSES
    elif dataset == "VisDA":
        cls = VISDA_CLASSES
    else:
        raise ValueError("Only DomainNet/VisDA supported in this script.")

    direct_user = (
        f"Does this image belong to one of the categories in the following list {cls}? "
        "Please format the answer csv format with keys unknown and class_name separated by ','. "
        "Example: unknown: False, class_name: 'cat'"
    )
    summary_pred_template = (
        f"Does this image belong to one of the categories in the following list {cls} "
        "based on the following summary: {}? "
        "Please format the answer csv format with keys unknown and class_name separated by ','. "
        "Example: unknown: False, class_name: 'cat'"
    )
    return system_prompt, summary_prompt, direct_user, summary_pred_template


def load_target_dataloader(dataset_name: str, t_idx: int, seed: int):
    import sys

    orig = sys.argv
    try:
        sys.argv = [orig[0]]
        args = build_args()
    finally:
        sys.argv = orig

    args.dataset = dataset_name
    args.t_idx = t_idx
    set_random_seed(seed)
    target_data_list = open(Path(args.target_data_dir) / "image_unida_list.txt", "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    loader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=1)
    return args, loader


def build_client(model: str, openai_api_key: Optional[str], gemini_api_key: Optional[str]):
    if is_gemini_model(model):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini models. Install with `pip install google-genai`."
            ) from exc
        key = gemini_api_key or os.getenv("GEMINI_API_KEY") or openai_api_key
        if not key:
            raise ValueError("Missing Gemini API key. Provide --gemini-api-key or set GEMINI_API_KEY.")
        if key.startswith("sk-"):
            raise ValueError(
                "Gemini models require a Google API key. Provide --gemini-api-key (or GEMINI_API_KEY), not an OpenAI key."
            )
        return genai.Client(api_key=key)
    return OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()


def call_openai_with_image(
    client,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    image_b64: str,
    max_completion_tokens: int,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"}},
            ],
        }
    )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    text = resp.choices[0].message.content or ""
    return text, usage_to_dict(resp)


def call_gemini_with_image(
    client,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    image_bytes: bytes,
    max_completion_tokens: int,
):
    from google import genai
    from google.genai import types

    cfg = {"max_output_tokens": max_completion_tokens}
    if system_prompt:
        cfg["system_instruction"] = system_prompt

    retries = 5
    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(**cfg),
                contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), user_prompt],
            )
            text = getattr(resp, "text", "") or ""
            return text, gemini_usage_to_dict(resp)
        except genai.errors.ServerError:
            if i == retries - 1:
                raise
            time.sleep(2**i)

    raise RuntimeError("Gemini request failed after retries.")


def call_model_with_image(
    client,
    model: str,
    system_prompt: Optional[str],
    user_prompt: str,
    image_b64: Optional[str],
    image_bytes: Optional[bytes],
    max_completion_tokens: int,
):
    if is_gemini_model(model):
        if image_bytes is None:
            raise ValueError("image_bytes is required for Gemini requests.")
        return call_gemini_with_image(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            max_completion_tokens=max_completion_tokens,
        )
    if image_b64 is None:
        raise ValueError("image_b64 is required for OpenAI requests.")
    return call_openai_with_image(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_b64=image_b64,
        max_completion_tokens=max_completion_tokens,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run 3 API calls per image and compute true API cost.")
    p.add_argument("--dataset", choices=["DomainNet", "VisDA"], default="DomainNet")
    p.add_argument("--target-domain-index", type=int, default=1)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. For Gemini, use --gemini-api-key or GEMINI_API_KEY.",
    )
    p.add_argument(
        "--gemini-api-key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key. If omitted, reads GEMINI_API_KEY env var.",
    )
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-completion-tokens", type=int, default=300)
    p.add_argument("--input-price-per-1m", type=float, default=None)
    p.add_argument("--output-price-per-1m", type=float, default=None)
    p.add_argument("--out-dir", type=str, default="llm_data/cost_computation_live")
    p.add_argument("--save-per-request", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args.model, args.api_key, args.gemini_api_key)

    prices = get_prices(args.model)
    in_price = args.input_price_per_1m if args.input_price_per_1m is not None else prices["input"]
    out_price = args.output_price_per_1m if args.output_price_per_1m is not None else prices["output"]
    if in_price is None or out_price is None:
        raise ValueError("Missing pricing. Provide --input-price-per-1m and --output-price-per-1m.")

    _, loader = load_target_dataloader(args.dataset, args.target_domain_index, args.seed)
    system_prompt, summary_prompt, direct_user, summary_pred_template = build_prompts(args.dataset)

    rows = []
    request_logs = []
    processed = 0
    for i, (imgs_train, _, imgs_idx, ground_truth, private) in enumerate(loader):
        if i < args.start_index:
            continue
        if processed >= args.max_images:
            break

        image_path = str(list(imgs_train)[0])
        if is_gemini_model(args.model):
            image_bytes = read_image_bytes(image_path)
            image_b64 = None
        else:
            image_b64 = encode_image(image_path)
            image_bytes = None
        idx_val = int(imgs_idx.cpu().numpy()[0])
        gt = str(list(ground_truth)[0])
        priv = bool(private.cpu().numpy()[0])

        # 1) Direct call
        t0 = time.perf_counter()
        d_text, d_usage = call_model_with_image(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=direct_user,
            image_b64=image_b64,
            image_bytes=image_bytes,
            max_completion_tokens=args.max_completion_tokens,
        )
        t1 = time.perf_counter()
        d_parsed = parse_response_text(d_text)
        d_cost = per_request_cost(d_usage, in_price, out_price)

        # 2) Summary call
        t2 = time.perf_counter()
        s_text, s_usage = call_model_with_image(
            client=client,
            model=args.model,
            system_prompt=None,
            user_prompt=summary_prompt,
            image_b64=image_b64,
            image_bytes=image_bytes,
            max_completion_tokens=args.max_completion_tokens,
        )
        t3 = time.perf_counter()
        s_cost = per_request_cost(s_usage, in_price, out_price)

        # 3) Summary-pred call
        t4 = time.perf_counter()
        sp_text, sp_usage = call_model_with_image(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=summary_pred_template.format(s_text),
            image_b64=image_b64,
            image_bytes=image_bytes,
            max_completion_tokens=args.max_completion_tokens,
        )
        t5 = time.perf_counter()
        sp_parsed = parse_response_text(sp_text)
        sp_cost = per_request_cost(sp_usage, in_price, out_price)

        row = {
            "idx": idx_val,
            "ground_truth": gt,
            "private": priv,
            "img_url": image_path,
            "direct_text": d_text,
            "summary_text": s_text,
            "summary_pred_text": sp_text,
            "direct_unknown": d_parsed["unknown"],
            "direct_class_name": d_parsed["class_name"],
            "summary_pred_unknown": sp_parsed["unknown"],
            "summary_pred_class_name": sp_parsed["class_name"],
            "direct_latency_sec": t1 - t0,
            "summary_latency_sec": t3 - t2,
            "summary_pred_latency_sec": t5 - t4,
            "direct_prompt_tokens": d_usage["prompt_tokens"],
            "direct_completion_tokens": d_usage["completion_tokens"],
            "direct_total_tokens": d_usage["total_tokens"],
            "summary_prompt_tokens": s_usage["prompt_tokens"],
            "summary_completion_tokens": s_usage["completion_tokens"],
            "summary_total_tokens": s_usage["total_tokens"],
            "summary_pred_prompt_tokens": sp_usage["prompt_tokens"],
            "summary_pred_completion_tokens": sp_usage["completion_tokens"],
            "summary_pred_total_tokens": sp_usage["total_tokens"],
            "direct_cost_usd": d_cost,
            "summary_cost_usd": s_cost,
            "summary_pred_cost_usd": sp_cost,
        }
        rows.append(row)
        processed += 1

        if args.save_per_request:
            request_logs.extend(
                [
                    {"idx": idx_val, "stage": "direct", "latency_sec": t1 - t0, **d_usage, "cost_usd": d_cost},
                    {"idx": idx_val, "stage": "summary", "latency_sec": t3 - t2, **s_usage, "cost_usd": s_cost},
                    {"idx": idx_val, "stage": "summary_pred", "latency_sec": t5 - t4, **sp_usage, "cost_usd": sp_cost},
                ]
            )

    if not rows:
        raise RuntimeError("No images processed.")

    per_image = pd.DataFrame(rows)
    per_image_path = out_dir / f"{args.dataset}_domain{args.target_domain_index}_{args.model}_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    metric_rows = [
        aggregate_variant(per_image, args.model, "V"),
        aggregate_variant(per_image, args.model, "T"),
        aggregate_variant(per_image, args.model, "V+T"),
    ]

    v = metric_rows[0]
    vt = metric_rows[2]
    vt["overhead_calls_vplusT_vs_V"] = (vt["calls_per_image"] - v["calls_per_image"]) / v["calls_per_image"]
    vt["overhead_latency_vplusT_vs_V"] = (vt["latency_mean_sec"] - v["latency_mean_sec"]) / v["latency_mean_sec"]
    vt["overhead_cost_vplusT_vs_V"] = (vt["cost_per_image_usd"] - v["cost_per_image_usd"]) / v["cost_per_image_usd"]

    out_metrics = pd.DataFrame(metric_rows)
    out_metrics["variant"] = pd.Categorical(out_metrics["variant"], categories=["V", "T", "V+T"], ordered=True)
    out_metrics = out_metrics.sort_values("variant")
    metrics_path = out_dir / f"{args.dataset}_domain{args.target_domain_index}_{args.model}_cost_metrics.csv"
    out_metrics.to_csv(metrics_path, index=False)

    if args.save_per_request:
        req_path = out_dir / f"{args.dataset}_domain{args.target_domain_index}_{args.model}_per_request.csv"
        pd.DataFrame(request_logs).to_csv(req_path, index=False)

    print(out_metrics.to_string(index=False))
    print(f"\nSaved per-image logs to: {per_image_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
