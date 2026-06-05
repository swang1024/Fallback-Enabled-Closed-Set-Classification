#!/usr/bin/env python3
"""
CLIP zero-shot classifier (CLIP-ViT-L) with simple unknown-detection.

Usage examples:
  # ImageFolder layout: data_root/class_x/*.jpg
  python main_5_clip_rebuttal.py --data-root /path/to/data_root \
    --prompt "a photo of a {classname}" --threshold 0.18 --batch-size 32 \
    --output-csv clip_preds.csv

  # Using explicit images CSV with column `image_path` and classes file
  python main_5_clip_rebuttal.py --images-csv images_list.csv --images-col image_path \
    --classes-file classes.txt --prompt "a photo of a {classname}" --threshold 0.18

This script uses Hugging Face Transformers' CLIPModel + CLIPProcessor and PyTorch.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
except Exception as e:
    raise ImportError(
        "Missing dependencies: install `transformers` and `torch` (and pillow)."
        " Example: pip install torch transformers pillow"
    ) from e

from PIL import Image


class ClipZeroShotClassifier:
    """Simple wrapper for CLIP zero-shot classification with thresholded unknown detection.

    - Loads `openai/clip-vit-large-patch14` by default (CLIP-ViT-L family).
    - Create text prompts like `prompt.format(classname=classname)` to build text embeddings.
    - Predict images in batches, compute cosine similarity to text embeddings.
    - If top similarity < threshold, returns label 'unknown'.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.text_embeddings = None  # (num_classes, dim)
        self.classnames: List[str] = []

    def load_model(self):
        if self.model is None:
            print(f"Loading CLIP model {self.model_name} on {self.device} ...")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def build_text_embeddings(self, classnames: List[str], prompt: str = "a photo of a {classname}"):
        """Compute and store normalized text embeddings for provided classnames.

        Returns a tensor of shape (num_classes, dim) normalized to unit length.
        """
        self.load_model()
        self.classnames = list(classnames)
        prompts = [prompt.format(classname=c) for c in self.classnames]
        # Use processor to tokenize text
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            text_feats = self.model.get_text_features(**inputs)
            text_feats = F.normalize(text_feats, p=2, dim=1)
        self.text_embeddings = text_feats
        return self.text_embeddings

    def _batch_images(self, image_paths: List[Path], batch_size: int) -> Iterable[List[Path]]:
        for i in range(0, len(image_paths), batch_size):
            yield image_paths[i : i + batch_size]

    def predict_paths(self, image_paths: List[Path], batch_size: int = 32, threshold: float = 0.18) -> List[Tuple[str, float]]:
        """Predict labels for the given list of image `Path`s.

        Returns list of (predicted_label, max_similarity).
        If max_similarity < threshold, predicted_label is 'unknown'.
        """
        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings not built. Call build_text_embeddings(...) first.")
        self.load_model()
        results: List[Tuple[str, float]] = []

        for batch_paths in self._batch_images(image_paths, batch_size):
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    # fallback: record unknown with very low similarity
                    images.append(None)
                    continue
                images.append(img)

            # Prepare inputs; processor accepts list of PIL images
            imgs_for_proc = [im for im in images if im is not None]
            if len(imgs_for_proc) == 0:
                # all failed
                for _ in batch_paths:
                    results.append(("unknown", -999.0))
                continue

            with torch.no_grad():
                inputs = self.processor(images=imgs_for_proc, return_tensors="pt").to(self.device)
                image_feats = self.model.get_image_features(**inputs)
                image_feats = F.normalize(image_feats, p=2, dim=1)

            # Now iterate through batch_paths mapping to image_feats indices
            img_feat_iter = iter(image_feats)
            for im in images:
                if im is None:
                    results.append(("unknown", -999.0))
                    continue
                feat = next(img_feat_iter)
                # cosine similarities with text embeddings
                sims = (feat.unsqueeze(0) @ self.text_embeddings.T).squeeze(0)
                # sims shape: (num_classes,)
                max_val, idx = torch.max(sims, dim=0)
                max_val_f = float(max_val.cpu().item())
                if max_val_f < threshold:
                    results.append(("unknown", max_val_f))
                else:
                    results.append((self.classnames[int(idx.cpu().item())], max_val_f))

        return results


def infer_classes_from_imagefolder(root: str) -> List[str]:
    # classes are subdirectory names
    p = Path(root)
    classes = [d.name for d in sorted(p.iterdir()) if d.is_dir()]
    return classes


def read_images_csv(images_csv: str, images_col: str = "image") -> List[Path]:
    rows = []
    with open(images_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if images_col not in reader.fieldnames:
            raise ValueError(f"images_col '{images_col}' not found in CSV fields: {reader.fieldnames}")
        for r in reader:
            rows.append(Path(r[images_col]))
    return rows


def write_output_csv(out_csv: str, image_paths: List[Path], preds: List[Tuple[str, float]]):
    # To support extra columns, we need ground truth and private/unknown flags
    # If ground truth is available, pass it as a list; else fallback to None
    ground_truths = getattr(write_output_csv, "ground_truths", None)
    classes_list = getattr(write_output_csv, "classes_list", None)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "predicted", "max_similarity", "ground_truth", "private", "unknown"])
        for idx, (p, (lab, sim)) in enumerate(zip(image_paths, preds)):
            gt = ground_truths[idx] if ground_truths is not None and idx < len(ground_truths) else ""
            # private: True if ground truth not in classes_list
            private = str(gt not in classes_list) if gt != "" and classes_list is not None else ""
            # unknown: True if predicted not in classes_list
            unknown = str(lab not in classes_list) if lab != "" and classes_list is not None else ""
            writer.writerow([str(p), lab, f"{sim:.6f}", gt, private, unknown])


def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--data-root", type=str, help="Root directory with class subfolders (ImageFolder layout)")
    group.add_argument("--images-csv", type=str, help="CSV listing image paths for prediction")

    p.add_argument("--use-dataloader", action="store_true", help="Use dataloader logic from main_1_direct_prompting.py for dataset loading")
    p.add_argument("--images-col", type=str, default="image", help="Column name in images CSV containing image paths")
    p.add_argument("--classes-file", type=str, help="Text file listing classnames (one per line). If omitted and --data-root used, classes will be inferred from subfolders.")
    p.add_argument("--prompt", type=str, default="a photo of a {classname}", help="Prompt template used for zero-shot (must contain {classname})")
    p.add_argument("--threshold", type=float, default=0.19, help="Similarity threshold below which prediction is 'unknown' (cosine similarity)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-csv", type=str, default="clip_predictions_DomainNet_1.csv")
    p.add_argument("--model-name", type=str, default="openai/clip-vit-large-patch14", help="Hugging Face model name for CLIP")

    return p.parse_args()


def main():
    args = parse_args()

    # Dataloader logic from main_1_direct_prompting.py
    if args.use_dataloader:
        # Import required modules
        import sys
        import torch
        from pathlib import Path
        from torchvision import transforms
        # Dataset imports
        sys.path.append(str(Path(__file__).parent))
        from dataset.dataset import SFUniDADataset_BLIP, SFUniDADataset, INaturalist_UniDA
        from torchvision.datasets import INaturalist
        from config.model_config import build_args
        from net_utils import set_random_seed
        from torch.utils.data.dataloader import DataLoader
        import random

        # Call build_args() in a clean argv environment so it doesn't try to parse
        # this script's custom CLI flags (which would raise "unrecognized arguments").
        import sys
        orig_argv = sys.argv
        try:
            sys.argv = [orig_argv[0]]
            cfg = build_args()
        finally:
            sys.argv = orig_argv
        set_random_seed(2025)

        # Allow CLI overrides to the configuration from build_args().
        # Map any CLI arg whose name exists on cfg and is not None.
        for key, val in vars(args).items():
            if val is None:
                continue
            # special-case: --data-root should map to cfg.target_data_dir when present
            if key == "data_root":
                setattr(cfg, "target_data_dir", val)
                continue
            if hasattr(cfg, key):
                try:
                    setattr(cfg, key, val)
                except Exception:
                    # ignore if assignment not supported
                    pass

        image_paths = []
        classes = []

        if cfg.dataset == "DomainNet" or cfg.dataset == "VisDA":
            if cfg.dataset == "DomainNet":
                target_data_dir = cfg.target_data_dir
            else:
                # prefer CLI-provided data_root if given, otherwise default path
                target_data_dir = cfg.target_data_dir if hasattr(cfg, "target_data_dir") and cfg.target_data_dir else "/hpc/group/carin/sw361/data/VisDA/validation/"
            target_data_list = open(os.path.join(target_data_dir, "image_unida_list.txt"), "r").readlines()
            target_dataset = SFUniDADataset(cfg, target_data_dir, target_data_list, d_type="target", preload_flg=True)
            # Use data_list for image paths and class names
            image_paths = [Path(os.path.join(target_data_dir, item[0])) for item in target_dataset.data_list]
            # Class names: get unique label names from tgt_labels dict if available, else from data_list
            if hasattr(target_dataset, "src_labels") and isinstance(target_dataset.src_labels, list):
                classes = sorted(list(set(target_dataset.src_labels)))
                print(classes)
            else:
                # fallback: use first part of path as class name
                classes = sorted(list(set([item[0].split('/')[0].replace('_', ' ') for item in target_dataset.data_list])))

        elif args.dataset == "iNaturalist":
            target_type = 'class'
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_ = INaturalist(
                root='/hpc/group/carin/sw361/data/',
                version='2021_valid',
                target_type=target_type,
                transform=transform,
                download=True,
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
            classes = label_names_list
            # Get image paths
            for j in range(len(dataset_)):
                img_path = dataset_.imgs[j][0]
                image_paths.append(Path(img_path))

        else:
            raise SystemExit("Unknown dataset for dataloader option.")

    else:
        # Standard logic
        if args.classes_file:
            with open(args.classes_file) as fh:
                classes = [ln.strip() for ln in fh if ln.strip()]
        elif args.data_root:
            classes = infer_classes_from_imagefolder(args.data_root)
        else:
            classes = []

        if args.images_csv:
            image_paths = read_images_csv(args.images_csv, args.images_col)
        else:
            image_paths = []
            for c in classes:
                class_dir = Path(args.data_root) / c
                if not class_dir.exists():
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.webp"):
                    image_paths.extend(sorted(class_dir.glob(ext)))

    if len(classes) == 0:
        raise SystemExit("No classes found. Provide --classes-file, --data-root, or use --use-dataloader with a valid dataset.")

    print(f"Found {len(classes)} classes and {len(image_paths)} images to classify.", flush=True)

    # clf = ClipZeroShotClassifier(model_name=args.model_name, device=args.device)
    # clf.load_model()
    # clf.build_text_embeddings(classes, prompt=args.prompt)

    # preds = clf.predict_paths(image_paths, batch_size=args.batch_size, threshold=args.threshold)
    # # Prepare ground truth and classes list for output
    # ground_truths = None
    # classes_list = classes
    # if args.use_dataloader and hasattr(locals().get('target_dataset', None), 'data_list'):
    #     ground_truths = [target_dataset.tgt_labels[int(item[1])] for item in target_dataset.data_list]
    # # Attach as attributes for write_output_csv
    # write_output_csv.ground_truths = ground_truths
    # write_output_csv.classes_list = classes_list
    # write_output_csv(args.output_csv, image_paths, preds)
    # print(f"Wrote predictions to {args.output_csv}")

    # --- Post-process output CSV for accuracy and private/unknown stats ---
    import pandas as pd
    df = pd.read_csv(args.output_csv)
    # Per-class accuracy
    class_acc = {}
    for cls in classes:
        cls_rows = df[df['ground_truth'] == cls]
        if len(cls_rows) == 0:
            acc = None
        else:
            acc = (cls_rows['predicted'] == cls).sum() / len(cls_rows)
        class_acc[cls] = acc
    print("Per-class accuracy:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc if acc is not None else 'N/A'}")
    # kwn_acc = sum(list(class_acc.values())[:6]) / 6
    kwn_acc = sum((list(class_acc.values()))[:150]) / 150
    print(kwn_acc)

    # Total number of samples with private == True
    private_true_count = (df['private'] == True).sum()
    print(f"Total samples with private == True: {private_true_count}")

    # Number of samples with both private == True and unknown == True
    private_and_unknown_count = ((df['private'] == True) & (df['unknown'] == True)).sum()
    print(f"Samples with private == True and unknown == True: {private_and_unknown_count}")

    perc = private_and_unknown_count / private_true_count
    print(f"unknown accuracy", perc)

    ukn_acc = perc
    # H-score calculation
    hscore = 2 * kwn_acc * ukn_acc / (kwn_acc + ukn_acc)
    print("H score", hscore)
if __name__ == "__main__":
    main()
