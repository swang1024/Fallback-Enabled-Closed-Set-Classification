"""
Correlated Error Analysis Script

This script analyzes the correlated errors between two models:
1. opensrc_main_1_inaturalist.py (direct prompting)
2. opensrc_main_1_summary_pred_inaturalist.py (summary-based prediction)

It identifies cases where both models:
- Made incorrect predictions (compared to ground truth)
- Made the EXACT SAME incorrect prediction
- Produced a shared wrong label that is NOT in the predefined label list for that dataset

It also calculates the proportion of such correlated errors relative to the total dataset.
"""

import os
import re
import pandas as pd
from pathlib import Path
import argparse


DATASET_CONFIG = {
    "DomainNet": {
        "domains": ["Painting", "Real", "Sketch"],
        "shared_class_num": 150,
        "source_private_class_num": 50,
    },
    "VisDA": {
        "domains": ["train", "validation"],
        "shared_class_num": 6,
        "source_private_class_num": 3,
    },
    "Office": {
        "domains": ["Amazon", "Dslr", "Webcam"],
        "shared_class_num": 10,
        "source_private_class_num": 10,
    },
    "OfficeHome": {
        "domains": ["Art", "Clipart", "Product", "Realworld"],
        "shared_class_num": 10,
        "source_private_class_num": 5,
    },
}


def normalize_label(label):
    """Normalize labels for robust string comparison."""
    if pd.isna(label):
        return ""
    text = str(label).strip().strip("'\"").replace("_", " ")
    return " ".join(text.split()).lower()


def infer_metadata_from_filename(file_path):
    """Infer dataset metadata from result filename."""
    name = Path(file_path).name
    dataset = None
    for candidate in ["INaturalist", "DomainNet", "VisDA", "OfficeHome", "Office"]:
        if candidate.lower() in name.lower():
            dataset = candidate
            break

    target_domain_match = re.search(r"target_domain(\d+)", name, flags=re.IGNORECASE)
    random_seed_match = re.search(r"randomseed(\d+)", name, flags=re.IGNORECASE)
    inat_target_type_match = re.search(r"(class|phylum)_INaturalist", name, flags=re.IGNORECASE)

    return {
        "dataset": dataset,
        "target_domain_idx": int(target_domain_match.group(1)) if target_domain_match else None,
        "random_seed": int(random_seed_match.group(1)) if random_seed_match else None,
        "inaturalist_target_type": inat_target_type_match.group(1).lower() if inat_target_type_match else None,
    }


def merge_metadata(primary, secondary):
    """Merge two inferred metadata dictionaries, preferring primary when available."""
    merged = dict(primary)
    for key, value in secondary.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    return merged


def extract_source_labels_from_image_list(list_file, source_class_num):
    """Read source-class label names from image_unida_list.txt."""
    label_id_to_name = {}
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            try:
                label_id = int(parts[-1])
            except ValueError:
                continue
            if label_id >= source_class_num or label_id in label_id_to_name:
                continue
            label_id_to_name[label_id] = rel_path.split("/")[0].replace("_", " ")

    return [label_id_to_name[i] for i in range(source_class_num) if i in label_id_to_name]


def load_predefined_labels_from_dataset_files(dataset, target_domain_idx, repo_root):
    """Load predefined label list for Office/OfficeHome/VisDA/DomainNet."""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset for file-based label loading: {dataset}")

    config = DATASET_CONFIG[dataset]
    source_class_num = config["shared_class_num"] + config["source_private_class_num"]
    base_dir = repo_root / "data" / dataset

    candidate_files = []
    domains = config["domains"]
    if target_domain_idx is not None and 0 <= target_domain_idx < len(domains):
        candidate_files.append(base_dir / domains[target_domain_idx] / "image_unida_list.txt")
    candidate_files.extend(base_dir / domain / "image_unida_list.txt" for domain in domains)

    best_labels = []
    best_file = None
    for file_path in candidate_files:
        if not file_path.exists():
            continue
        labels = extract_source_labels_from_image_list(file_path, source_class_num)
        if len(labels) > len(best_labels):
            best_labels = labels
            best_file = file_path
        if len(labels) == source_class_num:
            break

    if not best_labels:
        raise ValueError(
            f"Could not load predefined labels for dataset={dataset}. "
            f"Expected image list files under: {base_dir}"
        )

    print(f"Using predefined labels from: {best_file} (count={len(best_labels)})")
    return {normalize_label(lbl) for lbl in best_labels}


def read_label_list_file(label_file):
    """Load a label list CSV/TXT and return normalized labels."""
    file_path = Path(label_file)
    if not file_path.exists():
        raise ValueError(f"Label list file not found: {label_file}")

    if file_path.suffix.lower() == ".txt":
        with open(file_path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Label list file is empty: {label_file}")
        preferred_cols = ["class label", "class_name", "class name", "label"]
        chosen_col = None
        for col in preferred_cols:
            if col in df.columns:
                chosen_col = col
                break
        if chosen_col is None:
            chosen_col = df.columns[0]
        labels = df[chosen_col].dropna().astype(str).tolist()

    normalized = {normalize_label(lbl) for lbl in labels if normalize_label(lbl)}
    if not normalized:
        raise ValueError(f"No valid labels found in label list file: {label_file}")
    return normalized


def load_inaturalist_labels_from_local_files(
    repo_root,
    target_domain_idx,
    random_seed,
    ground_truth_labels_norm,
):
    """Find and load the best matching INaturalist predefined label list from local label files."""
    search_dirs = [repo_root, repo_root / "llm_data"]
    candidate_files = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        candidate_files.extend(search_dir.glob("*INaturalist*label*.csv"))
        candidate_files.extend(search_dir.glob("*label*INaturalist*.csv"))

    candidate_files = sorted(set(candidate_files))

    filtered_candidates = []
    for file_path in candidate_files:
        name = file_path.name
        if target_domain_idx is not None and f"target_domain{target_domain_idx}" not in name:
            continue
        if random_seed is not None and f"randomseed{random_seed}" not in name:
            continue
        filtered_candidates.append(file_path)

    if not filtered_candidates:
        raise ValueError(
            "Could not locate INaturalist predefined label file. "
            "Please provide --predefined_label_file."
        )

    best_labels = None
    best_file = None
    best_score = -1
    for file_path in filtered_candidates:
        try:
            labels = read_label_list_file(file_path)
        except Exception:
            continue
        score = len(labels & ground_truth_labels_norm)
        if score > best_score or (score == best_score and best_labels is not None and len(labels) > len(best_labels)):
            best_score = score
            best_labels = labels
            best_file = file_path
        elif score == best_score and best_labels is None:
            best_score = score
            best_labels = labels
            best_file = file_path

    if not best_labels:
        raise ValueError(
            "Found INaturalist label files but could not parse usable labels. "
            "Please provide --predefined_label_file."
        )

    print(f"Using INaturalist predefined labels from: {best_file} (count={len(best_labels)}, overlap={best_score})")
    return best_labels


def load_results(direct_file, summary_file):
    """
    Load results from both direct prompting and summary prediction files.
    
    Args:
        direct_file: Path to direct prompting results CSV
        summary_file: Path to summary prediction results CSV
    
    Returns:
        df_direct: DataFrame with direct prompting results
        df_summary: DataFrame with summary prediction results
    """
    print(f"Loading direct prompting results from: {direct_file}")
    df_direct = pd.read_csv(direct_file)
    
    print(f"Loading summary prediction results from: {summary_file}")
    df_summary = pd.read_csv(summary_file)
    
    print(f"Direct prompting samples: {len(df_direct)}")
    print(f"Summary prediction samples: {len(df_summary)}")
    
    return df_direct, df_summary


def resolve_predefined_label_set(direct_file, summary_file, ground_truth_series, predefined_label_file=None):
    """Resolve predefined labels for the dataset."""
    metadata = merge_metadata(
        infer_metadata_from_filename(direct_file),
        infer_metadata_from_filename(summary_file),
    )
    dataset = metadata["dataset"]
    if dataset is None:
        raise ValueError(
            "Could not infer dataset from input filenames. "
            "Use filenames that contain dataset names (e.g., DomainNet/VisDA/INaturalist), "
            "or provide --predefined_label_file."
        )

    if predefined_label_file:
        labels = read_label_list_file(predefined_label_file)
        print(f"Using predefined labels from user-provided file: {predefined_label_file} (count={len(labels)})")
        return labels, metadata

    repo_root = Path(__file__).resolve().parent
    ground_truth_labels_norm = {normalize_label(x) for x in ground_truth_series if normalize_label(x)}

    if dataset == "INaturalist":
        labels = load_inaturalist_labels_from_local_files(
            repo_root=repo_root,
            target_domain_idx=metadata["target_domain_idx"],
            random_seed=metadata["random_seed"],
            ground_truth_labels_norm=ground_truth_labels_norm,
        )
    else:
        labels = load_predefined_labels_from_dataset_files(
            dataset=dataset,
            target_domain_idx=metadata["target_domain_idx"],
            repo_root=repo_root,
        )

    return labels, metadata


def analyze_correlated_errors(df_direct, df_summary, direct_file, summary_file, predefined_label_file=None):
    """
    Analyze correlated errors between the two models.
    
    Args:
        df_direct: DataFrame with direct prompting results
        df_summary: DataFrame with summary prediction results
        direct_file: Path to direct prompting results CSV
        summary_file: Path to summary prediction results CSV
        predefined_label_file: Optional path to label list file
    
    Returns:
        dict with analysis results
    """
    # Merge dataframes on 'idx' to align samples
    df_merged = pd.merge(
        df_direct, 
        df_summary, 
        on='idx', 
        suffixes=('_direct', '_summary')
    )
    
    print(f"\nMerged dataset size: {len(df_merged)}")
    
    # Clean class names (remove quotes if present)
    df_merged['predicted_direct'] = df_merged['predicted class name_direct'].fillna("").astype(str).str.strip().str.strip("'\"")
    df_merged['predicted_summary'] = df_merged['predicted class name_summary'].fillna("").astype(str).str.strip().str.strip("'\"")
    df_merged['ground_truth'] = df_merged['ground truth_direct'].fillna("").astype(str).str.strip()
    df_merged['predicted_direct_norm'] = df_merged['predicted_direct'].map(normalize_label)
    df_merged['predicted_summary_norm'] = df_merged['predicted_summary'].map(normalize_label)
    df_merged['ground_truth_norm'] = df_merged['ground_truth'].map(normalize_label)
    
    # Verify ground truth is the same in both (sanity check)
    if not (df_merged['ground truth_direct'] == df_merged['ground truth_summary']).all():
        print("WARNING: Ground truth labels don't match between files!")

    predefined_label_set, metadata = resolve_predefined_label_set(
        direct_file=direct_file,
        summary_file=summary_file,
        ground_truth_series=df_merged['ground_truth'],
        predefined_label_file=predefined_label_file,
    )
    df_merged['predicted_direct_in_predefined'] = df_merged['predicted_direct_norm'].isin(predefined_label_set)
    df_merged['predicted_summary_in_predefined'] = df_merged['predicted_summary_norm'].isin(predefined_label_set)
    
    # Identify errors for each model
    df_merged['direct_wrong'] = df_merged['predicted_direct'] != df_merged['ground_truth']
    df_merged['summary_wrong'] = df_merged['predicted_summary'] != df_merged['ground_truth']

    # Both wrong and the same wrong prediction
    df_merged['same_wrong_prediction'] = (
        df_merged['direct_wrong'] & 
        df_merged['summary_wrong'] & 
        (df_merged['predicted_direct_norm'] == df_merged['predicted_summary_norm'])
    )

    # Updated correlated error: same wrong prediction and that label is outside predefined list
    df_merged['correlated_error'] = (
        df_merged['same_wrong_prediction'] &
        (~df_merged['predicted_direct_in_predefined']) &
        (~df_merged['predicted_summary_in_predefined'])
    )
    
    # Calculate statistics
    total_samples = len(df_merged)
    direct_errors = df_merged['direct_wrong'].sum()
    summary_errors = df_merged['summary_wrong'].sum()
    both_wrong = (df_merged['direct_wrong'] & df_merged['summary_wrong']).sum()
    same_wrong_prediction_total = df_merged['same_wrong_prediction'].sum()
    correlated_errors = df_merged['correlated_error'].sum()
    
    # Additional categories
    both_correct = (~df_merged['direct_wrong'] & ~df_merged['summary_wrong']).sum()
    direct_wrong_only = (df_merged['direct_wrong'] & ~df_merged['summary_wrong']).sum()
    summary_wrong_only = (~df_merged['direct_wrong'] & df_merged['summary_wrong']).sum()
    both_wrong_diff_pred = both_wrong - same_wrong_prediction_total
    same_wrong_in_predefined = same_wrong_prediction_total - correlated_errors
    
    results = {
        'total_samples': total_samples,
        'dataset': metadata['dataset'],
        'target_domain_idx': metadata['target_domain_idx'],
        'random_seed': metadata['random_seed'],
        'predefined_label_count': len(predefined_label_set),
        'direct_errors': direct_errors,
        'summary_errors': summary_errors,
        'both_wrong': both_wrong,
        'same_wrong_prediction_total': same_wrong_prediction_total,
        'same_wrong_in_predefined': same_wrong_in_predefined,
        'correlated_errors': correlated_errors,
        'both_wrong_diff_pred': both_wrong_diff_pred,
        'both_correct': both_correct,
        'direct_wrong_only': direct_wrong_only,
        'summary_wrong_only': summary_wrong_only,
        'direct_accuracy': (total_samples - direct_errors) / total_samples,
        'summary_accuracy': (total_samples - summary_errors) / total_samples,
        'correlated_error_rate': correlated_errors / total_samples,
        'correlated_error_proportion_of_both_wrong': correlated_errors / both_wrong if both_wrong > 0 else 0,
        'correlated_error_proportion_of_same_wrong': correlated_errors / same_wrong_prediction_total if same_wrong_prediction_total > 0 else 0,
    }
    
    return results, df_merged


def print_results(results):
    """
    Print analysis results in a readable format.
    
    Args:
        results: Dictionary with analysis results
    """
    print("\n" + "="*70)
    print("CORRELATED ERROR ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nDataset Size: {results['total_samples']}")
    print(f"Dataset: {results['dataset']}")
    if results['target_domain_idx'] is not None:
        print(f"Target Domain Index: {results['target_domain_idx']}")
    if results['random_seed'] is not None:
        print(f"Random Seed: {results['random_seed']}")
    print(f"Predefined Label Count: {results['predefined_label_count']}")
    
    print(f"\n--- Individual Model Performance ---")
    print(f"Direct Prompting Errors: {results['direct_errors']} ({results['direct_errors']/results['total_samples']*100:.2f}%)")
    print(f"Direct Prompting Accuracy: {results['direct_accuracy']*100:.2f}%")
    print(f"\nSummary Prediction Errors: {results['summary_errors']} ({results['summary_errors']/results['total_samples']*100:.2f}%)")
    print(f"Summary Prediction Accuracy: {results['summary_accuracy']*100:.2f}%")
    
    print(f"\n--- Error Analysis ---")
    print(f"Both Models Correct: {results['both_correct']} ({results['both_correct']/results['total_samples']*100:.2f}%)")
    print(f"Only Direct Wrong: {results['direct_wrong_only']} ({results['direct_wrong_only']/results['total_samples']*100:.2f}%)")
    print(f"Only Summary Wrong: {results['summary_wrong_only']} ({results['summary_wrong_only']/results['total_samples']*100:.2f}%)")
    print(f"Both Models Wrong: {results['both_wrong']} ({results['both_wrong']/results['total_samples']*100:.2f}%)")
    
    print(f"\n--- Correlated Error Analysis ---")
    print(f"Same Wrong Prediction (both wrong, same label): {results['same_wrong_prediction_total']}")
    print(f"Same Wrong Prediction within Predefined List: {results['same_wrong_in_predefined']}")
    print(f"Correlated Errors (same wrong label NOT in predefined list): {results['correlated_errors']}")
    print(f"Proportion of Total Dataset: {results['correlated_error_rate']*100:.2f}%")
    print(f"Proportion of Cases Where Both Wrong: {results['correlated_error_proportion_of_both_wrong']*100:.2f}%")
    print(f"Proportion of Same-Wrong Cases Outside Predefined List: {results['correlated_error_proportion_of_same_wrong']*100:.2f}%")
    print(f"\nBoth Wrong with Different Predictions: {results['both_wrong_diff_pred']} ({results['both_wrong_diff_pred']/results['total_samples']*100:.2f}%)")
    
    print("\n" + "="*70)


def save_correlated_error_samples(df_merged, output_file):
    """
    Save samples with correlated errors to a CSV file for inspection.
    
    Args:
        df_merged: Merged DataFrame with error annotations
        output_file: Path to save the correlated error samples
    """
    correlated_df = df_merged[df_merged['correlated_error']].copy()
    
    # Select relevant columns
    output_cols = [
        'idx',
        'ground_truth',
        'predicted_direct',
        'predicted_summary',
        'predicted_direct_in_predefined',
        'predicted_summary_in_predefined',
        'private_direct',
        'unknown_direct',
        'unknown_summary',
        'img url_direct'
    ]
    
    correlated_df = correlated_df[output_cols]
    correlated_df.columns = [
        'idx',
        'ground_truth',
        'direct_prediction',
        'summary_prediction',
        'direct_prediction_in_predefined',
        'summary_prediction_in_predefined',
        'is_private',
        'direct_unknown',
        'summary_unknown',
        'image_path'
    ]
    
    correlated_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(correlated_df)} correlated error samples to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze correlated errors between direct and summary prediction models')
    parser.add_argument('--direct_file', type=str, required=True,
                        help='Path to direct prompting results CSV file')
    parser.add_argument('--summary_file', type=str, required=True,
                        help='Path to summary prediction results CSV file')
    parser.add_argument('--output_dir', type=str, default='correlated_error_analysis',
                        help='Directory to save output files')
    parser.add_argument('--predefined_label_file', type=str, default=None,
                        help='Optional path to CSV/TXT file containing predefined label list')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    df_direct, df_summary = load_results(args.direct_file, args.summary_file)
    
    # Analyze correlated errors
    results, df_merged = analyze_correlated_errors(
        df_direct,
        df_summary,
        direct_file=args.direct_file,
        summary_file=args.summary_file,
        predefined_label_file=args.predefined_label_file,
    )
    
    # Print results
    print_results(results)
    
    # Save correlated error samples
    base_name = Path(args.direct_file).stem
    output_file = os.path.join(args.output_dir, f'{base_name}_correlated_errors.csv')
    save_correlated_error_samples(df_merged, output_file)
    
    # Save summary statistics
    stats_file = os.path.join(args.output_dir, f'{base_name}_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("CORRELATED ERROR ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Direct File: {args.direct_file}\n")
        f.write(f"Summary File: {args.summary_file}\n\n")
        f.write(f"Dataset Size: {results['total_samples']}\n\n")
        f.write(f"Dataset: {results['dataset']}\n")
        if results['target_domain_idx'] is not None:
            f.write(f"Target Domain Index: {results['target_domain_idx']}\n")
        if results['random_seed'] is not None:
            f.write(f"Random Seed: {results['random_seed']}\n")
        f.write(f"Predefined Label Count: {results['predefined_label_count']}\n\n")
        f.write(f"Direct Prompting Accuracy: {results['direct_accuracy']*100:.2f}%\n")
        f.write(f"Summary Prediction Accuracy: {results['summary_accuracy']*100:.2f}%\n\n")
        f.write(f"Both Models Correct: {results['both_correct']} ({results['both_correct']/results['total_samples']*100:.2f}%)\n")
        f.write(f"Both Models Wrong: {results['both_wrong']} ({results['both_wrong']/results['total_samples']*100:.2f}%)\n")
        f.write(f"Same Wrong Prediction (both wrong, same label): {results['same_wrong_prediction_total']}\n")
        f.write(f"Same Wrong Prediction within Predefined List: {results['same_wrong_in_predefined']}\n")
        f.write(f"Correlated Errors (same wrong label NOT in predefined list): {results['correlated_errors']} ({results['correlated_error_rate']*100:.2f}%)\n")
        f.write(f"Proportion of Cases Where Both Wrong: {results['correlated_error_proportion_of_both_wrong']*100:.2f}%\n")
        f.write(f"Proportion of Same-Wrong Cases Outside Predefined List: {results['correlated_error_proportion_of_same_wrong']*100:.2f}%\n")
    
    print(f"\nSaved statistics to: {stats_file}")


if __name__ == "__main__":
    main()
