# Correlated Error Analysis

This tool analyzes correlated errors between two models:
- **Direct Prompting Model** (`opensrc_main_1_inaturalist.py`)
- **Summary-based Prediction Model** (`opensrc_main_1_summary_pred_inaturalist.py`)

## What It Does

The script identifies cases where:
1. Both models made **incorrect predictions** (compared to ground truth)
2. Both models made the **exact same wrong prediction**
3. The shared wrong prediction is **outside the dataset's predefined label list**

It also calculates the proportion of such correlated errors relative to the total dataset.

## Usage

### Basic Command

```bash
python correlated_error_eval.py \
    --direct_file <path_to_direct_prompting_results.csv> \
    --summary_file <path_to_summary_prediction_results.csv> \
    --output_dir <output_directory> \
    [--predefined_label_file <path_to_label_list_csv_or_txt>]
```

### Example Usage

```bash
# Llama model analysis
python correlated_error_eval.py \
    --direct_file llm_data/class_INaturalist_target_domain1_llama_v8_randomseed0.csv \
    --summary_file llm_data/class_INaturalist_target_domain1_llama_v8_summary_pred.csv \
    --output_dir correlated_error_analysis/llama_domain1_v8

# Qwen model analysis
python correlated_error_eval.py \
    --direct_file llm_data/class_INaturalist_target_domain1_qwen_v8_randomseed0.csv \
    --summary_file llm_data/class_INaturalist_target_domain1_qwen_v8_randomseed0_summary_pred.csv \
    --output_dir correlated_error_analysis/qwen_domain1_v8
```

### Batch Analysis

Use the provided shell script to analyze multiple configurations:

```bash
./run_correlated_error_analysis.sh
```

## Input Files

Both CSV files should have the following columns:
- `idx`: Sample index
- `ground truth`: True class label
- `predicted class name`: Model's predicted class
- `private`: Whether the class is private/unknown
- `unknown`: Whether the model flagged it as unknown
- `img url`: Path to the image file

## Output Files

The script generates two output files in the specified output directory:

### 1. Correlated Errors CSV
**File**: `<base_name>_correlated_errors.csv`

Contains all samples where both models made the same wrong prediction **outside** the predefined label list:
- `idx`: Sample index
- `ground_truth`: True class label
- `direct_prediction`: Direct model's (wrong) prediction
- `summary_prediction`: Summary model's (wrong) prediction
- `direct_prediction_in_predefined`: Whether direct prediction is in predefined list
- `summary_prediction_in_predefined`: Whether summary prediction is in predefined list
- `is_private`: Whether the class is private
- `direct_unknown`: Whether direct model flagged as unknown
- `summary_unknown`: Whether summary model flagged as unknown
- `image_path`: Path to the image

### 2. Statistics Summary
**File**: `<base_name>_statistics.txt`

Contains summary statistics:
- Dataset size
- Individual model accuracies
- Error breakdown by category
- Correlated error counts and proportions

## Metrics Explained

### Key Metrics

1. **Correlated Errors**: Number of samples where both models made the exact same wrong prediction **and** that predicted label is **not** in the dataset's predefined label list

2. **Proportion of Total Dataset**: `correlated_errors / total_samples`
   - What percentage of the entire dataset has this out-of-list correlated error

3. **Proportion of Cases Where Both Wrong**: `correlated_errors / both_wrong`
   - Among all cases where both models were wrong, what percentage are out-of-list correlated errors

4. **Proportion of Same-Wrong Cases Outside Predefined List**: `correlated_errors / same_wrong_prediction_total`
   - Among same-wrong cases, what percentage are outside the predefined label list

### Error Categories

- **Both Correct**: Both models predicted correctly
- **Only Direct Wrong**: Only the direct model was wrong
- **Only Summary Wrong**: Only the summary model was wrong
- **Both Wrong with Same Prediction**: Same-wrong cases (split into in-list vs out-of-list)
- **Both Wrong with Different Predictions**: Both wrong but different mistakes

## Example Output

```
======================================================================
CORRELATED ERROR ANALYSIS RESULTS
======================================================================

Dataset Size: 60050

--- Individual Model Performance ---
Direct Prompting Errors: 46775 (77.89%)
Direct Prompting Accuracy: 22.11%

Summary Prediction Errors: 40181 (66.91%)
Summary Prediction Accuracy: 33.09%

--- Error Analysis ---
Both Models Correct: 12352 (20.57%)
Only Direct Wrong: 7517 (12.52%)
Only Summary Wrong: 923 (1.54%)
Both Models Wrong: 39258 (65.38%)

--- Correlated Error Analysis ---
Correlated Errors (same wrong prediction): 2667
Proportion of Total Dataset: 4.44%
Proportion of Cases Where Both Wrong: 6.79%

Both Wrong with Different Predictions: 36591 (60.93%)
======================================================================
```

## Interpretation

The analysis shows:
- **4.44%** of all samples have correlated errors (both models make the same mistake)
- When both models are wrong (65.38% of cases), they agree on the wrong answer only **6.79%** of the time
- This suggests the models make mostly **independent errors**, with relatively low correlation

## Requirements

- Python 3.6+
- pandas
- numpy

Install dependencies:
```bash
pip install pandas numpy
```

## Files

- `correlated_error_eval.py`: Main analysis script
- `run_correlated_error_analysis.sh`: Batch analysis examples
- `README_correlated_error.md`: This documentation
