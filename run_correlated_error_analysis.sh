#!/bin/bash

# Example script to run correlated error analysis
# Adjust the file paths according to your actual output files

# Example 1: Llama model, domain 1, randomseed 0
python correlated_error_eval.py \
    --direct_file llm_data/class_INaturalist_target_domain1_llama_v8_randomseed0.csv \
    --summary_file llm_data/class_INaturalist_target_domain1_llama_v8_summary_pred.csv \
    --output_dir correlated_error_analysis/llama_domain1_v8

# Example 2: Qwen model, domain 1, randomseed 0
python correlated_error_eval.py \
    --direct_file llm_data/class_INaturalist_target_domain1_qwen_v8_randomseed0.csv \
    --summary_file llm_data/class_INaturalist_target_domain1_qwen_v8_randomseed0_summary_pred.csv \
    --output_dir correlated_error_analysis/qwen_domain1_v8

# Example 3: Llama model with v13, domain 1, randomseed 0
python correlated_error_eval.py \
    --direct_file llm_data/class_INaturalist_target_domain1_llama_v13_randomseed0.csv \
    --summary_file llm_data/class_INaturalist_target_domain1_llama_v13_summary_pred_randomseed0.csv \
    --output_dir correlated_error_analysis/llama_domain1_v13

echo "Analysis complete! Check the correlated_error_analysis directory for results."
