#!/bin/bash

#SBATCH -e %A_%a_opda.err

#SBATCH -o %A_%a_opda.out

#SBATCH --account=carin

#SBATCH --partition=scavenger-gpu --gres=gpu:a5000:1

#SBATCH -c 4

#SBATCH --mem=80G

set -euo pipefail

cd /hpc/group/carin/sw361/ChatGPT_exp

# Use local cache directories for HF artifacts.
export HF_HOME=/hpc/group/carin/sw361/ChatGPT_exp
export HF_HUB_CACHE=/hpc/group/carin/sw361/ChatGPT_exp/hub
export TRANSFORMERS_CACHE=/hpc/group/carin/sw361/ChatGPT_exp/hub

# # Default to Llama; requires HF_TOKEN and approved access to the gated model.
# export MODEL_NAMES="${MODEL_NAMES:-llama3.2-vision}"

# if [[ "${MODEL_NAMES}" == *"llama3.2-vision"* || "${MODEL_NAMES}" == *"meta-llama/Llama-3.2-11B-Vision-Instruct"* ]]; then
#   if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
#     echo "ERROR: Llama gated model selected but no Hugging Face token found."
#     echo "Set HF_TOKEN in your environment before sbatch, e.g.:"
#     echo "  export HF_TOKEN=hf_xxx"
#     echo "Also make sure your HF account has approved access to:"
#     echo "  https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct"
#     exit 2
#   fi
# fi

# export CUDA_VISIBLE_DEVICES=$gpuid

# python correlated_error_eval.py \
#      --direct_file llm_data/VisDA_target_domain1_qwen_v8.csv \
#      --summary_file llm_data/VisDA_target_domain1_qwen_v13_summary_pred.csv \
#      --output_dir correlated_error_analysis/output

python paradox_analysis.py 

# python self_consistency.py
# python self_consistency_rephrase.py
# python self-consistency-baseline-metrics.py
# python cost_computation.py \
#    --dataset DomainNet \
#    --target-domain-index 1 \
#    --model gpt-4o-mini \
#    --max-images 100 \
#    --save-per-request

# python cost_computation.py \
#    --dataset DomainNet \
#    --target-domain-index 1 \
#    --model gemini-2.0-flash \
#    --max-images 100 \
#    --save-per-request

# python test_scoring.py
# python scoring_performance_analysis.py
# python performance_analysis.py
# python test_label_explanation.py
# python scoring_performance_analysis_summary_vs_label.py
# python scoring_performance_analysis_summary_vs_label_kwn_vs_unk.py
# python scoring_performance_analysis_summary_vs_llm_preds.py
# python test_summary.py

# python main_1.py
# python main_2.py
# python main_3.py
# python main_4.py

# python main_3_test_.py

# python opensrc_main_1.py
# python opensrc_main_2.py
# python opensrc_main_3.py

# python opensrc_main_1_summary_feats.py
# python opensrc_main_1_summary.py
# python opensrc_main_1_inaturalist.py
# python opensrc_main_1_summary_pred_inaturalist.py
# python opensrc_main_2_inaturalist.py
# python opensrc_main_3_inaturalist.py
# python opensrc_main_3.5_eval_plot.py
# python opensrc_main_3_summary_inaturalist.py
# python opensrc_main_3_summary_pred.py
# python opensrc_main_3_summary_pred_ablation.py
# python opensrc_main_4_inaturalist.py
# python opensrc_main_4.py

# python main_concat.py

# python convert_hf_model.py --dest_fn $/hpc/group/carin/sw361/blip3.pt

# python opensrc_test_label_explanation.py

# python plot_barplot.py
# python plot_confusion_matrix.py
# python plot_radar_plots.py
# python plot_stacked_barplot.py

# python preliminary_performance_analysis.py
# python assessment.py

# write code in correlated_error_eval.py to check the results from @opensrc_main_1_inaturalist.py and @opensrc_main_1_summary_pred_inaturalist.py are both wrong predictions (compared to ground truth), but the wrong predictions are the exact same labels
# and what proportion is the sample size compared to the whole dataset

# write code in correlated_error_eval.py to ALSO check if both wrong predictions (that are the exact same labels) are NOT within the predefined label list for each dataset, and what proportion is the sample size compared to the whole dataset, and update the correlated_error_rate
# check what the predefined label list is for each dataset in the code for @main_1_direct_prompting.py and @opensrc_main_1_inaturalist.py


# For each image, compute the model's log-probability of the ground-truth label under both arms:

# logp_visual = log P(GT label | image, prompt) from Pass 1
# logp_textual = log P(GT label | summary, prompt) from Pass 3

# Then:

# gap = logp_textual − logp_visual