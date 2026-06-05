#!/bin/bash

#SBATCH -e %A_%a_opda.err

#SBATCH -o %A_%a_opda.out

#SBATCH -A scavenger-h200

#SBATCH -p scavenger-h200 --gres=gpu:h200:1  -t 24:00:00

#SBATCH -c 12

#SBATCH --mem=120G
#SBATCH --export=ALL

# set -euo pipefail

# cd /hpc/group/carin/sw361/ChatGPT_exp

# # Use local cache directories for HF artifacts.
# export HF_HOME=/hpc/group/carin/sw361/ChatGPT_exp
# export HF_HUB_CACHE=/hpc/group/carin/sw361/ChatGPT_exp/hub
# export TRANSFORMERS_CACHE=/hpc/group/carin/sw361/ChatGPT_exp/hub

# # Default to Llama; requires HF token and approved access to the gated model.
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

# python self_consistency.py
# python self_consistency_rephrase.py

# python self-consistency-baseline-metrics.py

# python paradox_analysis.py 

# python correlated_error_eval.py \
#      --direct_file llm_data/phylum_INaturalist_target_domain1_4o-mini_v8_randomseed2.csv \
#      --summary_file llm_data/phylum_INaturalist_target_domain1_qwen_v13_summary_pred_randomseed2.csv \
#      --output_dir correlated_error_analysis/output

# python test_scoring.py
# python scoring_performance_analysis.py
# python performance_analysis.py
# python test_label_explanation.pyphylum_INaturalist
# python scoring_performance_analysis_summary_vs_label.py
# python scoring_performance_analysis_summary_vs_label_kwn_vs_unk.py
# python scoring_performance_analysis_summary_vs_llm_preds.py
# python test_summary.py

# python main_1.py
# python main_2.py
# python main_1_direct_prompting.py 
# python main_1_direct_prompting_gemini.py 
# python main_2_summary_gen_gemini.py
# python main_2.5_summary_pred_gemini.py
# python main_3.py
# python main_3_summary_pred.py
# python main_4.py
# python main_debug.py

# python main_1_direct_prompting_gemini.py 
# python main_2_summary_gen_gemini.py
# python main_2.5_summary_pred_gemini.py
# python main_3_summary_pred.py

# python main_3_test_.py

# python opensrc_main_1.py
# python opensrc_main_2.py
# python opensrc_main_3.py

# python opensrc_main_1_summary_feats.py
# python opensrc_main_1_summary.py
# python opensrc_main_1_inaturalist.py
# python opensrc_main_1_summary_pred_inaturalist.py
# python opensrc_main_1_summary_pred.py
# python opensrc_main_2_inaturalist.py
# python opensrc_main_2.py
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
# python plot_grid_stacked_bars.py

# python preliminary_performance_analysis.py
