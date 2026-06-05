#!/bin/bash

#SBATCH -e %A_%a_opda.err

#SBATCH -o %A_%a_opda.out

#Account: institution_h200

#SBATCH --partition=gpu --gres=gpu:h200

#SBATCH -c 4

#SBATCH --mem=80G

# export CUDA_VISIBLE_DEVICES=$gpuid

python self_consistency.py
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