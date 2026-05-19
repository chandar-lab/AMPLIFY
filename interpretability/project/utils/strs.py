"""String constants and configuration paths used throughout the interpretability project.

This module centralizes file paths, model configurations, and other string constants
to ensure consistency across the codebase.
"""
from pathlib import Path

# Random seed for reproducibility across experiments
SEED = 184502
# Maximum protein sequence length (in amino acids) for processing
PROTEIN_LENGTH_CUTOFF = 1024

# Base directory paths for data, results, and models
base_dir = Path('/home/mila/s/shawn.whitfield/scratch')
data_dir = base_dir / 'data'
mmseqs_dir = data_dir / 'mmseqs_clusters'
model_dir = base_dir / 'models'
results_dir = base_dir / 'results'
linear_probe_model_dir = model_dir / 'linear_probes'
linear_probe_results_dir = results_dir / 'linear_probing'
linear_probe_metrics_dir = linear_probe_results_dir / 'metrics'
linear_probe_figures_dir = linear_probe_results_dir / 'figures'
linear_probe_compiled_results_dir = linear_probe_results_dir / 'compiled_results'
knn_results_dir = results_dir / 'knns'
knn_figures_dir = knn_results_dir / 'figures'
knn_compiled_results_dir = knn_results_dir / 'compiled_results'
pca_results_dir = results_dir / 'pca'
pca_figures_dir = pca_results_dir / 'figures'
pca_compiled_results_dir = pca_results_dir / 'compiled_results'
embedding_dir = data_dir / 'embeddings'
interventions_dir = base_dir / 'interventions'
interventions_figures_dir = interventions_dir / 'figures'
interventions_results_dir = interventions_dir / 'results'
interventions_compiled_results_dir = interventions_dir / 'compiled_results'

# Dictionary mapping model shorthand names to their HuggingFace identifiers and architecture details
# 'num_hiddens' indicates the number of transformer layers (excluding embedding layer)
plms = {
    'esm2_8m': {'full_name': "facebook/esm2_t6_8M_UR50D", 'num_hiddens': 6},  # 6 hidden layers + embedding layer
    'esm2_35m': {'full_name': "facebook/esm2_t12_35M_UR50D",'num_hiddens': 12}, # 12 hidden layers + embedding layer
    'esm2_150m': {'full_name': "facebook/esm2_t30_150M_UR50D",'num_hiddens': 30}, # 30 hidden layers + embedding layer
    'esm2_650m': {'full_name': "facebook/esm2_t33_650M_UR50D", 'num_hiddens': 33},# 33 hidden layers + embedding layer
    'amplify_120m': {'full_name': "chandar-lab/AMPLIFY_120M", 'num_hiddens': 24},# 24 hidden layers, 2048 max position embeddings, 640 hidden size https://huggingface.co/nvidia/AMPLIFY_120M
    'mila_amplify_120m_100000': {'full_name': "Lolalb/MILA_U100_baseline_100000", 'num_hiddens': 24},
    'mila_amplify_120m_200000': {'full_name': "Lolalb/MILA_U100_baseline_200000", 'num_hiddens': 24},
    'mila_amplify_120m_300000': {'full_name': "Lolalb/MILA_U100_baseline_300000", 'num_hiddens': 24},
    'mila_amplify_120m_400000': {'full_name': "Lolalb/MILA_U100_baseline_400000", 'num_hiddens': 24},
    'mila_amplify_120m_500000': {'full_name': "Lolalb/MILA_U100_baseline_500000", 'num_hiddens': 24},
    'mila_amplify_120m_600000': {'full_name': "Lolalb/MILA_U100_baseline_600000", 'num_hiddens': 24},
    'mila_amplify_120m_700000': {'full_name': "Lolalb/MILA_U100_baseline_700000", 'num_hiddens': 24},
    'mila_amplify_120m_800000': {'full_name': "Lolalb/MILA_U100_baseline_800000", 'num_hiddens': 24},
    'mila_amplify_120m_900000': {'full_name': "Lolalb/MILA_U100_baseline_900000", 'num_hiddens': 24},
    'mila_amplify_120m_1000000': {'full_name': "Lolalb/MILA_U100_baseline_1000000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_100000': {'full_name': "Lolalb/MILA_U50_baseline_100000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_200000': {'full_name': "Lolalb/MILA_U50_baseline_200000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_300000': {'full_name': "Lolalb/MILA_U50_baseline_300000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_400000': {'full_name': "Lolalb/MILA_U50_baseline_400000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_500000': {'full_name': "Lolalb/MILA_U50_baseline_500000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_600000': {'full_name': "Lolalb/MILA_U50_baseline_600000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_700000': {'full_name': "Lolalb/MILA_U50_baseline_700000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_800000': {'full_name': "Lolalb/MILA_U50_baseline_800000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_900000': {'full_name': "Lolalb/MILA_U50_baseline_900000", 'num_hiddens': 24},
    'mila_amplify_120m_u50_1000000': {'full_name': "Lolalb/MILA_U50_baseline_1000000", 'num_hiddens': 24},
    'amplify_350m': {'full_name': "chandar-lab/AMPLIFY_350M", 'num_hiddens': 32},# 32 hidden layers, 2048 max position embeddings, 960 hidden size https://huggingface.co/nvidia/AMPLIFY_350M
    'samplify_120m': {'full_name': "chandar-lab/SaAMPLIFY_120M",'num_hiddens': 24},
    'samplify_350m': {'full_name': "chandar-lab/SaAMPLIFY_350M", 'num_hiddens': 32},
          }