import itertools

from project.utils.strs import plms

MODEL_NAMES = [
    # "chandar-lab/AMPLIFY_120M",
    # "chandar-lab/AMPLIFY_350M",
    # "chandar-lab/SaAMPLIFY_120M",
    # "chandar-lab/SaAMPLIFY_350M",
    # "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    # "facebook/esm2_t30_150M_UR50D",
    # "facebook/esm2_t33_650M_UR50D",
    # "Lolalb/MILA_U100_baseline_100000",
    # "Lolalb/MILA_U100_baseline_200000",
    # "Lolalb/MILA_U100_baseline_300000",
    # "Lolalb/MILA_U100_baseline_400000",
    # "Lolalb/MILA_U100_baseline_500000",
    # "Lolalb/MILA_U100_baseline_600000",
    # "Lolalb/MILA_U100_baseline_700000",
    # "Lolalb/MILA_U100_baseline_800000",
    # "Lolalb/MILA_U100_baseline_900000",
    # "Lolalb/MILA_U100_baseline_1000000",
    # "Lolalb/MILA_U50_baseline_100000",
    # "Lolalb/MILA_U50_baseline_200000",
    # "Lolalb/MILA_U50_baseline_300000",
    # "Lolalb/MILA_U50_baseline_400000",
    # "Lolalb/MILA_U50_baseline_500000",
    # "Lolalb/MILA_U50_baseline_600000",
    # "Lolalb/MILA_U50_baseline_700000",
    # "Lolalb/MILA_U50_baseline_800000",
    # "Lolalb/MILA_U50_baseline_900000",
    # "Lolalb/MILA_U50_baseline_1000000"
]

models_to_hiddens = {v['full_name']: v['num_hiddens'] for v in plms.values()}

MODEL_LAYERS = {model_name: list(range(models_to_hiddens[model_name])) for model_name in MODEL_NAMES
}

TARGET_DATASETS = [
    # 'prot_param',
    'uniprot_peptide',
    # 'interpro_conserved_site',
    # 'biomap_localization_prediction',
    # 'interpro_repeat',
    # 'biomap_ssp_q3',
    # 'biomap_ssp_q8',
    # 'uniprot_secondary_structure',
    # 'interpro_domain',
    # 'interpro_family',
    # 'interpro_homologous_superfamily',
    # 'uniprot_functional_sites',
    # 'interpro_binding_site',
    # 'biomap_metal_ion_binding',
    # 'interpro_active_site',
    # 'uniprot_topology',
    # 'uniprot_post_translational_modification',
    # 'uniprot_phosphorylation',
    # 'uniprot_lipidation',
    # 'GO_cc',
    # 'GO_mf',
    # 'GO_bp',
]

# # Add the other datasets - different splits
# TARGET_DATASETS_SPLITS = [f"{ds}_split1" for ds in TARGET_DATASETS] # + [f"{ds}_split2" for ds in TARGET_DATASETS]

# TARGET_DATASETS = TARGET_DATASETS_SPLITS #TARGET_DATASETS +

add_cutoff = True
if add_cutoff:
    TARGET_DATASETS = [f"{dataset}_512_cutoff" for dataset in TARGET_DATASETS]

all_configs = []

for model_name in MODEL_NAMES:

    # 1. Generate the list of layer strings for this model
    layers = MODEL_LAYERS.get(model_name)

    # 2. Create the cross-product of (Layer, Dataset)
    permutations = itertools.product(layers, TARGET_DATASETS)

    # 3. Format the final config string: MODEL_NAME LAYER_TO_USE TARGET_DATASET
    for layer, dataset in permutations:
        # Note: The model name contains slashes, which is fine for the array entry
        # but requires special handling when passed to read -r -a
        config_string = f'"{model_name}" {layer} {dataset}'
        all_configs.append(config_string)

# Save the configurations to a file
with open("linear_probe_configs.txt", "w") as f:
    for config in all_configs:
        f.write(f"{config}\n")

print(f"Generated {len(all_configs)} total configurations in job_configs.txt")
print(f"Submit with: sbatch --array=0-{len(all_configs) - 1} your_script_name.sh")
