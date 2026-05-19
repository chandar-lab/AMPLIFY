from project.utils.strs import plms
import itertools

MODEL_NAMES = [
    "chandar-lab/AMPLIFY_120M",
    "chandar-lab/AMPLIFY_350M",
    "chandar-lab/SaAMPLIFY_120M",
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
]

models_to_hiddens = {v['full_name']: v['num_hiddens'] for v in plms.values()}

# def get_model_layers(models_to_hiddens, MODEL_NAMES, interval:int=2):
#     MODEL_LAYERS = {}
#     for model_name in MODEL_NAMES:
#         # Get the total number of layers.
#         # Assuming models_to_hiddens[model_name] gives the index of the last layer.
#         last_layer = models_to_hiddens[model_name]
        
#         # 1. Get the 0th and every second layer
#         # The stop value is last_layer + 1 to ensure last_layer is included if it's even.
#         layers = list(range(0, last_layer + 1, interval))
        
#         # 2. Explicitly add the last layer if it was missed (i.e., if last_layer is odd)
#         if last_layer not in layers:
#             layers.append(last_layer)
        
#         # 3. Ensure the list is sorted and unique (optional, but good practice)
#         layers = sorted(list(set(layers)))
        
#         MODEL_LAYERS[model_name] = layers
        
#     return MODEL_LAYERS

# MODEL_LAYERS = get_model_layers(models_to_hiddens=models_to_hiddens, MODEL_NAMES=MODEL_NAMES, interval=1)

MODEL_LAYERS = {model_name: list(range(models_to_hiddens[model_name])) for model_name in MODEL_NAMES
}

# MODEL_LAYERS = {
#     "chandar-lab/AMPLIFY_120M": [0,5,10,15,20,24],
#     "chandar-lab/AMPLIFY_350M": [0,5,10,15,20,25,30,32],
#     "chandar-lab/SaAMPLIFY_120M": [0,5,10,15,20,24],
#     "facebook/esm2_t6_8M_UR50D": [0,2,4,6],
#     "facebook/esm2_t12_35M_UR50D": [0,5,10,12],
#     "facebook/esm2_t30_150M_UR50D": [0,5,10,15,20,25,30],
#     "facebook/esm2_t33_650M_UR50D": [0,5,10,15,20,25,30,33],
# }

TARGET_DATASETS = [
	"interpro_conserved_site",
	"interpro_repeat",
	"interpro_binding_site",
	"interpro_active_site",
    "interpro_domain",
    "interpro_family",
    "interpro_homologous_superfamily",
    "GO_bp",
    "GO_cc",
    "GO_mf",
]

# Add the other datasets - different splits
TARGET_DATASETS_SPLITS = [f"{ds}_split1" for ds in TARGET_DATASETS] + [f"{ds}_split2" for ds in TARGET_DATASETS]

TARGET_DATASETS = TARGET_DATASETS + TARGET_DATASETS_SPLITS

DISTANCE_METRICS = ['euclidean', 'cosine'] #, 'cosine'

models_to_shorthands = {v['full_name']: k for k,v in plms.items()}

all_configs = []

for model_name in MODEL_NAMES:

    # 1. Generate the list of layer strings for this model
    layers = MODEL_LAYERS.get(model_name)

    # 2. Create the cross-product of (Layer, Dataset)
    permutations = itertools.product(layers, TARGET_DATASETS)

    # 3. Format the final config string: MODEL_NAME LAYER_TO_USE TARGET_DATASET
    for layer, dataset in permutations:
        for distance_metric in DISTANCE_METRICS:
            # Note: The model name contains slashes, which is fine for the array entry
            # but requires special handling when passed to read -r -a
            config_string = f'{models_to_shorthands[model_name]} {layer} {dataset} {distance_metric}'
            all_configs.append(config_string)

# Save the configurations to a file
with open("nearest_neighbor_configs.txt", "w") as f:
    for config in all_configs:
        f.write(f"{config}\n")

print(f"Generated {len(all_configs)} total configurations in job_configs.txt")
print(f"Submit with: sbatch --array=0-{len(all_configs) - 1} your_script_name.sh")
