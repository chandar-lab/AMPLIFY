"""Utility functions for protein sequence processing, embedding extraction, and analysis.

This module provides a wide range of helper functions for:
- Device management (GPU/MPS/CPU selection)
- FASTA file parsing
- Gene Ontology (GO) term processing and visualization
- Protein sequence analysis (properties, scrambling, ambiguity handling)
- Embedding extraction and caching from PLMs
- Distance matrix computation and PCA/UMAP visualization
- Model checkpoint management and probe loading

Many functions are used by scripts in `project/scripts/` and by the main training pipeline.
"""
from project.utils.strs import SEED

def ambiguities(sequence):
    """Adjust noncanonical amino acids to "standard" ones.
    
    See: https://biopython-tutorial.readthedocs.io/en/latest/notebooks/03%20-%20Sequence%20Objects.html
    
    Substitutions:
    - U (Selenocysteine) -> C (Cysteine)
    - O (Pyrrolysine) -> L (Lysine) - Note: BioPython treats O as Lysine in some contexts, here mapped to L? Wait, O is Pyrrolysine, structurally similar to Lysine.
    - B (Asparagine or Aspartic acid) -> N (Asparagine)
    - Z (Glutamine or Glutamic acid) -> Q (Glutamine)
    - J (Leucine or Isoleucine) -> L (Leucine)
    - X (Unknown) -> G (Glycine)

    This is a approximation to allow basic calculation on full protein sequences.
    """
    return sequence.replace("U", "C").replace("O", "L").replace("B", "N").replace("Z", "Q").replace("J", "L").replace("X", "G")

def protein_analysis(sequences: list[str]) -> list[dict]:
    """Take a protein sequence, calculate properties with BioPython, and return them.
    
    Calculated properties:
    - length: Length of the sequence.
    - molecular_weight: MW from Protein sequence.
    - instability_index: Guruprasad et al. 1990 method. >40 means unstable.
    - flexibility: Vihinen, 1994 method. Optimized for window=9.
    - gravy: Grand average of hydropathy (Kyte and Doolittle).
    - isoelectric_point: pH at which molecule is neutral.
    - charge_at_pH: Charge at pH 4.7, 7.2, and 8.0.
    - secondary_structure_fraction: Fraction of helix, turn, and sheet.
    - amino_acid_percent: Percentage of each amino acid.
    
    Args:
        sequences: List of protein sequences (strings).
        
    Returns:
        List of dictionaries, each containing calculated properties for a sequence.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    results = []
    for sequence in sequences:
        # Make sequence analysis tool with the sequence inside it
        protein = ProteinAnalysis(ambiguities(sequence))
        # get the requested property
        properties = {
                    'length': len(sequence),
                    'molecular_weight': protein.molecular_weight(),
                    'mass': protein.molecular_weight(),
                    'instability_index': protein.instability_index(),
                    'isoelectric_point': protein.isoelectric_point(),
                    'amino_acid_percent': protein.amino_acids_percent,
                    'flexibility': protein.flexibility(),
                    'gravy': protein.gravy(),
                    # 'molar_extinction_coefficient': protein.molar_extinction_coefficient(),
                    'secondary_structure_fraction': protein.secondary_structure_fraction(),
                    "charge_at_ph4_7": protein.charge_at_pH(4.7), # lysosomal pH https://www.nature.com/articles/nrm2820
                    "charge_at_ph7_2": protein.charge_at_pH(7.2), # general physiological pH https://www.nature.com/articles/nrm2820
                    "charge_at_ph8": protein.charge_at_pH(8.0), # mitochondrial pH https://www.nature.com/articles/nrm2820
                    }
        # Unpack Amino Acid Percentages (dictionary)
        for aa, percent in protein.amino_acids_percent.items():
            properties[f"percent_{aa.lower()}"] = percent
        
        # Unpack Secondary Structure Fraction (tuple)
        for name, value in zip(['helix', 'turn', 'sheet'], protein.secondary_structure_fraction()):
            properties[f"fraction_{name}_aas"] = value

        results.append(properties)
    return results

def check_correlation_plotly(df, **kwargs):
    """Make interactive clustergram plot of dataframe correlations using Dash Bio.
    
    Args:
        df: Polars DataFrame.
        **kwargs: Additional arguments passed to dash_bio.Clustergram.
        
    Returns:
        dash_bio.Clustergram object.
    """
    import dash_bio
    import polars as pl
    # Select only non-string columns
    df = df.select(pl.exclude(pl.String))
    # Calculate pearson correlations between all 
    corr_matrix = df.corr()
    # Make an interactive clustermap with plotly dash
    return dash_bio.Clustergram(
        data=corr_matrix.to_pandas(),
        column_labels=list(df.columns),
        row_labels=list(df.columns),
        **kwargs,
    )

def one_hot_polars_column(column_name, df, cols_to_keep, delimiter=';', occurrence_threshold=5, only_get_positive=True):
    """
    One-hot encode a Polars DataFrame column that contains delimited strings.
    
    Args:
        column_name: Name of column to explode and one-hot encode.
        df: Input DataFrame.
        cols_to_keep: List of other columns to retain in output.
        delimiter: Separator for string splitting.
        occurrence_threshold: Minimum count for a term to be included as a category.
        only_get_positive: If True, filter out rows that have no active categories.
        
    Returns:
        Polars DataFrame with one-hot encoded columns.
    """
    import polars as pl

    # Get unique terms' value counts
    unique_term_counts = df[column_name].str.split(delimiter).explode().value_counts(sort=True)
    # Threshold and remove null 
    terms_above_threshold = unique_term_counts.filter((unique_term_counts['count'] > occurrence_threshold) & ~pl.col(column_name).is_null()).sort(by=column_name)
    # Turn into a list
    unique_occurrences_list = terms_above_threshold[column_name].to_list()
    # Remove whitespace on ends and None values
    unique_occurrences_list = [x.strip() for x in unique_occurrences_list if x is not None and x != ""]
    # Turn into a series and back to list to drop duplicates
    unique_occurrences_list = pl.Series(unique_occurrences_list).unique().sort().to_list()

    print(f"unique occurrences with count > occurrence_threshold: {len(unique_occurrences_list)}")
    # Create one-hot encoded columns and select only the columns we want to keep + one-hot columns
    onehot_df = df.with_columns([
        pl.col(column_name).str.contains(category, literal=True).alias(f"{category}").fill_null(False)
        for category in unique_occurrences_list
    ]).select(cols_to_keep + unique_occurrences_list)

    if only_get_positive:
        return onehot_df.filter(onehot_df[unique_occurrences_list].sum_horizontal() > 0)

    return onehot_df

def load_config(dataset_name, dataset_config_dir:str = '/home/mila/s/shawn.whitfield/projects/AMPLIFY-private/interpretability/project/configs/datamodule/dataset'):
    """
    Load dataset configuration from a YAML file.

    Args:
        dataset_name: Name of the dataset (filename without .yaml).
        dataset_config_dir: Directory containing config files.

    Returns:
        Dictionary containing dataset configuration.
    """
    from pathlib import Path
    import yaml
    dataset_config_dir = Path(dataset_config_dir)
    with open(dataset_config_dir / f"{dataset_name}.yaml", 'r') as f:
        dataset_config_dict = yaml.safe_load(f)
    return dataset_config_dict

def filter_onehot_df(data_df, annotation_min_overlap = 1, annotation_max_overlap = 3, min_label_count=2):
    """Take a one-hot df generated with one_hot_polars_column and make a subset 
    with a specified amount of permitted label overlap

    Assume the first two columns are 'id', 'sequence' and the rest are boolean columns of labels
    """
    import polars as pl
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    annotation_columns = [col for col in data_df.columns if (data_df[col].dtype != pl.String)]

    # Calculate the sum of all boolean columns
    one_hot_df = data_df.with_columns(
        pl.sum_horizontal(pl.col(c) for c in annotation_columns).alias("annotation_count")
    )
    # Filter to columns with at only one horizontal to get a subset df
    one_hot_df = one_hot_df.filter((pl.col('annotation_count') >= annotation_min_overlap) & (pl.col('annotation_count') <= annotation_max_overlap)) 
    # Get the labels (for each row, which columns are 1s)
    combined_label_expr = pl.concat_str(
            [
                pl.when(pl.col(col)).then(pl.lit(col)).otherwise(pl.lit(None))
                for col in annotation_columns
            ],
            separator=";",
            ignore_nulls=True
        ).alias("combined_labels")
    one_hot_df = one_hot_df.with_columns(
        combined_label_expr
    )
     # drop rows where label num is <2
    one_hot_df = one_hot_df.filter(
        pl.col("combined_labels").count().over("combined_labels") >= min_label_count,
        )
    # Drop columns where where the column sum is zero
    sum_df = one_hot_df.sum()
    columns_to_keep = [
    col for col in sum_df.columns if sum_df[col].item() != 0
    ]
    one_hot_df = one_hot_df.select(columns_to_keep)

    # Convert labels to category numbers for e.g. knn preds
    combined_labels = np.array(one_hot_df['combined_labels'].to_list())
    le = LabelEncoder()
    le.fit(combined_labels)
    cat_labels = le.transform(combined_labels)

    label_dict = dict(zip(combined_labels, cat_labels))

    one_hot_df = one_hot_df.with_columns(
        pl.col('combined_labels').replace_strict(label_dict).alias('label_num')
    )

    return one_hot_df, label_dict, le

def get_cached_embeddings(sequences, model_shorthand, layer_nums:list[0,1], as_numpy=True):
    """
    Retrieve embeddings cached with get_embeddings.py
    """
    from pathlib import Path
    import torch
    embedding_dir = Path('/home/mila/s/shawn.whitfield/scratch/data/embeddings/')
    embedding_model_dir = embedding_dir / model_shorthand

    all_embeddings = {}

    for layer_num in layer_nums:
        filename = embedding_model_dir / f"{model_shorthand}_layer_{layer_num}_mean_embeddings.pt"
        # Load pre-cached embeddings
        embedding_dict = torch.load(filename, map_location='cpu', weights_only=True) # format: {'ids':list: ids, 'sequences':list : seqs, 'embeddings': torch.Tensor(num_seqs, embedding_dim)}
        cached_sequences = embedding_dict['sequences']
        cached_embeddings = embedding_dict['embeddings']
        # Find where the requested sequences are in the list
        sequence_to_index = {seq: i for i, seq in enumerate(cached_sequences)}
        seq_inds = [sequence_to_index[seq] for seq in sequences]
            
        # Get those indices in the cached embeddings
        requested_tensor = cached_embeddings[seq_inds,:]

        if as_numpy:
            all_embeddings[layer_num] = requested_tensor.numpy()
        else:
            all_embeddings[layer_num] = requested_tensor

        if requested_tensor.shape[0] != len(sequences):
            print('mismatch in expected size! sequences may be missing')

    return all_embeddings

def pca_layerwise_embeddings(layerwise_embeddings:dict):
    f"""
    Perform PCA on layerwise embeddings, a dictionary of layer_num: embeddings.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from project.utils.strs import SEED
    
    layerwise_pca = {}

    for layer_num, embeddings in layerwise_embeddings.items():

        ss = StandardScaler()
        # Standarize the embeddings
        standardized_embeddings = ss.fit_transform(embeddings)

        # make a pca
        pca = PCA(n_components=2, random_state=SEED)
        # Fit the PCA on the standardized embeddings
        result = pca.fit_transform(
            standardized_embeddings
        )

        layerwise_pca[layer_num] = {
            'pca_0': result[:,0],
            'pca_1': result[:,1]
        }
    
    return layerwise_pca

def create_batches(total_size: int, batch_size: int):
    """
    A lightweight generator that yields the start and end indices for batches.

    Args:
        total_size: The total number of elements to be batched (e.g., number of proteins).
        batch_size: The maximum size of each batch.

    Yields:
        A tuple (start_index, end_index) for each batch.
    """
    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        yield start_idx, end_idx

def concatenate_with_padding(tensor_list:list):
    """
    Concatenates a list of tensors, padding the sequence dimension (dim 1) to the max length.

    Args:
        tensor_list: List of tensors of shape (batch_size, seq_len, ...).

    Returns:
        Concatenated tensor.
    """
    import torch

    if len(tensor_list) <=1:
        return tensor_list[0]
    
    # Scan through tensor_list to find longest tensor
    max_len = 0
    for t in tensor_list:
        tensor_seq_length = t.shape[1]
        if tensor_seq_length > max_len:
            # update max length if necessary
            max_len = tensor_seq_length

    # Pad each tensor if necessary
    padded_tensor_list = []
    for t in tensor_list:
        tensor_seq_length = t.shape[1]
        if tensor_seq_length < max_len:
            padding_needed = max_len - tensor_seq_length
            # Pad the first dimension (sequence length)
            padded_t = torch.nn.functional.pad(t, (0,0,0,padding_needed), 'constant', 0.0)
            
            padded_tensor_list.append(padded_t)
        else:
            padded_tensor_list.append(t)

    # Now we can do a concatenate and return
    return torch.concatenate(padded_tensor_list)

def get_embeddings(sequences:list, protein_language_model, batch_size: int = 16, layer_nums:list = [0,1], pooled=True, as_numpy=False):
    """
    Extracts embeddings from a PLM for a list of sequences.

    Args:
        sequences: List of protein sequences.
        protein_language_model: Loaded PLM model.
        batch_size: Batch size for inference.
        layer_nums: List of layer indices to extract embeddings from.
        pooled: If True, returns mean-pooled (protein-level) embeddings. 
                If False, returns unpooled (residue-level) embeddings.
        as_numpy: Unused argument (kept for compatibility), output is torch tensors or dict of tensors.

    Returns:
        Dictionary mapping layer_num to concatenated embeddings tensor.
        If pooled=False, also returns concatenated attention masks.
    """

    from collections import defaultdict
    import torch

    all_embeddings = defaultdict(list)
    all_attention_masks = []
    for i in create_batches(len(sequences), batch_size=batch_size):
        # Select sequences indexing on the batch tuple produced
        seqs = sequences[i[0]:i[1]]
        # Get embeddings for all proteins, at the chosen layer
        with torch.no_grad():
        # Through PLM
            hidden_state, attention_mask, attentions = protein_language_model(
                seqs # pass a whole batch of sequences
            )  # (batch_size, sequence_length, hidden_dim)

            # Collect from specified layers
            for layer_num in layer_nums:
                if pooled:
                    # Aggregate to protein-level embeddings by taking the mean across the amino acids
                    embeddings_at_layer = (hidden_state[layer_num] * attention_mask.unsqueeze(-1)).sum(
                        dim=1
                    ) / attention_mask.sum(dim=1, keepdim=True) # Use attention mask to avoid padding influence

                    all_embeddings[layer_num].append(embeddings_at_layer.detach().to('cpu'))
                else:
                    all_embeddings[layer_num].append(hidden_state[layer_num].detach().to('cpu'))
                all_attention_masks.append(attention_mask)

    if pooled:
        all_embeddings = {layer_num: torch.concatenate(embeddings_list) for layer_num, embeddings_list in all_embeddings.items()}
        return all_embeddings
    else:
        all_embeddings = {layer_num: concatenate_with_padding(embeddings_list) for layer_num, embeddings_list in all_embeddings.items()}
        return all_embeddings, concatenate_with_padding(all_attention_masks)

def scramble_sequences(sequences:list, seed:int = SEED):
    """
    Randomly scramble a set of protein sequence (leaving the initiator methionine untouched)
    """
    import numpy as np

    def scramble_sequence(sequence:str, random_number_generator):
        """
        Randomly scramble an entire protein sequence (leaving the initiator methionine untouched)
        """
        # Shuffle all but the start methionine, to improve biological plausibility (make it look more like a protein)
        to_shuffle = list(sequence)[1:]
        # Inplace shuffle the list
        random_number_generator.shuffle(to_shuffle)
        # Join it back into a string
        return sequence[0]+ ''.join(to_shuffle)

    # Make one rng generator so we don't have to initialize each time we scramble a sequence
    rng = np.random.default_rng(seed=seed)     

    scrambled_sequences = []
    for seq in sequences:
        scrambled_sequences.append(scramble_sequence(seq, rng))

    return scrambled_sequences

def set_device():
    """Set the device to MPS, GPU, or CPU based on availability."""
    import torch

    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def reset_to_pytorch_defaults(model):
    """
    Resets all learnable parameters in the model to PyTorch's default 
    initialization (e.g., Kaiming uniform/normal, Xavier uniform/normal).
    """
    import torch.nn as nn
    def reset_module_params(module):
        # Check if the module has a reset_parameters method
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        
    model.apply(reset_module_params)

    # We're not going to be training
    for param in model.parameters():
        param.requires_grad = False

def get_distance_matrices(embeddings, distance_metric:str, seed:int=SEED):
    """
    Computes distance matrices for embeddings and generated controls.

    Controls:
    - Shuffled embeddings: Columns (features) are shuffled independently.
    - Shuffled positions: Rows (samples) are shuffled independently.

    Args:
        embeddings: Input embedding matrix (N, D).
        distance_metric: Metric name for scipy.spatial.distance.cdist.
        seed: Random seed.

    Returns:
        Dictionary containing original and control matrices and embeddings.
    """
    from scipy.spatial.distance import cdist
    import numpy as np
	# Calculate the distance matrix for all embeddings
    distance_matrix = cdist(embeddings, embeddings, metric = distance_metric)

    # Control: randomize each embedding dimension to see if we destroy the signal (column shuffle)
    rng = np.random.default_rng(seed=SEED)
    shuffled_embeddings = embeddings.copy()
    # Shuffle each column independently
    [rng.shuffle(shuffled_embeddings[:,i]) for i in range(shuffled_embeddings.shape[1])]
    shuffled_distance_matrix = cdist(shuffled_embeddings, shuffled_embeddings, metric=distance_metric)

    # Control: randomize each embedding position (row shuffle)
    rng = np.random.default_rng(seed=seed) # Re-seed or use the existing rng if desired
    shuffled_positions = embeddings.copy()
    # Shuffle each row independently
    [rng.shuffle(shuffled_positions[i,:]) for i in range(shuffled_positions.shape[0])] 
    shuffled_position_distance_matrix = cdist(shuffled_positions, shuffled_positions, metric=distance_metric)

    embedding_treatments = {'embeddings': {'matrix': distance_matrix, 'embeddings': embeddings},
                                'shuffled_embeddings': {'matrix': shuffled_distance_matrix, 'embeddings': shuffled_embeddings},
                                'shuffled_positions': {'matrix': shuffled_position_distance_matrix, 'embeddings': shuffled_positions}}
    return embedding_treatments

def shuffle_tensor(row, pad_val = -100):
    """
    Helper function to shuffle the non-padded part of a tensor
    """
    import torch
    # Guard against 0-dim tensors (scalars) resulting from batch-of-1 iteration
    if row.dim() == 0:
        # If it's a scalar, unsqueeze it to make it 1D
        row = row.unsqueeze(0)
    row_copy = row.clone().detach()
    # Identify where we have real values and not the padding
    mask = torch.where(row_copy != pad_val, True, False)
    # Select down to the part to shuffle
    part_to_shuffle = row_copy[mask]
    # Shuffle it
    permuted = part_to_shuffle[torch.randperm(len(part_to_shuffle))]
    # Update the original row with the shuffled part
    row_copy[0:len(part_to_shuffle)] = permuted
    return row_copy