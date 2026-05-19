# %%
# Standard imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import List
from collections import defaultdict
import polars as pl
import os
import gc # Garbage Collection

# Project specific imports (assuming these are available)
from project.algorithms.networks.protein_language_model import ProteinLanguageModel
from project.utils.strs import plms
from project.utils.functions import set_device

# %%
# --- Configuration ---
data_dir = Path('/home/mila/s/shawn.whitfield/scratch/data')
embedding_dir = data_dir / 'embeddings'
data_path = data_dir / 'datasets/h_sapiens_proteome/uniprotkb_AND_model_organism_9606_2025_09_09_annotated.parquet.gz'

model_shorthands = ['esm2_35m'] #'esm2_8m',  'amplify_120m', 'samplify_120m', 'esm2_150m'  'amplify_350m', 'esm2_650m'
BATCH_SIZE = 16
RAW_SEQUENCE_BATCH_LIMIT = 400 # Save raw embeddings every n sequences
# Set device to 'cuda' if available for faster computation
DEVICE = set_device()

# --- Custom Dataset for PyTorch DataLoader ---
class ProteinDataset(Dataset):
    """Custom Dataset for loading protein sequences."""
    def __init__(self, ids, sequences):
        self.ids = ids
        self.sequences = sequences

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # The PLM handles tokenization/padding later, so we just return the raw data
        return {'id': self.ids[idx], 'sequence': self.sequences[idx]}

def concatenate_raw_embeddings(raw_embeddings_batches: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenates a list of raw token-level embedding tensors (from different batches)
    into a single tensor, handling variable sequence lengths via padding.

    Args:
        raw_embeddings_batches (List[torch.Tensor]): A list where each element is 
                                                     a tensor of shape (BatchSize, SeqLen, Features).

    Returns:
        torch.Tensor: A single concatenated tensor of shape (TotalSequences, MaxSeqLen, Features).
    """
    if not raw_embeddings_batches:
        print("Input list of raw embedding batches is empty.")
        return torch.tensor([])

    # Pass 1: Find the absolute maximum sequence length across all batches
    max_len = 0
    for tensor in raw_embeddings_batches:
        # tensor.shape[1] is the SeqLen for that batch
        max_len = max(max_len, tensor.shape[1])

    # Pass 2: Pad and Concatenate
    final_tensors = []
    print(f"Max sequence length found: {max_len}. Starting padding...")
    
    for tensor in tqdm(raw_embeddings_batches, desc=f"Padding and Concatenating raw embeddings"):
        # Calculate padding needed for the current batch's tensor
        to_pad = max_len - tensor.shape[1]
        
        # F.pad(input, pad, mode='constant', value=0)
        # pad is (padding_left_dimN, padding_right_dimN, ..., padding_left_dim0, padding_right_dim0)
        # For a (Batch, SeqLen, Features) tensor, we pad dim 1 (SeqLen).
        # Padding is (pad_left_dim2, pad_right_dim2, pad_left_dim1, pad_right_dim1, ...)
        padded_tensor = F.pad(tensor, (0, 0, 0, to_pad), 'constant', 0.0)
        final_tensors.append(padded_tensor)

    # Concatenate all padded tensors along the batch dimension (dim=0)
    concatenated_tensor = torch.cat(final_tensors, dim=0)
    return concatenated_tensor

def concatenate_attention_masks(raw_mask_batches: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenates a list of attention mask tensors (from different batches)
    into a single tensor, handling variable sequence lengths via padding.

    Args:
        raw_mask_batches (List[torch.Tensor]): A list where each element is 
                                               a mask tensor of shape (BatchSize, SeqLen).

    Returns:
        torch.Tensor: A single concatenated tensor of shape (TotalSequences, MaxSeqLen).
    """
    if not raw_mask_batches:
        print("Input list of raw attention mask batches is empty.")
        return torch.tensor([])

    # Pass 1: Find the absolute maximum sequence length across all batches
    max_len = 0
    for tensor in raw_mask_batches:
        # tensor.shape[1] is the SeqLen for that batch
        max_len = max(max_len, tensor.shape[1])

    # Pass 2: Pad and Concatenate
    final_tensors = []
    print(f"Max sequence length found: {max_len}. Starting padding...")
    
    # Use enumerate for simple iteration if tqdm is not installed
    # for i, tensor in enumerate(raw_mask_batches): 
    #     print(f"Padding and Concatenating mask batch {i+1}...")
    
    # Keeping the structure from your original example:
    # Replace tqdm iteration with a standard loop if tqdm isn't available
    # Assuming 'tqdm' is available or can be removed for simplicity:
    # for tensor in tqdm(raw_mask_batches, desc=f"Padding and Concatenating attention masks"):
    for tensor in raw_mask_batches: # Standard loop as alternative to tqdm
        # Calculate padding needed for the current batch's tensor
        to_pad = max_len - tensor.shape[1]
        
        # F.pad(input, pad, mode='constant', value=0)
        # pad is a tuple: (padding_left_dimN, padding_right_dimN, ..., padding_left_dim0, padding_right_dim0)
        # For a (Batch, SeqLen) tensor (2 dims), we pad dim 1 (SeqLen).
        # The pad tuple for 2 dimensions is (pad_left_dim1, pad_right_dim1, pad_left_dim0, pad_right_dim0)
        # We want to pad on the right of SeqLen (dim 1): (0, to_pad).
        # Since we are padding the last dimension, the full tuple is just (0, to_pad).
        # We use a constant value of 0.0, which is standard for padding in attention masks.
        padded_tensor = F.pad(tensor, (0, to_pad), 'constant', 0.0)
        
        # Ensure the padded mask is of the correct type (usually torch.bool or torch.int/long)
        # Since F.pad with a float value might change the dtype, we explicitly convert it back
        # to the original tensor's dtype (assuming it was integer or boolean).
        padded_tensor = padded_tensor.to(tensor.dtype)
        
        final_tensors.append(padded_tensor)

    # Concatenate all padded tensors along the batch dimension (dim=0)
    concatenated_tensor = torch.cat(final_tensors, dim=0)
    return concatenated_tensor


class RawEmbeddingsSaver:
    """Manages the accumulation and periodic saving of raw token-level embeddings."""
    def __init__(self, model_shorthand: str, layer_num: int, output_dir: Path, batch_size: int = 1000):
        self.output_dir = output_dir / f"layer_{layer_num}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_shorthand = model_shorthand
        self.layer_num = layer_num
        
        # Batching parameters
        self.sequence_batch_limit = batch_size # Save file after this many sequences
        self.current_file_index = 0
        
        # Accumulated data lists (RAM storage)
        self.ids: List[str] = []
        self.sequences: List[str] = []
        self.raw_embeddings: List[torch.Tensor] = [] # List of (BatchSize, SeqLen, Dim) tensors
        self.masks: List[torch.Tensor] = [] # List of (BatchSize, SeqLen) tensors

    def add_batch(self, ids: List[str], sequences: List[str], raw_embeddings: torch.Tensor, mask: torch.Tensor):
        """Adds a new batch to the accumulator."""
        current_size = len(self.ids)
        new_size = current_size + len(ids)
        
        # Accumulate data
        self.ids.extend(ids)
        self.sequences.extend(sequences)
        # Raw embeddings and mask are already on CPU if you follow the recommendation
        self.raw_embeddings.append(raw_embeddings) 
        self.masks.append(mask)

        # Check if we should save and clear
        if new_size >= self.sequence_batch_limit:
            self._save_and_clear()

    def _save_and_clear(self):
        """Pads, concatenates, saves the accumulated data, and resets the lists."""
        if not self.ids:
            return

        print(f"  -> Saving raw batch file {self.current_file_index} ({len(self.ids)} sequences)...")
        
        # Use existing utility functions for padding and concatenation
        # Note: concatenate_raw_embeddings needs to handle tensors on CPU
        final_embeddings = concatenate_raw_embeddings(self.raw_embeddings)
        final_masks = concatenate_attention_masks(self.masks)

        save_path = self.output_dir / f'{self.model_shorthand}_layer_{self.layer_num}_aa_level_embeddings_{self.current_file_index:03d}.pt'
        
        torch.save({
            'ids': self.ids,
            'sequences': self.sequences,
            'masks': final_masks,
            'embeddings': final_embeddings,
        }, save_path)
        
        # Clean up in-memory accumulation
        self.ids = []
        self.sequences = []
        self.raw_embeddings = []
        self.masks = []
        self.current_file_index += 1
        
        # Free up memory (crucial after saving large tensors)
        del final_embeddings
        del final_masks
        gc.collect() 
        torch.cuda.empty_cache()

    def finalize(self):
        """Saves any remaining data upon loop completion."""
        self._save_and_clear()
        print(f"  -> Finalized raw saving for layer {self.layer_num}. Total files: {self.current_file_index}.")


# --- Data Loading and Preparation ---
print("Loading data...")
try:
    df = pl.read_parquet(data_path)
    all_ids = df['id'].to_list()
    all_sequences = df['sequence'].to_list()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Instantiate the Dataset and DataLoader
protein_dataset = ProteinDataset(all_ids, all_sequences)
# The DataLoader will handle batching; 'collate_fn' is not strictly needed 
# if the PLM handles the batch of strings directly, as your original script did.
protein_dataloader = DataLoader(
    protein_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2, # Use some workers for better I/O
    pin_memory=True if DEVICE.type == 'cuda' else False
)
print(f"Data loaded. Total sequences: {len(protein_dataset)}")

# --- Main Embedding Loop ---
for model_shorthand in model_shorthands:
    print(f"\nProcessing model: **{model_shorthand}**...")

    model_embedding_dir = embedding_dir / model_shorthand
    model_embedding_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize PLM outside the try block for scope, set to None for cleanup later
    plm = None 
    
    try:
        model_name = plms[model_shorthand]['full_name']
        
        # Instantiate PLM and move to the appropriate device
        plm = ProteinLanguageModel(model_name=model_name, layer_to_use=None).to(DEVICE)
        plm.eval() # Set model to evaluation mode

        # Structure for sequence-level data (mean, norm_mean)
        accumulated_sequence_data = defaultdict(lambda: {'mean': [], 'norm_mean': []})
        # Structure for token-level data (raw)
        raw_savers = {} 

        all_ids = []
        all_sequences = []
        
        # Loop over batches from the DataLoader
        for batch in tqdm(protein_dataloader, desc=f"Batches for {model_shorthand}",miniters=500):
            ids = batch['id']
            sequences = batch['sequence']
            
            # Pass sequences through PLM
            with torch.no_grad():
                # Use mixed precision (BF16) for L40S for speed and memory
                # with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16): 
                hidden_states, attention_mask, _ = plm(sequences)
                
                attention_mask = attention_mask.to('cpu') 

                all_ids.extend(ids)
                all_sequences.extend(sequences)

                # Process embeddings for all layers in the batch
                for layer_num, hidden_state in enumerate(hidden_states):
                    

                    hidden_state = hidden_state.to('cpu')
                    # RAW (Token-level) Embeddings
                    embeddings = (hidden_state * attention_mask.unsqueeze(-1))
                    
                    # # Handle RAW (Token-level) Embeddings using the Saver
                    # if layer_num not in raw_savers:
                    #     raw_savers[layer_num] = RawEmbeddingsSaver(
                    #         model_shorthand=model_shorthand, 
                    #         layer_num=layer_num, 
                    #         output_dir=model_embedding_dir,
                    #         batch_size=RAW_SEQUENCE_BATCH_LIMIT
                    #     )
                    # # Pass the CPU-moved raw embeddings to the saver
                    # raw_savers[layer_num].add_batch(
                    #     ids=ids, 
                    #     sequences=sequences, 
                    #     raw_embeddings=embeddings, 
                    #     mask=attention_mask
                    # )
                    
                    # Handle Sequence-level Embeddings (Mean/Norm_Mean)
                    non_padding_count = attention_mask.sum(dim=1, keepdim=True)
                    mean_embeddings = embeddings.sum(dim=1) / non_padding_count
                    
                    # normalized_token_embeddings = F.normalize(embeddings, dim=-1)
                    # normalized_mean_embeddings = normalized_token_embeddings.sum(dim=1) / non_padding_count
                    
                    # Accumulate sequence-level results 
                    accumulated_sequence_data[layer_num]['mean'].append(mean_embeddings)
                    # accumulated_sequence_data[layer_num]['norm_mean'].append(normalized_mean_embeddings)

            torch.cuda.empty_cache() 

        # --- CONSOLIDATED FILE SAVING (One file per layer, excluding raw) ---
        print(f"\n**Saving consolidated sequence embeddings for {model_shorthand}...**")

        # Save Mean and Norm_Mean (which were accumulated fully)
        for layer_num, data in accumulated_sequence_data.items():
            # model_embedding_layer_dir = model_embedding_dir / f"layer_{layer_num}"
            # model_embedding_layer_dir.mkdir(parents=True, exist_ok=True)
            
            mean_embeddings = torch.cat(data['mean'], dim=0)
            # normalized_mean_embeddings = torch.cat(data['norm_mean'], dim=0)
           
            mean_save_path = model_embedding_dir / f'{model_shorthand}_layer_{layer_num}_mean_embeddings.pt'
            torch.save({'ids': all_ids, 'sequences': all_sequences, 'embeddings': mean_embeddings}, mean_save_path)

            # norm_mean_save_path = model_embedding_layer_dir / f'{model_shorthand}_layer_{layer_num}_normalized_mean_embeddings.pt'
            # torch.save({'ids': all_ids, 'embeddings': normalized_mean_embeddings}, norm_mean_save_path)

            print(f" -> Saved consolidated layer {layer_num} MEAN/NORM_MEAN embeddings.")

        # Finalize the RAW embedding savers (saves any remaining data)
        for saver in raw_savers.values():
            saver.finalize()
            del saver 

        # Clean up model and memory
        del plm 
        gc.collect() 
        torch.cuda.empty_cache()
        print(f'Completed and saved embeddings for **{model_shorthand}**')

    except Exception as e:
        print(f"An error occurred while processing {model_shorthand}: {e}")
        # Ensure cleanup even on error
        if plm is not None:
            del plm
        # Attempt to clean up savers if they were initialized before the crash
        if 'raw_savers' in locals():
            # If the crash happened mid-batch, these files will be incomplete/corrupted, 
            # so we shouldn't attempt to finalize them, but just clear the memory.
            del raw_savers
        gc.collect() 
        torch.cuda.empty_cache()
        continue

print('\nscript complete')