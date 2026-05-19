"""Wrapper for pre-trained protein language models (PLMs) for embedding extraction.

This module provides a unified interface for loading and using various PLMs (ESM2, AMPLIFY, etc.)
from HuggingFace, extracting embeddings from specific layers, and handling model-specific
tokenization and attention mask differences.
"""
import torch as t
import torch.nn as nn

from project.utils.functions import set_device
from project.utils.strs import embedding_dir, plms

# Set the device (GPU/MPS/CPU) for model operations
DEVICE = set_device()

import os

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure embedding directory exists for caching
embedding_dir.mkdir(parents=True, exist_ok=True)


class ProteinLanguageModel(nn.Module):
    """
    A wrapper for pre-trained protein language models (PLMs) that tokenizes input sequences
    and extracts embeddings from specified transformer layers.
    
    Supports multiple PLM architectures (ESM2, AMPLIFY, MILA variants) and handles
    architecture-specific differences in tokenization, attention masks, and layer indexing.
    
    Attributes:
        model_name: HuggingFace model identifier (e.g., "facebook/esm2_t6_8M_UR50D")
        layer_to_use: Which transformer layer to extract embeddings from (None = all layers)
        normalize_embeddings: Whether to L2-normalize embeddings before returning
        apply_attention_mask: Whether to zero out padding tokens in embeddings
        ignore_embedding_layer: For ESM2, whether to exclude the embedding layer from indexing
    """

    def __init__(
        self,
        model_name: str,
        layer_to_use: int | None,
        ignore_embedding_layer=True,
        normalize_embeddings: bool = False,  # Whether to normalize embeddings (along dim -1) before returning
        apply_attention_mask: bool = False,  # Whether to return embeddings that have had the attention mask already applied, to zero out padding tokens
    ):
        super().__init__()

        if 'mila' in model_name.lower():
            # Get everything before the last underscore as the model name for reading from huggingface
            self.model_name = '_'.join(model_name.split('_')[:-1])
            # The last part is the revision number (from 100k to 1M, in 100k increments)
            self.revision = str(model_name.split('_')[-1])
        else:
            self.model_name = model_name
            self.revision = None
        # Get the appropriate shorthand from our plms dict
        for key, info in plms.items():
            if info["full_name"] == model_name:
                self.model_shorthand = key
                self.total_layers = info["num_hiddens"]
                break
        self.model, self.tokenizer = self.load_plm(self.model_name, self.revision)
        if isinstance(layer_to_use, float):
            # We've passed relative depth, a float in [0,1], need to convert to a layer number
            self.relative_depth = layer_to_use
            self.layer_to_use = self.get_layer_to_use(layer_to_use)
        else:
            #
            self.layer_to_use = layer_to_use
            # Calculate the relative depth from the given layer number (if layers exist)
            if self.layer_to_use is not None and self.total_layers > 0:
                self.relative_depth = self.get_relative_depth(self.layer_to_use)
            else:
                self.relative_depth = -1.0
        self.normalize_embeddings = normalize_embeddings
        self.apply_attention_mask = apply_attention_mask
        self.ignore_embedding_layer = ignore_embedding_layer  # for esm2 models, whether to ignore the first layer (embedding layer) or not

    def forward(self, x: str | list[str]) -> t.Tensor:
        """
        Pass sequences through a protein language model. Tokenizes, extracts hidden states, and applies attention mask to hidden states to zero out padding tokens. returns hidden states, attention mask, and attentions.
        """

        if "esm2" in self.model_name.lower():
            # Tokenize
            tokenized_input = self.tokenizer(x, return_tensors="pt", padding=True).to(DEVICE)
            # Collect tokenized inputs and attention mask
            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]

        elif "amplify" in self.model_name.lower() or "mila" in self.model_name.lower():
            # Amplify weirdness means we need to pad tokenization to multiple of 8
            tokenized_input = self.tokenizer(x, return_tensors="pt", padding=True, truncation=False, pad_to_multiple_of=8).to(
                DEVICE
            )
            # With amplify, need to convert attention mask
            input_ids, attention_mask = tokenized_input["input_ids"], t.where(
                tokenized_input["attention_mask"].bool(), float(0.0), float("-inf")
            )
        else:
            raise ValueError(
                f"Unsupported model name: {self.model_name}. Supported models include ESM2 and AMPLIFY."
            )

        # Pass through model and get outputs
        with t.no_grad():
            outputs = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True
            )
        # Get hiddens
        hidden_states = outputs.hidden_states  # Extract the hidden states
        attentions = outputs.attentions  # Extract the attention matrices
        # Return hiddens at the desired layer, or all hiddens

        if "amplify" in self.model_name.lower():
            # Fix the attention mask back to something that resembles sanity ie. mimic ESM2
            attention_mask = (~attention_mask.bool()).long()
        elif "mila" in self.model_name.lower():
            # Fix the attention mask back to something that resembles sanity ie. mimic ESM2
            attention_mask = (~attention_mask.bool()).long()
        elif "esm" in self.model_name.lower():
            # ignore the first layer of the hidden states
            hidden_states = hidden_states[1:]

        # Apply attention mask to hidden states
        if self.apply_attention_mask:
            hidden_states = [
                hidden_state * attention_mask.unsqueeze(-1) for hidden_state in hidden_states
            ]  # Apply attention mask to zero out padding tokens
        if self.normalize_embeddings:
            hidden_states = [
                nn.functional.normalize(hidden_state, dim=-1) for hidden_state in hidden_states
            ]  # Normalize the embeddings of each amino acid

        if self.layer_to_use is not None:
            return hidden_states[self.layer_to_use], attention_mask, attentions[self.layer_to_use]
        else:
            return hidden_states, attention_mask, attentions

    def load_plm(self, model_name: str, revision: str | None = None):
        """Load a pre-trained protein language model from Hugging Face.

        Args:
            model_name (str): The name or path of the model to load. Supported models include:
                - Any ESM2 model (e.g., facebook/esm2_t6_8M_UR50D, facebook/esm2_t33_650M_UR50D, etc.)
                - AMPLIFY models (e.g., chandar-lab/AMPLIFY_120M)
            revision (str | None): The specific model revision to load (e.g., branch name, tag, or commit hash).
        Returns:
            model: The loaded pre-trained model.
            tokenizer: The corresponding tokenizer.
        """
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        # Determine the appropriate model class and settings based on model name
        if "esm2" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif "amplify" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        elif "mila" in model_name.lower():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, revision=revision, trust_remote_code=True
                )
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision=revision)
            except Exception as e:
                print(
                    f"Error loading model {model_name}: {e}, if you got CUDA_HOME does not exist, unable to compile CUDA op(s) error, make sure you have loaded the appropriate cudatoolkit module: 'module load cudatoolkit/{t.version.cuda}'"
                )
        else:
            raise ValueError(
                f"Unsupported model name: {model_name}. Supported models include ESM2, AMPLIFY and AMPLIFY V1.5 (Lolalb/MILA_U100_baseline)"
            )
        model.eval()  # Set the model to evaluation mode - we're not going to be training them

        # Freeze the model parameters to prevent training
        for param in model.parameters():
            param.requires_grad = False
        # Move the model to GPU or MPS if available
        model.to(DEVICE)
        return model, tokenizer

    def get_relative_depth(self, layer_num):
        """
        Given a layer number, return the relative depth of that layer for the plm
        """
        return round((layer_num) / (self.total_layers - 1), 2)

    def get_layer_to_use(self, relative_depth):
        """Given a specified relative_depth, get the closest layer to use"""
        return round(relative_depth * (self.total_layers - 1))

    def get_special_token_counts(self) -> tuple[int, int]:
        if "esm2" in self.model_name.lower():
            return 1, 1
        elif "amplify" in self.model_name.lower():
            return 1, 1
        elif "mila" in self.model_name.lower():
            return 0, 0

