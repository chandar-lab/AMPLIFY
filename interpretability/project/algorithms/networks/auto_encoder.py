"""Sparse autoencoder architectures for interpretability analysis of protein language models.

This module implements various sparse autoencoder (SAE) architectures for discovering
interpretable features in PLM activations. SAEs learn a sparse dictionary of features
that can reconstruct the original activations.

References:
    - TransposeWeightAutoEncoder: Based on "SPARSE AUTOENCODERS FIND HIGHLY INTERPRETABLE 
      FEATURES IN LANGUAGE MODELS" (https://arxiv.org/pdf/2309.08600)
    - UntiedWeightAutoEncoder: Based on "InterPLM: Discovering Interpretable Features in 
      Protein Language Models via Sparse Autoencoders" (https://arxiv.org/abs/2412.12101)
"""
from abc import ABC, abstractmethod

import torch as t
import torch.nn as nn
import torch.nn.init as init
from huggingface_hub import hf_hub_download
from hydra.utils import get_class


def make_model(sae_target: str, **kwargs):
    """Factory function to instantiate an SAE model from a class path string.
    
    Args:
        sae_target: Fully qualified class path (e.g., "project.algorithms.networks.auto_encoder.UntiedWeightAutoEncoder")
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        An instance of the specified SAE model class
    """
    return get_class(sae_target)(**kwargs)


class AutoEncoder(nn.Module, ABC):
    """Abstract base class for autoencoders."""

    def __init__(self, input_dim: int, dict_size: int, **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size

        self.encoder = nn.Linear(input_dim, dict_size)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights and normalize the feature dictionary."""
        init.xavier_uniform_(self.encoder.weight)
        with t.no_grad():
            self.encoder.weight.data = nn.functional.normalize(self.encoder.weight.data, dim=1)

    @abstractmethod
    def encode(self, x: t.Tensor) -> t.Tensor:
        """Abstract method for encoding input tensor."""
        pass

    @abstractmethod
    def decode(self, c: t.Tensor) -> t.Tensor:
        """Abstract method for decoding dictionary."""
        pass

    @abstractmethod
    def compute_sparsity_loss(self, sparse_latent: t.Tensor) -> t.Tensor:
        """Compute sparsity loss specific to this SAE type."""
        pass

    def forward(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    @staticmethod
    def from_pretrained(
        plm_model: str,
        plm_layer: int,
        device: str | None = None,
    ) -> "UntiedWeightAutoEncoder":
        """Load a pretrained autoencoder from Hugging Face InterPLM repository."""

        # Validate inputs
        pretrained_models = ["esm2-8m", "esm2-650m"]
        pretrained_layers = {
            "esm2-8m": [1, 2, 3, 4, 5, 6],
            "esm2-650m": [1, 9, 18, 24, 30, 33],
        }

        if plm_model not in pretrained_models:
            raise ValueError(f"Invalid ESM model: {plm_model}, options: {pretrained_models}")
        if plm_layer not in pretrained_layers[plm_model]:
            raise ValueError(
                f"Invalid layer for {plm_model}: {plm_layer}, options: {pretrained_layers[plm_model]}"
            )

        model_path = hf_hub_download(
            repo_id=f"Elana/InterPLM-{plm_model}",
            filename=f"layer_{plm_layer}/ae_normalized.pt",
        )  # Download model from HF

        state_dict = t.load(model_path, map_location=t.device(device), weights_only=True)

        dict_size, input_dim = state_dict["encoder.weight"].shape

        instance = UntiedWeightAutoEncoder(input_dim=input_dim, dict_size=dict_size)

        instance.load_state_dict(state_dict, strict=False)
        instance.to(device)
        instance.eval()

        # Disable gradients for inference
        for param in instance.parameters():
            param.requires_grad = False

        return instance


class TransposeWeightAutoEncoder(AutoEncoder):
    """
    Autoencoder with tied weights where decoder is the transpose of encoder matrix.
     based on the equations from the paper :
     SPARSE AUTOENCODERS FIND HIGHLY INTERPRETABLE FEATURES IN LANGUAGE MODELS
     https://arxiv.org/pdf/2309.08600

    c = ReLU(Mx + b)
    x̂ = M^T c

    Where M is the feature dictionary matrix (normalized row-wise).
    """

    def __init__(self, input_dim: int, dict_size: int, **kwargs) -> None:
        super().__init__(input_dim, dict_size)
        self.relu = nn.ReLU()

    def encode(self, x: t.Tensor) -> t.Tensor:
        self.encoder.weight.data = nn.functional.normalize(self.encoder.weight.data, dim=1)
        return self.relu(self.encoder(x))

    def decode(self, c: t.Tensor) -> t.Tensor:
        # Use the transpose of the normalized encoder weight
        encoder_weight = nn.functional.normalize(self.encoder.weight, dim=1)
        return nn.functional.linear(c, encoder_weight.t())

    def compute_sparsity_loss(self, sparse_latent: t.Tensor) -> t.Tensor:
        return t.mean(t.abs(sparse_latent))  # standard L1 penalty


class UntiedWeightAutoEncoder(AutoEncoder):
    """Autoencoder with separate learnable weights and bias for both encoder and decoder, based on the equations from the paper
    InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders
    https://arxiv.org/abs/2412.12101

    c = ReLU(M(x - b_e) + b)
    x̂ = M'c + b_e

    Where M is the feature dictionary matrix (normalized row-wise)
    """

    def __init__(self, input_dim: int | None, dict_size: int, **kwargs) -> None:
        super().__init__(input_dim, dict_size)
        self.bias = nn.Parameter(t.zeros(input_dim))
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.relu = nn.ReLU()

        self._initialize_decoder_weights()

    def _initialize_decoder_weights(self) -> None:
        init.xavier_uniform_(self.decoder.weight)

    def encode(self, x: t.Tensor) -> t.Tensor:
        self.encoder.weight.data = nn.functional.normalize(
            self.encoder.weight.data, dim=1
        )  # todo should we no_grad this operation ?
        return self.relu(self.encoder(x - self.bias))

    def decode(self, c: t.Tensor) -> t.Tensor:
        # normalization not necessary when not tied
        return self.decoder(c) + self.bias

    def compute_sparsity_loss(self, sparse_latent: t.Tensor) -> t.Tensor:
        return t.mean(t.abs(sparse_latent))  # standard L1 penalty


class TopKAutoEncoder(AutoEncoder):
    """Autoencoder that keeps only the top-k activations in the hidden layer.

    c = TopK(Mx + b) x̂ = M'c + b'

    Where M is the feature dictionary matrix (normalized row-wise).
    """

    def __init__(self, input_dim: int, dict_size: int, k: int, **kwargs) -> None:
        super().__init__(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.k = k
        assert k <= dict_size, f"k ({k}) must be <= dict_size ({dict_size})"

    def encode(self, x: t.Tensor) -> t.Tensor:
        self.encoder.weight.data = nn.functional.normalize(self.encoder.weight.data, dim=1)
        return self._apply_topk(self.encoder(x))

    def _apply_topk(self, activations: t.Tensor) -> t.Tensor:
        topk_values, topk_indices = t.topk(activations, self.k, dim=-1)

        sparse_activations = t.zeros_like(activations)
        sparse_activations.scatter_(
            -1, topk_indices, topk_values
        )  # Create sparse tensor with only top-k activations
        # TODO: test if this implementation does not break gradient backpropagation

        return sparse_activations

    def decode(self, c: t.Tensor) -> t.Tensor:
        return self.decoder(c)

    def compute_sparsity_loss(self, sparse_latent: t.Tensor) -> t.Tensor:
        return t.tensor(0.0, device=sparse_latent.device)
