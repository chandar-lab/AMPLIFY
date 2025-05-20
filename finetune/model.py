"""define the models we train"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from protein_dataset import obtain_real_residue_mask


class ContrastiveLearningModel(nn.Module):
    """
    CL model class
    """

    def __init__(self, seq_dim: int, struc_dim: int, output_dim: int):
        """
        define the CL model here
        Args:
            seq_dim:
            struc_dim:
            output_dim:
        """
        super(ContrastiveLearningModel, self).__init__()
        self.linear_seq = nn.Linear(seq_dim, output_dim)
        self.linear_struct = nn.Linear(struc_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        struc_embeddings: torch.Tensor,
        weights: torch.Tensor,
    ) -> List:
        """
        Args:
            seq_embeddings:
            struc_embeddings:
            weights:

        Returns: cl_loss

        """
        projected_seq = self.linear_seq(seq_embeddings)
        projected_struct = self.linear_struct(struc_embeddings)

        projected_seq = F.normalize(projected_seq, p=2, dim=-1)
        projected_struct = F.normalize(projected_struct, p=2, dim=-1)

        logit_scale = self.logit_scale.exp().clip(min=0.01)  # follow CLIP paper setting

        similarity_matrix = logit_scale * torch.matmul(
            projected_seq, projected_struct.T
        )

        residue_num = seq_embeddings.size(0)
        labels = torch.arange(residue_num).to(similarity_matrix.device)

        loss_seq_to_struct = F.cross_entropy(
            similarity_matrix,
            labels,
            size_average=False,
            reduce=False,
        )

        loss_struct_to_seq = F.cross_entropy(
            similarity_matrix.T,
            labels,
            size_average=False,
            reduce=False,
        )

        return loss_seq_to_struct.squeeze(), loss_struct_to_seq.squeeze()


class AmplifyClassifier(nn.Module):
    """
    Deepspeed only support single-model training
    so we have to merge all models into one.
    """

    def __init__(
        self,
        prt_model_name: str,
        num_labels: int = 26,
        seq_D: int = 640,
        struc_D: int = 384,
        output_D: int = 384,
    ):
        """
        initialize all three models into one model
        Args:
            pretrained_model_name_or_path:
            num_labels:
            seq_D:
            struc_D:
            output_D:
        """
        super().__init__()

        if prt_model_name in ["chandar-lab/AMPLIFY_120M", "chandar-lab/AMPLIFY_350M"]:
            self.trunk = AutoModel.from_pretrained(
                prt_model_name, trust_remote_code=True
            )
        else:
            self.trunk = AutoModelForMaskedLM.from_pretrained(
                prt_model_name, trust_remote_code=True
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.trunk.config.hidden_size, self.trunk.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.trunk.config.hidden_size, num_labels),
        )
        self.cl_model = ContrastiveLearningModel(
            seq_dim=seq_D, struc_dim=struc_D, output_dim=output_D
        )

    def main_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        frozen_trunk: bool = True,
        normalize_hidden_states: bool = True,
        layer_idx: int = -1,
    ):
        """
        main forward to obtain logit and hidden states
        Args:
            input_ids:
            attention_mask:
            frozen_trunk:
            normalize_hidden_states:
            layer_idx:

        Returns: logit_mlm, logit_cls, h

        """

        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            model_output = self.trunk(
                input_ids, attention_mask, output_hidden_states=True
            )
            h = model_output.hidden_states[layer_idx]
        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)

        logit_mlm = model_output.logits
        logit_cls = self.classifier(h)

        return logit_mlm, logit_cls, h

    def cl_forward(
        self,
        seq_embeddings: torch.Tensor,
        struc_embeddings: torch.Tensor,
        cl_weights: torch.Tensor,
    ) -> List:
        """
        compute the constrastive learning loss
        Args:
            seq_embeddings:
            struc_embeddings:
            cl_weights:

        Returns: inter_loss

        """
        loss_seq_to_struct, loss_struct_to_seq = self.cl_model(
            seq_embeddings, struc_embeddings, cl_weights
        )
        return loss_seq_to_struct, loss_struct_to_seq


class BaseDownstreamModel(nn.Module):
    """
    Base class for downstream models.
    This class handles loading the pretrained trunk model based on the given model name
    and an optional fine-tuned checkpoint.
    """

    def __init__(self, prt_model_name: str, ft_model_path: str = None):
        """
        Args:
            prt_model_name: Name or identifier of the pretrained model.
            ft_model_path: Optional path to fine-tuned weights.
        """
        super().__init__()
        if ft_model_path:
            # Load configuration from pretrained model
            trunk_config = AutoConfig.from_pretrained(
                prt_model_name, trust_remote_code=True
            )
            if prt_model_name in [
                "chandar-lab/AMPLIFY_120M",
                "chandar-lab/AMPLIFY_350M",
            ]:
                # For these models, load using AutoModel.from_config
                self.trunk = AutoModel.from_config(trunk_config, trust_remote_code=True)
            else:
                # ESM-based models
                # Otherwise, use AutoModelForMaskedLM.from_config
                self.trunk = AutoModelForMaskedLM.from_config(
                    trunk_config, trust_remote_code=True
                )
            # Load the fine-tuned weights
            state_dict = torch.load(ft_model_path, map_location="cpu")
            self.trunk.load_state_dict(state_dict)
        else:
            # Load pretrained model without fine-tuning weights
            if prt_model_name in [
                "chandar-lab/AMPLIFY_120M",
                "chandar-lab/AMPLIFY_350M",
            ]:
                self.trunk = AutoModel.from_pretrained(
                    prt_model_name, trust_remote_code=True
                )
            elif prt_model_name in [
                "facebook/esm2_t6_8M_UR50D",
                "facebook/esm2_t12_35M_UR50D",
                "facebook/esm2_t30_150M_UR50D",
                "facebook/esm2_t33_650M_UR50D",
            ]:
                self.trunk = AutoModelForMaskedLM.from_pretrained(
                    prt_model_name, trust_remote_code=True
                )
            elif prt_model_name == "ism":
                self.trunk = AutoModel.from_pretrained(
                    "checkpoint/ISM/ism_model"
                )
            elif prt_model_name == "esm-s":
                self.trunk = AutoModel.from_pretrained(
                    "checkpoint/ESM-s/esm_s_model"
                )
            else:
                raise ValueError(
                    f"Unsupported pretrained model name: {prt_model_name}."
                )

        if prt_model_name == "ism":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "checkpoint/ISM/ism_model"
            )
        elif prt_model_name == "esm-s":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "checkpoint/ESM-s/esm_s_model"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                prt_model_name, trust_remote_code=True
            )


class PointPredictionModel(BaseDownstreamModel):
    """
    Model for point prediction tasks.

    This model is suitable for predicting properties of individual residues (residue-level)
    or entire proteins (protein-level), such as residue functionality or protein stability.

    Args:
        prt_model_name: Name or identifier of the pretrained model.
        ft_model_path: Optional path to fine-tuned weights.
        hidden_size: Hidden dimension size for the classifier head.
        task_num_labels: Output dimension for the task (e.g., 1 for regression, or number of classes for classification).
        task_output_type: Specifies the prediction level; "residue" for per-residue predictions (output shape: (B, L, task_num_labels)),
                      "protein" for pooled protein-level predictions (output shape: (B, task_num_labels)).
        normalization: Whether to apply normalization on the final output (i.e., scaling by std and adding mean).
        target_mean: Mean value for output normalization.
        target_std: Standard deviation for output normalization.
    """

    def __init__(
        self,
        prt_model_name: str,
        ft_model_path: str = None,
        # hidden_size: int = 128,
        task_num_labels: int = 1,
        task_output_type: str = "residue",  # "residue" or "protein"
        normalization: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__(prt_model_name, ft_model_path)
        self.task_output_type = task_output_type.lower()
        # Classifier head for point prediction with input dimension equal to trunk hidden size
        hidden_size = self.trunk.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.trunk.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, task_num_labels),
        )
        self.normalization = normalization

        self.register_buffer("mean", torch.tensor([target_mean], dtype=torch.float))
        self.register_buffer("std", torch.tensor([target_std], dtype=torch.float))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        frozen_trunk: bool = True,
        normalize_hidden_states: bool = True,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Forward pass for point prediction.

        Args:
            input_ids: Tensor of token IDs with shape (B, L).
            attention_mask: Tensor with shape (B, L) indicating valid token positions.
            frozen_trunk: If True, the trunk part is frozen (no gradients will be computed).
            normalize_hidden_states: If True, apply L2 normalization to the trunk's hidden states.
            layer_idx: Index of the hidden state layer to extract from the trunk; default is the last layer.

        Returns:
            If task_output_type=="residue", returns predictions with shape (B, L, task_num_labels);
            if task_output_type=="protein", returns pooled predictions with shape (B, task_num_labels).
        """
        # Obtain hidden states from the trunk; requires the model to output hidden_states
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            trunk_output = self.trunk(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            h = trunk_output.hidden_states[layer_idx]  # h has shape (B, L, D)

        if self.task_output_type == "protein":
            # for protein-level task, we need to average all residue embeddings to obtain protein-level embed
            real_residue_mask = obtain_real_residue_mask(input_ids, self.tokenizer)
            mask = real_residue_mask.unsqueeze(-1)
            h = (h * mask).sum(dim=1) / mask.sum(dim=1)

        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)

        pred = self.classifier(h)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


class ContactPredictionModel(BaseDownstreamModel):
    """
    Model for pairwise (interaction) prediction tasks.

    This model is designed for predicting interactions between pairs of residues (or proteins),
    such as residue-residue contact prediction. The model first obtains token-level representations h
    with shape (B, L, D) from the trunk, then constructs pairwise features of shape (B, L, L, 2*D)
    by concatenating representations for every pair of positions, and finally applies a small classifier
    to obtain predictions for each pair.

    Args:
        prt_model_name: Name or identifier of the pretrained model.
        ft_model_path: Optional path to fine-tuned weights.
        hidden_size: Hidden dimension size for the pairwise classifier head.
        task_num_labels: Output dimension for each pair (e.g., 2 for binary classification).
        normalization: Whether to apply normalization on the final output (i.e., scaling by std and adding mean).
        target_mean: Mean value for output normalization.
        target_std: Standard deviation for output normalization.
    """

    def __init__(
        self,
        prt_model_name: str,
        ft_model_path: str = None,
        # hidden_size: int = 128,
        task_num_labels: int = 1,
        normalization: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__(prt_model_name, ft_model_path)
        # Pairwise classifier with input dimension equal to 2 * trunk hidden size
        hidden_size = 128 # self.trunk.config.hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.trunk.config.hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, task_num_labels),
        )
        self.normalization = normalization
        self.register_buffer("mean", torch.tensor([target_mean], dtype=torch.float))
        self.register_buffer("std", torch.tensor([target_std], dtype=torch.float))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        frozen_trunk: bool = True,
        normalize_hidden_states: bool = True,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Forward pass for pairwise (interaction) prediction.

        Args:
            input_ids: Tensor of token IDs with shape (B, L).
            attention_mask: Tensor with shape (B, L) indicating valid token positions.
            frozen_trunk: If True, the trunk part is frozen (no gradients will be computed).
            normalize_hidden_states: If True, apply L2 normalization to the trunk's hidden states.
            layer_idx: Index of the hidden state layer to extract from the trunk; default is the last layer.

        Returns:
            A tensor of predictions with shape (B, L, L, task_num_labels), where for each pair (i, j) in the sequence,
            a prediction is output (e.g., a score for contact probability).
        """
        # Obtain token-level representations from the trunk with shape (B, L, D)
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            trunk_output = self.trunk(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            h = trunk_output.hidden_states[layer_idx]  # h has shape (B, L, D)

        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)

        B, L, D = h.shape
        # Construct pairwise features:
        # Expand h to obtain h_i and h_j for each token position,
        # then concatenate along the last dimension to get a tensor of shape (B, L, L, 2*D)
        h_i = h.unsqueeze(2)  # Shape: (B, L, 1, D)
        h_j = h.unsqueeze(1)  # Shape: (B, 1, L, D)
        pairwise_features = torch.cat(
            [h_i.expand(-1, -1, L, -1), h_j.expand(-1, L, -1, -1)], dim=-1
        )

        # pred
        pred = self.classifier(pairwise_features)  # Shape: (B, L, L, task_num_labels)

        pred = (pred + pred.transpose(1, 2)) / 2

        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


class PPIModel(BaseDownstreamModel):
    """
    Model for PPI task.
    """

    def __init__(
        self,
        prt_model_name: str,
        ft_model_path: str = None,
        task_num_labels: int = 2,
        # hidden_size: int = 128,
        normalization: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__(prt_model_name, ft_model_path)
        hidden_size = self.trunk.config.hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.trunk.config.hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, task_num_labels),
        )
        self.normalization = normalization
        self.register_buffer("mean", torch.tensor([target_mean], dtype=torch.float))
        self.register_buffer("std", torch.tensor([target_std], dtype=torch.float))

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        frozen_trunk: bool = True,
        normalize_hidden_states: bool = True,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            output1 = self.trunk(
                input_ids_1, attention_mask=attention_mask_1, output_hidden_states=True
            )
            h1 = output1.hidden_states[layer_idx]  # shape: (B, L, D)
        real_mask1 = (
            obtain_real_residue_mask(input_ids_1, self.tokenizer)
            .unsqueeze(-1)
            .to(h1.dtype)
        )
        h1 = (h1 * real_mask1).sum(dim=1) / real_mask1.sum(dim=1)

        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            output2 = self.trunk(
                input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True
            )
            h2 = output2.hidden_states[layer_idx]  # shape: (B, L, D)
        real_mask2 = (
            obtain_real_residue_mask(input_ids_2, self.tokenizer)
            .unsqueeze(-1)
            .to(h2.dtype)
        )
        h2 = (h2 * real_mask2).sum(dim=1) / real_mask2.sum(dim=1)

        if normalize_hidden_states:
            h1 = torch.nn.functional.normalize(h1, p=2, dim=-1)
            h2 = torch.nn.functional.normalize(h2, p=2, dim=-1)

        h = torch.cat([h1, h2], dim=-1)  # shape: (B, 2*D)

        pred = self.classifier(h)  # shape: (B, 1)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred
