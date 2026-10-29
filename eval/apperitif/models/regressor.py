from typing import Any, Dict
import torch
from torch import Tensor

from apperitif.models.Encoder import SequenceEncoder, TokenEncoder, StructureEncoder


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, activation="gelu"):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        if activation == "relu":
            self.act = torch.nn.ReLU()
        elif activation == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):  # TODO optimization
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Regressor(torch.nn.Module):
    def __init__(
        self,
        trunk: (
            SequenceEncoder | TokenEncoder
        ),  # Potentially Encoder instead of SequenceEncoder (TokenEncoder would be covered too)
        use_low_rank: bool = False,
        reduction_method: str = "mean",
        hidden_size: int = 512,
        output_size: int = 1,
        head_precision: torch.dtype = torch.float32,
        trunk_autocast_precision: torch.dtype = torch.float16,
    ):
        super(Regressor, self).__init__()
        self.trunk = trunk
        self.head_precision = head_precision
        self.head = MLP(trunk.hidden_size, hidden_size, output_size).to(self.head_precision)
        self.trunk_autocast_precision = trunk_autocast_precision

        self.use_low_rank = use_low_rank
        if self.use_low_rank:
            raise NotImplementedError("Low-rank regression is not implemented yet.")
        self.reduction_method = reduction_method
        assert self.reduction_method in [
            "mean",
            "sum",
            "cls",
        ], f"{self.reduction_method} is an invalid reduction method. Must be one of ['mean', 'sum', 'cls']"

    def forward(
        self,
        batch: Dict[str, Tensor],
        frozen_backbone=False,
        device_type: str = "cuda",
        **kwargs: Any,
    ) -> Tensor:
        """
        Forward pass of the regressor.

        Args:
            batch: Dictionary containing the input data.
            frozen_backbone: Whether to freeze the trunk during forward pass.
            device_type: Device type for autocast.
            **kwargs: Additional arguments for the trunk model.

        Returns:
            Tensor: Output of the regressor.
        """
        if frozen_backbone:
            self.trunk.eval()  # To have similar behavior as the precomputed case
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device_type,
                    dtype=self.trunk_autocast_precision,
                ),
            ):
                embeddings = self.trunk(batch, reduction_method=self.reduction_method, **kwargs)
        else:
            self.trunk.train()
            with (
                torch.autocast(
                    device_type=device_type,
                    dtype=self.trunk_autocast_precision,
                ),
            ):
                embeddings = self.trunk(batch, reduction_method=self.reduction_method, **kwargs)

        # Wrap head operations in autocast to ensure computation matches head_precision
        with torch.autocast(device_type=device_type, dtype=self.head_precision):
            return self.head(embeddings)


class StructureRegressor(torch.nn.Module):
    def __init__(
        self,
        trunk: StructureEncoder,  # Potentially Encoder instead of SequenceEncoder (TokenEncoder would be covered too)
        use_low_rank: bool = False,
        hidden_size: int = 512,
        use_attn_maps: bool = False,  # Whether to use the attention maps or embeddings from the trunk for structure prediction
        output_size: int = 2,
        head_precision: torch.dtype = torch.float32,
        trunk_autocast_precision: torch.dtype = torch.float16,
        # dim_total_attn_maps: int | None = None,  # If not None, the attention maps will be used as input to the head
    ):
        super(StructureRegressor, self).__init__()
        self.trunk = trunk
        self.head_precision = head_precision
        self.trunk_autocast_precision = trunk_autocast_precision
        # Enforce two-class output for contact prediction (neg, pos)
        if output_size != 2:
            raise ValueError(
                f"StructureRegressor must output 2 logits per pair (neg,pos). Got output_size={output_size}"
            )

        self.use_low_rank = use_low_rank
        if self.use_low_rank:
            raise NotImplementedError("Low-rank regression is not implemented yet.")
        self.use_attn_maps = use_attn_maps
        self.use_attn_maps = trunk.use_attn_maps

        if self.use_attn_maps:
            self.head = MLP(self.trunk.num_total_attn_maps, hidden_size, output_size).to(self.head_precision)
        else:
            self.head = MLP(2 * self.trunk.hidden_size, hidden_size, output_size).to(self.head_precision)

    # TODO change the forward to match Xtrimo
    def forward(
        self,
        batch: Dict[str, Tensor],
        frozen_backbone=False,
        use_precomputed_embeddings=False,
        device_type: str = "cuda",
        **kwargs: Any,
    ) -> Tensor:
        """
        Forward pass of the structure regressor.

        Args:
            batch: Dictionary containing the input data.
            frozen_backbone: Whether to freeze the trunk during forward pass.
            use_precomputed_embeddings: If True, use precomputed embeddings from batch["embeddings"] and bypass trunk.
            device_type: Device type for autocast.
            **kwargs: Additional arguments for the trunk model.

        Returns:
            Tensor: Nested tensor of contact maps with shape (seq_len, seq_len, output_size) for each sequence.
        """
        # Use precomputed embeddings if specified
        if use_precomputed_embeddings:
            # Bypass the trunk and use precomputed embeddings directly
            embeddings = batch["embeddings"]
            # Get the device of the head parameters
            head_device = next(self.head.parameters()).device
            # Convert list of tensors to nested tensor if needed
            if isinstance(embeddings, list):
                # Move to correct device and dtype with non-blocking transfer
                embeddings = torch.nested.as_nested_tensor(
                    [emb.to(device=head_device, dtype=self.head_precision, non_blocking=True) for emb in embeddings]
                )
            else:
                embeddings = embeddings.to(device=head_device, dtype=self.head_precision, non_blocking=True)
        elif frozen_backbone:
            self.trunk.eval()
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device_type,
                    dtype=self.trunk_autocast_precision,
                ),
            ):
                embeddings = self.trunk(batch, **kwargs)
        else:
            self.trunk.train()
            with (
                torch.autocast(
                    device_type=device_type,
                    dtype=self.trunk_autocast_precision,
                ),
            ):
                embeddings = self.trunk(batch, **kwargs)

        # Wrap head operations in autocast to ensure fp16 computation
        with torch.autocast(device_type=device_type, dtype=self.head_precision):
            if self.use_attn_maps:
                all_contact_maps = []
                for embedding in embeddings:
                    all_contact_maps.append(
                        self.head(embedding.permute(1, 2, 0))
                    )  # Shape: (seq_len, seq_len, output_size)
                all_contact_maps = torch.nested.as_nested_tensor(all_contact_maps)
            else:
                all_contact_maps = []
                for embedding in embeddings:
                    # embedding has shape (seq_len, hidden_size)
                    # contact(i,j) = mlp(concat(embedding_i, embedding_j))
                    pairwise_features = torch.cat(
                        [
                            embedding.unsqueeze(1).repeat(1, embedding.shape[0], 1),
                            embedding.unsqueeze(0).repeat(embedding.shape[0], 1, 1),
                        ],
                        dim=-1,
                    )
                    # Shape: (seq_len, seq_len, 2 * hidden_size)
                    contact_embedding = self.head(pairwise_features)  # Shape: (seq_len, seq_len, output_size)
                    # Symmetrize the contact map
                    contact_embedding = (contact_embedding + contact_embedding.transpose(0, 1)) / 2
                    all_contact_maps.append(contact_embedding)
                all_contact_maps = torch.nested.as_nested_tensor(all_contact_maps)
        return all_contact_maps
