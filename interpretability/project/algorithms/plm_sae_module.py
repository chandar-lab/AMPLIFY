"""Lightning module for training Sparse Autoencoders (SAE) on language model activations."""

import dataclasses
from dataclasses import dataclass
from logging import getLogger

import hydra_zen
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from project.algorithms.networks.auto_encoder import AutoEncoder, make_model
from project.algorithms.networks.protein_language_model import ProteinLanguageModel

logger = getLogger(__name__)


@hydra_zen.hydrated_dataclass(
    target=ProteinLanguageModel,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class ProteinLanguageModelConfig:
    model_name: str
    layer_to_use: int


@hydra_zen.hydrated_dataclass(
    target=make_model,
    frozen=False,
    unsafe_hash=True,
    populate_full_signature=True,
)
class SAEConfig:
    """Configuration for the Sparse Autoencoder.

    Supports two modes:
    1. Pretrained loading: specify compatible HF plm_model and plm_layer
    2. Manual creation: specify sae_target, dict_size, etc.
    """

    plm_model: str | None = None  # "esm2-8m" or "esm2-650m"
    plm_layer: int | None = None  # embedding layer the SAE was pretrained on

    sae_target: str = "project.algorithms.networks.auto_encoder.TransposeWeightAutoEncoder"
    input_dim: int | None = None
    dict_size: int = 2048  # Dictionary size for SAE
    k: int | None = None  # For TopKAutoEncoder only

    sparsity_weight: float = 1e-3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    adam_betas: tuple[float, float] = (0.9, 0.999)

    use_scheduler: bool = True
    warmup_steps: int = 1000

    log_every_n_steps: int = 100


class SAELightningModule(LightningModule):
    """Lightning module for training Sparse Autoencoders on frozen PLM activations."""

    def __init__(
        self,
        plm_config: ProteinLanguageModelConfig,
        sae_config: SAEConfig,
        training_config: TrainingConfig,
        **kwargs,
    ):
        """
        Initializes the SAELightningModule.

        Args:
            plm_config: Configuration for the frozen Protein Language Model.
            sae_config: Configuration for the Sparse Autoencoder.
            training_config: Configuration for training hyperparameters.
        """
        super().__init__()
        self.plm_config = plm_config
        self.sae_config = sae_config
        self.training_config = training_config

        self.save_hyperparameters(
            dict(
                plm_config=dataclasses.asdict(plm_config),
                sae_config=dataclasses.asdict(sae_config),
                training_config=dataclasses.asdict(training_config),
            )
        )  # Save hyperparameters with WandB for reproducibility

        # Initialize components (will be created in configure_model)
        self.protein_language_model: ProteinLanguageModel | None = None
        self.sae: AutoEncoder | None = None

    def configure_model(self) -> None:
        """Initialize the models here after the module is created to avoid creating large objects
        if loading from checkpoint."""
        if self.protein_language_model is not None:
            return  # Already configured

        logger.info(f"Configuring models on device: {self.device}")

        self.protein_language_model = hydra_zen.instantiate(self.plm_config)

        hidden_size = self.protein_language_model.model.config.hidden_size
        if self.sae_config.plm_model:
            logger.info("Loading pretrained SAE from Hugging Face...")
            self.sae = AutoEncoder.from_pretrained(
                plm_model=self.sae_config.plm_model,
                plm_layer=self.sae_config.plm_layer,
                device=str(self.device),
            )
            assert (
                self.sae.input_dim == hidden_size
            ), f"SAE input dimension mismatch: {self.sae.input_dim} != {hidden_size}"
        else:
            logger.info("Creating new SAE from scratch...")
            self.sae_config.input_dim = hidden_size
            self.sae = hydra_zen.instantiate(self.sae_config)

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the SAE
        Args:
            batch: dict containing input tensors

        Returns:
            Dictionary containing SAE outputs and metrics
        """

        with torch.inference_mode(): 
            # We use inference_mode() for efficiency as we don't need gradients for the PLM
            activations, attention_mask, attentions = self.protein_language_model(
                seq
            )  # [batch, max_seq_len, hidden_dim]

            mask = attention_mask.unsqueeze(-1).expand_as(
                activations
            )  # [batch, max_seq_len, hidden_dim]
            activations = activations * mask
            activations = rearrange(activations, "b s h -> (b s) h")
            attention_mask = rearrange(attention_mask, "b s -> (b s)")

            # Normalize activations before passing into MLPs
            # Note: L2-normalization is used here. Z-score normalization might be statistically 
            # more appropriate but can introduce data poisoning/leakage issues across the batch.
            activations = F.normalize(
                activations, dim=-1
            )

        reconstructions, sparse_latent = self.sae(activations)
        return {
            "original_activations": activations,
            "reconstructions": reconstructions,
            "sparse_latent": sparse_latent,
            "attention_mask": attention_mask,
            "num_valid_tokens": mask.sum().item(),
        }

    def compute_loss(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute SAE training loss."""

        original = outputs["original_activations"]
        reconstructed = outputs["reconstructions"]
        sparse_latent = outputs["sparse_latent"]
        attention_mask = outputs["attention_mask"]

        valid_indices = attention_mask.bool()
        if valid_indices.sum() > 0:  # Only compute loss on valid (non-padded) tokens
            valid_original = original[valid_indices]
            valid_reconstructed = reconstructed[valid_indices]
            valid_sparse_latent = sparse_latent[valid_indices]

            reconstruction_loss = F.mse_loss(valid_reconstructed, valid_original)
            sparsity_loss = self.sae.compute_sparsity_loss(valid_sparse_latent)

        else:  # weird edge case with no valid tokens in the entire mini-batch
            reconstruction_loss = torch.tensor(0.0, device=original.device, requires_grad=True)
            sparsity_loss = torch.tensor(0.0, device=original.device, requires_grad=True)

        # Total loss
        loss = reconstruction_loss + self.sae_config.sparsity_weight * sparsity_loss

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs = batch["sequence"]

        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs)
        self.log_metrics(loss, prefix="train")
        return loss["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs = batch["sequence"]

        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs)
        self.log_metrics(loss, prefix="val")
        return loss["loss"]

    def log_metrics(self, loss: dict[str, torch.Tensor], prefix: str = "") -> None:
        """Log metrics with optional prefix for train/val distinction."""
        for key, value in loss.items():
            if prefix:
                log_key = f"{prefix}/{key}"
            else:
                log_key = key
            self.log(log_key, value)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.sae.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            eps=self.training_config.adam_epsilon,
            betas=self.training_config.adam_betas,
        )

        if not self.training_config.use_scheduler:
            return optimizer
        else:
            raise NotImplementedError("CosineAnnealingLR is not implemented")
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=1000, # Placeholder value. TODO: Determine correct T_max based on actual training steps/epochs. 
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    def on_train_start(self) -> None:
        logger.info("Starting SAE training...")
        logger.info(f"SAE type: {type(self.sae).__name__}")
        logger.info(f"Input dim: {self.sae.input_dim}")
        logger.info(f"Dict size: {self.sae.dict_size}")
        if hasattr(self.sae, "k"):
            logger.info(f"Top-k: {self.sae.k}")

    def on_validation_start(self) -> None:
        logger.info("Starting SAE validation...")
