import os
import sys
import torch
import signal
import shutil
import logging
from tqdm import tqdm
from typing import Optional
from types import FrameType
from omegaconf import OmegaConf, DictConfig

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed

from ..model import (
    AMPLIFYConfig,
    AMPLIFYForMaskedLM,
    AMPLIFYForSequenceClassification,
    AMPLIFYForTokenClassification,
    AMPLIFYModel,
    ProteinTokenizer,
    TokenizerConfig,
)
from ..metric import TrainingLogger
from ..dataset import get_dataloader, CollatorConfig, DataLoaderConfig
from ..scheduler import get_scheduler
from ..optimizer import get_optimizer

# Suppress warning that sequences are being truncated during pre-training.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    logger: TrainingLogger,
    split_name: str,
) -> None:
    """Run one evaluation pass and accumulate eval counters in ``logger``.

    Args:
        model: Wrapped AMPLIFY model.
        dataloader: Validation dataloader for one eval split.
        logger: Logger that stores eval counters until next log.
        split_name: Eval split name.
    """
    # Switch to eval mode so dropout/batchnorm are deterministic.
    model.eval()

    # Forward each validation batch without autograd state.
    with torch.inference_mode():
        for batch in dataloader:
            output = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                position_ids=batch["position_ids"],
                attention_mask=batch.get("attention_mask"),
                cu_seqlens=batch.get("cu_seqlens"),
                max_seqlen=batch.get("max_seqlen"),
            )
            logger.add_eval_batch(split_name, batch, output)

    # Restore training mode for the next optimizer step.
    model.train()


def save_hf_checkpoint(
    model: torch.nn.Module,
    tokenizer: ProteinTokenizer,
    accelerator: Accelerator,
    hf_checkpoint_dir: str,
    step: int,
) -> None:
    """Save a Hugging Face checkpoint for inference/loading.

    Args:
        model: Accelerate-wrapped model (unwrapped before save).
        tokenizer: Tokenizer saved next to model weights/config.
        accelerator: Accelerator instance for coordinated saving.
        hf_checkpoint_dir: Parent directory for ``checkpoint_<step>`` folders.
        step: Global training step used in checkpoint folder name.
    """
    # Save model weights/config + tokenizer. Unwraps mixed-precision wrappers.
    path = os.path.join(hf_checkpoint_dir, f"checkpoint_{step}")
    os.makedirs(path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)

    # Make local checkpoints self-contained for Hugging Face Auto* loading.
    AMPLIFYConfig.register_for_auto_class()
    AMPLIFYModel.register_for_auto_class("AutoModel")
    AMPLIFYForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
    AMPLIFYForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
    AMPLIFYForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
    tokenizer.register_for_auto_class("AutoTokenizer")
    unwrapped_model.config.auto_map = {
        "AutoConfig": "amplify.AMPLIFYConfig",
        "AutoModel": "amplify.AMPLIFYModel",
        "AutoModelForMaskedLM": "amplify.AMPLIFYForMaskedLM",
        "AutoModelForSequenceClassification": "amplify.AMPLIFYForSequenceClassification",
        "AutoModelForTokenClassification": "amplify.AMPLIFYForTokenClassification",
    }

    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(path)


def trainer(cfg: DictConfig) -> None:
    """Main AMPLIFY training entrypoint.

    Args:
        cfg: Hydra config with ``model``, ``data``, ``training_args`` and ``wandb`` sections.
    """
    # Resolve output folders and detect latest checkpoint when resuming.
    full_checkpoint_dir = os.path.join(cfg.training_args.output_dir, "checkpoints")
    hf_checkpoint_dir = os.path.join(cfg.training_args.output_dir, "hf_checkpoints")
    latest_checkpoint = None

    # Resume policy: fresh runs clear previous state; resume runs pick latest checkpoint_*.
    if cfg.training_args.resume_from_checkpoint is False:
        shutil.rmtree(full_checkpoint_dir, ignore_errors=True)
        shutil.rmtree(hf_checkpoint_dir, ignore_errors=True)
    elif os.path.exists(full_checkpoint_dir):
        checkpoints = [
            d for d in os.listdir(full_checkpoint_dir) if d.startswith("checkpoint_") and d.split("_")[1].isdigit()
        ]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda d: int(d.split("_")[1]))
    os.makedirs(hf_checkpoint_dir, exist_ok=True)

    # Configure Accelerate runtime (DDP/FSDP, grad accumulation, checkpoint policy).
    project_config = ProjectConfiguration(
        cfg.training_args.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.training_args.save_total_limit,
        iteration=0 if latest_checkpoint is None else int(latest_checkpoint.split("_")[1]) + 1,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=cfg.training_args.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=cfg.seed,
            even_batches=False,
            non_blocking=cfg.training_args.dataloader_pin_memory,
        ),
    )

    # Initialize Weights & Biases experiment tracking.
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    wandb_config = OmegaConf.to_container(cfg)
    wandb_config["distributed_type"] = accelerator.distributed_type
    wandb_config["mixed_precision"] = accelerator.mixed_precision

    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": wandb_config,
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.training_args.resume_from_checkpoint,
            }
        },
    )

    # Seed all RNGs for deterministic shuffling and masking.
    set_seed(cfg.seed)

    # Hardware optimization flags for NVIDIA GPUs.
    torch.set_float32_matmul_precision("high")
    if cfg.training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model + optimizer + scheduler from config values.
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("tokenizer")
    model = AMPLIFYForMaskedLM(AMPLIFYConfig(**model_config))
    optimizer = get_optimizer(
        model,
        type=cfg.training_args.optimizer.type,
        lr=cfg.training_args.optimizer.lr,
        weight_decay=cfg.training_args.optimizer.weight_decay,
        betas=tuple(cfg.training_args.optimizer.betas),
        eps=cfg.training_args.optimizer.eps,
        fused=cfg.training_args.optimizer.fused,
    )
    scheduler = get_scheduler(
        optimizer,
        lr_scheduler_type=cfg.training_args.scheduler.type,
        num_warmup_steps=cfg.training_args.scheduler.num_warmup_steps,
        num_training_steps=cfg.training_args.scheduler.num_training_steps,
        scheduler_specific_kwargs=OmegaConf.to_container(cfg.training_args.scheduler.get("kwargs"), resolve=True),
    )

    # Build tokenizer/collator and train/validation dataloaders.
    tokenizer = ProteinTokenizer(TokenizerConfig(**cfg.model.tokenizer), max_length=cfg.training_args.max_length)
    assert model.config.vocab_size == len(tokenizer), "Model and Tokenizer vocab_size are different"

    # Logger is checkpointed so counters resume correctly after restarts.
    logger = TrainingLogger()
    accelerator.register_for_checkpointing(logger)
    logger_json_path = os.path.join(cfg.wandb.dir, "wandb", "metrics.jsonl")

    # Configure how sequences are masked, truncated, and padded/packed for MLM.
    packing = cfg.training_args.get("packing", False)
    col_cfg = CollatorConfig(
        max_length=cfg.training_args.max_length,
        packing=packing,
        mlm_probability=cfg.training_args.mlm_probability,
        pad_to_multiple_of=cfg.training_args.pad_to_multiple_of,
        random_truncate=cfg.training_args.random_truncate,
        exclude_special_tokens_from_masking=cfg.training_args.exclude_special_tokens_from_masking,
        mlm=cfg.training_args.mlm,
    )

    # Train split is merged into one dataloader.
    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        col_cfg=col_cfg,
        dl_cfg=DataLoaderConfig(
            paths=cfg.data.train_files,
            shuffle=cfg.data.shuffle,
            drop_last=cfg.training_args.dataloader_drop_last,
            num_proc=cfg.data.num_proc,
            per_device_batch_size=cfg.training_args.per_device_train_batch_size,
            per_device_batch_tokens=cfg.training_args.per_device_train_batch_tokens,
            dataloader_num_workers=cfg.training_args.dataloader_num_workers,
            persistent_workers=cfg.training_args.dataloader_persistent_workers,
            pin_memory=cfg.training_args.dataloader_pin_memory,
            prefetch_factor=cfg.training_args.dataloader_prefetch_factor,
            merge=True,
        ),
    )
    # Validation splits stay separate to log per-split metrics.
    eval_dataloaders = get_dataloader(
        tokenizer=tokenizer,
        col_cfg=col_cfg,
        dl_cfg=DataLoaderConfig(
            paths=cfg.data.validation_files,
            shuffle=False,
            drop_last=cfg.training_args.dataloader_drop_last,
            num_proc=cfg.data.num_proc,
            per_device_batch_size=cfg.training_args.per_device_eval_batch_size,
            per_device_batch_tokens=cfg.training_args.per_device_eval_batch_tokens,
            dataloader_num_workers=cfg.training_args.dataloader_num_workers,
            persistent_workers=cfg.training_args.dataloader_persistent_workers,
            pin_memory=cfg.training_args.dataloader_pin_memory,
            prefetch_factor=cfg.training_args.dataloader_prefetch_factor,
        ),
    )

    # Let Accelerate place/wrap objects for the selected distributed mode.
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    eval_dataloaders = {
        name: accelerator.prepare(eval_dataloader) for name, eval_dataloader in eval_dataloaders.items()
    }
    # Used for reporting only; the length of token-budget dataloaders are approximate.
    logger.steps_per_epoch = len(train_dataloader) / cfg.training_args.gradient_accumulation_steps

    # Restore full training state; skip already-consumed batches when resuming mid-epoch.
    if cfg.training_args.resume_from_checkpoint and latest_checkpoint is not None:
        accelerator.load_state(os.path.join(full_checkpoint_dir, latest_checkpoint))
        num_batches_to_skip = logger.num_batches_in_epoch
    else:
        num_batches_to_skip = 0

    # Persist state on preemption-friendly SIGTERM (e.g., SLURM).
    def handler(signum: int, frame: Optional[FrameType]) -> None:
        """Save a full checkpoint and exit cleanly on SIGTERM."""
        print(f"Signal {signum} received on rank {accelerator.process_index}, checkpointing...")
        accelerator.save_state()
        accelerator.wait_for_everyone()
        print(f"Done on rank {accelerator.process_index}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)

    # Main train loop until max_steps is reached.
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=logger.num_steps,
        total=cfg.training_args.max_steps,
        disable=(cfg.training_args.disable_tqdm or not accelerator.is_main_process),
    )

    while cfg.training_args.max_steps > logger.num_steps:
        # Keep sampler shuffling deterministic across epochs in distributed mode.
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(logger.num_epochs)

        # On resume, skip mini-batches already consumed in the current epoch.
        if num_batches_to_skip > 0:
            print(
                f"Resuming from step {logger.num_steps}, epoch {logger.num_epochs}. "
                f"Skipping {num_batches_to_skip} batches in current epoch."
            )
            dataloader = accelerator.skip_first_batches(train_dataloader, num_batches_to_skip)
            num_batches_to_skip = 0
        else:
            dataloader = train_dataloader

        for batch in dataloader:
            # Accelerate controls when gradients are synchronized.
            with accelerator.accumulate(model):
                # Forward pass through AMPLIFY (model computes loss internally).
                output = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch.get("attention_mask"),
                    cu_seqlens=batch.get("cu_seqlens"),
                    max_seqlen=batch.get("max_seqlen"),
                )

                # Backpropagate current micro-batch loss.
                accelerator.backward(output.loss)

                # Accumulate train metrics on GPU for this micro-batch.
                logger.add_train_batch(batch, output, output.loss)

                # Clip gradients on sync boundaries.
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training_args.optimizer.max_grad_norm or float("inf")
                    )

                # Accelerate makes these no-ops on non-sync (gradient accumulation) steps.
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                logger.num_batches_in_epoch += 1

                if accelerator.sync_gradients:
                    # Scheduler is stepped on optimizer-sync steps only.
                    scheduler.step()

                    pbar.update(1)
                    logger.num_steps += 1

                    # Periodic evaluation on all validation splits.
                    if cfg.training_args.eval_steps > 0 and logger.num_steps % cfg.training_args.eval_steps == 0:
                        for name, eval_dataloader in eval_dataloaders.items():
                            evaluate(model, eval_dataloader, logger, name)

                    # Periodic logging: reduce metrics across ranks and log to wandb.
                    if cfg.training_args.logging_steps > 0 and logger.num_steps % cfg.training_args.logging_steps == 0:
                        logger.log(optimizer, accelerator, logger_json_path)

                    # Periodic full checkpoint (model + optimizer + scheduler + logger state).
                    if cfg.training_args.save_steps > 0 and logger.num_steps % cfg.training_args.save_steps == 0:
                        torch.cuda.empty_cache()
                        accelerator.save_state()

                    # Periodic HF checkpoint (model weights + tokenizer only, for inference).
                    if cfg.training_args.hf_save_steps > 0 and logger.num_steps % cfg.training_args.hf_save_steps == 0:
                        torch.cuda.empty_cache()
                        save_hf_checkpoint(
                            model,
                            tokenizer,
                            accelerator,
                            hf_checkpoint_dir,
                            logger.num_steps,
                        )

                    if logger.num_steps >= cfg.training_args.max_steps:
                        break

        logger.num_epochs += 1
        logger.num_batches_in_epoch = 0

    # Clean shutdown for progress bar + tracker backends.
    pbar.close()
    accelerator.end_training()
