import os
import re
import sys
import torch
import signal
import shutil
from tqdm import tqdm
from typing import Tuple
from omegaconf import OmegaConf, DictConfig

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from deepspeed.utils import safe_get_full_fp32_param

from amplify.config import config_schema, ConfigError
from amplify.model import AMPLIFY, AMPLIFYConfig
from amplify.metric import Metrics
from amplify.loss import get_loss
from amplify.dataset import get_dataloader
from amplify.scheduler import get_scheduler
from amplify.optimizer import get_optimizer


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    vocab_size: int,
) -> Tuple[int, int, int]:
    """Evaluate the model on the dataloader provided."""
    model.eval()
    sum_val_loss, num_val_correct, num_val_pred = 0, 0, 0
    with torch.no_grad():
        for x, y, pad_mask in dataloader:
            logits = model(x, pad_mask).logits
            val_loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            num_val_pred += torch.sum(y != -100).item()
            sum_val_loss += val_loss.item() * torch.sum(y != -100).item()
            num_val_correct += torch.sum(torch.argmax(logits, dim=-1) == y).item()
    model.train()
    return num_val_pred, sum_val_loss, num_val_correct


def trainer(cfg: DictConfig) -> None:
    """Entrypoint for training a model with the given configuration."""
    config_check = config_schema.validate(cfg)
    if not config_check.is_ok():
        raise ConfigError(config_check)
    it = 0
    chk_dir = os.path.join(cfg.trainer.dir, "checkpoints")

    # Delete the folder if resume is disabled and folder exists
    if cfg.trainer.resume is False:
        shutil.rmtree(chk_dir, ignore_errors=True)
    elif os.path.exists(chk_dir):
        it = max(int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in os.listdir(chk_dir))
        # Remove empty checkpoint folders
        while len(os.listdir(os.path.join(chk_dir, f"checkpoint_{it}"))) == 0:
            shutil.rmtree(os.path.join(chk_dir, f"checkpoint_{it}"), ignore_errors=True)
            it -= 1

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.max_checkpoints,
        iteration=it + 1,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        project_config=project_config,
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.trainer.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.trainer.tf32)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Model, optimizer, and learning rate scheduler
    model = AMPLIFY(AMPLIFYConfig(**cfg.model, **cfg.tokenizer))
    optimizer = get_optimizer(model, **cfg.optimizer)
    scheduler = get_scheduler(optimizer, **cfg.scheduler)

    # Log the number of parameters (this logs to the console instead of wandb)
    print({"model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    # Get the dtype for the pad_mask and class_weights
    dtype_pad_mask, dtype_class_weight = torch.float32, torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            dtype_class_weight = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            dtype_class_weight = torch.bfloat16

    # Train and validation Dataloaders
    train_dataloader = get_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.train,
        **cfg.trainer.train,
        merge=True,
        return_labels=False,
        dtype=dtype_pad_mask,
    )
    eval_dataloaders = get_dataloader(
        **cfg.tokenizer,
        **cfg.dataset.validation,
        **cfg.trainer.validation,
        merge=False,
        return_labels=False,
        dtype=dtype_pad_mask,
    )

    # Prepare for distributed training
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}

    # Get loss functions
    train_loss_fn = get_loss(accelerator.device, **cfg.tokenizer, **cfg.trainer.train, dtype=dtype_class_weight)
    val_loss_fn = get_loss(accelerator.device, **cfg.tokenizer, **cfg.trainer.validation, dtype=dtype_class_weight)

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume and os.path.exists(os.path.join(cfg.trainer.dir, "checkpoints")):
        accelerator.load_state()
        skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["num_batches_in_epoch"])

    # Save the model when receiving the signal SIGTERM
    def handler(signum, frame):
        print(f"Signal {signum} received on rank {accelerator.process_index}, checkpointing...")
        accelerator.save_state()
        accelerator.wait_for_everyone()
        print(f"Done on rank {accelerator.process_index}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["num_steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    while cfg.trainer.max_steps > metrics["num_steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = train_dataloader if skipped_train_dataloader is None else skipped_train_dataloader

        for x, y, pad_mask in dataloader:
            # Increment the number of batches
            metrics["local_num_batches"] += 1

            # Gradient accumulation logic
            if metrics["local_num_batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    logits = model(x, pad_mask).logits
                    train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1))

                    # Update local metrics
                    metrics["num_batches_in_epoch"] += 1
                    metrics["local_num_samples"] += x.shape[0]
                    metrics["local_num_tokens"] += (pad_mask == 0).sum().item()
                    metrics["local_num_train_pred"] += torch.sum(y != -100).item()
                    metrics["local_sum_train_loss"] += train_loss.item() * torch.sum(y != -100).item()
                    metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                    # Backpropagation
                    accelerator.backward(train_loss)
            else:
                logits = model(x, pad_mask).logits
                train_loss = train_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), y.view(-1))

                # Log metrics and update progress bar
                pbar.update(1)
                metrics["num_steps"] += 1
                metrics["num_batches_in_epoch"] += 1
                metrics["local_num_samples"] += x.shape[0]
                metrics["local_num_tokens"] += (pad_mask == 0).sum().item()
                metrics["local_num_train_pred"] += torch.sum(y != -100).item()
                metrics["local_sum_train_loss"] += train_loss.item() * torch.sum(y != -100).item()
                metrics["local_num_train_correct"] += torch.sum(torch.argmax(logits, dim=-1) == y).item()

                # Backpropagation
                accelerator.backward(train_loss)

                # Evaluate the model
                if metrics["num_steps"] % cfg.trainer.eval_steps == 0:
                    for k, v in eval_dataloaders.items():
                        num_val_pred, sum_val_loss, num_val_correct = evaluate(
                            model,
                            v,
                            val_loss_fn,
                            cfg.tokenizer.vocab_size,
                        )
                        metrics[f"local_{k}_sum_val_loss"] = sum_val_loss
                        metrics[f"local_{k}_num_val_correct"] = num_val_correct
                        metrics[f"local_{k}_num_val_pred"] = num_val_pred

                # Gradient clipping
                if cfg.trainer.gradient_clipping is not None and cfg.trainer.gradient_clipping > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()

                # Reset the gradient
                optimizer.zero_grad()

                # Save the model from the main process
                if metrics["num_steps"] % cfg.trainer.save_steps == 0:
                    accelerator.save_state()

                if metrics["num_steps"] >= cfg.trainer.max_steps:
                    break

        metrics["num_epochs"] += 1
        metrics["num_batches_in_epoch"] = 0

        # "Remove" the skipped dataloader once exhausted
        skipped_train_dataloader = None

    pbar.close()
    accelerator.end_training()
