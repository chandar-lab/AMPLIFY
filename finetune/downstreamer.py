"""main code related to train/test"""

import gc
import logging
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torchmetrics
import wandb
from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import (
    AUROC,
    Accuracy,
    MatthewsCorrCoef,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from tqdm import tqdm
from transformers import AutoTokenizer

from metric import Fmax, LongRangePrecisionAtL
from model import ContactPredictionModel, PointPredictionModel, PPIModel
from protein_dataset import create_transform_collate, obtain_real_residue_mask


@dataclass
class EvalParams:
    task_name: str
    task_output_type: str
    task_num_labels: int
    tokenizer: AutoTokenizer
    data_type: torch.dtype
    label_type: torch.dtype
    device: torch.device
    metric_fn: torchmetrics.Metric


# Use logger
logger = logging.getLogger(__name__)


@torch.no_grad()
def downstream_valid(
    model: torch.nn.Module,
    dataloader_valid: DataLoader,
    eval_params: EvalParams,
) -> torch.Tensor:
    """
    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader_valid (DataLoader): A DataLoader that provides validation data batches.
        eval_params (EvalParams): An object containing evaluation parameters such as device, data type, label type,
                                  task name, task output type, number of labels, tokenizer, and the metric function.

    Returns:
        torch.Tensor: The computed validation metric.
    """
    model.eval()
    eval_params.metric_fn.reset()

    labels = []
    outputs = []

    if eval_params.task_name == "Bo1015/contact_prediction_binary":
        effective_Ls = []

    for batch in dataloader_valid:
        # Convert to correct dtype and move to GPU
        input_ids = batch["input_ids"].to(torch.long).to(eval_params.device)
        attention_mask = (
            batch["attention_mask"].to(eval_params.data_type).to(eval_params.device)
        )
        real_residue_mask = obtain_real_residue_mask(input_ids, eval_params.tokenizer)
        label = batch["labels"].to(eval_params.label_type).to(eval_params.device)
        if eval_params.task_name in ["saprot_data/HumanPPI"]:
            input_ids_2 = batch["input_ids_2"].to(torch.long).to(eval_params.device)
            attention_mask_2 = (
                batch["attention_mask_2"]
                .to(eval_params.data_type)
                .to(eval_params.device)
            )
            output = model(
                input_ids,
                attention_mask,
                input_ids_2,
                attention_mask_2,
                frozen_trunk=True,
            )
        else:
            output = model(input_ids, attention_mask, frozen_trunk=True)

        if eval_params.task_name == "Bo1015/contact_prediction_binary":
            for i in range(output.shape[0]):
                labels.append(label[i])
                outputs.append(output[i])
                effective_Ls.append(int(real_residue_mask[i].sum().item()))
        else:
            if eval_params.task_output_type == "residue":
                # extract the real residue excluding pad/bos/eos
                label = label[real_residue_mask == True]
                output = output[real_residue_mask == True]
            labels.append(label)
            outputs.append(output)
        # break

    if eval_params.task_name not in ["Bo1015/contact_prediction_binary"]:
        preds = (
            torch.cat(outputs, dim=0).view(-1, eval_params.task_num_labels).squeeze()
        )
        labels = torch.cat(labels, dim=0)
        val_metric = eval_params.metric_fn(preds, labels)
    else:
        for pred, gt, effective_L in zip(outputs, labels, effective_Ls):
            eval_params.metric_fn.update(pred, gt, effective_L)
        val_metric = eval_params.metric_fn.compute()

    model.train()
    eval_params.metric_fn.reset()

    return val_metric


def downstream(config, task_name):
    """
    Run the downstream training and evaluation pipeline.

    This function loads the dataset, initializes the tokenizer, model, optimizer, scheduler,
    and evaluation metrics based on the given task, and then conducts training with periodic
    validation (or testing) and checkpointing. It also supports resuming from checkpoints and
    logs key metrics using wandb.

    Args:
        config: Configuration object containing training parameters.
        task_name (str): Identifier for the downstream task.

    Returns:
        None
    """
    logger.info(f"config: {config}, task_name: {task_name}")

    # Define device: we only use one GPU for downstream tasks.
    task_config = OmegaConf.load("config/task.yaml")
    device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")


    # Define save_dir
    prt_model_safe = config.prt_model_name.split("/")[-1]
    output_dir = os.path.join(
        "output",
        f"{prt_model_safe}",
        f"{config.ft_model_path}",
        f"{task_name}",
        f"{config.seed}"
    )

    if config.ft_model_path != "None":
        config.ft_model_path = os.path.join('checkpoint', config.ft_model_path, 'best/model_trunk.pt')
    else:
        config.ft_model_path = None

    latest_ckpt_dir = os.path.join(output_dir, "latest")
    best_ckpt_dir = os.path.join(output_dir, "best")
    os.makedirs(latest_ckpt_dir, exist_ok=True)
    os.makedirs(best_ckpt_dir, exist_ok=True)

    # Use wandb
    wandb_id = "_".join(output_dir.split("/")[-3:])
    wandb.init(
        project="structure-aware-plm",
        name=wandb_id,
        entity="drug-discovery-amgen",
        config=OmegaConf.to_container(config, resolve=True),
        id=wandb_id,
        dir=output_dir,
        resume=config.get("resume", True),
        mode="offline",
    )
    logger.info("wandb initialized.")

    # Define tokenizer
    if config.prt_model_name == "ism":
        tokenizer = AutoTokenizer.from_pretrained(
            "checkpoint/ISM/ism_model"
        )
    elif config.prt_model_name == "esm-s":
        tokenizer = AutoTokenizer.from_pretrained(
            "checkpoint/ESM-s/esm_s_model"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.prt_model_name, trust_remote_code=True
        )
    logger.info("Tokenizer loaded.")

    # Load data
    if task_name.split("/")[0] == "Bo1015":
        all_data = load_dataset(task_name)
    elif task_name.split("/")[0] == "saprot_data":
        all_data = load_from_disk(task_name)
    else:
        raise ValueError(
            f"Unsupported task: {task_name}. Please check your task name and data source."
        )

    all_data = all_data.rename_column("label", "labels")
    if task_name in ["Bo1015/fitness_prediction"]:

        def string2float(sample):
            sample["labels"] = float(sample["labels"])
            return sample

        for data_key in all_data:
            all_data[data_key] = all_data[data_key].map(string2float)

    normalization = task_config[task_name].loss_type == "regression"
    if normalization:
        target_mean, target_std = np.mean(all_data["train"]["labels"]), np.std(
            all_data["train"]["labels"]
        )
    else:
        target_mean, target_std = 0, 1
    logger.info("Dataset loaded and processed.")

    # Define transform_fn and collate_fn
    task_output_type = task_config[task_name].output_type
    transform_fn, collate_fn = create_transform_collate(
        task_name, task_output_type, tokenizer, max_len=config.get("max_len", 2048)
    )
    all_data.set_transform(transform_fn)
    logger.info("Collate_fn set.")

    # Define data loaders
    num_workers = 4
    train_sampler = DistributedSampler(all_data["train"], num_replicas=1, rank=0)
    dataloader_train = DataLoader(
        all_data["train"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    if "valid" in all_data:
        dataloader_valid = DataLoader(
            all_data["valid"],
            collate_fn=collate_fn,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        dataloader_valid = None

    dataloader_test = DataLoader(
        all_data["test"],
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
    )
    logger.info("Data loaders created.")

    # Load fine-tuned model/tokenizer
    task_num_labels = task_config[task_name].num_labels
    data_type = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        config.precision
    ]
    if task_name in ["Bo1015/contact_prediction_binary"]:
        # residue-residue
        model = ContactPredictionModel(
            prt_model_name=config.prt_model_name,
            ft_model_path=config.ft_model_path,
            task_num_labels=task_num_labels,
        )
    elif task_name in ["saprot_data/HumanPPI"]:
        # protein-protein
        model = PPIModel(
            prt_model_name=config.prt_model_name,
            ft_model_path=config.ft_model_path,
            task_num_labels=task_num_labels,
        )
    else:
        # residue or protein
        model = PointPredictionModel(
            prt_model_name=config.prt_model_name,
            ft_model_path=config.ft_model_path,
            task_num_labels=task_num_labels,
            task_output_type=task_output_type,
            normalization=normalization,
            target_mean=target_mean,
            target_std=target_std,
        )
    model.eval()
    # model = torch.compile(model)
    model = model.to(device).to(dtype=data_type)

    logger.info("Downstream model loaded and moved to device.")

    # Build the loss, data type and optimizer
    task_loss_type = task_config[task_name].loss_type
    loss_fn = {
        "classification": torch.nn.CrossEntropyLoss(),
        "regression": torch.nn.MSELoss(),
        "multi_classification": torch.nn.BCEWithLogitsLoss(),
    }[task_loss_type]
    label_type = {
        "classification": torch.long,
        "regression": data_type,
        "multi_classification": data_type,
    }[task_loss_type]

    if config.frozen_trunk:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.trunk.parameters(), "lr": 1e-4},
                {"params": model.classifier.parameters(), "lr": 1e-3},
            ],
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

    logger.info("Loss, data type and optimizer created.")

    # Build scheduler
    updates_per_epoch = max(len(dataloader_train) // config.opt_interval, 1)
    warmup_updates = updates_per_epoch * 2
    total_updates = updates_per_epoch * config.n_epochs

    def lr_lambda(current_step: int):
        if current_step < warmup_updates:
            return float(current_step) / float(max(1, warmup_updates))
        else:
            progress = (current_step - warmup_updates) / float(
                max(1, total_updates - warmup_updates)
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return 0.01 + 0.99 * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info("Scheduler created.")

    # Build evaluation metric
    metric_type = task_config[task_name].metric
    metric_fn = {
        "accuracy": Accuracy(task="multiclass", num_classes=max(task_num_labels, 2)),
        "mcc": MatthewsCorrCoef(task="binary"),
        "spearman": SpearmanCorrCoef(),
        "auc": AUROC(task="binary"),
        "pcc": PearsonCorrCoef(),
        "long_range_precision_at_L": LongRangePrecisionAtL(top_factor=5),
        "fmax": Fmax(),
    }[metric_type]
    metric_fn = metric_fn.to(device)
    logger.info(f"Evaluation metric {metric_type} initialized.")

    # Eval params for evaluation
    eval_params = EvalParams(
        task_name=task_name,
        task_output_type=task_output_type,
        task_num_labels=task_num_labels,
        tokenizer=tokenizer,
        data_type=data_type,
        label_type=label_type,
        device=device,
        metric_fn=metric_fn,
    )
    logger.info(f"Evaluation params set for evaluation.")

    # Training record and restore
    start_epoch = 0
    best_val_metric = float("-inf")
    if dataloader_valid is None:
        best_test_metric = float("-inf")

    resume_state_path = os.path.join(latest_ckpt_dir, "training_state.pt")
    if config.get("resume", True) and os.path.exists(resume_state_path):
        state = torch.load(resume_state_path, map_location=device)
        start_epoch = state.get("epoch", 0) + 1
        dataloader_train.sampler.set_epoch(start_epoch)
        best_val_metric = state.get("best_val_metric", float("-inf"))
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
            logger.info("Optimizer state loaded from checkpoint.")
        if "scheduler_state" in state:
            scheduler.load_state_dict(state["scheduler_state"])
            logger.info("Scheduler state loaded from checkpoint.")

        if config.frozen_trunk:
            ckpt_path = os.path.join(latest_ckpt_dir, "model_classifier.pt")
            if os.path.exists(ckpt_path):
                model.classifier.load_state_dict(
                    torch.load(ckpt_path, map_location=device)
                )
                logger.info("Classifier weights loaded from checkpoint for resuming.")
        else:
            ckpt_path = os.path.join(latest_ckpt_dir, "model.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                logger.info("Full model weights loaded from checkpoint for resuming.")
        logger.info(
            f"Resuming training from epoch {start_epoch} with best metric {best_val_metric}."
        )
    else:
        logger.info("No resume checkpoint found, starting training from scratch.")

    # time counter
    global_step = 0
    start = time.time()

    for epoch in range(start_epoch, config.n_epochs):
        if config.frozen_trunk:
            model.classifier.train()
            model.trunk.eval()
        else:
            model.train()

        train_loss = []
        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.n_epochs}")
        for index, batch in enumerate(progress_bar):
            global_step += 1

            # Convert to correct dtype and move to GPU
            input_ids = batch["input_ids"].to(torch.long).to(device)
            attention_mask = batch["attention_mask"].to(data_type).to(device)
            labels = batch["labels"].to(label_type).to(device)

            if task_name in ["saprot_data/HumanPPI"]:
                # only protein-protein tasks involve inputting more than one protein
                input_ids_2 = batch["input_ids_2"].to(torch.long).to(device)
                attention_mask_2 = batch["attention_mask_2"].to(data_type).to(device)

                output = model(
                    input_ids,
                    attention_mask,
                    input_ids_2,
                    attention_mask_2,
                    frozen_trunk=config.frozen_trunk,
                )
            else:
                output = model(
                    input_ids,
                    attention_mask,
                    frozen_trunk=config.frozen_trunk,
                )

            if task_loss_type == "multi_classification":
                # if it is a multi_classification task, the labels should be like a vector instead of a scalar
                loss = loss_fn(output, labels)
            elif task_loss_type in ["classification", "regression"]:
                loss = loss_fn(
                    output.view(-1, task_num_labels).squeeze(), labels.view(-1)
                )

            train_loss.append(loss.item())

            # Update the parameters
            loss = loss / config.opt_interval
            loss.backward()
            if global_step % config.opt_interval == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                effective_step = global_step // config.opt_interval

                cost = time.time() - start
                remain_iterations = (config.n_epochs - start_epoch) * len(
                    dataloader_train
                ) - global_step
                estimated_time = cost / global_step * remain_iterations / 60 / 60

                train_metric = {
                    "epoch": epoch,
                    "step": effective_step,
                    "train_loss": np.mean(train_loss),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "ETA_hours": estimated_time,
                }

                progress_bar.set_postfix(train_metric)
                wandb.log(train_metric)
                logger.info(f"Epoch {epoch}, Step {effective_step}: {train_metric}")

                train_loss = []
                # break

        # Save latest model
        state_to_save = {
            "epoch": epoch,
            "best_val_metric": best_val_metric,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(state_to_save, os.path.join(latest_ckpt_dir, "training_state.pt"))
        logger.info(f"Checkpoint saved for epoch {epoch}.")
        if config.frozen_trunk:
            model_weight_path = os.path.join(latest_ckpt_dir, "model_classifier.pt")
            torch.save(model.classifier.state_dict(), model_weight_path)
            logger.info(
                f"Checkpoint saved for epoch {epoch}: classifier weights stored at {model_weight_path}."
            )
        else:
            model_weight_path = os.path.join(latest_ckpt_dir, "model.pt")
            torch.save(model.state_dict(), model_weight_path)
            logger.info(
                f"Checkpoint saved for epoch {epoch}: full model weights stored at {model_weight_path}."
            )

        # Evaluate per epoch
        if dataloader_valid is not None:
            val_metric = downstream_valid(
                model=model, dataloader_valid=dataloader_valid, eval_params=eval_params
            )
            wandb.log({"epoch": epoch, "val_metric": val_metric})
            logger.info(f"Epoch {epoch}: Validation {metric_type} = {val_metric}")

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_state = {
                    "epoch": epoch,
                    "best_val_metric": best_val_metric,
                }
                torch.save(best_state, os.path.join(best_ckpt_dir, "training_state.pt"))
                if config.frozen_trunk:
                    torch.save(
                        model.classifier.state_dict(),
                        os.path.join(best_ckpt_dir, "model_classifier.pt"),
                    )
                    logger.info(
                        f"Best model updated at epoch {epoch}: classifier checkpoint saved."
                    )
                else:
                    torch.save(
                        model.state_dict(), os.path.join(best_ckpt_dir, "model.pt")
                    )
                    logger.info(
                        f"Best model updated at epoch {epoch}: full model checkpoint saved."
                    )
        else:
            test_metric = downstream_valid(
                model=model, dataloader_valid=dataloader_test, eval_params=eval_params
            )
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                if config.frozen_trunk:
                    torch.save(
                        model.classifier.state_dict(),
                        os.path.join(best_ckpt_dir, "model_classifier.pt"),
                    )
                    logger.info(
                        f"Best model updated at epoch {epoch}: classifier checkpoint saved."
                    )
                else:
                    torch.save(
                        model.state_dict(), os.path.join(best_ckpt_dir, "model.pt")
                    )
                    logger.info(
                        f"Best model updated at epoch {epoch}: full model checkpoint saved."
                    )
            logger.info(
                f"Epoch {epoch}: Test {metric_type} = {test_metric} (Best so far: {best_test_metric})"
            )

            if epoch == config.n_epochs - 1:
                torch.save(
                    {
                        "test_metric": best_test_metric,
                    },
                    os.path.join(best_ckpt_dir, "test_metric.pt"),
                )

            wandb.log(
                {
                    "epoch": epoch,
                    "test_metric": test_metric,
                    "best_test_metric": best_test_metric,
                }
            )
        # break
    if dataloader_valid is not None:
        if config.frozen_trunk:
            best_ckpt = os.path.join(best_ckpt_dir, "model_classifier.pt")
            if os.path.exists(best_ckpt):
                model.classifier.load_state_dict(
                    torch.load(best_ckpt, map_location=device)
                )
                logger.info(
                    "Best classifier checkpoint loaded for final test evaluation."
                )
        else:
            best_ckpt = os.path.join(best_ckpt_dir, "model.pt")
            if os.path.exists(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
                logger.info(
                    "Best full model checkpoint loaded for final test evaluation."
                )

        final_test_metric = downstream_valid(
            model=model, dataloader_valid=dataloader_test, eval_params=eval_params
        )

        torch.save(
            {"test_metric": final_test_metric},
            os.path.join(best_ckpt_dir, "test_metric.pt"),
        )
        logger.info(f"Final Test {metric_type}: {final_test_metric}")
        wandb.log({"final_test_metric": final_test_metric})

    wandb.finish()
    logger.info("Downstream Finished!")

    torch.cuda.empty_cache()
    gc.collect()
