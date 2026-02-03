"""main code related to train/test"""

import functools
import json
import logging
import os
import time
from typing import List

import numpy as np
import torch
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DataLoaderConfiguration
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import constants
from model import AmplifyClassifier
from protein_dataset import ProteinDataset, collate_fn


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    accelerator: Accelerator,
    seq_vocab_size: int,
    struc_vocab_size: int,
    loss_weight: List,
    data_type: torch.dtype,
    batch_size: int,
) -> List:
    """
    Compute validation loss across all GPUs and return averaged loss.
    Args:
        model:
        val_dataloader:
        loss_fn:
        accelerator:
        seq_vocab_size:
        struc_vocab_size:

    Returns: avg_val_loss

    """

    model.eval()
    val_loss_list = []

    mlm_loss_list = []
    intra_loss_list = []
    inter_loss_list = []

    MASK_DENOMINATOR_VAL = constants.MEAN_MASK_SEQ_LEN * batch_size
    DENOMINATOR_VAL = constants.MEAN_SEQ_LEN * batch_size

    for batch in val_dataloader:
        seq_tokens = batch["seq_tokens"].to(torch.long)
        attention_mask = batch["attention_mask"].to(data_type)
        real_residue_mask = batch["real_residue_mask"].to(torch.long)
        seq_labels = batch["seq_labels"].to(torch.long)
        struc_labels = batch["struc_labels"].to(torch.long)
        struc_embeddings = batch["struc_embeddings"].to(data_type)
        weights = batch["weights"].to(data_type)
        cl_weights = batch["cl_weights"].to(data_type)

        logit_mlm, logit_cls, hidden_state = model.main_forward(
            seq_tokens, attention_mask, frozen_trunk=True
        )

        mlm_loss = loss_fn(logit_mlm.view(-1, seq_vocab_size), seq_labels.view(-1))
        mlm_loss = mlm_loss.view(logit_mlm.shape[0], logit_mlm.shape[1])
        mlm_loss = (
            torch.sum(mlm_loss * weights.view(weights.shape[0], -1))
            / MASK_DENOMINATOR_VAL
        )

        intra_loss = loss_fn(
            logit_cls.view(-1, struc_vocab_size), struc_labels.view(-1)
        )
        intra_loss = intra_loss.view(logit_cls.shape[0], logit_cls.shape[1])
        intra_loss = (
            torch.sum(intra_loss * weights.view(weights.shape[0], -1)) / DENOMINATOR_VAL
        )

        seq_embeddings = hidden_state[real_residue_mask == True]
        loss_seq_to_struct, loss_struct_to_seq = model.cl_forward(
            seq_embeddings, struc_embeddings, cl_weights
        )

        inter_loss = (
            torch.sum(loss_seq_to_struct) / DENOMINATOR_VAL
            + torch.sum(loss_struct_to_seq) / DENOMINATOR_VAL
        ) / 2.0

        loss = (
            loss_weight[0] * mlm_loss
            + loss_weight[1] * intra_loss
            + loss_weight[2] * inter_loss
        )
        val_loss_list.append(loss)
        mlm_loss_list.append(mlm_loss)
        intra_loss_list.append(intra_loss)
        inter_loss_list.append(inter_loss)

    # Gather all losses from all GPUs
    if len(val_loss_list) == 0:
        avg_val_loss = float("inf")
    else:
        val_loss_tensor = torch.stack(val_loss_list)
        gathered_loss = accelerator.gather(val_loss_tensor)
        avg_val_loss = gathered_loss.mean().item()

        mlm_loss_tensor = torch.stack(mlm_loss_list)
        gathered_mlm_loss = accelerator.gather(mlm_loss_tensor)
        avg_mlm_loss = gathered_mlm_loss.mean().item()

        intra_loss_tensor = torch.stack(intra_loss_list)
        gathered_intra_loss = accelerator.gather(intra_loss_tensor)
        avg_intra_loss = gathered_intra_loss.mean().item()

        inter_loss_tensor = torch.stack(inter_loss_list)
        gathered_inter_loss = accelerator.gather(inter_loss_tensor)
        avg_inter_loss = gathered_inter_loss.mean().item()

    model.train()

    return avg_val_loss, (avg_mlm_loss, avg_intra_loss, avg_inter_loss)


def train(config: DictConfig):
    """
    training process

    Args:
        config:

    Returns:

    """

    prt_model_safe = config.prt_model_name.split("/")[-1]
    ref_model_safe = (
        config.reference_model.split("/")[1] if config.reference_model else "None"
    )
    output_dir = os.path.join(
        "checkpoint",
        f"{prt_model_safe}_{ref_model_safe}_"
        f"{'_'.join(map(str, config.loss_weight))}_"
        f"{config.sample_mode}_"
        f"{str(config.ratio)}_"
        f"{config.struc_token_type}_"
        f"{config.struc_embed_type}_"
        f"{config.seed}",
    )

    latest_ckpt_dir = os.path.join(output_dir, "latest")
    best_ckpt_dir = os.path.join(output_dir, "best")
    os.makedirs(best_ckpt_dir, exist_ok=True)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_clipping=1.0,
        gradient_accumulation_steps=config.opt_interval,
    )

    dataloader_config = DataLoaderConfiguration(
        use_seedable_sampler=True,
    )

    accelerator = Accelerator(
        mixed_precision=config.precision,
        deepspeed_plugin=deepspeed_plugin,
        dataloader_config=dataloader_config,
    )

    if accelerator.is_main_process:
        # configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # configure wandb
        wandb.init(
            project="structure-aware-plm",
            name=output_dir.split("/")[-1],
            entity="drug-discovery-amgen",
            config=OmegaConf.to_container(config, resolve=True),
            id=output_dir.split("/")[-1],
            dir=output_dir,
            resume=config.resume,
            mode=config.get("wandb_mode", "offline"),
        )
        train_metric_file = os.path.join(output_dir, "train_metric.json")
        valid_metric_file = os.path.join(output_dir, "valid_metric.json")

    # build model
    struc_vocab_size = {
        "foldseek": 26,
        "pst": 4096 + 2,
        "protoken": 512 + 2,
        "aido": 512 + 2,
    }[config.struc_token_type]

    prt_model_name2seq_D = {
        "chandar-lab/AMPLIFY_120M": 640,
        "chandar-lab/AMPLIFY_350M": 960,
        "facebook/esm2_t6_8M_UR50D": 320,
        "facebook/esm2_t12_35M_UR50D": 480,
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t33_650M_UR50D": 1280,
    }
    seq_D = prt_model_name2seq_D[config.prt_model_name]

    struc_D = {"af2": 384, "gearnet": 512}[config.struc_embed_type]

    output_D = min(seq_D, struc_D)
    model = AmplifyClassifier(
        config.prt_model_name,
        num_labels=struc_vocab_size,
        seq_D=seq_D,
        struc_D=struc_D,
        output_D=output_D,
    )

    model = torch.compile(model)

    if config.get("reference_model", None):
        try:
            if config.prt_model_name not in [
                "chandar-lab/AMPLIFY_120M",
                "chandar-lab/AMPLIFY_350M",
            ]:
                model.trunk.gradient_checkpointing_enable()
                if accelerator.is_main_process:
                    logger.info("Gradient checkpointing enabled for trunk!")
        except AttributeError:
            if accelerator.is_main_process:
                logger.info(
                    "Warning: trunk does not support gradient_checkpointing_enable()."
                )
        reference_seq_D = prt_model_name2seq_D[config.reference_prt_model_name]
        reference_output_D = min(reference_seq_D, struc_D)
        reference_model = AmplifyClassifier(
            config.reference_prt_model_name,
            num_labels=struc_vocab_size,
            seq_D=reference_seq_D,
            struc_D=struc_D,
            output_D=reference_output_D,
        )
        trunk_state_dict = torch.load(
            os.path.join(config.reference_model, "model_trunk.pt")
        )
        reference_model.trunk.load_state_dict(trunk_state_dict)
        cl_model_state_dict = torch.load(
            os.path.join(config.reference_model, "model_cl_model.pt")
        )
        reference_model.cl_model.load_state_dict(cl_model_state_dict)
        classifier_state_dict = torch.load(
            os.path.join(config.reference_model, "model_classifier.pt")
        )
        reference_model.classifier.load_state_dict(classifier_state_dict)
        reference_model.eval()
        if accelerator.is_main_process:
            logger.info("Reference_model loaded successfully!")
        reference_model = torch.compile(reference_model)

    tokenizer = AutoTokenizer.from_pretrained(
        config.prt_model_name, trust_remote_code=True
    )

    # define train data
    dataset_train = ProteinDataset(
        data_type=config.train_data_type,
        struc_token_type=config.struc_token_type,
        struc_embed_type=config.struc_embed_type,
        prefix_path=config.prefix_path
    )
    collate_fn_with_tokenizer = functools.partial(
        collate_fn, tokenizer=tokenizer, struc_token_type=config.struc_token_type
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=4,
    )

    dataset_val = ProteinDataset(
        data_type=config.get("valid_data_type", "valid"),
        struc_token_type=config.struc_token_type,
        struc_embed_type=config.struc_embed_type,
        prefix_path=config.prefix_path
    )

    sampler_val = DistributedSampler(
        dataset=dataset_val,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler_val,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=4,
    )

    # Build the loss, optimizer, and scheduler
    loss_fn = torch.nn.CrossEntropyLoss(
        size_average=False, reduce=False, ignore_index=-100
    )

    param_groups = [
        {"params": model.trunk.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": model.cl_model.parameters(), "lr": 1e-3},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

    MASK_DENOMINATOR = constants.MEAN_MASK_SEQ_LEN * config.batch_size
    DENOMINATOR = constants.MEAN_SEQ_LEN * config.batch_size * config.ratio

    # prepare
    model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val
    )

    data_type = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        config.precision
    ]

    if config.get("reference_model", None):
        reference_model = reference_model.to(accelerator.device).to(dtype=data_type)

    updates_per_epoch = len(dataloader_train) // config.opt_interval
    warmup_updates = updates_per_epoch * 2
    total_updates = updates_per_epoch * config.n_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates
    )
    scheduler = accelerator.prepare(scheduler)

    # Resume logic
    start_epoch = 0
    start_iteration = 0
    best_val_loss = float("inf")

    # If resume=True, load the latest checkpoint if exists
    skipped_train_dataloader = None
    if config.get("resume", False):
        if os.path.exists(latest_ckpt_dir):
            accelerator.print(f"Resuming from checkpoint: {latest_ckpt_dir}")
            accelerator.load_state(latest_ckpt_dir)

            state_path = os.path.join(latest_ckpt_dir, "training_state.pt")
            if os.path.exists(state_path):
                state_dict = torch.load(state_path, map_location="cpu")
                start_iteration = state_dict.get(
                    "iteration", 0
                )  # num of batch within epoch
                start_epoch = state_dict.get("epoch", 0) + int(
                    start_iteration / len(dataloader_train)
                )
                start_iteration = start_iteration % len(dataloader_train)

                best_val_loss = state_dict.get("best_val_loss", float("inf"))

                scheduler_state = state_dict.get("scheduler_state", None)
                if scheduler_state is not None:
                    scheduler.load_state_dict(scheduler_state)

            dataloader_train.set_epoch(start_epoch)
            skipped_train_dataloader = accelerator.skip_first_batches(
                dataloader_train, start_iteration
            )

    global_step = (
        start_epoch * len(dataloader_train) + start_iteration
    )  # num of batches in total

    start_global_step = global_step
    start = time.time()

    for epoch in range(start_epoch, config.n_epochs):
        model.train()
        train_loss = []
        train_mlm_loss = []
        train_intra_loss = []
        train_inter_loss = []

        dataloader = (
            dataloader_train
            if skipped_train_dataloader is None
            else skipped_train_dataloader
        )

        if accelerator.is_main_process:
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch}",
                total=len(dataloader),
                bar_format="{desc} {postfix}\n",
                mininterval=999999,
            )
        else:
            progress_bar = dataloader

        for index, batch in enumerate(progress_bar):
            global_step += 1
            # load data
            seq_tokens = batch["seq_tokens"].to(torch.long)
            attention_mask = batch["attention_mask"].to(data_type)
            real_residue_mask = batch["real_residue_mask"].to(torch.long)
            seq_labels = batch["seq_labels"].to(torch.long)
            struc_labels = batch["struc_labels"].to(torch.long)
            struc_embeddings = batch["struc_embeddings"].to(data_type)
            # weights = batch["weights"].to(data_type)
            cl_weights = batch["cl_weights"].to(data_type)

            # part0: obtain logit and hidden_state
            logit_mlm, logit_cls, hidden_state = model.main_forward(
                seq_tokens, attention_mask, frozen_trunk=False
            )
            if config.get("reference_model", None):
                with torch.no_grad():
                    (
                        _,
                        reference_logit_cls,
                        reference_hidden_state,
                    ) = reference_model.main_forward(
                        seq_tokens, attention_mask, frozen_trunk=False
                    )

            # loss1: mlm -- masked language modeling
            mlm_loss = loss_fn(
                logit_mlm.view(-1, tokenizer.vocab_size), seq_labels.view(-1)
            )
            mlm_loss = mlm_loss.view(logit_mlm.shape[0], logit_mlm.shape[1])
            mlm_loss = mlm_loss[seq_labels != -100]
            mlm_loss = torch.sum(mlm_loss) / MASK_DENOMINATOR

            # loss2: intra loss -- structure token prediction
            intra_loss = loss_fn(
                logit_cls.view(-1, struc_vocab_size), struc_labels.view(-1)
            )
            intra_loss = intra_loss.view(logit_cls.shape[0], logit_cls.shape[1])

            # below step ensures the masked residues are selected
            intra_loss_data = intra_loss.detach().clone()
            intra_loss_data = intra_loss_data[struc_labels != -100]

            intra_loss = intra_loss[struc_labels != -100]
            if config.get("reference_model", None):
                reference_intra_loss = loss_fn(
                    reference_logit_cls.view(-1, struc_vocab_size),
                    struc_labels.view(-1),
                )
                reference_intra_loss = reference_intra_loss.view(
                    reference_logit_cls.shape[0], reference_logit_cls.shape[1]
                )

                reference_intra_loss = reference_intra_loss[struc_labels != -100]
                excess_intra_loss = intra_loss_data - reference_intra_loss
            else:
                excess_intra_loss = intra_loss_data

            K = int(config.ratio * excess_intra_loss.shape[0])
            if config.sample_mode == "loss_large":
                _, topk_index = torch.topk(excess_intra_loss, K)
            elif config.sample_mode == "loss_small":
                _, topk_index = torch.topk(-excess_intra_loss, K)

            intra_loss = torch.sum(intra_loss[topk_index]) / DENOMINATOR

            # loss3: inter loss
            seq_embeddings = hidden_state[real_residue_mask == True]
            loss_seq_to_struct, loss_struct_to_seq = model.cl_forward(
                seq_embeddings, struc_embeddings, cl_weights
            )

            # below step ensures the masked residues are selected
            loss_seq_to_struct_data, loss_struct_to_seq_data = (
                loss_seq_to_struct.detach().clone(),
                loss_struct_to_seq.detach().clone(),
            )

            if config.get("reference_model", None):
                reference_seq_embeddings = reference_hidden_state[
                    real_residue_mask == True
                ]
                with torch.no_grad():
                    (
                        reference_loss_seq_to_struct,
                        reference_loss_struct_to_seq,
                    ) = reference_model.cl_forward(
                        reference_seq_embeddings, struc_embeddings, cl_weights
                    )
                excess_loss_seq_to_struct = (
                    loss_seq_to_struct_data - reference_loss_seq_to_struct
                )
                excess_loss_struct_to_seq = (
                    loss_struct_to_seq_data - reference_loss_struct_to_seq
                )

            else:
                excess_loss_seq_to_struct = loss_seq_to_struct_data
                excess_loss_struct_to_seq = loss_struct_to_seq_data

            K = int(config.ratio * excess_loss_seq_to_struct.shape[0])
            if config.sample_mode == "loss_large":
                _, topk_index = torch.topk(excess_loss_seq_to_struct, K)
            elif config.sample_mode == "loss_small":
                _, topk_index = torch.topk(-excess_loss_seq_to_struct, K)

            loss_seq_to_struct = torch.sum(loss_seq_to_struct[topk_index]) / DENOMINATOR

            K = int(config.ratio * excess_loss_struct_to_seq.shape[0])
            if config.sample_mode == "loss_large":
                _, topk_index = torch.topk(excess_loss_struct_to_seq, K)
            elif config.sample_mode == "loss_small":
                _, topk_index = torch.topk(-excess_loss_struct_to_seq, K)
            loss_struct_to_seq = torch.sum(loss_struct_to_seq[topk_index]) / DENOMINATOR

            inter_loss = (loss_seq_to_struct + loss_struct_to_seq) / 2.0

            # final loss
            loss = (
                config.loss_weight[0] * mlm_loss
                + config.loss_weight[1] * intra_loss
                + config.loss_weight[2] * inter_loss
            )

            accelerator.backward(loss)

            # record loss
            train_loss.append(accelerator.gather(loss).detach().float().mean().item())
            train_mlm_loss.append(
                accelerator.gather(mlm_loss).detach().float().mean().item()
            )
            train_intra_loss.append(
                accelerator.gather(intra_loss).detach().float().mean().item()
            )
            train_inter_loss.append(
                accelerator.gather(inter_loss).detach().float().mean().item()
            )

            # validation
            if global_step % config.opt_interval == 0:
                scheduler.step()
                effective_step = global_step // config.opt_interval

                if accelerator.is_main_process:
                    cost = time.time() - start
                    remain_iterations = (
                        config.n_epochs * len(dataloader_train) - global_step
                    )
                    estimated_time = (
                        cost
                        / (global_step - start_global_step)
                        * remain_iterations
                        / 60
                        / 60
                    )

                    train_metric = {
                        "step": effective_step,
                        "train_loss": np.mean(train_loss),
                        "mlm_loss": np.mean(train_mlm_loss),
                        "intra_loss": np.mean(train_intra_loss),
                        "inter_loss": np.mean(train_inter_loss),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "ETA": estimated_time,
                    }
                    progress_bar.set_postfix(train_metric)
                    wandb.log(train_metric)
                    with open(train_metric_file, "a") as f:
                        f.write(json.dumps(train_metric) + "\n")

                train_loss = []
                train_mlm_loss = []
                train_intra_loss = []
                train_inter_loss = []

                if effective_step % config.eval_steps == 0:
                    if config.get("reference_model", None):
                        reference_model = reference_model.to("cpu")

                    val_loss, (mlm_loss, intra_loss, inter_loss) = validate(
                        model,
                        dataloader_val,
                        loss_fn,
                        accelerator,
                        tokenizer.vocab_size,
                        struc_vocab_size,
                        config.loss_weight,
                        data_type,
                        config.batch_size,
                    )

                    if config.get("reference_model", None):
                        reference_model = reference_model.to(accelerator.device)

                    if accelerator.is_main_process:
                        logger.info(
                            f"Validation at step {effective_step}, loss: {val_loss}, mlm_loss: {mlm_loss}, intra_loss: {intra_loss}, inter_loss: {inter_loss}"
                        )
                        valid_metric = {
                            "val_loss": val_loss,
                            "val_mlm_loss": mlm_loss,
                            "val_intra_loss": intra_loss,
                            "val_inter_loss": inter_loss,
                            "step": effective_step,
                        }
                        wandb.log(valid_metric)
                        with open(valid_metric_file, "a") as f:
                            f.write(json.dumps(valid_metric) + "\n")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "iteration": index,
                                    "best_val_loss": best_val_loss,
                                    "mlm_loss": mlm_loss,
                                    "intra_loss": intra_loss,
                                    "inter_loss": inter_loss,
                                },
                                os.path.join(best_ckpt_dir, "training_state.pt"),
                            )
                            logger.info(
                                f"Best model updated at step {effective_step} with val_loss {val_loss}"
                            )
                            torch.save(
                                model.trunk.state_dict(),
                                os.path.join(best_ckpt_dir, "model_trunk.pt"),
                            )

                            torch.save(
                                model.classifier.state_dict(),
                                os.path.join(best_ckpt_dir, "model_classifier.pt"),
                            )

                            torch.save(
                                model.cl_model.state_dict(),
                                os.path.join(best_ckpt_dir, "model_cl_model.pt"),
                            )

                # save in case the job is killed
                if effective_step % config.save_steps == 0:
                    if accelerator.is_main_process:
                        os.makedirs(latest_ckpt_dir, exist_ok=True)
                        torch.save(
                            {
                                "epoch": epoch,
                                "iteration": index + 1,
                                "best_val_loss": best_val_loss,
                                "scheduler_state": scheduler.state_dict(),
                            },
                            os.path.join(latest_ckpt_dir, "training_state.pt"),
                        )
                        logger.info(f"Save at step {effective_step}")
                    accelerator.save_state(latest_ckpt_dir)

        skipped_train_dataloader = None
    if accelerator.is_main_process:
        logger.info("Training Finished!")
        wandb.finish()