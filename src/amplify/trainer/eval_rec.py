"""Evaluation of a pretrained recAMPLIFY checkpoint.

Loads an HF checkpoint directory written by ``save_hf_checkpoint`` (or by the original
pipeline -- the state_dict layout is identical), runs one pass over each validation split and
reports masked-LM loss, masked-token accuracy and, for GMR variants, the expected number of
recursive cycles.

Batches are padded, not packed, matching what the original eval path forces, so the numbers
are directly comparable to the published ones.
"""

import logging
import os
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed

from ..dataset import CollatorConfig, DataLoaderConfig, get_dataloader
from ..metric import TrainingLogger
from ..model import ProteinTokenizer, RecAMPLIFY, TokenizerConfig

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def eval_trainer(cfg: DictConfig) -> Dict[str, float]:
    """Evaluate ``training_args.checkpoint_dir`` on every split in ``data.validation_files``.

    Args:
        cfg: Hydra config. Requires ``training_args.checkpoint_dir``; honours the optional
            ``training_args.eval_l_cycles`` to re-run a checkpoint at a different depth.

    Returns:
        Mapping of metric name to value, as printed.
    """
    checkpoint_dir = cfg.training_args.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError(
            "training_args.checkpoint_dir must be set, e.g. "
            "+training_args.checkpoint_dir=/path/to/hf_checkpoints/checkpoint_100000"
        )

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=cfg.seed, even_batches=False
        )
    )
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    if cfg.training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator.print(f"[eval] loading {checkpoint_dir}")
    model, info = RecAMPLIFY.from_pretrained(checkpoint_dir, output_loading_info=True)
    for kind in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if info.get(kind):
            raise RuntimeError(f"checkpoint {kind}: {info[kind]}")

    # Re-run at a different recursion depth than the checkpoint was trained at.
    eval_l_cycles = cfg.training_args.get("eval_l_cycles")
    if eval_l_cycles is not None:
        accelerator.print(f"[eval] overriding L_cycles {model.config.L_cycles} -> {eval_l_cycles}")
        model.set_num_cycles(eval_l_cycles)

    cf = model.config
    recur = cf.num_core_layers - cf.no_recur_layers - cf.no_exit_layers
    accelerator.print(
        f"[eval] nc={cf.num_core_layers} nr={cf.no_recur_layers} ne={cf.no_exit_layers} "
        f"L={cf.L_cycles} ctx={cf.context_type} ponder={cf.use_ponder_act} "
        f"gate={cf.ponder_gate_mode} min_steps={cf.ponder_act_min_steps} | "
        f"depth={cf.no_recur_layers + (cf.L_cycles + 1) * recur + cf.no_exit_layers}"
    )

    tokenizer = ProteinTokenizer(TokenizerConfig(**cfg.model.tokenizer), max_length=cfg.training_args.max_length)
    assert model.config.vocab_size == len(tokenizer), "Model and Tokenizer vocab_size are different"

    # Padded batches, and no random truncation, so the pass is deterministic.
    eval_dataloaders = get_dataloader(
        tokenizer=tokenizer,
        col_cfg=CollatorConfig(
            max_length=cfg.training_args.max_length,
            packing=False,
            mlm_probability=cfg.training_args.mlm_probability,
            pad_to_multiple_of=None,
            random_truncate=False,
            exclude_special_tokens_from_masking=cfg.training_args.exclude_special_tokens_from_masking,
            mlm=True,
        ),
        dl_cfg=DataLoaderConfig(
            paths=cfg.data.validation_files,
            shuffle=False,
            drop_last=False,
            num_proc=cfg.data.num_proc,
            per_device_batch_size=cfg.training_args.per_device_eval_batch_size,
            per_device_batch_tokens=cfg.training_args.per_device_eval_batch_tokens,
            dataloader_num_workers=cfg.training_args.dataloader_num_workers,
            persistent_workers=False,
            pin_memory=cfg.training_args.dataloader_pin_memory,
            prefetch_factor=cfg.training_args.dataloader_prefetch_factor,
        ),
    )

    model = accelerator.prepare(model)
    model.eval()
    logger = TrainingLogger()

    for name, dataloader in eval_dataloaders.items():
        dataloader = accelerator.prepare(dataloader)
        with torch.inference_mode():
            for batch in dataloader:
                output = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    position_ids=batch["position_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                logger.add_eval_batch(name, batch, output)

    # Reduce across ranks and report. num_steps stays 0: this is a single snapshot.
    metrics: Dict[str, float] = {}
    for name, acc in logger._eval.items():
        reduced = accelerator.reduce(
            torch.tensor([acc["pred"], acc["loss_sum"], acc["correct_sum"], acc["cycles_sum"], acc["cycles_n"]],
                         dtype=torch.float64, device=accelerator.device),
            reduction="sum",
        ).tolist()
        pred, loss_sum, correct_sum, cycles_sum, cycles_n = reduced
        metrics[f"eval/{name}/loss"] = loss_sum / pred
        metrics[f"eval/{name}/accuracy"] = correct_sum / pred
        if cycles_n > 0:
            metrics[f"eval/{name}/avg_cycles"] = cycles_sum / cycles_n
            metrics[f"eval/{name}/expected_depth"] = (
                cf.no_recur_layers + (cycles_sum / cycles_n) * recur + cf.no_exit_layers
            )

    if accelerator.is_main_process:
        print()
        for k, v in metrics.items():
            print(f"  {k:34s} {v:.4f}")
    return metrics
