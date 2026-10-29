import os
import re
from tqdm import tqdm
import torch
import logging
import pandas as pd
import polars as pl

from typing import Dict, Any

from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from torch.utils.data import DataLoader

from apperitif.evaluation.Evaluator import Evaluator
from apperitif.collator import DataCollatorWithPacking


class PseudoPPLEvaluator(Evaluator):
    """
    Evaluator for pseudo-perplexity evaluation.
    """

    def __init__(
        self,
        packing: bool = True,
        batch_size: int = 128,
        save_steps: int = 1000,
        num_workers: int = 4,
    ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=None, level=logging.INFO)

        super().__init__()

        self.packing = packing
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.num_workers = num_workers

    @property
    def name(cls):
        return "pseudo_ppl"

    def _compute_metric(self, model_name: str, data_name: str, output_path: str):
        output_dir = os.path.join(output_path, "pseudo_ppl", model_name, data_name)
        parquet_files = [f for f in os.listdir(output_dir) if f.endswith(".parquet")]

        expected_schema = {
            "log_prob": pl.Float32,
            "id": pl.Utf8,
            "masked_index": pl.Int32,
        }

        print(f"Loading {len(parquet_files)} files from {output_dir}")
        df = pl.concat([pl.read_parquet(os.path.join(output_dir, f)).cast(expected_schema) for f in parquet_files])

        ppl = df.group_by("id").mean()

        return ppl

    def __call__(self, model, tokenizer, dataset: Dataset, model_name: str, data_name: str, output_path: str) -> Dict[str, Any]:
        """
        Evaluate the pseudo-perplexity.

        Args:
            model: Encoder model to evaluate.

        Returns:
            Dictionary containing the evaluation metrics.
        """

        # Define paths
        output_path = os.path.join(output_path, self.name, model_name, data_name)
        os.makedirs(output_path, exist_ok=True)

        project_config = ProjectConfiguration(
            project_dir=output_path,
            automatic_checkpoint_naming=False,
        )
        accelerator = Accelerator(
            project_config=project_config,
            mixed_precision="fp16",
        )

        # Put model in evaluation mode
        self.logger.info(f"Loading model {model_name}")
        model.eval()

        if self.packing:
            collate_fn = DataCollatorWithPacking()
            dataset.set_transform(
                lambda x: tokenizer(x["sequence"], truncation=False, padding=False, return_tensors=None)
                | {"id": x["id"]}
                | {"masked_index": x["masked_index"]}
                | {"target": tokenizer.convert_tokens_to_ids(x["target"])}
            )
        else:
            dataset.set_transform(
                lambda x: tokenizer(x["sequence"], truncation=False, padding=True, return_tensors="pt", pad_to_multiple_of=8)
                | {"id": x["id"]}
                | {"masked_index": x["masked_index"]}
                | {"target": tokenizer.convert_tokens_to_ids(x["target"])}
            )

        # Dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn if self.packing else None,
            shuffle=False,
            prefetch_factor=2,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        model, dataloader = accelerator.prepare(model, dataloader)

        # Process bar
        total_batches = len(dataloader)
        pbar = tqdm(
            desc=f"Evaluating metric {self.name} on model {model_name}",
            unit="step",
            initial=0,
            total=total_batches,
            disable=(not accelerator.is_main_process),
            smoothing=0.02,
        )

        # Temporary dict to store data in between dumps
        results = {
            "log_prob": [],
            "id": [],
            "masked_index": [],
        }

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        start_idx = 0
        output_files = os.listdir(output_path)
        if len(output_files) > 0:
            # Find the maximum index from files named data_<int>.parquet
            parquet_indices = [
                int(re.match(r"data_(\d+)\.parquet", f).group(1)) for f in output_files if re.match(r"data_(\d+)\.parquet", f)
            ]
            latest = max(parquet_indices) if parquet_indices else 0
            if latest > 0:
                start_idx = latest + 1
                self.logger.info(f"Resuming from index {start_idx}")

        # Iterate through batches
        for idx, batch in enumerate(dataloader):
            # Skip already processed batches
            if idx < start_idx:
                pbar.update(1)
                continue

            # Update local batch counter
            input_ids, ids, max_seqlen, cu_seqlens, position_ids, attention_mask, masked_index = (
                batch["input_ids"],
                batch["id"],
                batch.get("max_seqlen", None),
                batch.get("cu_seqlens", None),
                batch.get("position_ids", None),
                batch.get("attention_mask", None),
                batch.get("masked_index", None),
            )
            targets = batch["target"]

            # Compute model output
            with torch.no_grad(), accelerator.autocast():
                # Forward pass
                if self.packing:
                    output = model(
                        input_ids,
                        position_ids=position_ids,
                        max_seqlen=max_seqlen,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    if not "esm" in model_name:
                        attention_mask = torch.where(attention_mask.to(torch.bool), float(0.0), float("-inf")).to(torch.float16)
                    output = model(
                        input_ids,
                        attention_mask,
                    )

            if self.packing:
                # Find the masked token for each sequence
                logits_flat = output["logits"].view(-1, output["logits"].size(-1))  # [total_length, vocab_size]
                masked_token = (input_ids.view(-1) == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

                # Get logits at masked positions
                logits = logits_flat[masked_token]

            else:
                # Find the masked token for each sequence
                masked_token = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)

                # Get logits at masked positions
                logits = output["logits"][masked_token[:, 0], masked_token[:, 1]]

            log_probs = torch.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[torch.arange(logits.size(0)), targets]

            results["masked_index"].extend(masked_index.cpu().numpy())
            results["id"].extend(ids)
            results["log_prob"].extend(target_log_probs.cpu().numpy())

            pbar.update(1)

            if idx > 0 and idx % self.save_steps == 0:
                print(idx, self.save_steps)
                gathered = accelerator.gather_for_metrics(
                    [
                        {
                            "id": id_,
                            "masked_index": idx_,
                            "log_prob": lp_,
                        }
                        for id_, idx_, lp_ in zip(results["id"], results["masked_index"], results["log_prob"])
                    ]
                )

                if accelerator.is_main_process:
                    df = pd.DataFrame(gathered)
                    output_file = os.path.join(output_path, f"data_{idx:05d}.parquet")
                    df.to_parquet(output_file, index=False, compression="gzip")
                    self.logger.info(f"Data saved in file {output_file}")

                results = {"log_prob": [], "id": [], "masked_index": []}

        gathered = accelerator.gather_for_metrics(
            [
                {
                    "id": id_,
                    "masked_index": idx_,
                    "log_prob": lp_,
                }
                for id_, idx_, lp_ in zip(results["id"], results["masked_index"], results["log_prob"])
            ]
        )

        if accelerator.is_main_process:
            df = pd.DataFrame(gathered)
            output_file = os.path.join(output_path, f"data_{idx:05d}.parquet")
            df.to_parquet(output_file, index=False, compression="gzip")
            self.logger.info(f"Data saved in file {output_file}")

        pbar.close()
        accelerator.end_training()

        with open(os.path.join(output_path, "done.txt"), "w") as f:
            f.write("done")
