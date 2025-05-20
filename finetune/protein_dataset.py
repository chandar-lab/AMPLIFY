"""load protein seq data and structure data"""

import functools
import math
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)


class ProteinDataset(Dataset):
    """
    our own protein dataset
    """

    def __init__(
        self,
        data_type: str = "train",
        struc_token_type: str = "foldseek",
        struc_embed_type: str = "af2",
        prefix_path: str = "/network/scratch/c/can.chen/datasets/pdb_data"
    ):
        """
        Args:
            data_type:
            struc_embed_type:
            struc_token_type:
            prefix_path:
            reweight:

        """
        key_prefix_path = os.path.join(prefix_path, "important_data")
        self.keys = np.load(
            os.path.join(key_prefix_path, "key_names_" + data_type + ".npy")
        )

        self.key2seq_token = np.load(
            os.path.join(key_prefix_path, "key_name2seq_token.npy"), allow_pickle=True
        ).item()

        struc_token_path = {
            "foldseek": "key_name2foldseek_token.npy",
            "pst": "key_name2pst_token.npy",
            "protoken": "key_name2protoken_token.npy",
            "aido": "key_name2aido_token.npy",
        }[struc_token_type]

        self.key2struc_token = np.load(
            os.path.join(key_prefix_path, struc_token_path), allow_pickle=True
        ).item()

        self.embed_prefix_path = {
            "af2": os.path.join(prefix_path, "af2_embedding"),
            "gearnet": os.path.join(prefix_path, "gearnet_embedding"),
        }[struc_embed_type]
        self.struc_D = {"af2": 384, "gearnet": 512}[struc_embed_type]
        self.crop_length = 2046  # 384

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        seq_token = self.key2seq_token[key]
        struc_token = self.key2struc_token[key]
        weight = 1.0  # self.key2weight[key]

        embed_path = os.path.join(self.embed_prefix_path, f"{key}.npy")
        if os.path.exists(embed_path):
            struc_embedding = torch.from_numpy(np.load(embed_path))
        else:
            struc_embedding = torch.zeros(len(seq_token), self.struc_D)

        if len(seq_token) > self.crop_length:
            # start_idx = random.randint(0, len(seq_token) - self.crop_length)
            start_idx = torch.randint(
                0, len(seq_token) - self.crop_length + 1, (1,)
            ).item()
            seq_token = seq_token[start_idx : start_idx + self.crop_length]
            struc_token = np.hstack(
                [
                    struc_token[0],
                    struc_token[start_idx + 1 : start_idx + 1 + self.crop_length],
                    struc_token[-1],
                ]
            )
            struc_embedding = struc_embedding[start_idx : start_idx + self.crop_length]

        return {
            "seq_token": seq_token,
            "struc_token": struc_token,
            "struc_embedding": struc_embedding,
            "weight": weight,
        }


def collate_fn(
    batch: Dict, tokenizer: AutoTokenizer, struc_token_type: str = "foldseek"
) -> Dict:
    """
    collate the batch to the form we want to perform structural fine-tuning

    Args:
        batch:
        tokenizer:
        struc_token_type:

    Returns:

    """

    # collate seq tokens
    seq_tokens = [item["seq_token"] for item in batch]
    output = tokenizer(
        seq_tokens,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    if tokenizer.name_or_path in [
        "chandar-lab/AMPLIFY_120M",
        "chandar-lab/AMPLIFY_350M",
    ]:
        output["attention_mask"] = torch.where(
            output["attention_mask"] == 1,
            torch.zeros_like(output["attention_mask"], dtype=torch.float),
            torch.full_like(output["attention_mask"], -float("inf"), dtype=torch.float),
        )

    seq_tokens, attention_mask = output["input_ids"], output["attention_mask"]

    seq_labels = seq_tokens.clone()

    # identify real residues
    pad_mask = seq_tokens == tokenizer.pad_token_id
    if tokenizer.name_or_path in [
        "chandar-lab/AMPLIFY_120M",
        "chandar-lab/AMPLIFY_350M",
    ]:
        bos_mask = seq_tokens == tokenizer.bos_token_id
    elif tokenizer.name_or_path in [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    ]:
        bos_mask = seq_tokens == tokenizer.cls_token_id
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")

    eos_mask = seq_tokens == tokenizer.eos_token_id
    real_residue_mask = ~(pad_mask + bos_mask + eos_mask)

    # random masking
    probability_matrix = torch.full(seq_tokens.shape, 0.15)
    probability_matrix.masked_fill_(pad_mask | bos_mask | eos_mask, value=0.0)
    masked_ids = torch.bernoulli(probability_matrix).bool()
    seq_tokens[masked_ids] = tokenizer.mask_token_id

    # mask labels
    seq_labels[~masked_ids] = -100

    # collate struc tokens as struc labels
    struc_tokens = [torch.from_numpy(item["struc_token"]) for item in batch]
    struc_pad_id = -100
    struc_labels = torch.full(
        size=(len(struc_tokens), seq_tokens.shape[1]),
        fill_value=struc_pad_id,
        dtype=torch.long,
    )
    for i, p in enumerate(struc_tokens):
        struc_labels[i, : p.shape[0]] = p

    if struc_token_type == "foldseek":
        bos_mask = struc_labels == 3
        eos_mask = struc_labels == 4
    elif struc_token_type == "protoken":
        bos_mask = struc_labels == 513
        eos_mask = struc_labels == 512
    elif struc_token_type == "aido":
        bos_mask = struc_labels == 513
        eos_mask = struc_labels == 512
    else:
        raise ValueError(f"Unsupported struc_token_type: {struc_token_type}")

    struc_labels[bos_mask | eos_mask] = -100

    # collate embeddings
    struc_embeddings = torch.cat([item["struc_embedding"] for item in batch], dim=0)

    # collate weights
    weights = torch.tensor([item["weight"] for item in batch], dtype=torch.float)

    # collate cl_weights
    cl_weights = []
    for item in batch:
        cl_weights = cl_weights + [
            item["weight"] for _ in range(len(item["seq_token"]))
        ]
    cl_weights = torch.tensor(cl_weights, dtype=torch.float)

    return {
        "seq_tokens": seq_tokens,
        "attention_mask": attention_mask,
        "real_residue_mask": real_residue_mask,
        "seq_labels": seq_labels,
        "struc_labels": struc_labels,
        "struc_embeddings": struc_embeddings,
        "weights": weights,
        "cl_weights": cl_weights,
    }


def contact_collate_fn(batch: list, tokenizer: AutoTokenizer, max_len: int) -> dict:
    """
    Custom collate function for contact prediction tasks.
    This function tokenizes each sample's sequence without pre-padding, then manually pads both
    the tokenized input (input_ids, attention_mask) and the contact map labels to the maximum
    sequence length in the batch.

    Args:
        batch (list): A list of samples, where each sample is a dict with keys:
            - "seq": protein sequence (string)
            - "labels": list of contacts
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing sequences, providing pad_token_id etc.

    Returns:
        dict: A batch dictionary with keys:
            - "input_ids": Tensor of shape (B, max_len)
            - "attention_mask": Tensor of shape (B, max_len)
            - "labels": Tensor of shape (B, max_len, max_len)
    """

    tokenized_samples = []
    label_list = []
    seq_lengths = []

    # Process each sample in the batch
    for sample in batch:
        # Tokenize the sequence without padding
        encoding = tokenizer(
            sample["seq"],
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            padding=False,
        )

        if tokenizer.name_or_path in [
            "chandar-lab/AMPLIFY_120M",
            "chandar-lab/AMPLIFY_350M",
        ]:
            encoding["attention_mask"] = torch.where(
                encoding["attention_mask"] == 1,
                torch.zeros_like(encoding["attention_mask"], dtype=torch.float),
                torch.full_like(
                    encoding["attention_mask"], -float("inf"), dtype=torch.float
                ),
            )
        # Remove the extra batch dimension: shape (1, L) -> (L,)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        tokenized_samples.append(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        L = input_ids.shape[0]
        seq_lengths.append(L)

        # Convert the original contact list to a full contact map matrix (shape: (L, L))
        contact_map = torch.zeros((L, L), dtype=torch.long)
        # fill BOS and EOS with -100
        contact_map[0, :] = -100
        contact_map[:, 0] = -100
        contact_map[-1, :] = -100
        contact_map[:, -1] = -100

        for pair in sample["labels"]:
            i, j = pair[0] + 1, pair[1] + 1
            if i < L and j < L:  # some seq len is more than max_len
                contact_map[i, j] = 1
                contact_map[j, i] = 1
        label_list.append(contact_map)

    # Determine the maximum sequence length in this batch
    max_len = math.ceil(max(seq_lengths) / 8) * 8

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    # Pad each sample's input_ids, attention_mask, and label to max_len (and max_len x max_len for label)
    for i, ts in enumerate(tokenized_samples):
        seq_len = ts["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids with the tokenizer's pad_token_id
        pad_input = torch.full(
            (pad_len,), tokenizer.pad_token_id, dtype=ts["input_ids"].dtype
        )
        padded_input = torch.cat([ts["input_ids"], pad_input], dim=0)
        padded_input_ids.append(padded_input)

        # Pad attention_mask with zeros
        pad_mask = torch.zeros(pad_len, dtype=ts["attention_mask"].dtype)
        padded_mask = torch.cat([ts["attention_mask"], pad_mask], dim=0)
        padded_attention_mask.append(padded_mask)

        # Pad label matrix: original shape is (seq_len, seq_len)
        label = label_list[i]
        if pad_len > 0:
            # Pad additional rows with zeros
            pad_rows = torch.full((pad_len, label.shape[1]), -100, dtype=label.dtype)
            label = torch.cat([label, pad_rows], dim=0)
            # Then pad columns with zeros
            pad_cols = torch.full((label.shape[0], pad_len), -100, dtype=label.dtype)
            label = torch.cat([label, pad_cols], dim=1)
        padded_labels.append(label)

    # Stack padded tensors to form the final batch
    batch_dict = {
        "input_ids": torch.stack(padded_input_ids),  # shape: (B, max_len)
        "attention_mask": torch.stack(padded_attention_mask),  # shape: (B, max_len)
        "labels": torch.stack(padded_labels),  # shape: (B, max_len, max_len)
    }

    return batch_dict


def obtain_real_residue_mask(seq_tokens, tokenizer):
    # identify real residues
    pad_mask = seq_tokens == tokenizer.pad_token_id
    if tokenizer.name_or_path in [
        "chandar-lab/AMPLIFY_120M",
        "chandar-lab/AMPLIFY_350M",
    ]:
        bos_mask = seq_tokens == tokenizer.bos_token_id
    elif tokenizer.name_or_path in [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
        "checkpoint/ISM/ism_model",
        "checkpoint/ESM-s/esm_s_model",
    ]:
        bos_mask = seq_tokens == tokenizer.cls_token_id
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")

    eos_mask = seq_tokens == tokenizer.eos_token_id
    real_residue_mask = ~(pad_mask + bos_mask + eos_mask)
    return real_residue_mask


def create_transform_collate(
    task_name: str, task_output_type: str, tokenizer, max_len: int = 2048
):
    """
    Create a transform function based on task name and output type.

    Args:
        task_name (str): The name of the task.
        task_output_type (str): The output type of the task, e.g., "residue" or "protein".
        tokenizer: The tokenizer to use for data processing.
        max_len (int): The maximum sequence length (default: 2048).

    Returns:
        A function that transforms an input sample into a tokenized format,
        or None if no transformation is needed.
    """
    if task_name == "Bo1015/contact_prediction_binary":
        # For contact prediction binary task, a simple transform is applied.
        transform_fn = lambda x: {"seq": x["seq"], "labels": x["labels"]}
        collate_fn = functools.partial(
            contact_collate_fn, tokenizer=tokenizer, max_len=max_len
        )
    elif task_name == "saprot_data/HumanPPI":
        # For HumanPPI task, tokenize two sequences separately.
        def transform_fn(x):
            tokenized_1 = tokenizer(
                x["seq_1"],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
            )
            tokenized_2 = tokenizer(
                x["seq_2"],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
            )

            if tokenizer.name_or_path in [
                "chandar-lab/AMPLIFY_120M",
                "chandar-lab/AMPLIFY_350M",
            ]:
                tokenized_1["attention_mask"] = torch.where(
                    tokenized_1["attention_mask"] == 1,
                    torch.zeros_like(tokenized_1["attention_mask"], dtype=torch.float),
                    torch.full_like(
                        tokenized_1["attention_mask"], -float("inf"), dtype=torch.float
                    ),
                )
                tokenized_2["attention_mask"] = torch.where(
                    tokenized_2["attention_mask"] == 1,
                    torch.zeros_like(tokenized_2["attention_mask"], dtype=torch.float),
                    torch.full_like(
                        tokenized_2["attention_mask"], -float("inf"), dtype=torch.float
                    ),
                )

            return {
                "input_ids": tokenized_1["input_ids"].squeeze(0),
                "attention_mask": tokenized_1["attention_mask"].squeeze(0),
                "input_ids_2": tokenized_2["input_ids"].squeeze(0),
                "attention_mask_2": tokenized_2["attention_mask"].squeeze(0),
                "labels": x["labels"],
            }

        collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True, return_tensors="pt"
        )

    else:
        if task_output_type == "residue":
            # For residue tasks, tokenize and add bos/eos markers to labels.
            def transform_fn(x):
                tokenized = tokenizer(
                    x["seq"],
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    max_length=max_len,
                    truncation=True,
                )

                if tokenizer.name_or_path in [
                    "chandar-lab/AMPLIFY_120M",
                    "chandar-lab/AMPLIFY_350M",
                ]:
                    tokenized["attention_mask"] = torch.where(
                        tokenized["attention_mask"] == 1,
                        torch.zeros_like(
                            tokenized["attention_mask"], dtype=torch.float
                        ),
                        torch.full_like(
                            tokenized["attention_mask"],
                            -float("inf"),
                            dtype=torch.float,
                        ),
                    )

                tokenized["labels"] = [
                    [-100.0] + label_ + [-100.0] for label_ in x["labels"]
                ]
                return tokenized

            collate_fn = DataCollatorForTokenClassification(tokenizer, padding=True)

        elif task_output_type == "protein":
            # For protein tasks, simply tokenize the sequence.
            def transform_fn(x):
                tokenized = tokenizer(
                    x["seq"],
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    max_length=max_len,
                    truncation=True,
                )

                if tokenizer.name_or_path in [
                    "chandar-lab/AMPLIFY_120M",
                    "chandar-lab/AMPLIFY_350M",
                ]:
                    tokenized["attention_mask"] = torch.where(
                        tokenized["attention_mask"] == 1,
                        torch.zeros_like(
                            tokenized["attention_mask"], dtype=torch.float
                        ),
                        torch.full_like(
                            tokenized["attention_mask"],
                            -float("inf"),
                            dtype=torch.float,
                        ),
                    )

                tokenized["labels"] = x["labels"]
                return tokenized

            collate_fn = DataCollatorWithPadding(
                tokenizer=tokenizer, padding=True, return_tensors="pt"
            )

        else:
            transform_fn, collate_fn = None, None
            # If no transform is required or unknown type, return None.

    return transform_fn, collate_fn
