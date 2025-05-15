import os
import yaml
import torch
import pickle
import numpy as np
from transformers import EsmForMaskedLM, AutoTokenizer
from amplify.model import AMPLIFY, AMPLIFYConfig
from amplify.tokenizer import ProteinTokenizer
from itertools import islice


def load_csv_dataset(path: str, n_proteins: int | None = None):
    assert os.path.exists(path), f"{path} does not exist."
    assert path.endswith(".csv"), f"{path} is not a CSV."

    if path.endswith(".csv"):
        with open(path, "r") as file:
            next(file)  # skip the header
            return [row.strip().split(",") for row in islice(file, n_proteins)]


def load_pickle_dataset(path: str, n_proteins: int | None = None, max_length: int | None = None):
    assert os.path.exists(path), f"{path} does not exist."
    assert path.endswith(".pickle"), f"{path} is not a PICKLE."

    labels, proteins, dist_matrices = [], [], []
    with open(path, "rb") as f:
        for label, (protein, dist_matrix, fold_spans) in pickle.load(f).items():

            # Create a boolean mask for which residue is in the fold
            fold_idx = np.zeros(len(protein), dtype=bool)
            for start, stop in fold_spans:
                fold_idx[start:stop] = True
            protein = "".join(np.array(list(protein))[fold_idx])

            # Verify dimension of the distance matrix as there is an issue when the protein
            # sequence from the structure file has residues on either its N-term or C-term
            if len(protein) != dist_matrix.shape[0]:
                print(
                    f"Expected {label} dist_matrix to be of shape "
                    f"({len(protein)},{len(protein)}) but got {dist_matrix.shape}"
                )
                continue

            # ESM2 has maximum sequence length of 1024 (including <bos> and <eos>)
            if dist_matrix.shape[0] > max_length - 2:
                print(f"Skipped {label} because sequence length is longer than {max_length - 2}")
                continue

            labels.append(label)
            proteins.append(protein)
            dist_matrices.append(dist_matrix)

            if n_proteins is not None and len(labels) == n_proteins:
                break
        return (labels, proteins, dist_matrices)


def load_from_hf(model_path: str, tokenizer_path: str, fp16: bool = True):
    model = EsmForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float16 if fp16 else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def load_from_mila(model_path: str, config_path: str):
    assert os.path.exists(model_path), f"{model_path} does not exist."
    assert os.path.exists(config_path), f"{config_path} does not exist."
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
        model = AMPLIFY(AMPLIFYConfig(**cfg["model"], **cfg["tokenizer"]))
        model.load_state_dict(torch.load(model_path))
        tokenizer = ProteinTokenizer(**cfg["tokenizer"])
    return model, tokenizer


# From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
def symmetrize(x: torch.Tensor):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


# From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
def apc(x: torch.Tensor):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized
