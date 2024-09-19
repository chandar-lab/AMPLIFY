import torch
from torch.utils.data import DataLoader

from ..tokenizer import ProteinTokenizer

from .iterable_protein_dataset import IterableProteinDataset
from .data_collator import DataCollatorMLM


def get_dataloader(
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None,
    paths: dict,
    max_length: int,
    random_truncate: bool,
    return_labels: bool,
    num_workers: int,
    per_device_batch_size: int,
    samples_before_next_set: list | None = None,
    mask_probability: int = 0,
    span_probability: float = 0.0,
    span_max: int = 0,
    exclude_special_tokens_replacement: bool = False,
    padding: str = "max_length",
    pad_to_multiple_of: int = 8,
    dtype: torch.dtype = torch.float32,
    merge: bool = False,
    **kwargs,
) -> DataLoader:
    """Public wrapper for constructing a ``torch`` dataloader.

    Args:
        vocab_path (str): Path to the vocabulary file to load.
        pad_token_id (int): <PAD> token index in the vocab file.
        mask_token_id (int): <MASK> token index in the vocab file.
        bos_token_id (int): <BOS> token index in the vocab file.
        eos_token_id (int): <EOS> token index in the vocab file.
        unk_token_id (int): <UNK> token index in the vocab file.
        other_special_token_Unknown ids (list | None): List of other special tokens.
        paths (dict): Dict of name:paths to the CSV files to read.
        max_length (int): Maximum sequence length.
        random_truncate (bool): Truncate the sequence to a random subsequence of if longer than truncate.
        return_labels (bool): Return the protein labels.
        num_workers (int): Number of workers for the dataloader.
        per_device_batch_size (int): Batch size for each GPU.
        samples_before_next_set (list | None, optional): Number of samples of each dataset to return before moving
        to the next dataset (interleaving). Defaults to ``None``.
        mask_probability (int, optional): Ratio of tokens that are masked. Defaults to 0.
        span_probability (float, optional): Probability for the span length. Defaults to 0.0.
        span_max (int, optional): Maximum span length. Defaults to 0.
        exclude_special_tokens_replacement (bool, optional): Exclude the special tokens such as <BOS> or <EOS> from the
        replacement. Defaults to False.
        padding (str, optional): Pad the batch to the longest sequence or to max_length. Defaults to "max_length".
        pad_to_multiple_of (int, optional): Pad to a multiple of. Defaults to 8.
        dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.

    Returns:
        torch.utils.data.DataLoader
    """

    tokenizer = ProteinTokenizer(
        vocab_path,
        pad_token_id,
        mask_token_id,
        bos_token_id,
        eos_token_id,
        unk_token_id,
        other_special_token_ids,
    )
    collator = DataCollatorMLM(
        tokenizer,
        max_length,
        random_truncate,
        return_labels,
        mask_probability,
        span_probability,
        span_max,
        exclude_special_tokens_replacement,
        padding,
        pad_to_multiple_of,
        dtype,
    )

    if merge:
        return DataLoader(
            IterableProteinDataset(paths.values(), samples_before_next_set),
            per_device_batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        return {
            k: DataLoader(
                IterableProteinDataset([v], samples_before_next_set),
                per_device_batch_size,
                collate_fn=collator,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=True,
            )
            for k, v in paths.items()
        }
