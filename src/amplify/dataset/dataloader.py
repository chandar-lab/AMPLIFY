import gzip
import io
import math
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

import datasets
import numpy as np
import pyarrow.compute as pc
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from ..model import ProteinTokenizer


# Column names used for the protein sequence field across the AMPLIFY dataset exports.
# "Values" is the column in the canonical UniRef100 CSV snapshots.
_SEQUENCE_COLUMN_ALIASES = ("Values", "seq", "AA_seq", "protein_sequence")

# Extensions read via the FASTA generator rather than the CSV reader.
_FASTA_EXTENSIONS = (
    ".fasta", ".fa",
    ".fasta.gz", ".fa.gz",
    ".fasta.zip", ".fa.zip",
    ".fasta.tar.gz", ".fa.tar.gz",
    ".fasta.tgz", ".fa.tgz",
)


def _parse_fasta_lines(lines: Iterator[Union[str, bytes]]) -> Iterator[str]:
    """Yield one sequence per FASTA record, concatenating wrapped lines."""
    current: List[str] = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\r\n")
        if line.startswith(">"):
            if current:
                yield "".join(current)
            current = []
        else:
            residues = line.upper().replace(" ", "").replace("\t", "")
            if residues:
                current.append(residues)
    if current:
        yield "".join(current)


def iter_fasta_sequences(path: str) -> Iterator[str]:
    """Yield protein sequences from a FASTA file (plain, ``.gz``, ``.zip`` or ``.tar.gz``)."""
    lower = path.lower()
    if lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith((".fasta", ".fa")):
                    handle = tar.extractfile(member)
                    if handle is not None:
                        yield from _parse_fasta_lines(handle)
    elif lower.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            yield from _parse_fasta_lines(handle)
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as archive:
            for name in archive.namelist():
                if name.endswith((".fasta", ".fa")):
                    with archive.open(name) as handle:
                        yield from _parse_fasta_lines(handle)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            yield from _parse_fasta_lines(handle)


def _load_dataset_from_path(path: str, num_proc: int) -> "datasets.Dataset":
    """Load one dataset file, auto-detecting FASTA vs CSV and normalising the column name.

    CSV archives (``.csv.tar.gz``) are handled by ``datasets``, which extracts them.
    """
    if any(path.lower().endswith(ext) for ext in _FASTA_EXTENSIONS):
        return datasets.Dataset.from_generator(
            lambda p=path: ({"sequence": s} for s in iter_fasta_sequences(p)),
            features=datasets.Features({"sequence": datasets.Value("string")}),
        )
    dataset = load_dataset("csv", data_files=path, split="train", num_proc=num_proc)
    if "sequence" not in dataset.column_names:
        for alias in _SEQUENCE_COLUMN_ALIASES:
            if alias in dataset.column_names:
                return dataset.rename_column(alias, "sequence")
    return dataset


@dataclass
class CollatorConfig:
    """Configuration for :class:`DataCollator`.

    Attributes:
        max_length: Maximum sequence length after truncation.
        packing: Pack batch into a flat ``(1, T)`` buffer with ``cu_seqlens`` for FlashAttention.
        mlm_probability: Masking probability.
        pad_to_multiple_of: Pad sequence length to a multiple of this value.
        random_truncate: Sample a random window when truncating long sequences.
        exclude_special_tokens_from_masking: Never mask special tokens (BOS, EOS, etc.).
        mlm: Enable masked language modeling (should always be ``True``).
    """

    max_length: int = 512
    packing: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    random_truncate: bool = True
    exclude_special_tokens_from_masking: bool = True
    mlm: bool = True


@dataclass
class DataLoaderConfig:
    """Configuration for dataset loading and dataloader construction.

    Attributes:
        paths: Mapping of split name to CSV/FASTA file path, or a single path string.
        merge: If ``True``, merge all paths into one dataloader.
        shuffle: Shuffle the dataset each epoch.
        drop_last: Drop the last incomplete batch.
        per_device_batch_size: Fixed number of sequences per batch.
        per_device_batch_tokens: Token budget per batch (uses :class:`TokenBatchSampler`).
        dataloader_num_workers: Number of data-loading worker processes.
        persistent_workers: If ``True``, keep dataloader workers alive between epochs.
        pin_memory: If ``True``, copy tensors into pinned memory before returning them.
        prefetch_factor: Number of batches to prefetch per worker.
        num_proc: Number of processes for dataset loading/processing.
    """

    paths: Union[Dict[str, str], str] = field(default_factory=dict)
    merge: bool = False
    shuffle: bool = True
    drop_last: bool = False
    per_device_batch_size: Optional[int] = None
    per_device_batch_tokens: Optional[int] = None
    dataloader_num_workers: int = 0
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    num_proc: int = 1


class _EpochSampler:
    """Yields dataset indices, optionally shuffled. Epoch-aware for reproducibility."""

    def __init__(self, n: int, shuffle: bool = True, seed: int = 42) -> None:
        self.n = n
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            yield from np.random.default_rng(self.seed + self.epoch).permutation(self.n)
        else:
            yield from range(self.n)

    def __len__(self) -> int:
        return self.n

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class TokenBatchSampler:
    """Greedy token-budget batch sampler.

    Groups sequences into batches whose total token count stays within
    ``max_tokens``.  Sequence lengths are pre-computed at init time (via
    pyarrow) so iteration is a fast array lookup with no dataset random access.

    Follows PyTorch's ``BatchSampler(sampler, ...)`` convention: the inner
    ``.sampler`` attribute is an :class:`_EpochSampler`, which lets
    Accelerate's ``set_epoch`` propagation work without hacks.

    Args:
        lengths: Pre-computed sequence lengths (including BOS/EOS, clamped to max_length).
        max_tokens: Maximum total tokens per batch.
        shuffle: Shuffle indices each epoch.
        seed: Base random seed (combined with epoch for determinism).
        drop_last: Drop the last incomplete batch.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        max_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.drop_last = drop_last
        self.sampler = _EpochSampler(len(lengths), shuffle=shuffle, seed=seed)

    def __iter__(self):
        batch: List[int] = []
        budget = 0
        for idx in self.sampler:
            seq_len = int(self.lengths[idx])
            if batch and budget + seq_len > self.max_tokens:
                yield batch
                batch = []
                budget = 0
            batch.append(int(idx))
            budget += seq_len
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Approximate batch count estimated from mean sequence length."""
        mean_len = float(self.lengths.mean())
        seqs_per_batch = max(1, int(self.max_tokens / max(mean_len, 1)))
        return max(1, math.ceil(len(self.lengths) / seqs_per_batch))


class DataCollator:
    """MLM data collator with packing and padding modes.

    Masks a fixed fraction of non-special positions, then either packs the batch into a
    flat buffer for the FlashAttention varlen kernels or pads it to ``(B, S)``.

    Args:
        tokenizer: Protein tokenizer.
        col_cfg: Collator configuration.
    """

    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        col_cfg: CollatorConfig,
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.max_length = col_cfg.max_length
        self.packing = col_cfg.packing
        self.mlm_probability = col_cfg.mlm_probability
        self.random_truncate = col_cfg.random_truncate
        self.exclude_special_tokens_from_masking = col_cfg.exclude_special_tokens_from_masking
        self.mlm = col_cfg.mlm
        self.pad_to_multiple_of = col_cfg.pad_to_multiple_of

        self._tokenizer = tokenizer
        self._special_ids = np.array(sorted(tokenizer.all_special_ids), dtype=np.int64)

    def _special_mask_for(self, ids: np.ndarray) -> np.ndarray:
        """Return a boolean mask of positions that should never be MLM-masked."""
        if self.exclude_special_tokens_from_masking:
            return np.isin(ids, self._special_ids)
        return ids == self.pad_token_id

    def _ensure_min_one_mask(self, masked: np.ndarray, special: np.ndarray) -> None:
        """If no position is masked, randomly mask one non-special position (in-place)."""
        if not masked.any():
            candidates = np.where(~special)[0]
            if len(candidates) > 0:
                masked[candidates[np.random.randint(len(candidates))]] = True

    def _apply_mlm_masking(self, flat_ids: np.ndarray, lengths: np.ndarray) -> tuple:
        """Apply MLM masking to a flat 1D concatenation of sequences.

        Returns ``(masked_ids, labels)`` where *labels* has ``-100`` at unmasked positions.
        """
        special_mask = self._special_mask_for(flat_ids)
        prob = np.full(len(flat_ids), self.mlm_probability)
        prob[special_mask] = 0.0
        masked_indices = np.random.random(len(flat_ids)) < prob

        offset = 0
        for n in lengths:
            n = int(n)
            self._ensure_min_one_mask(masked_indices[offset : offset + n], special_mask[offset : offset + n])
            offset += n

        labels = flat_ids.copy()
        labels[~masked_indices] = -100
        masked_ids = flat_ids.copy()
        masked_ids[masked_indices] = self.mask_token_id
        return masked_ids, labels

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Tokenize all sequences in one call (Rust-backed for speed).
        encoded = self._tokenizer(
            [s["sequence"] for s in samples],
            truncation=True,
            max_length=self.max_length,
            random_truncate=self.random_truncate,
        )
        all_ids = [np.array(ids, dtype=np.int64) for ids in encoded["input_ids"]]
        lengths = np.array([len(ids) for ids in all_ids])

        if self.packing:
            return self._collate_packed(all_ids, lengths)
        return self._collate_padded(all_ids, lengths)

    def _collate_packed(self, all_ids: List[np.ndarray], lengths: np.ndarray) -> Dict[str, Any]:
        """Flat ``(1, T)`` buffer with ``cu_seqlens`` for flash_attn_varlen_func."""
        num_sequences = len(all_ids)
        total_tokens = int(lengths.sum())
        buf_size = total_tokens
        if self.pad_to_multiple_of is not None:
            buf_size = math.ceil(total_tokens / self.pad_to_multiple_of) * self.pad_to_multiple_of

        flat_ids = np.concatenate(all_ids)

        if self.mlm:
            flat_ids, labels = self._apply_mlm_masking(flat_ids, lengths)
        else:
            labels = np.full(total_tokens, -100, dtype=np.int64)

        # Pad output buffers to buf_size (a multiple of pad_to_multiple_of).
        input_ids = np.full(buf_size, self.pad_token_id, dtype=np.int64)
        input_ids[:total_tokens] = flat_ids
        labels_buf = np.full(buf_size, -100, dtype=np.int64)
        labels_buf[:total_tokens] = labels
        position_ids = np.zeros(buf_size, dtype=np.int64)
        position_ids[:total_tokens] = np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])

        # Extra entry maps the padding region [total_tokens, buf_size) as a dummy sequence.
        cu_seqlens = np.zeros(num_sequences + 2, dtype=np.int32)
        cu_seqlens[1 : num_sequences + 1] = np.cumsum(lengths)
        cu_seqlens[num_sequences + 1] = buf_size
        return {
            "input_ids": torch.from_numpy(input_ids).unsqueeze(0),
            "labels": torch.from_numpy(labels_buf).unsqueeze(0),
            "position_ids": torch.from_numpy(position_ids).unsqueeze(0),
            "cu_seqlens": torch.from_numpy(cu_seqlens),
            "max_seqlen": int(lengths.max()),
            "num_sequences": num_sequences,
            "num_tokens": total_tokens,
        }

    def _collate_padded(self, all_ids: List[np.ndarray], lengths: np.ndarray) -> Dict[str, Any]:
        """Padded ``(B, S)`` tensors with ``attention_mask``."""
        num_sequences = len(all_ids)
        seq_len = int(lengths.max())
        if self.pad_to_multiple_of is not None:
            seq_len = math.ceil(seq_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids = np.full((num_sequences, seq_len), self.pad_token_id, dtype=np.int64)
        for i, (ids, n) in enumerate(zip(all_ids, lengths)):
            input_ids[i, :n] = ids

        if self.mlm:
            # Flatten real tokens, apply masking, scatter back to 2D.
            real_ids = np.concatenate([input_ids[i, :int(n)] for i, n in enumerate(lengths)])
            masked_flat, labels_flat = self._apply_mlm_masking(real_ids, lengths)
            labels = np.full((num_sequences, seq_len), -100, dtype=np.int64)
            offset = 0
            for i, n in enumerate(lengths):
                n = int(n)
                input_ids[i, :n] = masked_flat[offset : offset + n]
                labels[i, :n] = labels_flat[offset : offset + n]
                offset += n
        else:
            labels = np.full((num_sequences, seq_len), -100, dtype=np.int64)

        cols = np.arange(seq_len)
        attention_mask = cols[None, :] < lengths[:, None]
        position_ids = np.where(attention_mask, cols[None, :], 0)

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(np.ascontiguousarray(attention_mask)),
            "position_ids": torch.from_numpy(np.ascontiguousarray(position_ids)),
        }


def get_dataloader(
    tokenizer: ProteinTokenizer,
    col_cfg: CollatorConfig,
    dl_cfg: DataLoaderConfig,
) -> Union[DataLoader, Dict[str, DataLoader]]:
    """Create training/evaluation dataloader(s) from CSV or FASTA input files.

    Batch sampling (``per_device_batch_size`` vs ``per_device_batch_tokens``)
    and collation (``col_cfg.packing``) are independent axes.

    Args:
        tokenizer: Initialized protein tokenizer.
        col_cfg: Collator configuration (masking strategy, sequence length, packing).
        dl_cfg: DataLoader configuration (paths, batch size/tokens, workers, etc.).

    Returns:
        A single :class:`DataLoader` when paths are merged, otherwise a mapping
        of split name to :class:`DataLoader`.
    """
    has_size = dl_cfg.per_device_batch_size is not None
    has_tokens = dl_cfg.per_device_batch_tokens is not None
    if has_size and has_tokens:
        raise ValueError("Cannot set both per_device_batch_size and per_device_batch_tokens")
    if not has_size and not has_tokens:
        raise ValueError("Must set either per_device_batch_size or per_device_batch_tokens")

    paths = dl_cfg.paths
    num_workers = dl_cfg.dataloader_num_workers

    collator = DataCollator(tokenizer, col_cfg)

    # Shared DataLoader kwargs.
    common_kwargs = dict(
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=dl_cfg.pin_memory,
        persistent_workers=dl_cfg.persistent_workers if num_workers > 0 else False,
        prefetch_factor=dl_cfg.prefetch_factor if num_workers > 0 else None,
    )

    def build(dataset: Any) -> DataLoader:
        assert "sequence" in dataset.column_names, (
            f"Dataset must have a 'sequence' column (or one of {_SEQUENCE_COLUMN_ALIASES}); "
            f"found: {dataset.column_names}"
        )
        dataset.set_format(columns=["sequence"])

        if has_tokens:
            # Pre-compute sequence lengths from Arrow (+2 for BOS/EOS, clamped to max_length).
            lengths = np.minimum(
                pc.utf8_length(dataset.data.column("sequence")).to_numpy() + 2,
                col_cfg.max_length,
            )
            batch_sampler = TokenBatchSampler(
                lengths=lengths,
                max_tokens=dl_cfg.per_device_batch_tokens,
                shuffle=dl_cfg.shuffle,
                drop_last=dl_cfg.drop_last,
            )
            return DataLoader(dataset=dataset, batch_sampler=batch_sampler, **common_kwargs)
        else:
            # Fixed sequence-count batching; Accelerate handles shuffle/epoch propagation.
            return DataLoader(
                dataset=dataset,
                batch_size=dl_cfg.per_device_batch_size,
                shuffle=dl_cfg.shuffle,
                drop_last=dl_cfg.drop_last,
                **common_kwargs,
            )

    # Merge all files into one dataloader, or build one dataloader per split.
    if dl_cfg.merge or isinstance(paths, str):
        file_list = [paths] if isinstance(paths, str) else list(paths.values())
        return build(
            datasets.concatenate_datasets([_load_dataset_from_path(p, dl_cfg.num_proc) for p in file_list])
        )

    return {name: build(_load_dataset_from_path(path, dl_cfg.num_proc)) for name, path in paths.items()}
