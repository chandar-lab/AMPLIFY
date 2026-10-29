import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import BatchEncoding, PreTrainedTokenizerFast


@dataclass
class TokenizerConfig:
    """Configuration used to build a :class:`ProteinTokenizer`.

    Attributes:
        vocab: Ordered list of tokens (index = token ID).
        pad_token: Padding token string.
        mask_token: Masking token string used for MLM.
        bos_token: Beginning-of-sequence token string.
        eos_token: End-of-sequence token string.
        unk_token: Unknown token string.
        other_special_tokens: Additional special tokens to register.
        ambiguous_tokens: Amino-acid tokens considered ambiguous (e.g. B, X, Z).
        remove_ambiguous: If ``True``, strip ambiguous tokens during tokenization.
        vocab_size: Optional explicit vocab size (inferred from ``vocab`` if omitted).
    """

    vocab: List[str]
    pad_token: str
    mask_token: str
    bos_token: str
    eos_token: str
    unk_token: str
    other_special_tokens: Optional[List[str]] = None
    ambiguous_tokens: Optional[List[str]] = None
    remove_ambiguous: bool = False
    vocab_size: Optional[int] = None


class ProteinTokenizer(PreTrainedTokenizerFast):
    """Character-level tokenizer for protein amino acid sequences.

    The tokenizer preserves Hugging Face fast-tokenizer behavior while adding
    AMPLIFY-specific preprocessing:
    - optional random-window truncation
    - optional removal of ambiguous amino-acid tokens
    - ``position_ids`` generation aligned with model inputs
    """

    def __init__(self, config: Optional[TokenizerConfig] = None, max_length: int = 2048, **kwargs):
        """Build from config or load from Hugging Face tokenizer assets.

        Args:
            config: Training-time tokenizer config. If ``None``, this instance is
                initialized from pretrained files via ``kwargs``.
            max_length: Default sequence length limit used by truncation/padding.
            **kwargs: Forwarded to :class:`PreTrainedTokenizerFast`.
        """
        if config is not None:
            # Training-time path: build a character-level WordPiece backend from vocab.
            token_to_id = {token: i for i, token in enumerate(config.vocab)}
            tokenizer_object = Tokenizer(WordPiece(vocab=token_to_id, unk_token=config.unk_token))

            # Split on every character so each amino acid becomes one token.
            tokenizer_object.pre_tokenizer = Split("", behavior="removed")

            # Automatically prepend BOS and append EOS (following ESM convention).
            tokenizer_object.post_processor = TemplateProcessing(
                single=f"{config.bos_token}:0 $A:0 {config.eos_token}:0",
                special_tokens=[
                    (config.bos_token, token_to_id[config.bos_token]),
                    (config.eos_token, token_to_id[config.eos_token]),
                ],
            )

            super().__init__(
                model_max_length=max_length,
                padding_side="right",
                truncation_side="right",
                pad_token=config.pad_token,
                bos_token=config.bos_token,
                eos_token=config.eos_token,
                unk_token=config.unk_token,
                mask_token=config.mask_token,
                model_input_names=["input_ids", "attention_mask", "position_ids"],
                tokenizer_object=tokenizer_object,
            )

            # Register additional special tokens (e.g. domain-specific markers).
            if config.other_special_tokens:
                for token in config.other_special_tokens:
                    if token not in token_to_id:
                        raise ValueError(f"other_special_tokens contains '{token}' which is not in the vocabulary")
                self.add_special_tokens({"additional_special_tokens": config.other_special_tokens})

            # Convert ambiguous amino-acid tokens (e.g. B, X, Z) to their IDs.
            if config.ambiguous_tokens is not None:
                ambiguous_token_ids = [token_to_id[tok] for tok in config.ambiguous_tokens]
            else:
                ambiguous_token_ids = []
            remove_ambiguous = config.remove_ambiguous
        else:
            # Pretrained path: restore from saved tokenizer assets.
            ambiguous_token_ids = kwargs.pop("ambiguous_token_ids", [])
            remove_ambiguous = kwargs.pop("remove_ambiguous", False)
            super().__init__(**kwargs)

        self.ambiguous_token_ids = list(ambiguous_token_ids)
        self.remove_ambiguous = remove_ambiguous

        # Persist AMPLIFY-specific fields in tokenizer_config.json.
        self.init_kwargs["ambiguous_token_ids"] = self.ambiguous_token_ids
        self.init_kwargs["remove_ambiguous"] = self.remove_ambiguous

        # Per-field padding values used by ``pad()``.
        self.key_to_padding = {
            "input_ids": self.pad_token_id,
            "attention_mask": 0,
            "special_tokens_mask": 1,
            "position_ids": 0,
        }

    def _truncate_sequences(
        self,
        encoded_inputs: Dict[str, List[List[int]]],
        max_length: Optional[int] = None,
        random_truncate: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """Truncate sequences longer than ``max_length``, preserving BOS/EOS.

        The first and last tokens (BOS/EOS) are kept intact; only the content
        between them is windowed, following ESM convention.

        Args:
            encoded_inputs: Tokenizer outputs with list-valued fields.
            max_length: Maximum sequence length (including BOS/EOS).
            random_truncate: If ``True``, sample a random content window;
                otherwise keep the left-most window.

        Returns:
            Truncated encoded inputs.
        """
        if max_length is None:
            return encoded_inputs

        for i, sequence in enumerate(encoded_inputs["input_ids"]):
            if len(sequence) > max_length:
                # Preserve BOS (first) and EOS (last); truncate content in between.
                content_max = max_length - 2
                if random_truncate:
                    offset = np.random.randint(0, len(sequence) - 2 - content_max + 1)
                else:
                    offset = 0
                for key in encoded_inputs:
                    vals = encoded_inputs[key][i]
                    encoded_inputs[key][i] = [vals[0]] + vals[1 + offset : 1 + offset + content_max] + [vals[-1]]

        return encoded_inputs

    def _remove_ambiguous_tokens(self, encoded_inputs: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        """Drop tokens listed in ``self.ambiguous_token_ids`` from all fields.

        Args:
            encoded_inputs: Tokenizer outputs with list-valued fields.

        Returns:
            Filtered encoded inputs.
        """
        filtered_inputs = {key: [] for key in encoded_inputs}

        for i, sequence in enumerate(encoded_inputs["input_ids"]):
            keep = [j for j, token in enumerate(sequence) if token not in self.ambiguous_token_ids]
            for key in encoded_inputs:
                filtered_inputs[key].append([encoded_inputs[key][i][j] for j in keep])

        return filtered_inputs

    def pad(
        self,
        encoded_inputs: Dict[str, List[List[int]]],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 8,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, List[List[int]]], BatchEncoding]:
        """Pad batch fields to a shared sequence length and optionally return tensors.

        Args:
            encoded_inputs: Tokenizer outputs or list of sample dictionaries.
            padding: If ``"longest"`` or ``True``, pad to the longest sequence.
            max_length: Maximum length to pad to.
            pad_to_multiple_of: Round target length up to this multiple.
            return_tensors: If set (e.g. ``"pt"``), wrap output in :class:`BatchEncoding`.

        Returns:
            Padded encoded inputs, optionally as :class:`BatchEncoding`.
        """
        # Convert list of sample dicts to dict of lists.
        if isinstance(encoded_inputs, list):
            encoded_inputs = {key: [d[key] for d in encoded_inputs] for key in encoded_inputs[0]}

        if max_length is None:
            max_length = self.model_max_length

        sequence_lengths = [len(sequence) for sequence in encoded_inputs["input_ids"]]
        if not sequence_lengths:
            if return_tensors is not None:
                return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
            return encoded_inputs

        # Cap target length to the longest sequence in the batch.
        if padding == "longest" or padding is True:
            max_length = min(max_length, max(sequence_lengths))

        # Round up to nearest multiple for hardware-friendly tensor shapes.
        if pad_to_multiple_of is not None:
            max_length = math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of

        # Pad each sequence to the target length with field-appropriate values.
        for i, seq_len in enumerate(sequence_lengths):
            pad_length = max_length - seq_len
            if pad_length > 0:
                for key in encoded_inputs:
                    encoded_inputs[key][i] += [self.key_to_padding[key]] * pad_length

        if return_tensors is not None:
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        return encoded_inputs

    def prepare_packed_batch(
        self,
        batch: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Pack a tokenized batch for FlashAttention varlen kernels.

        Accepts padded tensor batches ``(B, S)`` with ``attention_mask``,
        or ragged ``List[List[int]]`` from tokenizer outputs.

        Args:
            batch: Tokenizer output dict containing at least ``input_ids``.
                May also contain ``attention_mask``, ``position_ids``, and ``labels``.
            device: Target device for output tensors.

        Returns:
            Dict with:
                - ``input_ids``: ``(1, total_tokens)`` — all sequences concatenated.
                - ``position_ids``: ``(1, total_tokens)`` — per-sequence positions.
                - ``cu_seqlens``: ``(num_sequences + 1,)`` — cumulative lengths.
                - ``max_seqlen``: ``int`` — longest sequence in the batch.
                - ``labels``: ``(1, total_tokens)`` — only present if input has labels.
        """
        input_ids = batch["input_ids"]

        # Detect input format to normalize all fields uniformly.
        is_2d_tensor = isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
        is_single = not is_2d_tensor and not isinstance(input_ids[0], (list, tuple, torch.Tensor))

        def normalize(field):
            """Normalize a batch field to a list of per-sequence items."""
            if field is None:
                return None
            if is_2d_tensor and isinstance(field, torch.Tensor):
                return list(field)
            if is_single:
                return [field]
            return field

        input_ids = normalize(input_ids)
        mask_raw = normalize(batch.get("attention_mask"))
        pos_raw = normalize(batch.get("position_ids"))
        labels_raw = normalize(batch.get("labels"))

        target = torch.device(device) if device is not None else torch.as_tensor(input_ids[0]).device

        # Strip padding (via attention_mask) and collect per-sequence tensors.
        packed_ids, packed_pos, packed_labels = [], [], []
        for i, ids in enumerate(input_ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
            if mask_raw is not None:
                m = torch.as_tensor(mask_raw[i], dtype=torch.bool)
                ids = ids[m]
            packed_ids.append(ids.to(target))

            if pos_raw is not None:
                pos_i = torch.as_tensor(pos_raw[i], dtype=torch.long)
                if mask_raw is not None:
                    pos_i = pos_i[m]
                packed_pos.append(pos_i.to(target))
            else:
                packed_pos.append(torch.arange(len(ids), device=target))

            if labels_raw is not None:
                lab_i = torch.as_tensor(labels_raw[i], dtype=torch.long)
                if mask_raw is not None:
                    lab_i = lab_i[m]
                packed_labels.append(lab_i.to(target))

        # Concatenate into a single flat sequence and build cu_seqlens.
        seq_lengths = torch.tensor([len(s) for s in packed_ids], dtype=torch.int32, device=target)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=target), seq_lengths])
        cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)

        result = {
            "input_ids": torch.cat(packed_ids).unsqueeze(0),
            "position_ids": torch.cat(packed_pos).unsqueeze(0),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": seq_lengths.max().item(),
        }
        if packed_labels:
            result["labels"] = torch.cat(packed_labels).unsqueeze(0)
        return result

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        pad_to_multiple_of: Optional[int] = None,
        truncation: bool = False,
        random_truncate: bool = True,
        return_special_tokens_mask: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], BatchEncoding]:
        """Tokenize protein sequences with optional truncation/filtering/padding.

        Args:
            text: Single sequence or list of sequences.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            pad_to_multiple_of: If set, pad sequence length up to a multiple of this value.
            truncation: Whether to truncate sequences.
            random_truncate: If ``True``, truncate at random offset.
            return_special_tokens_mask: Whether to return special tokens mask.
            return_tensors: If ``"pt"``, return PyTorch tensors.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``position_ids`` and
            optionally ``special_tokens_mask``. Returns :class:`BatchEncoding`
            when ``return_tensors`` is set.
        """
        # Wrap single string as a list for uniform handling.
        single_input = isinstance(text, str)
        if single_input:
            text = [text]

        # Tokenize each amino-acid sequence into token IDs.
        encoded_inputs = super().__call__(
            text,
            padding=False,
            truncation=False,
            return_special_tokens_mask=return_special_tokens_mask,
            **kwargs,
        )

        if max_length is None:
            max_length = self.model_max_length

        if truncation:
            encoded_inputs = self._truncate_sequences(
                encoded_inputs,
                max_length=max_length,
                random_truncate=random_truncate,
            )

        # Generate position IDs (0, 1, 2, ...) for each sequence.
        encoded_inputs["position_ids"] = [list(range(len(seq))) for seq in encoded_inputs["input_ids"]]

        if self.remove_ambiguous and self.ambiguous_token_ids:
            encoded_inputs = self._remove_ambiguous_tokens(encoded_inputs)

        if padding:
            encoded_inputs = self.pad(
                encoded_inputs,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
            )

        # Unwrap single-input results back from list to plain values.
        if single_input and return_tensors is None:
            for key in encoded_inputs:
                encoded_inputs[key] = encoded_inputs[key][0]

        if return_tensors is not None:
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        return encoded_inputs
