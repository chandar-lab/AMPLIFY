import re

import torch
from typing import List
from torch import Tensor

vocab_ll = [
    "<pad>",
    "<unk>",
    "<mask>",
    "<bos>",
    "<eos>",
    "|",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "B",
]

def split_sequence(sequence: str):
    pattern = r'(<[^>]+>|[^<]+)'
    tokens = re.findall(pattern, sequence)
    result = []
    for token in tokens:
        if token.startswith('<') and token.endswith('>'):
            result.append(token)
        else:
            result.extend(list(token))
    return result

class ProteinTokenizer(object):
    def __init__(
        self,
        vocab_path: str,
        pad_token_id: int,
        mask_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        unk_token_id: int,
        other_special_token_ids: list | None,
        **kwargs,
    ):
        """Vocabulary comprising the amino acids, and the special tokens <unk>, <bos>, <eos>, <pad> and <mask>.

        Args:
            vocab_path (str): Path to the vocabulary file to load.
            pad_token_id (int): <PAD> token index.
            mask_token_id (int): <MASK> token index.
            bos_token_id (int): <BOS> token index.
            eos_token_id (int): <EOS> token index.
            unk_token_id (int): <UNK> token index.
            other_special_token_Unknown ids (list | None): List of additional special tokens.
        """
        self._token_to_id = dict()
        self._id_to_token = dict()

        for i, token in enumerate(vocab_ll):
            token = token.strip()
            self._token_to_id[token] = i
            self._id_to_token[i] = token

        # Padding token
        self.pad_token_id = pad_token_id
        self.pad_token = self._token_to_id.get(pad_token_id)

        # Beginning and end of sequence
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.bos_token = self._token_to_id.get(bos_token_id)
        self.eos_token = self._token_to_id.get(eos_token_id)

        # Mask token
        self.mask_token_id = mask_token_id
        self.mask_token = self._token_to_id.get(mask_token_id)

        # Unknown token
        self.unk_token_id = unk_token_id
        self.unk_token = self._id_to_token.get(unk_token_id)

        # Set of all special token indices
        self.special_token_ids = set()
        self.special_token_ids.add(pad_token_id)
        self.special_token_ids.add(mask_token_id)
        self.special_token_ids.add(bos_token_id)
        self.special_token_ids.add(eos_token_id)
        self.special_token_ids.add(unk_token_id)
        if other_special_token_ids is not None:
            self.special_token_ids.add(other_special_token_ids)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def encode(
        self,
        tokens: List[str] | str,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        random_truncate: bool = True,
        **kwargs,
    ) -> List | Tensor:
        """Encodes a list of tokens into a list or tensor of token indices.

        Args:
            tokens (List[str]): Sequence of tokens to encode.
            max_length (int | None, optional): Truncate the sequence to the specified length. Defaults to None.
            add_special_tokens (bool, optional): Add special tokens <bos> and <eos> at the start and end.. Defaults to True.
            random_truncate (bool, optional): Truncate the sequence to a random subsequence of if longer than truncate.
            Defaults to True.

        Returns:
            List | Tensor: Token indices.
        """
        if isinstance(tokens, str):
            tokens = split_sequence(tokens)
        token_ids = list(map(self.token_to_id, tokens))
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        if max_length is not None and max_length < len(token_ids):
            offset = int(torch.randint(0, len(token_ids) - max_length, (1,))) if random_truncate else 0
            token_ids = token_ids[offset : offset + max_length]
        return torch.as_tensor(token_ids, dtype=torch.long)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List | str:
        """Decodes a list or tensor of token ids into a list or string of tokens.

        Args:
            token_ids (List[int]): Token indices to decode.
            skip_special_tokens (bool, optional): Skip the special tokens <bos> and <eos> at the start and end.
            Defaults to True.

        Returns:
            List | str: Protein.
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            token_ids = token_ids[1:] if token_ids[1] in self.special_token_ids else token_ids
            token_ids = token_ids[:-1] if token_ids[-1] in self.special_token_ids else token_ids

        tokens = " ".join(map(self.id_to_token, token_ids))

        return tokens
