from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention, cross_entropy

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


_HALF_DTYPES = (torch.bfloat16, torch.float16)


class RMSNorm(nn.Module):
    """RMSNorm with manual weight casting to work around autocast bug.

    See https://github.com/pytorch/pytorch/issues/167308
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)


# FlashAttention is optional: only needed for packed-sequence training/inference.
try:
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_ as flash_apply_rotary_emb_qkv_
    from flash_attn.ops.activations import swiglu as flash_swiglu

    _flash_available = True
except ImportError:
    _flash_available = False


class RotaryEmbedding(nn.Module):
    """Rotary position embedding following LLaMA.

    Pre-computes cos/sin tables for all positions up to ``max_position_embeddings``
    so that the forward pass is a simple index lookup.
    """

    def __init__(self, config: "AMPLIFYConfig") -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self._build_tables(torch.device("cpu"))

    def _build_tables(self, device: torch.device) -> None:
        inv_freq = 1.0 / self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)
        freqs = torch.outer(torch.arange(self.max_position_embeddings, device=device).float(), inv_freq)
        self.rope_cos = freqs.cos()
        self.rope_sin = freqs.sin()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rebuild tables if not built yet (loading model) or if device has changed. Introduce a graph break.
        if self.rope_cos is None or self.rope_cos.device != x.device:
            self._build_tables(x.device)
        return self.rope_cos[position_ids], self.rope_sin[position_ids]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply LLaMA-style rotary embeddings to *x* (batch, seq, heads, dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class AMPLIFYConfig(PretrainedConfig):
    """Configuration for the AMPLIFY protein language model.

    Attributes:
        hidden_size: Dimension of hidden representations.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads per layer.
        intermediate_size: Dimension of the SwiGLU feed-forward inner layer.
        embedding_init_range: Uniform init bound for the token embedding.
        decoder_init_range: Uniform init bound for linear layers.
        norm_eps: Epsilon for RMSNorm layers.
        vocab_size: Number of tokens in the amino-acid vocabulary.
        max_position_embeddings: Maximum context length supported by RoPE.
        rope_theta: Base frequency for RoPE.
    """

    model_type = "amplify"

    def __init__(
        self,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        intermediate_size: int = 2560,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-05,
        vocab_size: int = 32,
        pad_token_id: int = 0,
        bos_token_id: int = 3,
        eos_token_id: int = 4,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


class EncoderBlock(nn.Module):
    """Transformer encoder block with RoPE attention and SwiGLU FFN."""

    def __init__(self, config: AMPLIFYConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Multi-head attention projections (query, key, value combined + output).
        self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # SwiGLU feed-forward network (gate + up projection, then down projection).
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        # Pre-normalization layers applied before attention and FFN respectively.
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pre-norm attention + SwiGLU FFN with residual connections.

        Returns (hidden_states, attn_weights).
        """
        batch_size, seq_len = hidden_states.shape[:2]

        # Fast path: packed sequences with fused FlashAttention kernels.
        if cu_seqlens is not None:
            total_tokens = seq_len
            hidden_states = hidden_states.view(-1, self.num_heads * self.head_dim)

            # Pre-norm + QKV projection + RoPE.
            normed = self.input_layernorm(hidden_states)
            qkv = self.qkv_proj(normed).view(1, total_tokens, 3, self.num_heads, self.head_dim)
            q, k, v = flash_apply_rotary_emb_qkv_(qkv, cos, sin).squeeze(0).unbind(1)

            # FlashAttention varlen.
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            ).view(-1, self.num_heads * self.head_dim)

            # Fused attention residual: hidden + attn_output @ o_proj.weight.T
            hidden_states = torch.addmm(hidden_states, attn_output, self.o_proj.weight.t())

            # Pre-norm + SwiGLU MLP with fused residual: hidden + activated @ down_proj.weight.T
            normed_mlp = self.post_attention_layernorm(hidden_states)
            gate, up = self.gate_up_proj(normed_mlp).chunk(2, dim=-1)
            hidden_states = torch.addmm(hidden_states, flash_swiglu(gate, up), self.down_proj.weight.t())

            # Restore 3D for the next layer or the final LM head.
            hidden_states = hidden_states.view(batch_size, total_tokens, -1)
            attn_weights = None
        # Standard path: padded batches with PyTorch SDPA.
        else:
            # Pre-norm before attention.
            normed = self.input_layernorm(hidden_states)
            # Project to Q, K, V and reshape to (batch, seq, heads, dim).
            qkv = self.qkv_proj(normed).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            # Apply rotary position embeddings, then transpose to (batch, heads, seq, dim).
            q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
            k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
            v = v.transpose(1, 2)

            if output_attentions:
                # Manual attention computation to return attention weights.
                attn_weights = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
                if attention_mask is not None:
                    attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
                attn_output = (attn_weights.softmax(-1) @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
            else:
                # Efficient fused attention (does not return attention weights).
                attn_output = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
                attn_weights = None

            # Attention residual connection.
            hidden_states = hidden_states + self.o_proj(attn_output)

            # Pre-norm + SwiGLU FFN + residual connection.
            gate, up = self.gate_up_proj(self.post_attention_layernorm(hidden_states)).chunk(2, dim=-1)
            hidden_states = hidden_states + self.down_proj(self.act_fn(gate) * up)

        return hidden_states, attn_weights


class AMPLIFYPreTrainedModel(PreTrainedModel):
    """Hugging Face base class with AMPLIFY-specific parameter initialization."""

    config_class = AMPLIFYConfig
    base_model_prefix = "amplify"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear layers and embeddings with uniform weights."""
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -self.config.decoder_init_range, self.config.decoder_init_range)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -self.config.embedding_init_range, self.config.embedding_init_range)


class AMPLIFYModel(AMPLIFYPreTrainedModel):
    """AMPLIFY base encoder without any task head.

    Returns contextual embeddings from the transformer encoder stack.
    Used for embedding extraction, the most common downstream use case for
    protein language models.

    Supports both padded batches (PyTorch SDPA) and packed sequences
    (FlashAttention varlen).
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        """Run the encoder stack and return contextual embeddings.

        Args:
            input_ids: Token IDs — ``(B, S)`` padded or ``(1, T)`` packed.
            position_ids: Position IDs — same shape as ``input_ids``.
            attention_mask: ``(B, S)`` boolean mask for padded mode.
            cu_seqlens: Cumulative sequence lengths for packed mode.
            max_seqlen: Maximum sequence length in the packed batch.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights (padded mode only).

        Returns:
            :class:`BaseModelOutput` with ``last_hidden_state``, optional
            ``hidden_states``, and optional ``attentions``.
        """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        hidden_states = self.embed_tokens(input_ids)

        # Packed mode: validate requirements and squeeze position_ids.
        if cu_seqlens is not None:
            if position_ids is None:
                raise ValueError("Packed sequences require explicit position_ids derived from cu_seqlens.")
            if output_attentions is True:
                raise ValueError("Packed sequences do not support returning attention weights.")
            if not _flash_available:
                raise ValueError("Packed sequences require FlashAttention.")
            if not hidden_states.is_cuda:
                raise ValueError("Packed sequences require CUDA.")
            if not (
                hidden_states.dtype in _HALF_DTYPES
                or (torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() in _HALF_DTYPES)
            ):
                raise ValueError("Packed sequences require half precision (autocast or explicit).")
            position_ids = position_ids.to(device=input_ids.device, dtype=torch.long).squeeze(0)
        # Standard mode: reshape attention mask and build position IDs.
        else:
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
            if position_ids is not None:
                position_ids = position_ids.to(device=input_ids.device, dtype=torch.long).expand(input_ids.shape[0], -1)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        # Compute rotary position embeddings.
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Standard path needs a head broadcast dimension: (B, S, D/2) -> (B, S, 1, D/2).
        if cu_seqlens is None:
            cos = cos.unsqueeze(2).to(hidden_states.dtype)
            sin = sin.unsqueeze(2).to(hidden_states.dtype)

        # Run through all encoder layers.
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = layer(
                hidden_states, attention_mask, cos, sin, output_attentions, max_seqlen, cu_seqlens
            )

            if output_attentions:
                all_attentions.append(attn_weights)

        # Final layer norm.
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class AMPLIFYForMaskedLM(AMPLIFYPreTrainedModel):
    """AMPLIFY encoder with a masked language model head.

    Used for pretraining (MLM), pseudo-perplexity scoring, zero-shot variant
    effect prediction, and MLM fine-tuning.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.amplify = AMPLIFYModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> MaskedLMOutput:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs — ``(B, S)`` padded or ``(1, T)`` packed.
            labels: MLM labels with ``-100`` for unmasked positions.
            position_ids: Position IDs — same shape as ``input_ids``.
            attention_mask: ``(B, S)`` boolean mask for padded mode.
            cu_seqlens: Cumulative sequence lengths for packed mode.
            max_seqlen: Maximum sequence length in the packed batch.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights (padded mode only).

        Returns:
            :class:`MaskedLMOutput` with loss, logits, hidden_states, attentions.
        """
        encoder_output = self.amplify(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.lm_head(encoder_output.last_hidden_state)

        loss = None
        if labels is not None:
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


class AMPLIFYForSequenceClassification(AMPLIFYPreTrainedModel):
    """AMPLIFY encoder with a sequence-level classification head.

    Mean-pools over non-padding tokens, then projects to ``num_labels``.
    Used for protein family classification, localization, solubility prediction.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.amplify = AMPLIFYModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Forward pass for sequence classification (padded mode only).

        Args:
            input_ids: Token IDs ``(B, S)``.
            labels: Class labels ``(B,)`` for cross-entropy or ``(B, num_labels)`` for regression.
            position_ids: Position IDs ``(B, S)``.
            attention_mask: Boolean mask ``(B, S)``, ``True`` for real tokens.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights.

        Returns:
            :class:`SequenceClassifierOutput` with loss, logits, hidden_states, attentions.
        """
        encoder_output = self.amplify(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Mean-pool over non-padding tokens.
        hidden_states = encoder_output.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.mse_loss(logits.squeeze(-1), labels.to(logits.dtype))
            else:
                loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


class AMPLIFYForTokenClassification(AMPLIFYPreTrainedModel):
    """AMPLIFY encoder with a per-token classification head.

    Projects each token representation to ``num_labels``.
    Used for binding site prediction, secondary structure, PTM site prediction.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.amplify = AMPLIFYModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> TokenClassifierOutput:
        """Forward pass for token classification (padded mode only).

        Args:
            input_ids: Token IDs ``(B, S)``.
            labels: Per-token labels ``(B, S)`` with ``-100`` for ignored positions.
            position_ids: Position IDs ``(B, S)``.
            attention_mask: Boolean mask ``(B, S)``.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights.

        Returns:
            :class:`TokenClassifierOutput` with loss, logits, hidden_states, attentions.
        """
        encoder_output = self.amplify(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.classifier(encoder_output.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )
