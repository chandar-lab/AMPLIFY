"""recAMPLIFY: AMPLIFY with recursive refinement and Gated Mixture Recurrence (GMR).

A non-recurrent *stem* and *exit* wrap a weight-tied *recurrent core* applied ``L`` times to
a residue-aligned latent state, giving effective depth at a fixed parameter budget.  **GMR**
decodes each sequence from a probability-weighted mixture of the states visited during
recursion, gating each cycle's residual update by a learned survival signal.

Five configurations (see ``README.md``):

==============  ================  ====================  =========================
``VARIANT``     ``use_ponder_act``  ``ponder_gate_mode``  description
==============  ================  ====================  =========================
baseline        False             --                    fixed ``L`` cycles, plain CE
halt_only       True              ``none``              halt head alone decides halting
gate_scalar     True              ``scalar``            gated residual, one channel
gate_lowrank    True              ``lowrank``           gated residual, ``rank`` channels
gate_perdim     True              ``perdim``            gated residual, one channel per feature
==============  ================  ====================  =========================

Module and config field names are the serialisation contract of the released checkpoints, so
they are kept stable even where a different name would read better; the paper's notation is
used for internal variables and docstrings instead.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import cross_entropy, rms_norm
from transformers.utils import ModelOutput

from .amplify import (
    _HALF_DTYPES,
    AMPLIFYConfig,
    AMPLIFYPreTrainedModel,
    EncoderBlock,
    RotaryEmbedding,
    _flash_available,
)

_GATE_MODES = ("none", "scalar", "lowrank", "perdim")
_CONTEXT_TYPES = ("zero", "zL0")


@dataclass
class RecMaskedLMOutput(ModelOutput):
    """Output of :class:`RecAMPLIFY`.

    Attributes:
        loss: Masked-LM cross-entropy on masked positions.
        logits: Vocabulary logits, same leading shape as ``input_ids``.
        ponder_cost: Per-sequence entropy regulariser ``(B,)``; ``None`` without GMR.
        avg_cycles: Expected number of recursive cycles; ``None`` without GMR.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None
    avg_cycles: Optional[torch.Tensor] = None


class RecAMPLIFYConfig(AMPLIFYConfig):
    """Configuration for :class:`RecAMPLIFY`.

    Field names match the original ``AMPLIFYSRMConfig`` because they are the wire format of
    every pretrained checkpoint's ``config.json``; that code's ``model_type="amplify_srm"`` is
    still accepted.

    Attributes:
        num_core_layers: TOTAL number of blocks, out of which the stem and exit are carved.
        no_recur_layers: Leading blocks run once before the loop (the stem).
        no_exit_layers: Trailing blocks run once after the loop (the exit).
        L_cycles: Number of recursive cycles.
        context_type: ``"zero"`` (context is 0) or ``"zL0"`` (context is ``z_init``).
        use_ponder_act: Enable GMR.
        ponder_gate_mode: ``"none"`` (halt-only), ``"scalar"``, ``"lowrank"`` or ``"perdim"``.
        ponder_gate_rank: Number of gate channels in ``"lowrank"`` mode.
        ponder_gate_init_bias: Gate-head bias init; ``sigmoid(4) ~ 0.98`` starts gates open.
        ponder_decode_perdim: Weight the mixture per feature instead of by a pooled scalar.
        ponder_budget_perchannel: Apply the cost per gate channel, then average.
        ponder_act_min_steps: Cycles during which halting is blocked.
        ponder_act_epsilon: Cumulative-mass threshold for the early exit.
    """

    model_type = "recamplify"
    # model_type written by the pre-release code; such config.json files are read as recamplify.
    legacy_model_type = "amplify_srm"

    def __init__(
        self,
        num_core_layers: int = 12,
        no_recur_layers: int = 2,
        no_exit_layers: int = 2,
        L_cycles: int = 6,
        context_type: str = "zero",
        use_ponder_act: bool = False,
        ponder_gate_mode: Optional[str] = "none",
        ponder_gate_rank: int = 16,
        ponder_gate_init_bias: float = 4.0,
        ponder_decode_perdim: bool = False,
        ponder_budget_perchannel: bool = False,
        ponder_act_min_steps: int = 1,
        ponder_act_epsilon: float = 1e-3,
        **kwargs,
    ):
        # Halt-only checkpoints serialise ponder_gate_mode as JSON null.
        ponder_gate_mode = ponder_gate_mode or "none"
        if context_type not in _CONTEXT_TYPES:
            raise ValueError(f"context_type must be one of {_CONTEXT_TYPES}, got {context_type!r}")
        if ponder_gate_mode not in _GATE_MODES:
            raise ValueError(f"ponder_gate_mode must be one of {_GATE_MODES}, got {ponder_gate_mode!r}")
        if no_recur_layers + no_exit_layers >= num_core_layers:
            raise ValueError(
                f"no_recur_layers + no_exit_layers ({no_recur_layers}+{no_exit_layers}) must leave at "
                f"least one recurrent block out of num_core_layers={num_core_layers}"
            )
        if use_ponder_act and not 0 <= ponder_act_min_steps < L_cycles:
            raise ValueError(f"ponder_act_min_steps must be in [0, L_cycles), got {ponder_act_min_steps}")
        if use_ponder_act and ponder_gate_mode != "none" and ponder_act_min_steps < 1:
            raise ValueError(f"gated GMR requires ponder_act_min_steps >= 1, got {ponder_act_min_steps}")

        # AMPLIFY has an untied decoder: every AMPLIFY and recAMPLIFY checkpoint stores its own
        # lm_head.weight. transformers >= 4.57 would otherwise auto-tie it to the input embedding.
        if kwargs.get("tie_word_embeddings", False):
            raise ValueError("recAMPLIFY never ties word embeddings; tie_word_embeddings must be False")
        kwargs["tie_word_embeddings"] = False
        # Ignored but accepted so old config.json files load without surprises:
        # truncBP_n (always full BPTT), ponder_act_lambda_p / ponder_act_prior_free (the cost is
        # always prior-free entropy), ponder_act_use_z (the heads always read dz),
        # ponder_gate_decode / ponder_eval_decode (always the weighted mixture), use_memory_init,
        # learnable_memory_init, use_carry_norm, vary_l_cycles, use_q_halt, detach_q_halt,
        # use_full_ponder_act, use_timestep_embed, dis_max_steps, z_H_init_std.
        super().__init__(**kwargs)

        self.num_core_layers = num_core_layers
        self.no_recur_layers = no_recur_layers
        self.no_exit_layers = no_exit_layers
        self.L_cycles = L_cycles
        self.context_type = context_type
        self.use_ponder_act = use_ponder_act
        self.ponder_gate_mode = ponder_gate_mode
        self.ponder_gate_rank = ponder_gate_rank
        self.ponder_gate_init_bias = ponder_gate_init_bias
        self.ponder_decode_perdim = ponder_decode_perdim
        self.ponder_budget_perchannel = ponder_budget_perchannel
        self.ponder_act_min_steps = ponder_act_min_steps
        self.ponder_act_epsilon = ponder_act_epsilon

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        """Read ``config.json``, accepting the pre-release ``model_type`` as this one."""
        config_dict, kwargs = super().get_config_dict(*args, **kwargs)
        if config_dict.get("model_type") == cls.legacy_model_type:
            config_dict["model_type"] = cls.model_type
        return config_dict, kwargs


class RecAMPLIFY(AMPLIFYPreTrainedModel):
    """Recursive AMPLIFY encoder with a masked language model head."""

    config_class = RecAMPLIFYConfig
    base_model_prefix = "recamplify"

    def __init__(self, config: RecAMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        hidden_size = config.hidden_size
        nc, nr, ne = config.num_core_layers, config.no_recur_layers, config.no_exit_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size, padding_idx=config.pad_token_id)
        # ONE flat list; the stem and exit are slices of it, not separate modules. This is what
        # makes `core_layers.{i}.*` in the pretrained checkpoints resolve.
        self.core_layers = nn.ModuleList([EncoderBlock(config) for _ in range(nc)])
        self.norm = nn.RMSNorm(hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size)
        self.rotary_emb = RotaryEmbedding(config)

        self._stem = slice(0, nr)
        self._recur = slice(nr, nc - ne)
        self._exit = slice(nc - ne, nc)

        # Frozen initial latent, broadcast to every token of every sequence. Persistent, so it
        # round-trips through the checkpoint.
        z_init = torch.empty(1, 1, hidden_size)
        nn.init.trunc_normal_(z_init, std=config.decoder_init_range)
        self.register_buffer("z_init", z_init)

        self.gate_channels = 0
        if config.use_ponder_act:
            # Built for every GMR variant, including the gated ones where it is never called:
            # the gated checkpoints contain halt_head.* and would not load without it.
            self.halt_head = nn.Linear(hidden_size, 1)
            nn.init.constant_(self.halt_head.bias, -5.0)
            # Load-compatibility only. The original filled this with a geometric prior for the
            # KL cost; the cost is now always prior-free entropy, so it is never read.
            self.register_buffer("ponder_prior", torch.zeros(config.L_cycles))

            if config.ponder_gate_mode != "none":
                self.gate_channels = {
                    "scalar": 1,
                    "lowrank": config.ponder_gate_rank,
                    "perdim": hidden_size,
                }[config.ponder_gate_mode]
                self.gate_head = nn.Linear(hidden_size, self.gate_channels)
                nn.init.constant_(self.gate_head.bias, float(config.ponder_gate_init_bias))
                if config.ponder_gate_mode == "lowrank":
                    # Row-softmaxed, so each feature is a convex mix of the k channels.
                    self.gate_assign = nn.Parameter(torch.empty(hidden_size, config.ponder_gate_rank))
                    nn.init.normal_(self.gate_assign, std=0.02)
                # The gated paths never call halt_head, so it receives no gradient and DDP
                # aborts on it ("parameters that were not used in producing loss"). Freezing
                # it makes DDP skip it while keeping halt_head.* in the state dict, which is
                # cheaper than the find_unused_parameters=True the original resorted to.
                self.halt_head.requires_grad_(False)

        # Re-inits every nn.Linear weight but touches no bias, so the head biases set above
        # survive and the head weights land on U(+/-decoder_init_range), as in the original.
        self.post_init()

    # ------------------------------------------------------------------ public API

    def set_num_cycles(self, num_cycles: int) -> None:
        """Set the number of recursive cycles ``L`` used at inference (test-time depth).

        GMR variants keep their halting rule: the forced halt moves to the new last cycle, and
        sequences may still exit earlier once their continuation mass falls below epsilon.
        """
        num_cycles = int(num_cycles)
        if self.config.use_ponder_act and num_cycles <= self.config.ponder_act_min_steps:
            raise ValueError(
                f"num_cycles must exceed ponder_act_min_steps={self.config.ponder_act_min_steps}, got {num_cycles}"
            )
        self.config.L_cycles = num_cycles

    # ------------------------------------------------------------------ helpers

    def _run(self, layers: slice, h: torch.Tensor, ctx: Tuple) -> torch.Tensor:
        """Run *h* through ``core_layers[layers]``; ``ctx`` carries the attention arguments."""
        attention_mask, cos, sin, max_seqlen, cu_seqlens = ctx
        for layer in self.core_layers[layers]:
            h, _ = layer(h, attention_mask, cos, sin, False, max_seqlen, cu_seqlens)
        return h

    @staticmethod
    def _sequence_ops(z: torch.Tensor, cu_seqlens: Optional[torch.Tensor]) -> Tuple[int, Callable, Callable]:
        """Build ``(num_sequences, bos, broadcast)`` for the packed or padded layout.

        ``bos(t)`` gathers each sequence's BOS latent into ``(B, D)``.  ``broadcast(v)`` expands a
        per-sequence ``(B,)`` or ``(B, C)`` value to a tensor broadcastable against ``z``, zeroing
        the packed padding tail.
        """
        if cu_seqlens is None:
            def bos(t: torch.Tensor) -> torch.Tensor:
                return t[:, 0]

            def broadcast(v: torch.Tensor) -> torch.Tensor:
                return (v.unsqueeze(-1) if v.dim() == 1 else v).unsqueeze(1)

            return z.shape[0], bos, broadcast

        # cu_seqlens is [0, ..., total_tokens, buf_size]: the trailing entry maps the padding
        # tail as a dummy sequence, hence the -2.
        num_sequences = cu_seqlens.numel() - 2
        starts = cu_seqlens[:num_sequences].long()
        counts = cu_seqlens[1 : num_sequences + 1].long() - starts
        total = int(counts.sum())
        seq_idx = z.new_zeros(z.shape[1], dtype=torch.long)
        seq_idx[:total] = torch.repeat_interleave(torch.arange(num_sequences, device=z.device), counts)
        real = (torch.arange(z.shape[1], device=z.device) < total).unsqueeze(-1)

        def bos(t: torch.Tensor) -> torch.Tensor:
            return t[0, starts]

        def broadcast(v: torch.Tensor) -> torch.Tensor:
            v = v.unsqueeze(-1) if v.dim() == 1 else v
            return (v[seq_idx] * real.to(v.dtype)).unsqueeze(0)

        return num_sequences, bos, broadcast

    def _chan_to_dim(self, v: torch.Tensor) -> torch.Tensor:
        """Map a per-channel value ``(B, C)`` to a per-feature value ``(B, D)``."""
        mode = self.config.ponder_gate_mode
        if mode == "perdim":
            return v
        if mode == "scalar":
            return v.expand(-1, self.config.hidden_size)
        return v.float() @ torch.softmax(self.gate_assign.float(), dim=1).t()

    # ------------------------------------------------------------ recursion paths

    def _recur_fixed(self, z: torch.Tensor, inj: torch.Tensor, ctx: Tuple) -> torch.Tensor:
        """Fixed-depth recursion: ``z_k = core(z_{k-1} + inj)`` for ``k = 1..L``."""
        for _ in range(self.config.L_cycles):
            z = self._run(self._recur, z + inj, ctx)
        return z

    def _recur_halt(
        self, z: torch.Tensor, inj: torch.Tensor, ctx: Tuple, bos: Callable, broadcast: Callable, num_seq: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Halt-only GMR; returns ``(z_weighted, p)`` with ``p`` of shape ``(B, T)``."""
        cfg = self.config
        surviving = z.new_ones(num_seq, dtype=torch.float32)
        halted = torch.zeros(num_seq, dtype=torch.bool, device=z.device)
        z_weighted = torch.zeros_like(z)
        bos_prev = None
        probs = []

        for k in range(cfg.L_cycles):
            proposal = self._run(self._recur, z + inj, ctx)
            # Freeze halted sequences before the head reads the state.
            z = torch.where(broadcast(halted), z, proposal)

            # dz: the detached BOS delta. The first cycle has no predecessor, so it feeds the
            # raw BOS latent (delta against zero), exactly as the original does.
            bos_cur = bos(z).detach()
            delta = bos_cur if bos_prev is None else bos_cur - bos_prev

            if k == cfg.L_cycles - 1:
                lam = surviving.new_ones(num_seq)          # forced halt: absorb the remainder
            else:
                lam = torch.sigmoid(self.halt_head(delta.to(z.dtype))[..., 0].float())
                if k < cfg.ponder_act_min_steps:
                    lam = torch.zeros_like(lam)

            p = surviving * lam
            surviving = surviving * (1.0 - lam)
            if k >= cfg.ponder_act_min_steps:
                probs.append(p)

            # Raw z, no RMSNorm -- this is the formula the halt-only checkpoint was trained
            # under. The gated path below norms instead; the asymmetry is intentional.
            z_weighted = z_weighted + broadcast(p.to(z.dtype)) * z

            halted = halted | (surviving < cfg.ponder_act_epsilon)
            bos_prev = bos_cur
            if self._early_exit(halted.all()):
                break

        return z_weighted, torch.stack(probs, dim=1)

    def _recur_gated(
        self, z: torch.Tensor, inj: torch.Tensor, ctx: Tuple, bos: Callable, broadcast: Callable, num_seq: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gated GMR; returns ``(z_weighted, p)`` with ``p`` of shape ``(B, T, C)``."""
        cfg = self.config
        surviving = z.new_ones(num_seq, self.gate_channels, dtype=torch.float32)
        z_weighted = torch.zeros_like(z)
        bos_prev = None
        masses = []

        for k in range(cfg.L_cycles):
            proposal = self._run(self._recur, z + inj, ctx)

            # dz over successive *proposals* -- not over the post-update state.
            bos_cur = bos(proposal).detach()
            delta = bos_cur if bos_prev is None else bos_cur - bos_prev
            gate = torch.sigmoid(self.gate_head(delta.to(z.dtype)).float())
            if k < cfg.ponder_act_min_steps:
                gate = torch.ones_like(gate)               # block closing
            if k == cfg.L_cycles - 1:
                gate = torch.zeros_like(gate)              # force halt

            # Survival ENTERING this cycle gates the update, so S_0 = 1 makes cycle 1 a full
            # replacement and the forced g=0 at k=L-1 does not freeze the last update.
            z = z + broadcast(self._chan_to_dim(surviving)).to(z.dtype) * (proposal - z)

            mass = surviving * (1.0 - gate)
            if k >= cfg.ponder_act_min_steps:
                masses.append(mass)
            surviving = surviving * gate

            weight = self._chan_to_dim(mass) if cfg.ponder_decode_perdim else mass.mean(dim=1, keepdim=True)
            # Stateless RMSNorm (no learned weight), not self.norm.
            z_weighted = z_weighted + broadcast(weight).to(z.dtype) * rms_norm(
                z, (z.shape[-1],), eps=cfg.norm_eps
            )

            bos_prev = bos_cur
            if self._early_exit((surviving.max(dim=1).values < cfg.ponder_act_epsilon).all()):
                break

        return z_weighted, torch.stack(masses, dim=1)

    @staticmethod
    def _early_exit(condition: torch.Tensor) -> bool:
        """Whether every sequence has fully halted, so the remaining cycles can be skipped.

        Reading the condition costs a device sync per cycle and, under ``torch.compile``, a graph
        break, so the check is skipped while compiling and all ``L`` cycles run.  That is NOT
        equivalent to the early exit: the residual mass (< epsilon) the exit would discard is
        instead absorbed by the forced halt at the last cycle, and the extra cycles enter the
        halting distribution, which shifts the entropy cost and ``avg_cycles`` slightly.
        """
        if torch.compiler.is_compiling():
            return False
        return bool(condition)

    # ------------------------------------------------------------------- ponder

    def _ponder(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Entropy cost ``(B,)`` and expected cycle count from the halting mass ``p``.

        Args:
            p: ``(B, T)`` for halt-only or ``(B, T, C)`` for gated, over the cycles where
               halting was allowed (``T = steps_done - min_steps``).
        """
        cfg = self.config
        gated = p.dim() == 3
        if not gated:
            p = p.unsqueeze(-1)
        num_cycles = p.shape[1]

        # Expected depth always uses the channel-averaged distribution, 1-indexed from
        # min_steps + 1 since earlier cycles cannot receive mass.
        pooled = p.mean(dim=-1)
        pooled = pooled / pooled.sum(dim=1, keepdim=True).clamp(min=1e-8)
        steps = torch.arange(1, num_cycles + 1, device=p.device, dtype=pooled.dtype)
        avg_cycles = (pooled * (steps + cfg.ponder_act_min_steps)).sum(dim=1).mean()

        if not cfg.ponder_budget_perchannel:
            p = p.mean(dim=-1, keepdim=True)
        # Halt-only uses the raw mass, as the original does; it sums to 1 unless the epsilon
        # early exit dropped a residual (< epsilon). The gated cost renormalises.
        if gated:
            p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # Negative entropy: penalises collapsing all mass onto a single cycle.
        cost = (p * p.clamp(min=1e-8).log()).sum(dim=1)

        return cost.mean(dim=-1), avg_cycles.detach()

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> RecMaskedLMOutput:
        """Forward pass for recursive masked language modeling.

        Args:
            input_ids: Token IDs -- ``(B, S)`` padded or ``(1, T)`` packed.
            labels: MLM labels with ``-100`` at unmasked positions.
            position_ids: Position IDs, same shape as ``input_ids``.
            attention_mask: ``(B, S)`` boolean mask for padded mode.
            cu_seqlens: Cumulative sequence lengths for packed mode (``B + 2`` entries).
            max_seqlen: Longest sequence in the packed batch.

        Returns:
            :class:`RecMaskedLMOutput` with loss, logits, ponder_cost and avg_cycles.
        """
        cfg = self.config
        hidden_states = self.embed_tokens(input_ids)

        # Packed mode: validate the FlashAttention requirements and flatten position_ids.
        if cu_seqlens is not None:
            if position_ids is None:
                raise ValueError("Packed sequences require explicit position_ids derived from cu_seqlens.")
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
        # Padded mode: reshape the mask for SDPA and build position IDs if absent.
        else:
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()
            if position_ids is not None:
                position_ids = position_ids.to(device=input_ids.device, dtype=torch.long).expand(
                    input_ids.shape[0], -1
                )
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if cu_seqlens is None:
            # SDPA needs a head broadcast dimension: (B, S, D/2) -> (B, S, 1, D/2).
            cos = cos.unsqueeze(2).to(hidden_states.dtype)
            sin = sin.unsqueeze(2).to(hidden_states.dtype)
        ctx = (attention_mask, cos, sin, max_seqlen, cu_seqlens)

        # Stem runs once; its output is re-injected at every cycle.
        stem_out = self._run(self._stem, hidden_states, ctx)

        z_0 = self.z_init.expand_as(stem_out)
        context = z_0 if cfg.context_type == "zL0" else torch.zeros_like(stem_out)
        injection = stem_out + context

        ponder_cost, avg_cycles = None, None
        if not cfg.use_ponder_act:
            z = self._recur_fixed(z_0, injection, ctx)
        else:
            num_seq, bos, broadcast = self._sequence_ops(stem_out, cu_seqlens)
            recur = self._recur_gated if cfg.ponder_gate_mode != "none" else self._recur_halt
            z, p = recur(z_0, injection, ctx, bos, broadcast, num_seq)
            ponder_cost, avg_cycles = self._ponder(p)

        # Read-out: one more core application, then the exit blocks. Note it injects only the
        # context, NOT stem_out -- so with context_type="zero" this is core(z) alone.
        features = self._run(self._recur, z + context, ctx)
        features = self._run(self._exit, features, ctx)
        logits = self.lm_head(self.norm(features))

        loss = None
        if labels is not None:
            # Mean CE over masked positions, and 0 (not NaN) when a batch has none, as in the
            # original. Dividing a sum by the count avoids a host sync on `labels.any()`.
            labels = labels.view(-1)
            loss = cross_entropy(logits.view(-1, logits.shape[-1]).float(), labels, reduction="sum")
            loss = loss / (labels != -100).sum().clamp(min=1)

        return RecMaskedLMOutput(
            loss=loss,
            logits=logits,
            ponder_cost=ponder_cost,
            avg_cycles=avg_cycles,
        )
