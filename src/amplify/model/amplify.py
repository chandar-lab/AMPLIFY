# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py
import yaml

import safetensors
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from xformers.ops import SwiGLU, memory_efficient_attention

from .rmsnorm import RMSNorm
from .rotary import precompute_freqs_cis, apply_rotary_emb
from ..tokenizer import ProteinTokenizer

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput

class DotDict(dict):
    """Dictionary that supports the dot notation to access attributes (similarly to HuggingFace)."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class AMPLIFYConfig(PretrainedConfig):
    model_type = "AMPLIFY"
    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        intermediate_size: int = 3840,
        dropout_prob: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        norm_eps: float = 1e-05,
        hidden_act: str = "SwiGLU",
        layer_norm_after_embedding: bool = False,
        layer_norm_before_last_layer: bool = True,
        vocab_size: int = 27,
        ffn_bias: bool = False,
        att_bias: bool = False,
        pad_token_id: int = 0,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.layer_norm_after_embedding = layer_norm_after_embedding
        self.layer_norm_before_last_layer = layer_norm_before_last_layer
        self.vocab_size = vocab_size
        self.ffn_bias = ffn_bias
        self.att_bias = att_bias
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: AMPLIFYConfig):
        """Initialize a EncoderBlock.

        Args:
            hidden_size (int): _description_
            num_attention_heads (int): _description_
            intermediate_size (int, optional): _description_. Defaults to 2048.
            dropout_prob (float, optional): _description_. Defaults to 0.1.
            activation (str, optional): _description_. Defaults to "relu".
            rms_norm (bool, optional): _description_. Defaults to True.
            norm_eps (float, optional): _description_. Defaults to 1e-5.
            pad_token_id (int, optional): _description_. Defaults to 0.
            max_length (int, optional): _description_. Defaults to 2048.
            ffn_bias (bool, optional): _description_. Defaults to False.
            att_bias (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        self.config = config
        self.d_head = config.hidden_size // config.num_attention_heads

        # Attention
        self.q = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.k = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.v = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.resid_dropout = nn.Dropout(config.dropout_prob)

        # Feedforward network
        act = config.hidden_act.lower()
        if act == "swiglu":
            # To keep the number of parameters and the amount of computation constant, we reduce the number of
            # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
            # avoid RuntimeError due to misaligned operand
            multiple_of = 8
            intermediate_size = int(2 * config.intermediate_size / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
            self.ffn = SwiGLU(
                config.hidden_size,
                intermediate_size,
                config.hidden_size,
                bias=config.ffn_bias
            )
        elif act == "relu":
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.ffn_bias),
                nn.ReLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.ffn_bias),
            )
        elif act == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.ffn_bias),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.ffn_bias),
            )
        else:
            raise ValueError(f"Unsupported hidden_act: {config.hidden_act}")

        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.ffn_dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool):
        attn, contact = self._att_block(self.attention_norm(x), pad_mask, freqs_cis, output_attentions)
        x = x + attn
        x = x + self._ff_block(self.ffn_norm(x))
        return x, contact

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Compute the attention weight
        attn_weights = None
        if output_attentions:
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if pad_mask is not None:
                attn_weights = attn_weights + pad_mask
            attn_weights = attn_weights.softmax(-1)

        # Compute the attention using xformers if the tensors are on GPU
        if x.is_cuda:
            # Input and output are of dimension (B, M, H, K) where B is the batch size, M the sequence length,
            # H the number of heads, and K the embeding size per head
            attn = memory_efficient_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_bias=pad_mask,
                p=self.config.dropout_prob if self.training else 0,
            )
        else:
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
            ).transpose(1, 2)

        attn_scores = self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.d_head))
        return (self.resid_dropout(attn_scores), attn_weights)
    
    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


class AMPLIFYPreTrainedModel(PreTrainedModel):
    config_class = AMPLIFYConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


class AMPLIFY(AMPLIFYPreTrainedModel):
    """The main model class.

       Args:
          config (amplify.model.amplify.AMPLIFYConfig): model configuration, usually defined from the Hydra configuration.
    """
    def __init__(self, config: AMPLIFYConfig, **kwargs):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.layer_norm_after_embedding:
            self.layer_norm_1 = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if config.layer_norm_before_last_layer:
            self.layer_norm_2 = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        
        # Initialize weights and apply final processing
        self.post_init()


    @classmethod
    def load(cls, checkpoint_path: str, config_path: str):

        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)

        model = AMPLIFY(AMPLIFYConfig(**cfg["model"], **cfg["tokenizer"]))

        if checkpoint_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(checkpoint_path)
        elif checkpoint_path.endswith(".pt"):
            state_dict = torch.load(checkpoint_path)
        else:
            raise ValueError(f"Expected checkpoint to be a `.pt` or `.safetensors` file.")

        model.load_state_dict(state_dict)
        tokenizer = ProteinTokenizer(**cfg["tokenizer"])
        return model, tokenizer


    def forward(self, src, pad_mask=None, output_hidden_states=False, output_attentions=False):
        # Initialize
        hidden_states, attentions = [], []

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "AMPLIFY expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
        freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)
        if self.config.layer_norm_after_embedding:
            x = self.layer_norm_1(x)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, pad_mask, freqs_cis, output_attentions)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Classification head with layer norm
        logits = self.decoder(self.layer_norm_2(x) if self.config.layer_norm_before_last_layer else x)

        # Return logits or the output of the last hidden layer
        return MaskedLMOutput(logits=logits, hidden_states=hidden_states, attentions=attentions)

