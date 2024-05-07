from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 10000
    rope_traditional: bool = False
    segment_size: int = 2052


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class InfiniAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.segment_size = args.segment_size

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

        self.gate = mx.full((1, self.n_heads, 1, 1), -100.0)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        memory: Optional[mx.array] = None,
        norm_term: Optional[mx.array] = None,
        no_memory_update: bool = False,
    ) -> mx.array:
        B, L, D = x.shape
        assert L == self.segment_size, f"Sequence length must be {self.segment_size}"

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if not no_memory_update:
            memory_output = self._retrieve_from_memory(
                queries, memory, norm_term
            )

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        if not no_memory_update:
            memory, norm_term = self._update_memory(
                keys, values, memory, norm_term
            )
            combined_output = mx.sigmoid(self.gate) * memory_output + (1 - mx.sigmoid(self.gate)) * attn_output
        else:
            combined_output = attn_output

        combined_output = combined_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(combined_output)

        if no_memory_update:
            memory = None
            norm_term = None

        return output, (keys, values), memory, norm_term

    def _retrieve_from_memory(self, queries, memory, norm_term):
        if memory is None or norm_term is None:
            return mx.zeros_like(queries)

        queries = nn.elu(queries) + 1
        memory_output = mx.dot(queries, memory)

        norm_term_broadcastable = mx.dot(queries, norm_term.transpose(0, 1))
        memory_output = memory_output / norm_term_broadcastable

        return memory_output

    def _update_memory(self, keys, values, memory, norm_term):
        keys = nn.elu(keys) + 1

        if memory is None:
            memory = mx.dot(keys.transpose(0, 2, 1, 3), values)
        else:
            memory = memory + mx.dot(keys.transpose(0, 2, 1, 3), values)

        if norm_term is None:
            norm_term = keys.sum(axis=2, keepdims=True)
        else:
            norm_term = norm_term + keys.sum(axis=2, keepdims=True)

        return memory, norm_term


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = InfiniAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        memory: Optional[mx.array] = None,
        norm_term: Optional[mx.array] = None,
        no_memory_update: bool = False,
    ) -> mx.array:
        r, cache, memory, norm_term = self.self_attn(self.input_layernorm(x), mask, cache, memory, norm_term, no_memory_update)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache, memory, norm_term


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        memory=None,
        norm_term=None,
        no_memory_update=False,
    ):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)
        if memory is None:
            memory = [None] * len(self.layers)
        if norm_term is None:
            norm_term = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e], memory[e], norm_term[e] = layer(h, mask, cache[e], memory[e], norm_term[e], no_memory_update)

        return self.norm(h), cache, memory, norm_term


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = GemmaModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        memory=None,
        norm_term=None,
        no_memory_update=False,
    ):
        out, cache, memory, norm_term = self.model(inputs, cache, memory, norm_term, no_memory_update)
        out = self.model.embed_tokens.as_linear(out)
        return out, cache, memory, norm_term

    @property
    def layers(self):
        return self.model.layers