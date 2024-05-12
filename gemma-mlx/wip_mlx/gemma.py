from typing import List, Optional, Tuple, Union
import mlx
from mlx.nn import Module, Linear, Embedding
from mlx.nn.init import normal
from dataclasses import dataclass

class GemmaRMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Removed debugging print statement
        self.eps = eps
        self.weight = mlx.core.zeros(dim)

    def forward(self, x):
        # Removed debugging print statement
        mean_sq = mlx.core.array.square(x).mean(axis=-1, keepdims=True)
        x = x * mlx.core.array.rsqrt(mean_sq + self.eps)
        return (1.0 + self.weight) * x

    def __call__(self, x):
        # Removed debugging print statement
        return self.forward(x)

class GemmaRotaryEmbedding(Module):
    def __init__(self, dim, num_heads, num_key_value_heads, head_dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim  # Added head_dim as a parameter
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = mlx.core.array.exp(mlx.core.array.log(mlx.core.array([self.base])) * (mlx.core.arange(0, self.dim, 2) / self.dim))

    def forward(self, x, position_ids, seq_len):
        # Calculate the frequencies for each position and head dimension
        freqs = mlx.core.array.exp(mlx.core.array([i for i in range(0, self.num_key_value_heads * seq_len * self.head_dim * 2, 2)]).astype(x.dtype) * (-mlx.core.array.log(mlx.core.array([self.base])) / self.head_dim))
        # Reshape freqs to match the target shape for broadcasting with q tensor
        freqs = freqs.reshape(1, self.num_key_value_heads, seq_len, self.head_dim)
        # Expand position_ids and freqs to match the expected shape for broadcasting
        position_ids = position_ids[:, None, :, None]
        # Multiply position_ids with freqs to get the correct frequency for each position
        freqs = position_ids * freqs
        # Generate cos and sin values for rotary embeddings
        cos_values = mlx.core.cos(freqs)
        sin_values = mlx.core.sin(freqs)
        return cos_values.astype(x.dtype), sin_values.astype(x.dtype)

    def __call__(self, x, position_ids, seq_len):
        return self.forward(x, position_ids, seq_len)

def rotate_half(x):
    # Split the input array into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Manually interleave elements of x1 and x2
    x1_reshaped = x1.reshape(-1, 1)
    x2_reshaped = x2.reshape(-1, 1)
    interleaved = mlx.core.zeros((x1_reshaped.shape[0], 2 * x1_reshaped.shape[1]), dtype=x.dtype)
    interleaved[:, ::2] = x1_reshaped
    interleaved[:, 1::2] = -x2_reshaped  # Reinstating negation for sine component
    interleaved = interleaved.reshape(x.shape)

    return interleaved

def apply_rotary_pos_emb(q, k, cos, sin, bsz, num_key_value_heads, head_dim, position_ids=None, unsqueeze_dim=1):
    # Calculate the actual sequence length for reshaping k
    # The sequence length should match the sequence length of the cos tensor for broadcasting
    actual_seq_len = cos.shape[-2]  # Use the second last dimension of cos for seq_len

    # Ensure k tensor has the correct number of elements before reshaping
    expected_size = bsz * num_key_value_heads * actual_seq_len * head_dim
    if k.size != expected_size:
        # If the size is double, it indicates that the tensor has been expanded incorrectly
        if k.size == expected_size * 2:
            # Reshape k to the correct size by selecting the first half of the sequence length
            k = k[:, :, :actual_seq_len, :]
        else:
            raise ValueError(f"Cannot reshape k tensor of size {k.size} into shape ({bsz}, {num_key_value_heads}, {actual_seq_len}, {head_dim}).")
    k = k.reshape(bsz, num_key_value_heads, actual_seq_len, head_dim)

    # Expand cos and sin to match the last dimension of q and k for broadcasting
    cos_expanded = mlx.core.broadcast_to(cos, (bsz, num_key_value_heads, actual_seq_len, head_dim))
    sin_expanded = mlx.core.broadcast_to(sin, (bsz, num_key_value_heads, actual_seq_len, head_dim))
    q_embed = (q * cos_expanded) + (rotate_half(q) * sin_expanded)
    k_embed = (k * cos_expanded) + (rotate_half(k) * sin_expanded)

    # No need to transpose k_embed as it should already be in the correct shape for attention computation
    return q_embed, k_embed

class GemmaMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = mlx.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = mlx.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = mlx.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_activation is None:
            hidden_activation = "gelu"
        else:
            hidden_activation = config.hidden_activation

        ACT2FN_MLX = {
            "relu": mlx.nn.relu,
            "gelu": mlx.nn.gelu,
            "tanh": mlx.core.tanh,
            "sigmoid": mlx.core.sigmoid,
        }

        self.act_fn = ACT2FN_MLX[hidden_activation]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down

def repeat_kv(hidden_states, n_rep: int):
    bsz, q_len, num_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Replicate hidden_states along a new dimension to create multiple copies
    expanded_states = mlx.core.broadcast_to(hidden_states[:, :, None, :, :], (bsz, q_len, n_rep, num_heads, head_dim))
    return expanded_states.reshape(bsz, q_len, n_rep * num_heads, head_dim)

class GemmaConfig:
    def __init__(self, hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
                 intermediate_size=3072, attention_dropout=0.1, hidden_dropout=0.1,
                 max_position_embeddings=512, type_vocab_size=2, vocab_size=30522,
                 layer_norm_eps=1e-12, initializer_range=0.02, pad_token_id=0,
                 segment_size=2048, attention_bias=True, rms_norm_eps=1e-6,
                 head_dim=64, num_key_value_heads=4, rope_theta=10000, hidden_activation="gelu"):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.segment_size = segment_size
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.hidden_activation = hidden_activation

class GemmaAttention(Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.cache = {}  # Initialize an empty cache

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = mlx.nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = mlx.nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = mlx.nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = mlx.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            self.num_heads,  # Pass the correct number of heads
            self.num_key_value_heads,  # Pass the correct number of key-value heads
            self.head_dim,  # Pass the correct head dimension
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        cache_position=None,
        use_cache=False,
    ):
        bsz, _, seq_len, *remaining_dims = hidden_states.shape
        if seq_len == 0:
            raise ValueError(f"Calculated seq_len is 0. bsz: {bsz}, num_heads: {self.num_heads}, head_dim: {self.head_dim}")

        # Flatten hidden_states for projection operations
        hidden_states_reshaped = hidden_states.reshape(-1, self.hidden_size)
        query_states = self.q_proj(hidden_states_reshaped)
        key_states = self.k_proj(hidden_states_reshaped)
        value_states = self.v_proj(hidden_states_reshaped)

        # Reshape for attention computation
        query_states = query_states.reshape(bsz, self.num_heads, -1, self.head_dim).transpose(0, 1, 2, 3)
        key_states = key_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)
        value_states = value_states.reshape(bsz, self.num_key_value_heads, self.head_dim, -1).transpose(0, 1, 3, 2)

        # Concatenate past_key_value[0] with key_states along the sequence length dimension (axis=2)
        # if past_key_value is provided and use_cache is True
        if past_key_value and past_key_value[0] is not None and use_cache:
            # Ensure past_key_value[0] has the correct shape for concatenation
            if past_key_value[0].shape != (bsz, self.num_key_value_heads, seq_len, self.head_dim):
                raise ValueError(f"Shape of past_key_value[0] {past_key_value[0].shape} does not match the expected shape (bsz, num_key_value_heads, seq_len, head_dim)")
            key_states = mlx.core.concatenate((past_key_value[0], key_states), axis=2)
            # Update correct_seq_len after concatenation
            correct_seq_len = key_states.shape[2]
        else:
            correct_seq_len = seq_len

        # Ensure the total size of key_states matches the expected size after reshaping
        expected_total_size = bsz * self.num_key_value_heads * correct_seq_len * self.head_dim
        if key_states.size != expected_total_size:
            raise ValueError(f"Size of key_states tensor {key_states.size} does not match the expected total size {expected_total_size}")

        # Apply rotary position embeddings to query and key states
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, bsz, self.num_key_value_heads, self.head_dim, position_ids)

        # Compute attention scores
        key_states = key_states.transpose(0, 1, 3, 2)
        attn_scores = query_states @ key_states
        if self.is_causal:
            inf_tensor = mlx.core.array([float('-inf') for _ in range(attn_scores.size)]).reshape(attn_scores.shape)
            attn_scores = mlx.core.where(attention_mask, inf_tensor, attn_scores)
        attn_scores = attn_scores / mlx.core.array.sqrt(mlx.core.array([self.head_dim], dtype=query_states.dtype))

        # Apply softmax to get the attention weights
        attn_weights = mlx.nn.softmax(attn_scores, axis=-1)
        # Instantiate dropout with the specified probability
        dropout_layer = mlx.nn.Dropout(p=self.attention_dropout)
        # Apply dropout to the attention weights if in training mode
        attn_weights = dropout_layer(attn_weights) if self.training else attn_weights

        # Compute attention output
        attn_output = attn_weights @ value_states
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Update cache with new computed states if use_cache is enabled
        if use_cache and self.layer_idx is not None and cache_position is not None:
            # Initialize the cache with zeros if it does not exist for the current layer and position
            if (self.layer_idx, cache_position) not in self.cache:
                self.cache[(self.layer_idx, cache_position)] = (mlx.core.zeros((bsz, self.num_key_value_heads, seq_len, self.head_dim)), mlx.core.zeros((bsz, self.num_key_value_heads, seq_len, self.head_dim)))
            # Reshape key_states and value_states to match the expected shape for caching
            key_states_reshaped = key_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)
            value_states_reshaped = value_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)
            # Update the cache with the reshaped key and value states
            self.cache[(self.layer_idx, cache_position)] = (key_states_reshaped, value_states_reshaped)
            # Set past_key_value to the updated cache entry
            past_key_value = self.cache[(self.layer_idx, cache_position)]

        return attn_output, attn_weights, past_key_value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class GemmaDecoderLayer(Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class GemmaModel(Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config  # Store the config as an instance attribute
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GemmaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds * mlx.core.sqrt(mlx.core.array([self.config.hidden_size], dtype=inputs_embeds.dtype))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                past_key_values = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

        # MLX does not have a BaseModelOutputWithPast class, so we return a tuple directly
        return (hidden_states, past_key_values, all_hidden_states, all_self_attns)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

if __name__ == "__main__":

    # Create a config with default values
    config = GemmaConfig()

    # Instantiate the model with the config
    model = GemmaModel(config)

    # Dummy input for testing
    input_ids = mlx.core.arange(0, config.max_position_embeddings).reshape(1, -1)
    attention_mask = mlx.core.ones_like(input_ids)

    # Generate position ids based on the length of the input sequence
    position_ids = mlx.core.arange(0, input_ids.shape[1]).reshape(1, -1)

    # Run the forward pass of the model with position_ids
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

if __name__ == "__main__":
    print("Starting main execution block")

    # Create a config with default values
    config = GemmaConfig()

    # Instantiate the model with the config
    model = GemmaModel(config)

    # Dummy input for testing
    input_ids = mlx.core.arange(0, config.max_position_embeddings).reshape(1, -1)
    attention_mask = mlx.core.ones_like(input_ids)

    # Generate position ids based on the length of the input sequence
    position_ids = mlx.core.arange(0, input_ids.shape[1]).reshape(1, -1)

    # Run the forward pass of the model with position_ids
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

    print("Completed main execution block")
