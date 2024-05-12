import unittest
from gemma import GemmaAttention, GemmaConfig, apply_rotary_pos_emb
from mlx.core import full

class TestGemmaAttention(unittest.TestCase):
    def test_caching(self):
        # Create a dummy GemmaConfig
        config = GemmaConfig(
            attention_dropout=0.1,
            hidden_size=768,
            num_attention_heads=12,
            head_dim=64,
            num_key_value_heads=3,
            max_position_embeddings=512,
            rope_theta=0.5,
            pad_token_id=0,
            vocab_size=50257,
            num_hidden_layers=6,
            rms_norm_eps=1e-5
        )

        # Initialize GemmaAttention with dummy config
        attention = GemmaAttention(config=config, layer_idx=0)

        # Create dummy tensors with known sizes
        hidden_states = full([1, 12, 512, 64], 0.1)
        attention_mask = full([1, 1, 1, 512], 1)
        position_ids = full([1, 512], 1)

        # Simulate a forward pass with caching enabled
        cache_position = 0
        use_cache = True
        output_attentions = False

        # First forward pass - no cache yet
        attn_output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            cache_position=cache_position,
            use_cache=use_cache
        )

        # Verify that cache is updated
        self.assertIn((0, cache_position), attention.cache)
        self.assertIsNotNone(attention.cache[(0, cache_position)][0])
        self.assertIsNotNone(attention.cache[(0, cache_position)][1])

        # Second forward pass - should use cache
        attn_output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            use_cache=use_cache
        )

        # Verify that cache is used
        self.assertEqual(past_key_value[0].shape, attention.cache[(0, cache_position)][0].shape)
        self.assertEqual(past_key_value[1].shape, attention.cache[(0, cache_position)][1].shape)

class TestApplyRotaryPosEmb(unittest.TestCase):
    def test_apply_rotary_pos_emb(self):
        # Create dummy tensors with known sizes
        q = full([1, 12, 512, 64], 0.1)
        k = full([1, 12, 512, 64], 0.1)
        cos = full([1, 12, 512, 64], 0.1)  # Adjusted the size to match q and k
        sin = full([1, 12, 512, 64], 0.1)  # Adjusted the size to match q and k
        bsz = 1
        num_heads = 12
        head_dim = 64

        # Call the apply_rotary_pos_emb function with the correct parameters
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, bsz, num_heads, head_dim)

        # Check if the function executes without errors
        self.assertIsNotNone(q_embed)
        self.assertIsNotNone(k_embed)

        # Check if the output tensors have the correct shapes
        self.assertEqual(q_embed.shape, q.shape)
        self.assertEqual(k_embed.shape, k.shape)

if __name__ == '__main__':
    unittest.main()
