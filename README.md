# Gemma 10M MLX

Gemma 2B with 10M context that runs on apple MLX. Our implementation uses **<32GB** of memory!

![Graphic of our implementation context](./images/graphic.png)

**Features:**

- 10M sequence length on Gemma 2B.
- Runs on less then 32GB of memory.
- Native inference on Apple Silicon using MLX.
- Highly performing retrieval - needle in hay stack.

## Quick Start

Install using pypi

```bash
pip install gemma-mlx
```

Download and run the model - view the model on [huggingface]().

```python
from gemma_mlx import gemma_10m

model = gemma_10m()

response = model.generate("Who is the president of the united states?")

print(response)
```

## Example

** Using a local PDF as context**

```python


```

## How does this work?

The largest bottleneck (in terms of memory) for LLMs is the KV cache. It grows quadratically in vanilla multi-head attention, thus limiting the size of your sequence length.

Our approach splits the attention in local attention blocks as outlined by [InfiniTransformer](https://arxiv.org/abs/2404.07143). We take those local attention blocks and apply recurrance to the local attention blocks for the final result of 10M context global atention.

A lot of the inspiration for our ideas comes from the [Transformer-XL](https://arxiv.org/abs/1901.02860) paper.
