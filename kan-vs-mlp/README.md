# KAN vs MLP in Transformers — Small-Scale Research Comparison

A self-contained experiment comparing **Kolmogorov-Arnold Networks (KANs)** vs **standard MLPs** as the feed-forward network (FFN) inside a GPT-style transformer, trained on WikiText-2.

## What's Being Compared?

In a standard transformer, each layer has:
1. **Multi-Head Self-Attention** — models token interactions
2. **FFN (Feed-Forward Network)** — processes each token independently

The FFN is traditionally a 2-layer MLP: `Linear → GELU → Linear`. **KANs** replace this with learnable B-spline activation functions on the edges, based on the [Kolmogorov-Arnold representation theorem](https://arxiv.org/abs/2404.19756).

```
MLP:  y = W₂ · GELU(W₁ · x)     ← fixed activation, learned weights
KAN:  y = Σ φⱼ(Σ φᵢⱼ(xᵢ))       ← learned activations on edges
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the benchmark (trains both MLP and KAN models)
python train.py

# Generate comparison plots
python plot_results.py
```

## Configuration

```bash
# Faster run (fewer epochs)
python train.py --epochs 5

# Use Apple Silicon GPU
python train.py --device mps

# Train only one model
python train.py --model kan
python train.py --model mlp
```

## Default Setup

| Setting | Value |
|---------|-------|
| Model size | ~1-2M params |
| Layers | 4 |
| Embedding dim | 128 |
| Heads | 4 |
| FFN dim | 512 |
| Sequence length | 128 |
| Dataset | WikiText-2 (byte-level) |
| Epochs | 10 |

## Project Structure

```
kan-vs-mlp/
├── kan_layer.py        # KAN implementation (B-spline)
├── model.py            # Small transformer (MLP or KAN FFN)
├── train.py            # Training benchmark
├── plot_results.py     # LinkedIn-ready visualizations
├── requirements.txt
├── README.md
├── results.json        # (generated after training)
└── plots/              # (generated after plotting)
    ├── loss_curves.png
    ├── perplexity.png
    ├── efficiency.png
    └── summary_panel.png   ← Best for LinkedIn!
```

## Key References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) — Original KAN paper
- [KAT: Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2409.10594) — KAN in vision transformers
- [Efficient-KAN](https://github.com/Blealtan/efficient-kan) — Efficient B-spline KAN implementation
- Inspired by [wave-field-llm](https://github.com/Pankh-AI/wave-field-llm)'s benchmarking methodology
