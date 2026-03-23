# Notebook 01: JEPA Fundamentals -- Math & Code From Scratch

## Overview

This notebook builds a complete understanding of the Joint Embedding Predictive Architecture (JEPA) from first principles. You start with the mathematical objective, implement a minimal working JEPA on toy data using only basic PyTorch (no ViT required), and then explore deep analyses of training dynamics, representation quality, and failure modes. By the end, you will have trained a JEPA, visualized its internals, and understood every equation it uses.

## Prerequisites

- Basic PyTorch (tensors, `nn.Module`, autograd)
- Matrix multiplication and linear algebra basics
- Understanding of what a neural network is and how gradient descent works
- No prior knowledge of JEPA, Vision Transformers, or self-supervised learning required

## Estimated Time

4-6 hours for a thorough study session (can be split across multiple sittings)

## Table of Contents

| Section | Cells | What You'll Learn | Key Outputs |
|---------|-------|-------------------|-------------|
| 1. What Problem Does JEPA Solve? | 0 | Why predicting representations beats predicting pixels; motivation for JEPA | Architecture dataflow diagram |
| 2. The JEPA Objective -- Mathematical Formulation | 1-2 | The core loss function, symbol-by-symbol breakdown of every term | Plot: `01_jepa_architecture_dataflow.png` |
| 2.5 The 5W+H of JEPA | 3-5 | Who created JEPA, what it does, where/when/why/how it works | Printed 5W+H summary |
| 3. Minimal JEPA Implementation | 6-8 | Overview of the toy implementation plan | Printed setup info |
| 3.1 Create Toy Data | 9-11 | How to represent images as patch sequences (16 patches, 32-dim) | Printed tensor shapes |
| 3.2 Masking Strategy | 12-15 | Random masking at 75%, how context/target splits work | Plot: `01_masking_spatial_grid.png` |
| 3.3 Build the Three Components | 16-18 | Online Encoder, Target Encoder, Predictor as simple MLPs | Printed model architectures |
| 3.4 The Training Loop | 19-25 | EMA update, stop-gradient, L1 loss, full training loop | Training loss curve, representation quality metrics |
| 3.7 Layer-by-Layer Activation Analysis | 26-28 | Deep dive into internal network behavior | Printed activation stats |
| 3.7.1 Activation Statistics | 29-31 | Mean, variance, kurtosis per layer | Printed statistics table |
| 3.7.2 Gradient Flow Analysis | 32-34 | How gradients propagate through the network | Gradient norm plots |
| 3.7.3 3D Loss Landscape | 35-37 | Visualizing the loss surface in 3D | 3D loss landscape plot |
| 3.7.4 Representation Space (PCA, t-SNE, UMAP) | 38-40 | 2D projections of learned representations | Scatter plots |
| 3.7.5 3D Representation Space | 41-43 | 3D PCA and t-SNE visualizations | 3D scatter plots |
| 3.7.6 Ablation Study | 44-46 | How hyperparameters affect JEPA training | Ablation comparison plots |
| 3.7.7 Attention Pattern Evolution | 47-49 | How attention changes during training | Attention heatmaps |
| 3.7.8 Mutual Information Analysis | 50-52 | Information flow between layers | MI plots |
| 3.6 Representation Collapse | 53 | What happens WITHOUT stop-gradient | Collapse demonstration |
| 3.5 Verify Representations | 54-58 | Proof that the encoder learned meaningful features | Representation quality metrics |
| 4. Key Differences: V-JEPA 2 vs Toy Model | 59 | What the real system adds (ViT, 3D RoPE, SwiGLU) | Comparison table |
| 5. Why L1 Loss and Not L2? | 60-62 | Mathematical and practical reasons for L1 | Loss comparison plots |
| 6.5 The apply_masks Gather Operation | 63 | How masking is implemented efficiently with torch.gather | Code walkthrough |
| 6. The Masking Strategy | 64-68 | Spatiotemporal block masking in V-JEPA 2 | Masking visualizations |
| 8. Complete 5W+1H Glossary | 69 | Every key term explained in 5W+1H format | Reference glossary |
| 9. Higher-Dimensional Visualizations (4D & 5D) | 70-76 | Advanced multi-dimensional analysis | 4D/5D plots |
| 10. Formal Mathematical Derivations | 77 | Rigorous proofs and derivations | Mathematical proofs |
| 7. Summary: JEPA Core Equations | 78 | Complete equation reference card | Equation summary |
| 11. Animated GIF Visualizations | 79-86 | EMA momentum comparison, collapse vs healthy training | GIFs: `01_ema_comparison.gif`, `01_collapse_vs_healthy.gif` |
| 12. Step-by-Step Intuition Builders | 87-92 | Concrete numbers through a forward pass, "What If" scenarios | Printed numerical traces |
| 13. Complete Lifecycle of a Patch | 93-95 | Tracing a single patch from pixels through training | Step-by-step walkthrough |
| 14. How Attention Actually Computes | 96-98 | Hand-computed attention with small numbers | Numerical attention trace |
| 15. Why JEPA Works: A Gentle Proof | 99-101 | Mathematical argument for why masking helps learning | Proof walkthrough |
| 16. LayerNorm vs BatchNorm | 102-104 | Why transformers use LayerNorm, not BatchNorm | Comparison |
| 17. Gradient Computation During Backprop | 105-107 | What happens in the backward pass | GIF: `01_gradient_backprop.gif` |
| 18. Tensor Dimensions Reference | 108-110 | Complete shape reference for every tensor in JEPA | Dimension table |
| 19. Self-Assessment | 111 | Test your understanding (with hidden answers) | 10+ questions with solutions |
| 20. Reading Guide for Every Plot and GIF | 112-116 | How to interpret each visualization | GIFs: `01_masking_strategies.gif`, `01_loss_landscape_evolution.gif` |
| 21. Formula Index | 117 | Every equation numbered and explained | Formula reference table |
| 22. Softmax Numerical Stability | 118-123 | LogSumExp trick, overflow/underflow prevention | Plots: `softmax_stability.png`, `logsumexp_trick.png` |
| 23. Complete Hand-Computed Forward Pass | 124-127 | Every number traced through a 2-patch, 4-dim JEPA | Plot: `hand_computed_forward_pass.png` |
| 24. JEPA Debugging Playbook | 128-133 | Systematic troubleshooting for common training failures | Plots: `jepa_diagnostics.png`, `training_failures.png` |
| 25. Training Monitoring Dashboard | 134-136 | Six key metrics to watch during training | Plot: `training_dashboard.png` |

## Key Formulas

| Formula | Meaning | Cell |
|---------|---------|------|
| `L = (1/|M|) * sum_i ||h_hat_i - sg[h_i^t]||_1` | JEPA Loss: average L1 distance between predicted and target representations for masked patches | 1 |
| `theta_bar <- m * theta_bar + (1-m) * theta` | EMA Update: target encoder slowly follows online encoder (m ~ 0.996) | 19 |
| `h = f_theta(x)` | Encoder: maps input patches to representation vectors | 16 |
| `h_hat_M = g_phi(h_ctx)` | Predictor: predicts masked patch representations from visible context | 16 |
| `Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V` | Self-Attention: weighted combination of values based on query-key compatibility | 96 |
| `Q = XW_Q, K = XW_K, V = XW_V` | QKV Projection: three learned linear transforms of input | 96 |
| `FFN(x) = W_2 * GELU(W_1 x + b_1) + b_2` | Feed-Forward Network: per-token nonlinear transformation | 78 |
| `x_out = x_in + f(x_in)` | Residual Connection: preserves information across layers | 78 |
| `LN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta` | LayerNorm: normalize each token to zero mean, unit variance | 102 |
| `dL_L1/dy_hat = sign(y_hat - y)` | L1 Gradient: constant magnitude (robust to outliers) | 60 |
| `Window ~ 1/(1-m)` | EMA Window: effective averaging window for momentum m | 78 |

## Key Visualizations

| File | Description |
|------|-------------|
| `plots/01_jepa_architecture_dataflow.png` | Complete JEPA architecture showing encoder, predictor, target encoder, and data flow |
| `plots/01_masking_spatial_grid.png` | How the 4x4 patch grid is masked (visible vs target patches) |
| `gifs/01_ema_comparison.gif` | Animated comparison of different EMA momentum values (0.9 vs 0.99 vs 0.999) |
| `gifs/01_collapse_vs_healthy.gif` | Side-by-side animation of healthy training vs representation collapse |
| `gifs/01_gradient_backprop.gif` | Animated visualization of gradient flow through the JEPA network |
| `gifs/01_masking_strategies.gif` | Different masking strategies compared (random, block, spatiotemporal) |
| `gifs/01_loss_landscape_evolution.gif` | How the loss landscape changes during training |
| `plots/softmax_stability.png` | Why naive softmax overflows and the fix |
| `plots/logsumexp_trick.png` | Step-by-step LogSumExp trick demonstration |
| `plots/hand_computed_forward_pass.png` | Every number in a tiny JEPA forward pass |
| `plots/jepa_diagnostics.png` | Health check diagnostics for JEPA training |
| `plots/training_failures.png` | Three training scenarios: healthy, collapse, and divergence |
| `plots/training_dashboard.png` | Six-panel monitoring dashboard for JEPA training |

## Self-Check Questions

1. What does JEPA predict -- pixels or representations? Why does this matter?
2. What are the THREE main components of JEPA, and what does each one do?
3. What happens if you remove the stop-gradient from the target encoder? Why?
4. Why does V-JEPA 2 use L1 loss instead of L2 (MSE)?
5. What is the EMA update rule, and why not just copy the online encoder weights?
6. How does masking ratio affect what the model learns? What happens at 50% vs 90%?
7. Explain the difference between the online encoder and target encoder in one sentence each.
8. What is representation collapse, and what mechanism prevents it in JEPA?
9. Why does the predictor only receive visible patch representations, not the mask tokens?
10. If the training loss drops to near-zero very quickly, is that good or bad? Why?

## Common Issues

- **Representation collapse**: If you remove stop-gradient or set EMA momentum too low, all outputs converge to the same constant vector. The loss looks good but representations are useless.
- **Section ordering**: Sections are not strictly numbered sequentially (e.g., 3.6 appears before 3.5). Follow the cell order, not the section numbers.
- **Large notebook**: At 137 cells, this is the largest notebook. Consider studying it in 2-3 sessions.
- **Toy model vs real V-JEPA 2**: The toy model uses MLPs, not Vision Transformers. This is intentional -- understand the algorithm first, architecture second.
- **GPU not required**: All code runs on CPU with toy data. No GPU needed for this notebook.

## Next Steps

- **Notebook 02** (V-JEPA 2 Architecture Deep Dive): Learn the real ViT encoder, 3D RoPE, SwiGLU, and how video is actually processed
- **Notebook 03** (V-JEPA 2-AC World Model): See how JEPA becomes a robot world model with action conditioning
