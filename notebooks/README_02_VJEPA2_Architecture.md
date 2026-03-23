# Notebook 02: V-JEPA 2 Architecture Deep Dive

## Overview

This notebook dissects the real V-JEPA 2 architecture by reading the actual source code from `refs/vjepa2/` and building minimal CPU-runnable versions of each component. You will understand how video becomes patch tokens via PatchEmbed3D, how 3D Rotary Position Embeddings encode spatiotemporal position, how SwiGLU replaces GELU in the feed-forward network, and how the full encoder and predictor work together. This is the bridge between the toy JEPA in Notebook 01 and the real production model.

## Prerequisites

- Complete Notebook 01 (JEPA Fundamentals) -- you need to understand the JEPA objective, loss function, and training loop
- Familiarity with self-attention (covered in Notebook 01, Section 14)
- Basic understanding of convolutions (for PatchEmbed3D)

## Estimated Time

3-4 hours for a focused study session

## Table of Contents

| Section | Cells | What You'll Learn | Key Outputs |
|---------|-------|-------------------|-------------|
| Title & Overview | 0 | Goals and structure of this notebook | -- |
| 0.5 The 5W+H of V-JEPA 2's Architecture | 2 | Who designed it, what it does, key design decisions | 5W+H summary |
| Key Term Definitions | 3 | Reference definitions for softmax, patch embedding, etc. | Glossary |
| 1. Video to Patches: PatchEmbed3D | 4-7 | How 3D convolution converts video [B,3,T,H,W] to patch tokens; tubelet size 2x16x16 | Plot: `02_patchembed3d_video_to_tokens.png` |
| 2. 3D Rotary Position Embeddings (3D RoPE) | 8-14 | How RoPE encodes temporal + spatial position; dimension splitting for depth/height/width | RoPE rotation plots, dimension split tables |
| 3.5 Self-Attention: Matrix Math in Detail | 15-21 | QKV projections, attention matrix computation, SwiGLU vs GELU comparison | Attention matrix heatmaps, SwiGLU plots |
| 3. Transformer Block: The Building Block | 22-24 | Pre-norm architecture, residual connections, SwiGLU FFN parameter comparison | Parameter count comparison |
| 4. Weight Initialization: Rescaling Trick | 25-27 | Why V-JEPA 2 rescales residual branches by 1/sqrt(2*depth) | Plot: rescaling factor across depths |
| 5. Mini V-JEPA 2 (CPU-Runnable) | 28-30 | Complete miniature encoder + predictor that runs on CPU | Forward pass output shapes |
| 5.5 Full Encoder Forward Pass with Shape Tracking | 31-33 | Every tensor shape at every layer of the encoder | Shape tracking table |
| 5.5.1 Tensor Statistics Across Layers | 34-35 | Mean, std, min, max through each transformer block | Statistics plots |
| 5.5.2 Multi-Head Attention Analysis | 36-37 | What each attention head specializes in | Attention head plots |
| 5.5.3 3D Patch Embeddings in Space | 38-39 | How patch vectors arrange in 3D embedding space | 3D scatter plot |
| 5.5.4 RoPE Frequency Spectrum & 3D Rotation | 40-41 | Frequency bands for temporal vs spatial dimensions | Frequency plots |
| 5.5.5 Layer-wise Representation Similarity (CKA) | 42-43 | How similar representations are between layers | CKA heatmap |
| 5.5.6 Gradient Norms & Parameter Statistics | 44-45 | Per-layer gradient health check | Gradient norm plots |
| 5.5.7 Token Mixing: Information Flow Between Patches | 46-48 | How attention enables patches to share information | Token mixing plots |
| 6. The Predictor: How It Works | 49-51 | Token flow: visible tokens + mask tokens through predictor | Predictor token flow diagram |
| 7. Model Sizes: V-JEPA 2 Family | 52-54 | ViT-Large, ViT-Huge, ViT-Giant parameter counts and FLOPs | Model size comparison table |
| 9. Complete 5W+1H Glossary | 55-59 | Every architecture term in 5W+1H format | 4D attention visualization, architecture search plots |
| 8. Summary: Key Points | 60 | Encoder (ViT-g: 40 blocks, 1408 dim, 22 heads) and predictor specs | Summary reference |
| 10. Animated GIF: Attention Through Layers | 61-62 | How attention patterns evolve from early to late layers | GIF: `02_attention_through_layers.gif` |
| 11. How 3D RoPE Works: Step-by-Step With Numbers | 63-65 | Concrete numerical walkthrough of RoPE rotation | Printed numerical trace |
| 12. The Predictor: Detailed Walkthrough | 65 | How the predictor processes visible + mask tokens | Detailed explanation |
| 13. The Residual Stream | 66-67 | How information flows through the residual path (the key insight most tutorials miss) | Residual stream analysis |
| 14. Self-Assessment | 68 | 5+ questions with hidden answers | Q&A with solutions |
| Hand-Computed Multi-Head Attention | 70-73 | Every number traced through 2-head, 4-patch, 8-dim attention | Plots: `_mha_attention_heads.png` |
| 3D RoPE: Numerical Walkthrough | 74-76 | Position encoding with concrete rotation matrices | Plots: `_rope_2d_rotation.png`, `_rope_3d_positions.png`, `_rope_attention_bias.png` |
| Weight Initialization Deep Dive | 77-78 | Why random numbers matter, init comparison | GIF: `02_init_comparison.gif`, Plot: `_init_comparison_static.png` |
| The Residual Stream: 12 Layers | 79-81 | Contribution analysis and layer similarity | Plots: `_residual_stream_contributions.png`, `_residual_layer_similarity.png` |

## Key Formulas

| Formula | Meaning | Cell |
|---------|---------|------|
| `PatchEmbed3D: Conv3d(3, D, kernel=(2,16,16), stride=(2,16,16))` | Converts video to patch tokens; tubelet = 2 frames x 16x16 pixels | 4 |
| `RoPE(x, pos) = x * cos(pos * freq) + rotate(x) * sin(pos * freq)` | Rotary position encoding: rotates query/key vectors by position-dependent angles | 8 |
| `SwiGLU(x) = (xW_1) * sigmoid(xW_1) * (xW_2)` | Gated feed-forward with Swish activation; replaces GELU FFN | 22 |
| `Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V` | Scaled dot-product attention | 15 |
| `x_out = x + (1/sqrt(2*depth)) * Block(LN(x))` | Residual with depth-dependent rescaling for stable training | 25 |
| `N_patches = (T/t_p) * (H/h_p) * (W/w_p)` | Number of patches from video dimensions and tubelet size | 4 |
| `d_k = D / num_heads` | Per-head dimension in multi-head attention | 15 |

## Key Visualizations

| File | Description |
|------|-------------|
| `plots/02_patchembed3d_video_to_tokens.png` | How PatchEmbed3D converts a video volume into a sequence of patch tokens |
| `plots/02_mha_head_specialization.png` | What different attention heads learn to focus on |
| `gifs/02_attention_through_layers.gif` | Animated attention patterns evolving from early (local) to late (global) layers |
| `gifs/02_weight_init.gif` | How weight initialization affects training dynamics |
| `gifs/02_init_comparison.gif` | Side-by-side comparison of different initialization strategies |
| `plots/_mha_attention_heads.png` | Hand-computed multi-head attention with every number shown |
| `plots/_rope_2d_rotation.png` | 2D visualization of how RoPE rotates embedding dimensions |
| `plots/_rope_3d_positions.png` | 3D position encoding for video patches (temporal + spatial) |
| `plots/_rope_attention_bias.png` | How RoPE creates distance-dependent attention bias |
| `plots/_init_comparison_static.png` | Static comparison of initialization methods |
| `plots/_residual_stream_contributions.png` | How much each layer contributes to the residual stream |
| `plots/_residual_layer_similarity.png` | Similarity between representations at different layers |

## Self-Check Questions

1. How many patches does a 224x224 image produce with patch_size=16? What about a video with 8 frames and tubelet size 2x16x16?
2. In multi-head attention with 6 heads and embed_dim=384, what is the per-head dimension?
3. Why does the FFN expand to a larger dimension and then contract back? What expansion factor does SwiGLU use vs standard GELU FFN?
4. What is the purpose of the 1/sqrt(d_k) scaling in attention? What would happen without it?
5. In 3D RoPE, how is the embedding dimension split among temporal, height, and width axes?
6. Why does V-JEPA 2 use SwiGLU instead of GELU in its feed-forward network?
7. What does the predictor receive as input -- all patches or only visible patches?
8. How does the weight initialization rescaling trick (1/sqrt(2*depth)) help with deep networks?

## Common Issues

- **3D RoPE dimension split**: The head dimension is split into 4 parts, not 3. The temporal axis gets 1/4, height gets 1/4, width gets 1/4, and the last 1/4 has no position encoding. This is specific to V-JEPA 2.
- **SwiGLU parameter count**: SwiGLU uses 8/3*D expansion instead of 4*D, but has an extra gate projection, so total parameters are similar.
- **Predictor vs Encoder**: The predictor is a separate, smaller transformer -- do not confuse it with the encoder. It has fewer layers and a different embedding dimension.
- **Model sizes**: ViT-Giant has 40 blocks with embed_dim=1408. The numbers in this notebook are for a miniature version that runs on CPU.

## Next Steps

- **Notebook 03** (V-JEPA 2-AC World Model): Learn how the architecture is extended with action conditioning for robotics
- **Notebook 06** (Hands-On): Work directly with the real V-JEPA 2 codebase
