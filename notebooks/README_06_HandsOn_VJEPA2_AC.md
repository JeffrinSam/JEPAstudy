# Notebook 06: Hands-On with V-JEPA 2-AC (Action-Conditioned World Model)

## Overview

This notebook is the practical capstone of the series. Instead of building toy models, you work directly with the real V-JEPA 2 codebase from `refs/vjepa2/`, building and running CPU-compatible versions of the actual encoder, AC predictor, ACBlock, and CEM planner. You will trace the real forward pass, understand the DROID dataset format, implement both training losses, learn how to load pretrained weights, and see exactly which files and lines to modify for each ICRA paper idea. This is your bridge from study to implementation.

## Prerequisites

- Complete Notebooks 01-05 (all prior material)
- Understanding of VisionTransformer, VisionTransformerPredictorAC, ACBlock, ACRoPEAttention
- Familiarity with the DROID dataset, teacher-forcing/autoregressive losses, and CEM planning
- The `refs/vjepa2/` directory should be present (cloned from facebookresearch/vjepa2)

## Estimated Time

3-4 hours for a focused study session (longer if experimenting with code modifications)

## Table of Contents

| Section | Cells | What You'll Learn | Key Outputs |
|---------|-------|-------------------|-------------|
| Title & Overview | 0-2 | What you will learn; setup and imports | Setup output |
| Section 2: Real Encoder -- VisionTransformer | 3-9 | Real `VisionTransformer` class: PatchEmbed3D, RoPE, Block; parameter table for all model sizes | Parameter count table, Plot: `06_ac_predictor_token_flow.png` (in later cell) |
| Key Components in VisionTransformer.__init__ | 5-7 | Three main pieces: PatchEmbed3D, Block stack, LayerNorm | Component breakdown |
| Section 3: Real AC Predictor -- VisionTransformerPredictorAC | 10-17 | Forward pass step-by-step: project, interleave, ACBlocks, extract | Plot: `06_causal_mask_and_tokens.png`, dimension trace |
| Forward Pass Step-by-Step | 12-13 | Exact operations in PredictorAC.forward(x, actions, states) | Printed shapes at each step |
| Understanding Every Dimension | 15-17 | Token sequence layout and causal mask visualization | Printed dimension table |
| Section 4: ACBlock & ACRoPEAttention Deep Dive | 18-25 | How ACBlock splits action vs spatial tokens for different RoPE | Plots: `06_rope_splitting.png`, `06_acblock_attention.png` |
| How ACRoPEAttention handles action vs spatial tokens | 20-21 | The critical split: RoPE for spatial tokens, learned PE for action tokens | RoPE splitting visualization |
| Why action tokens need different positional encoding | 22-23 | 3D spatial coordinates make no sense for action tokens | Explanation |
| Attention Trace | 24-25 | Full attention weight visualization across heads | Attention heatmap |
| 5. Real Data Format: DROID Dataset | 26-28 | 62 hours of diverse robot manipulation; action/state dimensions; data format | Plot: `06_droid_data.png` |
| 6. Loss Functions: Teacher-Forcing + Autoregressive | 30-34 | Implementing both loss components with real code structure | Plot: `06_tf_vs_ar_unrolled.png`, `06_loss_ablation.png` |
| Understanding the Loss Outputs | 33 | What jloss (TF) and sloss (AR) numbers mean | Loss interpretation |
| 7. Training Loop Walkthrough | 35-38 | Simplified training loop matching real `app/vjepa_droid/train.py` structure | Plot: `06_training_loop.png` |
| Understanding the Training Setup | 37 | TinyEncoder, TinyPredictor, TinyTargetEncoder -- how they map to real code | Component mapping |
| 8. CEM Planning | 39-42 | Running CEM with the world model: sample, evaluate, refine, select best | Plot: `06_cem_planning.png` |
| Reading the CEM Output | 42 | How to interpret CEM iteration plots: convergence from random to optimal | CEM interpretation guide |
| 9. Loading Pretrained Weights & Transfer Learning | 43-46 | Two-stage loading: Stage 1 (video pretrain) + Stage 2 (AC fine-tune); weight key remapping | Plot: `06_vram_freeze.png` |
| Smart Device Detection | 46 | Handling AMD GPUs, CUDA, MPS, and CPU fallback | Device detection code |
| 10. Where to Modify for Your ICRA Paper | 47-50 | Exact files and line numbers for each of the 4 paper ideas | Unified JEPA-VLA prototype code |
| Understanding the Unified JEPA-VLA Prototype | 50 | Modified predictor accepting language tokens alongside vision/action/state | Prototype analysis |
| 11. Fine-Tuning on Custom Data | 51-53 | Data format requirements, training configuration, compute budget | Training timeline |
| Training Timeline & Compute Budget | 53 | Full reproduction: 256 A100s for 3 days; your hardware: ViT-Small on CPU | Compute table |
| 12. Evaluation & Next Steps | 54-56 | World model evaluation (not policy evaluation), metrics, CEM-based evaluation | Plot: `06_evaluation.png` |
| Complete Roadmap | 56 | Phase 1 (done: notebooks) through Phase 4 (ICRA submission) | Roadmap |

## Key Formulas

| Formula | Meaning | Cell |
|---------|---------|------|
| `jloss = L1(predictor(z_gt_t, a_t), sg[z_gt_{t+1}])` | Teacher-forcing loss: predict from ground-truth context | 30 |
| `sloss = L1(predictor(z_hat_t, a_t), sg[z_gt_{t+1}])` | Autoregressive loss: predict from own previous predictions | 30 |
| `loss = 0.5*jloss + 0.5*sloss` | Combined training loss (equal weighting) | 30 |
| `CEM: mu_{k+1}, sigma_{k+1} = fit(top_K(samples))` | CEM update: refit distribution to best K samples | 39 |
| `load_pretrained: k.replace("backbone.", "")` | Weight key remapping for DDP-wrapped checkpoints | 43 |
| `VRAM ~ params * 4 bytes (fp32) or * 2 bytes (fp16)` | VRAM estimation for model loading | 43 |

## Key Visualizations

| File | Description |
|------|-------------|
| `plots/06_ac_predictor_token_flow.png` | Token flow through the AC predictor: input projection, interleaving, ACBlocks, output extraction |
| `plots/06_causal_mask_and_tokens.png` | Causal attention mask alongside the token sequence layout |
| `plots/06_rope_splitting.png` | How RoPE dimensions are split differently for action vs spatial tokens |
| `plots/06_acblock_attention.png` | Attention weights inside an ACBlock across multiple heads |
| `plots/06_droid_data.png` | DROID dataset format: 3D trajectory, action deltas, gripper state |
| `plots/06_tf_vs_ar_unrolled.png` | Unrolled comparison of teacher-forcing vs autoregressive loss computation |
| `plots/06_loss_ablation.png` | Ablation: effect of different TF/AR loss weightings |
| `plots/06_training_loop.png` | Simplified training loop flow diagram |
| `plots/06_cem_planning.png` | CEM optimization iterations showing convergence |
| `plots/06_vram_freeze.png` | VRAM usage with different freezing strategies (freeze encoder vs train all) |
| `plots/06_evaluation.png` | Evaluation metrics and methodology for world model assessment |

## Self-Check Questions

1. What are the three main pieces of `VisionTransformer.__init__`, and what does each do?
2. In the AC predictor's forward pass, what is the exact order of operations from input to output?
3. Why does `ACRoPEAttention` apply RoPE only to spatial tokens and not to action tokens?
4. What is the DROID dataset, and what are the dimensions of its action and state vectors?
5. What is the difference between jloss and sloss in the training code? Which one is harder to minimize, and why?
6. How does CEM select the best action sequence from 800 random candidates?
7. When loading pretrained weights, why do you need `k.replace("backbone.", "")`?
8. For the Unified JEPA-VLA paper idea, which specific files in `refs/vjepa2/` would you need to modify?
9. On your AMD RX 6700S, can you run ViT-Giant inference? What is the realistic alternative?
10. What is the evaluation methodology for a world model (as opposed to a direct policy)?

## Common Issues

- **No GPU required for learning**: All code in this notebook creates CPU-compatible miniature versions. GPU is only needed for full-scale training.
- **refs/vjepa2/ must exist**: The notebook references real source files. Make sure the repo is cloned at `refs/vjepa2/`.
- **AMD GPU support**: PyTorch ROCm support is limited. The notebook includes smart device detection that falls back to CPU if CUDA/ROCm is unavailable.
- **Pretrained weight files**: The notebook demonstrates the loading pattern but does not include actual checkpoint files (they are ~2GB each). Download from Meta's model hub if needed.
- **ViT-Giant is too large**: At ~1.3B parameters, ViT-Giant needs ~5GB+ VRAM just for inference. Use ViT-Large (304M params) or ViT-Small for prototyping.
- **DDP prefix stripping**: Real checkpoints saved with DistributedDataParallel have "backbone." or "module." prefixes that must be stripped when loading into a single-GPU or CPU model.

## Next Steps

- **Phase 1**: Set up LIBERO environment and run VLA-JEPA code from `refs/VLA-JEPA/`
- **Phase 2**: Implement your Unified JEPA-VLA modifications in the files identified in Section 10
- **Phase 3**: Run experiments on LIBERO benchmarks with ablation studies from Notebook 05
- **Phase 4**: Write and submit your ICRA paper using the structure from Notebook 05, Section 13
