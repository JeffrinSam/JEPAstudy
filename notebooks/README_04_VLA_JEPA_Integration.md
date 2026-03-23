# Notebook 04: JEPA-Based VLA Models -- The ICRA-Relevant Frontier

## Overview

This notebook provides a deep dive into two key 2025-2026 papers that integrate JEPA with Vision-Language-Action (VLA) models: JEPA-VLA (feature injection, Tsinghua University) and VLA-JEPA (world model integration, from the VLA-JEPA paper). You will understand how each approach works, compare their architectures and benchmark results, learn the flow matching action generation technique, and identify open research gaps. This is the critical notebook for connecting JEPA fundamentals to publishable ICRA research.

## Prerequisites

- Complete Notebooks 01-03 (JEPA fundamentals, V-JEPA 2 architecture, action-conditioned world model)
- Understanding of the JEPA loss, encoder/predictor architecture, and CEM planning
- Awareness of what VLAs are (RT-2, Octo, OpenVLA -- briefly introduced here)

## Estimated Time

3-4 hours for a focused study session

## Table of Contents

| Section | Cells | What You'll Learn | Key Outputs |
|---------|-------|-------------------|-------------|
| Title & Overview | 0-1 | Goals: understand the two key JEPA+VLA integration papers | -- |
| 0.5 The 5W+H of JEPA-Based VLAs | 2 | Who developed JEPA-VLA and VLA-JEPA, key differences | 5W+H summary |
| Part A: JEPA-VLA (Feature Injection) | 3-6 | How V-JEPA 2 is plugged into existing VLAs as a better visual encoder | Plot: `04_jepa_vla_vs_vla_jepa.png` |
| Leakage-Free Design | 7-8 | The critical innovation preventing the VLM from copying instead of predicting | Plot: `04_leakage_free_design.png` |
| Part B: VLA-JEPA (World Model) | 9-13 | Building a VLA with a JEPA world model as a core component | Forward pass output, flow matching plots |
| Part C: VLA-JEPA Two-Stage Training | 14-18 | Stage 1 (JEPA pretraining) and Stage 2 (VLA fine-tuning with gated cross-attention) | Gated cross-attention plots, flow matching demo |
| Part D: Comparative Analysis | 19-21 | Side-by-side comparison on LIBERO benchmarks | LIBERO benchmark comparison |
| Part F: End-to-End Pipeline | 22-27 | Complete pipeline trace from raw video to robot action | Pipeline trace table, flow visualization |
| Part F.2: Fusion Mechanism Comparison | 28-30 | Cross-attention, feature concatenation, gated fusion compared | Fusion comparison plots |
| Part F.3: Flow Matching Deep Dive | 31-33 | Vector fields in 2D and 3D; how noise becomes structured actions | Flow matching vector field plots |
| Part F.4: Action Distribution Analysis | 34-36 | Multi-modal action distributions in 2D and 3D | Action distribution plots |
| Part E: Open Research Gaps | 37 | Five ICRA paper opportunities: combining both approaches, safety, planning, sim-to-real | Gap analysis |
| Part G: Complete 5W+1H Glossary | 38-41 | VLA and integration terms with 4D/5D analysis | Glossary, 4D/5D plots |
| Part H: Flow Matching Mathematical Derivation | 42 | Linear interpolation path, velocity field, training objective, inference ODE | Full derivation |
| Part I: Animated GIF -- Flow Matching | 43-44 | Watch noise transform into robot actions step by step | GIF: `04_flow_matching.gif` |
| Part J: VLA Processes One Instruction | 45-47 | Complete trace of "pick up the red cup" through every neural network layer | VLA inference walkthrough plot |
| Part K: Action Chunking | 48-50 | Why robots predict multiple steps at once instead of single actions | Action chunking visualization |
| Part L: Self-Assessment | 51 | Questions on JEPA-VLA vs VLA-JEPA, flow matching, leakage-free design | Q&A with hidden answers |
| Flow Matching Intuitive Guide | 52-53 | From noise to robot actions -- visual intuition | GIF: `04_flow_2d.gif` |
| Flow Matching in Action Space | 54-56 | 7D robot action denoising, convergence analysis | Convergence table |
| End-to-End VLA Pipeline: Tensor Shapes | 57-59 | Every tensor shape traced through full VLA-JEPA pipeline | Pipeline trace diagram |
| VLA Model Comparison | 60-62 | Decision guide: which architecture for which task | Radar chart comparison |

## Key Formulas

| Formula | Meaning | Cell |
|---------|---------|------|
| `x_t = (1-t)*x_0 + t*x_1, t in [0,1]` | Flow matching linear interpolation path from noise x_0 to target x_1 | 42 |
| `u_t = dx_t/dt = x_1 - x_0` | Ground-truth velocity field (constant along linear path) | 42 |
| `L_FM = E[\|\|v_theta(x_t, t) - (x_1 - x_0)\|\|^2]` | Flow matching training loss: predict the velocity field | 42 |
| `x_{k+1} = x_k + dt * v_theta(x_k, t_k)` | Inference: Euler integration of learned velocity field | 42 |
| `Gated fusion: h = h_vlm + tanh(alpha) * CrossAttn(h_vlm, h_jepa)` | Gated cross-attention for fusing JEPA and VLM features | 14 |
| `noise -> R^{7x7} --[4 denoising steps]--> actions in R^{7x7}` | Flow matching in VLA-JEPA: 7 dims x 7 chunks, 4 steps | 54 |

## Key Visualizations

| File | Description |
|------|-------------|
| `plots/04_jepa_vla_vs_vla_jepa.png` | Side-by-side architecture comparison of JEPA-VLA (feature injection) vs VLA-JEPA (world model) |
| `plots/04_leakage_free_design.png` | How VLA-JEPA prevents information leakage in the world model |
| `gifs/04_flow_matching.gif` | Animated flow matching: noise transforms into structured robot actions |
| `gifs/04_flow_2d.gif` | 2D flow matching visualization with vector fields |
| `gifs/04_flow_2d_snapshots.png` | Static snapshots of 2D flow matching at key timesteps |
| `gifs/04_flow_7d_denoising.png` | 7D robot action denoising process |
| `gifs/04_vla_pipeline_trace.png` | Complete VLA-JEPA pipeline with tensor shapes at each stage |
| `gifs/04_vla_comparison.png` | Radar chart comparing different VLA architectures |

## Self-Check Questions

1. What is the key difference between JEPA-VLA (feature injection) and VLA-JEPA (world model integration)?
2. Why does flow matching use LINEAR interpolation between noise and target? What advantage does this have over diffusion?
3. Why is "leakage-free design" critical for VLA-JEPA? What would happen without it?
4. What are the two stages of VLA-JEPA training, and what data is used in each?
5. How many denoising steps does VLA-JEPA use for action generation, and why is this number important for real-time control?
6. What is gated cross-attention, and why use a learnable gate (tanh(alpha)) instead of simple addition?
7. Name three open research gaps where JEPA-VLA and VLA-JEPA could be improved for an ICRA paper.
8. What is action chunking, and why do robots predict multiple steps at once?

## Common Issues

- **JEPA-VLA vs VLA-JEPA naming**: These are two different papers with similar names. JEPA-VLA = plug V-JEPA 2 into an existing VLA. VLA-JEPA = build a new VLA that includes a JEPA world model. Do not confuse them.
- **Flow matching vs diffusion**: Flow matching is NOT the same as diffusion. It uses straight ODE paths (not curved SDEs), needs fewer steps (4-20 vs 50-1000), and produces deterministic outputs.
- **LIBERO benchmarks**: The comparison uses four LIBERO suites (Spatial, Object, Goal, Long). Each tests different capabilities.
- **Section lettering**: This notebook uses Parts A-L instead of numbered sections. The ordering is A, B, C, D, F, E, G, H, I, J, K, L (F comes before E in the cell order).

## Next Steps

- **Notebook 05** (ICRA Research Gaps): Identify specific paper ideas and plan your research contribution
- **Notebook 06** (Hands-On): Work with the real codebase to prototype modifications
