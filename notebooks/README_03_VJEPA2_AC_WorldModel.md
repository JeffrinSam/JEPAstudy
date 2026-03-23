# Notebook 03: V-JEPA 2-AC -- Action-Conditioned World Model for Robotics

## Overview

This notebook explains how V-JEPA 2 becomes a robot world model by conditioning on actions. You will learn how robot actions and states are represented, how the frame-causal attention mask prevents future information leakage, how action tokens are interleaved with spatial tokens, and how the model is trained with both teacher-forcing and autoregressive losses. The notebook culminates with CEM (Cross-Entropy Method) planning in latent space -- how a robot uses the world model to select actions by imagining future outcomes.

## Prerequisites

- Complete Notebook 01 (JEPA Fundamentals) -- understand the JEPA loss, EMA, stop-gradient
- Complete Notebook 02 (V-JEPA 2 Architecture) -- understand ViT encoder, 3D RoPE, transformer blocks
- Basic understanding of robot actions (joint positions/velocities, gripper state)

## Estimated Time

3-4 hours for a focused study session

## Table of Contents

| Section | Cells | What You'll Learn | Key Outputs |
|---------|-------|-------------------|-------------|
| Title & Overview | 0-1 | Goals: understand action conditioning for robotics | Setup output |
| 0.5 The 5W+H of V-JEPA 2-AC | 3 | Who developed it, what makes it different from base V-JEPA 2 | 5W+H summary |
| 1. Action & State Representation | 4-6 | How 7-DoF robot actions and proprioceptive states are formatted | Printed dimension info |
| 2. Frame-Causal Attention Mask | 7-9 | Block-lower-triangular mask preventing future leakage | Causal mask visualization |
| 3. Token Interleaving | 10-13 | How action tokens are inserted between frame patches in the sequence | Plot: `03_token_interleaving_strip.png` |
| 4. ACRoPEAttention | 14-16 | Different RoPE for action vs spatial tokens; why actions need separate positional encoding | Printed attention trace |
| 5.5 Teacher-Forcing vs Autoregressive Math | 17 | Detailed mathematical comparison of TF and AR losses | Equations |
| 5. Training Losses | 19-25 | Two loss components: teacher-forcing (jloss) + autoregressive (sloss); rollout error accumulation | Plot: `03_tf_vs_ar_dataflow.png`, loss curves |
| 6. Planning with CEM | 26-28 | How CEM samples, evaluates, and refines action sequences in latent space | CEM energy landscape plot |
| 7.5 Complete ACBlock Implementation | 29-31 | Building the full Action-Conditioned Block from scratch | ACBlock computation trace |
| 7.5.1 Causal Mask Construction | 32-34 | Step-by-step construction of the block-lower-triangular mask | Mask construction visualization |
| 7.5.2 Attention Weight Analysis | 35-37 | How much attention goes to action tokens vs frame tokens | Attention weight plots |
| 7.5.3 3D Rollout Quality vs Planning Horizon | 38-40 | How prediction quality degrades with longer rollouts | 3D rollout quality plot |
| 7.5.4 Action Embedding Space (t-SNE & PCA) | 41-43 | How different robot actions are represented in embedding space | t-SNE/PCA scatter plots |
| 7.5.5 Loss Decomposition | 44-46 | How TF and AR loss components balance during training | Loss decomposition plots |
| 8. Complete 5W+1H Glossary | 47-49 | World model and robotics terms in 5W+1H format | Glossary with key notation |
| 4D/5D Performance Analysis | 50-51 | Multi-dimensional analysis of model performance | 4D/5D plots |
| 9. Mathematical Deep Dive | 52 | Full TF loss, AR loss, combined loss, and CEM objective equations | Equations reference |
| 7. V-JEPA 2-AC Results Summary | 53 | Comparison vs Octo and Cosmos on benchmarks | Results table |
| 10. Animated GIFs | 54-59 | CEM planning optimization and rollout error accumulation | GIFs: `03_cem_planning.gif`, `03_rollout_error.gif` |
| 11. Robot Decision Loop | 60-62 | Complete 10Hz decision cycle: observe, encode, plan, act | Decision loop walkthrough |
| 12. Teacher-Forcing vs Autoregressive with Numbers | 63-65 | 5-step rollout traced through both training modes | GIF: `03_tf_vs_ar.gif` |
| 13. Self-Assessment | 66 | Questions on world models, causal attention, CEM | Q&A with hidden answers |
| 14. Robot Action Space | 67-70 | 7-DoF action vectors, trajectory visualization, action chunking | Plots: `_tmp_action_vec.png`, `_tmp_trajectory.png`, `_tmp_chunking.png` |
| 15. Frame-Causal Attention Deep Dive | 71-73 | Why the future must be hidden; mask comparison | GIF: `03_causal_mask.gif`, Plots: `_tmp_causal_mask.png`, `_tmp_causal_compare.png` |
| 16. Training Monitoring Dashboard | 74-76 | Key metrics for healthy vs collapsed training | Plots: `_tmp_dashboard_healthy.png`, `_tmp_dashboard_collapse.png` |
| 17. With vs Without Actions | 77-79 | Why action conditioning changes everything | GIF: `03_action_conditioning.gif`, Plots: `_tmp_action_cond.png`, `_tmp_error_compare.png` |

## Key Formulas

| Formula | Meaning | Cell |
|---------|---------|------|
| `L_TF = (1/(T-1)) * sum_t \|\|f(z_t^GT, a_t) - sg[z_{t+1}^GT]\|\|_1` | Teacher-Forcing Loss: predict next state given TRUE previous state and action | 52 |
| `L_AR = (1/(T-1)) * sum_t \|\|f(z_hat_t, a_t) - sg[z_{t+1}^GT]\|\|_1` | Autoregressive Loss: predict next state from model's OWN previous prediction | 52 |
| `L = alpha * L_TF + (1-alpha) * L_AR` | Combined Loss (alpha=0.5): TF for stable gradients, AR for robustness | 52 |
| `a* = argmin_{a_{1:H}} \|\|f(z_H, a_H) - z_goal\|\|_1` | CEM Planning Objective: find actions that bring predicted state closest to goal | 52 |
| `Frame-causal mask: M[i,j] = 1 if frame(i) >= frame(j)` | Block-lower-triangular attention mask; frame t sees only frames <= t | 7 |
| `Token sequence: [a_1, s_1, p_1...p_N, a_2, s_2, p_{N+1}...p_{2N}, ...]` | Interleaved action/state/patch token ordering per frame | 10 |

## Key Visualizations

| File | Description |
|------|-------------|
| `plots/03_token_interleaving_strip.png` | How action, state, and spatial patch tokens are interleaved in the sequence |
| `plots/03_tf_vs_ar_dataflow.png` | Side-by-side data flow for teacher-forcing vs autoregressive training |
| `gifs/03_cem_planning.gif` | Animated CEM optimization: samples converge from random to optimal action sequence |
| `gifs/03_rollout_error.gif` | How prediction error accumulates over longer autoregressive rollouts |
| `gifs/03_tf_vs_ar.gif` | Side-by-side comparison of teacher-forcing vs autoregressive rollouts with numbers |
| `gifs/03_causal_mask.gif` | Step-by-step construction and effect of the frame-causal attention mask |
| `gifs/03_action_conditioning.gif` | Comparison of predictions with vs without action conditioning |
| `plots/_tmp_action_vec.png` | Labeled 7-DoF action vector with real DROID dataset values |
| `plots/_tmp_trajectory.png` | 3D end-effector trajectory from the DROID dataset |
| `plots/_tmp_chunking.png` | Action chunking: predicting multiple steps at once |
| `plots/_tmp_causal_mask.png` | Block-lower-triangular causal attention mask |
| `plots/_tmp_causal_compare.png` | Comparison of causal vs full attention |
| `plots/_tmp_dashboard_healthy.png` | Training dashboard for a healthy run |
| `plots/_tmp_dashboard_collapse.png` | Training dashboard showing collapse |
| `plots/_tmp_action_cond.png` | Action-conditioned vs unconditioned predictions |
| `plots/_tmp_error_compare.png` | Error accumulation comparison |

## Self-Check Questions

1. What is the difference between a world model that operates in pixel space vs latent space? Why is latent space preferred?
2. Why does V-JEPA 2-AC use BOTH teacher-forcing and autoregressive training? What does each provide?
3. Why is the causal attention mask block-lower-triangular rather than token-level triangular?
4. How does CEM plan without using gradients? What are its steps?
5. Why do action tokens need different positional encoding (RoPE) than spatial patch tokens?
6. What happens to prediction quality as you roll out more steps autoregressively?
7. In the token sequence, what is the ordering of action, state, and patch tokens within a single frame?
8. How does the combined loss (alpha=0.5) prevent train-test distribution mismatch?
9. What is the 7-DoF action representation used in DROID, and what does each dimension control?
10. At 10Hz control, how much time does the robot have to plan each action? Why does CEM's 16-second planning time matter?

## Common Issues

- **Teacher-forcing vs autoregressive confusion**: Teacher-forcing gives the model ground-truth previous states; autoregressive gives it its own predictions. Both use the same model architecture but different inputs.
- **Token interleaving order**: Action and state tokens come BEFORE spatial patch tokens within each frame. The order is [action, state, patches] per frame.
- **CEM is slow**: At 800 candidates x 5 iterations, CEM takes ~16 seconds per action. This is a known limitation motivating research on amortized planning (Paper Idea 3 in Notebook 05).
- **Causal mask is per-frame, not per-token**: All patches within the same frame can attend to each other. The causality is at the frame level.
- **Section numbering**: Some sections appear out of numerical order (e.g., 7.5 before 8, then 7 after 9). Follow cell order.

## Next Steps

- **Notebook 04** (VLA-JEPA Integration): Learn how JEPA connects to Vision-Language-Action models for the ICRA frontier
- **Notebook 06** (Hands-On): Run the real V-JEPA 2-AC code with CPU-compatible versions
