# JEPA Study Guide: From Fundamentals to ICRA Research

A comprehensive, hands-on learning path for understanding **JEPA** (Joint Embedding Predictive Architecture) and its applications in robotics, VLA models, video prediction, and safe humanoid control.

**5 interactive Jupyter notebooks** with 180+ cells, 80+ visualizations (2D/3D/4D/5D), runnable toy models, layer-by-layer analysis, formal math, layman analogies, and complete 5W+1H glossaries for every key term.

---

## What's Inside

### Notebooks

| # | Notebook | Cells | Topics | Key Visualizations |
|---|----------|-------|--------|--------------------|
| 01 | **JEPA Fundamentals** | 52 | JEPA objective, L1 loss, EMA, masking, training loop | Loss landscapes (3D), t-SNE/PCA (2D+3D), ablation heatmaps, 4D/5D hyperparameter analysis, gradient flow, information theory |
| 02 | **V-JEPA 2 Architecture** | 43 | ViT encoder, 3D RoPE, SwiGLU, predictor, attention | Multi-head attention maps (per head per layer), 3D patch embeddings, RoPE frequency spectrum, CKA similarity, token trajectories (2D+3D), 5D architecture search |
| 03 | **V-JEPA 2-AC World Model** | 34 | Action conditioning, causal attention, CEM planning | ACBlock trace, step-by-step causal mask, 3D rollout quality, action embedding space (PCA/t-SNE), 4D/5D performance surfaces |
| 04 | **VLA-JEPA Integration** | 28 | JEPA-VLA, VLA-JEPA, flow matching, benchmarks | End-to-end pipeline trace, fusion comparison (4 methods), flow matching vector fields (2D+3D), multi-modal action distributions, 5D scaling laws |
| 05 | **ICRA Research Gaps** | 29 | 4 paper ideas, CBF safety, unified architecture | CBF safety contours + 3D surface, Unified JEPA-VLA implementation, ablation study design, radar charts, research landscape gap analysis |

### What Makes This Different

- **Layer-by-layer analysis**: Trace every tensor through every computation step with shape, mean, std, norm
- **5W+1H for every term**: Who, What, Where, When, Why, How + layman analogy for 30+ terms
- **Multi-dimensional visualizations**: 2D scatter, 3D surface, 4D (color+size), 5D (parallel coordinates)
- **Formal math + intuition**: LaTeX equations alongside plain-English explanations
- **Runnable on CPU**: All notebooks execute on CPU with toy models (no GPU required for learning)
- **ICRA paper ready**: Complete architecture code, ablation study designs, and compute planning

---

## Quick Start

```bash
# Clone
git clone <this-repo>
cd JEPAstudy

# Setup (using uv)
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy matplotlib scikit-learn scipy jupyter nbconvert

# Run notebooks
jupyter notebook notebooks/
```

### Prerequisites
- Python 3.11+
- No GPU required (all examples use CPU-runnable toy models)
- ~2GB disk space

---

## Learning Path

```
Phase 1: JEPA Fundamentals (Notebook 01)
  |- What is JEPA? Why predict in latent space?
  |- Build a toy JEPA from scratch
  |- Understand EMA, stop-gradient, L1 loss
  |- Visualize: loss landscape, representations, ablations

Phase 2: V-JEPA 2 Architecture (Notebook 02)
  |- How does the ViT encoder work?
  |- 3D RoPE for video position encoding
  |- SwiGLU activation, multi-head attention
  |- Trace tensors through every layer

Phase 3: World Models for Robotics (Notebook 03)
  |- Action-conditioned prediction
  |- Teacher-forcing vs autoregressive training
  |- CEM planning in latent space
  |- Causal attention masks

Phase 4: JEPA + VLA Integration (Notebook 04)
  |- JEPA-VLA: feature injection approach
  |- VLA-JEPA: world model approach
  |- Flow matching for action generation
  |- End-to-end pipeline analysis

Phase 5: ICRA Research (Notebook 05)
  |- 4 paper ideas with motivation
  |- Recommended: Unified JEPA-VLA
  |- CBF safety in latent space
  |- Compute planning for your hardware
```

---

## Key Concepts at a Glance

| Concept | One-Liner | Notebook |
|---------|-----------|----------|
| **JEPA** | Predict representations, not pixels | 01 |
| **EMA** | Slowly update target encoder for stable training | 01 |
| **Stop-Gradient** | Prevent representation collapse | 01 |
| **3D RoPE** | Encode video position through rotation | 02 |
| **SwiGLU** | Gated activation function for better FFNs | 02 |
| **World Model** | Predict future states given actions | 03 |
| **Teacher-Forcing** | Train with ground-truth inputs | 03 |
| **CEM Planning** | Find best actions by sampling + refining | 03 |
| **Flow Matching** | Generate actions by denoising noise | 04 |
| **VLA** | Vision + Language + Action for robots | 04 |
| **CBF** | Mathematical safety guarantee | 05 |

---

## Visualization Gallery

The notebooks contain 80+ visualizations across multiple dimensions:

**2D**: Attention heatmaps, loss curves, t-SNE scatter, cosine similarity matrices, contour plots
**3D**: Loss landscape surfaces, patch embeddings, token trajectories, CEM energy landscapes, CBF surfaces, rollout quality
**4D**: 3D scatter with color (4th dim), animated training evolution, faceted heatmap grids
**5D**: Parallel coordinates, 3D scatter with color + size, Pareto frontiers

---

## Reference Code

This repo includes cloned reference implementations:

| Directory | Source | Description |
|-----------|--------|-------------|
| `refs/vjepa2/` | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | V-JEPA 2 encoder, predictor, AC world model |
| `refs/VLA-JEPA/` | [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) | VLA-JEPA open-source implementation |

---

## ICRA Paper Direction

Notebook 05 proposes 4 paper ideas. The recommended direction:

**"Unified JEPA-VLA: Combining Feature Injection and World Models for Robot Manipulation"**

- Combines JEPA-VLA (better features) + VLA-JEPA (planning via world model)
- Complete architecture implementation in Notebook 05
- Ablation study design with expected LIBERO benchmark results
- Feasible on AMD Radeon RX 6700S (8GB VRAM) using V-JEPA 2-Small

---

## Key Papers

| Paper | Year | Link |
|-------|------|------|
| A Path Towards Autonomous Machine Intelligence (LeCun) | 2022 | [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| I-JEPA | 2023 | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| V-JEPA | 2024 | [Meta AI Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) |
| V-JEPA 2 | 2025 | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| JEPA-VLA | 2026 | [arXiv:2602.11832](https://arxiv.org/abs/2602.11832) |
| VLA-JEPA | 2026 | [arXiv:2602.10098](https://arxiv.org/abs/2602.10098) |

---

## Project Structure

```
JEPAstudy/
  README.md                              # This file
  pyproject.toml                         # Python project config
  notebooks/
    01_JEPA_Fundamentals.ipynb           # 52 cells - JEPA from scratch
    02_VJEPA2_Architecture.ipynb         # 43 cells - ViT deep dive
    03_VJEPA2_AC_WorldModel.ipynb        # 34 cells - Action-conditioned world model
    04_VLA_JEPA_Integration.ipynb        # 28 cells - JEPA + VLA integration
    05_ICRA_Research_Gaps.ipynb          # 29 cells - Research gaps & paper ideas
  refs/
    vjepa2/                              # V-JEPA 2 source code
    VLA-JEPA/                            # VLA-JEPA implementation
```

---

## License

This study guide is for educational and research purposes. Reference code in `refs/` retains its original licenses.
