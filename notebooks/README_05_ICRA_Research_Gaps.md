# Notebook 05: ICRA Research Gaps & Paper Ideas

## Overview

This notebook transitions from learning to doing research. It maps the current JEPA-for-robotics landscape, identifies four concrete publishable paper ideas, and recommends one to pursue: the "Unified JEPA-VLA" that combines feature injection with world model planning. You will find detailed paper outlines, ablation study designs, compute/memory profiling for your hardware (AMD RX 6700S), a 16-week research timeline, and practical guidance on writing and submitting to ICRA. This is your research planning notebook.

## Prerequisites

- Complete Notebooks 01-04 (JEPA fundamentals through VLA integration)
- Understanding of both JEPA-VLA and VLA-JEPA approaches
- Familiarity with CEM planning, flow matching, and the LIBERO benchmark suite

## Estimated Time

2-3 hours for reading; revisit throughout your research as a planning reference

## Table of Contents

| Section | Cells | What You'll Learn | Key Outputs |
|---------|-------|-------------------|-------------|
| Title & Overview | 0 | Goals: find a publishable ICRA contribution | ICRA 2026 deadline context |
| Landscape Map | 1 | ASCII diagram of what exists vs what's missing in JEPA for robotics | Landscape overview |
| 0.5 The 5W+H of JEPA for ICRA | 2 | Who publishes at ICRA, acceptance rates, what reviewers look for | 5W+H of ICRA research |
| Key Definitions | 3 | Quick reference for CBF, VLA, world model, etc. | Definition table |
| Paper Idea 1: Safe JEPA-VLA | 4-9 | JEPA + Control Barrier Functions in latent space for safety | CBF demo output, safety filter plots |
| Paper Idea 1 Deep Dive: CBF Math | 5-9 | Full CBF mathematical framework for JEPA latent space | CBF equations, safety filter visualization |
| Paper Idea 2: Unified JEPA-VLA | 10-11 | Combine feature injection + world model (the recommended paper) | Architecture description |
| Paper Idea 3: Amortized Planning | 12 | Replace CEM with a learned planner for real-time control | Planning speed analysis |
| Detailed Paper Structure | 13 | Full 6-page ICRA paper outline for "Unified JEPA-VLA" | Section-by-section outline |
| Paper Idea 4: Sim-to-Real Transfer | 14-16 | Use V-JEPA 2 to bridge the sim-to-real gap | Compute cost analysis |
| Recommended Paper: Idea 2 | 17-20 | Why Unified JEPA-VLA is the best starting point; draft abstract | Plot: `05_paper_ideas_decision_matrix.png`, draft abstract |
| Deep Analysis: CBF in Latent Space (2D & 3D) | 21-23 | Complete CBF safety visualization in JEPA latent space | CBF 2D/3D plots |
| Unified JEPA-VLA Architecture Pseudo-Code | 24-26 | Complete pseudo-code for the recommended architecture | Plot: `05_unified_jepa_vla_arch.png`, architecture output |
| Ablation Study Design | 27-29 | What to measure: WM ablation, language gating, pretraining, encoder sharing | Ablation study plots |
| Compute & Memory Profiling | 30-32 | Detailed VRAM/compute breakdown for AMD RX 6700S 8GB | Compute profiling plots |
| Compute Planning | 33 | Hardware sufficiency table with alternatives | Hardware table |
| Implementation Roadmap | 34 | 4-phase plan: reproduce baselines, implement, experiment, write | Phase checklist |
| Key References | 35 | Must-cite papers for your ICRA submission | Reference table |
| Quick-Start: VLA-JEPA Code | 36 | How to get the VLA-JEPA code running from refs/VLA-JEPA/ | Setup commands |
| Summary: Path to ICRA Paper | 37 | Visual timeline from now to submission | Timeline diagram |
| 5W+1H Glossary: Research & Safety Terms | 38-40 | CBF, ICRA, ablation, and other research terms | Research gap analysis plots |
| Animated GIF: CBF Safety Filter | 41-43 | Real-time trajectory modification by CBF | GIF: `05_cbf_safety.gif` |
| How to Write an ICRA Paper | 44 | Section-by-section guide with reviewer expectations | Writing guide |
| How CBFs Work in Latent Space | 45 | Full math: standard CBF to JEPA latent CBF | CBF derivation |
| Reproducible Experiments for ICRA | 46-48 | Statistical rigor: seeds, significance testing, ablation design | Experiment design visualization |
| Self-Assessment: Research Planning | 49 | Questions on paper selection, ablations, CBF, timeline | Q&A with hidden answers |
| Control Barrier Functions: Safety Made Simple | 50-52 | Intuitive CBF explanation with driving analogy | GIF: `05_cbf_navigation.gif` |
| CBF in Latent Space: Why Revolutionary | 53-55 | Latent-space CBF vs physical-space CBF | Plots: latent CBF, QP correction vectors |
| Statistical Rigor for ICRA | 56-58 | Power analysis, confidence intervals, minimum seeds | Plots: experiment bars, radar chart, statistical power |
| 16-Week Research Timeline | 59-61 | Week-by-week plan with compute estimates | Plots: Gantt chart, compute plan, VRAM estimation |
| ICRA Paper Checklist | 62 | Pre-submission checklist: structure, experiments, formatting | Checkbox list |

## Key Formulas

| Formula | Meaning | Cell |
|---------|---------|------|
| `h(x) >= 0 defines safe set S` | CBF safety condition: positive barrier value means safe | 5 |
| `dh/dt + alpha*h(x) >= 0` | CBF constraint: barrier must not decrease too fast (alpha > 0) | 5 |
| `h_latent(z) = h(Dec(z))` | Latent-space CBF: apply CBF to decoded latent state | 45 |
| `u* = argmin \|\|u - u_nom\|\| s.t. dh/dt + alpha*h >= 0` | CBF-QP: find closest safe action to the nominal (desired) action | 5 |
| `L_unified = L_vla + beta*L_worldmodel` | Unified JEPA-VLA combined loss: VLA policy loss + world model prediction loss | 24 |

## Key Visualizations

| File | Description |
|------|-------------|
| `plots/05_unified_jepa_vla_arch.png` | Complete architecture diagram for the Unified JEPA-VLA paper idea |
| `plots/05_paper_ideas_decision_matrix.png` | Decision matrix comparing all 4 paper ideas on novelty, feasibility, impact |
| `gifs/05_cbf_safety.gif` | Animated CBF safety filter modifying robot trajectory to avoid obstacles |
| `gifs/05_cbf_navigation.gif` | CBF-guided navigation avoiding obstacles in 2D space |
| `gifs/05_latent_cbf.png` | Latent space with CBF safety regions visualized |
| `gifs/05_cbf_qp_correction.png` | QP correction vectors showing how planned actions are modified for safety |
| `gifs/05_cbf_comparison.png` | Comparison of different CBF approaches |
| `gifs/05_experiment_bars.png` | Ablation study bar charts for experiment design |
| `gifs/05_radar_chart.png` | Multi-dimensional comparison of approaches |
| `gifs/05_statistical_power.png` | Statistical power analysis for experiment planning |
| `gifs/05_gantt_chart.png` | 16-week Gantt chart for research timeline |
| `gifs/05_compute_plan.png` | Compute budget allocation across research phases |
| `gifs/05_vram_estimation.png` | VRAM usage estimates for different model sizes on your GPU |

## Self-Check Questions

1. Why is the "Unified JEPA-VLA" paper idea recommended over the others? List at least three reasons.
2. What ablation studies would a reviewer expect in your ICRA paper? Name at least four.
3. How does a latent-space CBF differ from a physical-space CBF? What are the tradeoffs?
4. Why is CEM planning too slow for real-time humanoid control, and what is the proposed solution (Paper Idea 3)?
5. What is the minimum number of random seeds you should use for ICRA experiments? Why?
6. On your AMD RX 6700S 8GB, which V-JEPA 2 model size can you realistically run? What is the alternative?
7. What are the four LIBERO benchmark suites, and what does each test?
8. In the Unified JEPA-VLA, what does the world model contribute that simple feature injection does not?

## Common Issues

- **Hardware limitations**: The AMD RX 6700S (8GB) cannot run ViT-Giant. Plan to use ViT-Large or ViT-Small, or use CPU inference for prototyping.
- **Paper scope**: ICRA papers are 6+2 pages. Do not try to implement all four ideas -- pick one and do it well.
- **CBF guarantees in latent space**: Formal safety guarantees from physical-space CBFs do not automatically transfer to latent space. This is an open theoretical question that should be acknowledged.
- **Baseline reproduction**: Budget 4-6 weeks to reproduce baselines before implementing your contribution.
- **Some plots saved to gifs/ directory**: Several static PNG plots in this notebook are saved to the `gifs/` directory rather than `plots/` due to the code structure.

## Next Steps

- **Notebook 06** (Hands-On): Work directly with the V-JEPA 2-AC codebase and prototype modifications for your paper
- **Begin Phase 1**: Set up LIBERO environment, get VLA-JEPA code running from `refs/VLA-JEPA/`
- **Draft your paper abstract**: Use the template in cells 17-20 as a starting point
