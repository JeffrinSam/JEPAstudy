# JEPA Study Guide for Robotics, VLAs, and Safe AI

A beginner-friendly learning path for understanding JEPA (Joint Embedding Predictive Architecture)
and its applications in robotics, video prediction, VLA models, and safe humanoid control.

---

## Learning Path Overview

```
Phase 1: Foundations (weeks 1-3)
  |- Self-supervised learning basics
  |- Contrastive learning vs generative models
  |- Vision Transformers (ViT)
  |
Phase 2: JEPA Core (weeks 3-5)
  |- LeCun's vision paper
  |- I-JEPA (images)
  |- V-JEPA / V-JEPA 2 (video)
  |
Phase 3: Robotics + World Models (weeks 5-8)
  |- World models concept
  |- V-JEPA 2-AC (action-conditioned)
  |- VLA models (RT-2, Octo, OpenVLA, pi0)
  |- JEPA-VLA integration
  |
Phase 4: Safe Control for Humanoids (weeks 8-10)
  |- Control theory basics
  |- Control Barrier Functions (CBFs)
  |- Neural CBFs
  |- SHIELD / CBF-RL for humanoids
  |- Integrating safety with learned representations
```

---

## Phase 1: Foundations

### What you need to know first

**Self-Supervised Learning (SSL):**
SSL trains models on unlabeled data by creating "pretext tasks" -- e.g., masking parts
of an image and predicting what's missing. This is how JEPA learns.

**Three families of SSL (and why JEPA is different):**

| Approach           | How it learns                          | Limitation                        |
|--------------------|----------------------------------------|-----------------------------------|
| Generative         | Reconstruct pixels (MAE, GPT, VAE)    | Wastes capacity on irrelevant details |
| Contrastive        | Pull similar, push different (SimCLR, CLIP) | Needs careful augmentations    |
| **JEPA**           | **Predict in latent space**            | **Newer, still being explored**   |

**Vision Transformers (ViT):**
JEPA uses ViT as its backbone. A ViT splits an image into patches, treats each patch
like a "word", and processes them with a transformer.

### Beginner resources
- 3Blue1Brown: "Neural Networks" series on YouTube
- Andrej Karpathy: "Let's build GPT from scratch" (for transformer intuition)
- Lilian Weng's blog: "Self-Supervised Representation Learning"
  https://lilianweng.github.io/posts/2019-11-10-self-supervised/
- ViT paper (read the intro + figures): https://arxiv.org/abs/2010.11929

---

## Phase 2: JEPA Core

### The Big Idea

JEPA predicts **abstract representations** of missing content, NOT the raw pixels.

```
Traditional (MAE):    Image -> Mask patches -> Predict PIXELS of masked patches
JEPA:                 Image -> Mask patches -> Predict REPRESENTATIONS of masked patches
                                                (in a learned latent space)
```

Why does this matter? Because the real world is full of unpredictable details (exact
texture of grass, precise water ripple pattern). By predicting in latent space, JEPA
can focus on what's *semantically important* and ignore noise.

### Key Papers (read in this order)

1. **LeCun's Vision Paper** (2022) -- Read Sections 1-4 only to start
   "A Path Towards Autonomous Machine Intelligence"
   https://openreview.net/pdf?id=BZ5a1r-kVsf
   - Introduces the 6-module architecture: perception, world model, cost, memory, actor, configurator
   - JEPA is the foundation for the **world model** module

2. **I-JEPA** (2023) -- The image version
   https://arxiv.org/abs/2301.08243
   GitHub: https://github.com/facebookresearch/ijepa
   - Predicts representations of masked image blocks from visible context
   - No pixel reconstruction, no hand-crafted augmentations

3. **V-JEPA** (2024) -- Extended to video
   Blog: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
   GitHub: https://github.com/facebookresearch/jepa

4. **V-JEPA 2** (2025) -- The big one: 1M+ hours of video, zero-shot robotics
   https://arxiv.org/abs/2506.09985
   GitHub: https://github.com/facebookresearch/vjepa2

5. **VL-JEPA** (2025) -- Vision-Language JEPA
   https://arxiv.org/abs/2512.10942

### Hands-on
- **EB-JEPA** (Educational examples): https://github.com/facebookresearch/eb_jepa
  Includes a simple action-conditioned video JEPA for a 2D navigation task.
  **Start here for code.**

---

## Phase 3: Robotics + World Models

### What is a World Model?

A world model predicts "what happens next" given the current state and an action.
Think of it as an internal simulation of the world.

```
State(t) + Action(t) --> World Model --> Predicted State(t+1)
```

JEPA does this in **latent space** (fast, abstract) instead of **pixel space** (slow, detailed).

### V-JEPA 2-AC: JEPA as a Robot World Model

- Pre-trained on internet video (general physics understanding)
- Post-trained on just 62 hours of unlabeled robot video
- Plans by searching action sequences in latent space
- 80% success on pick-and-place vs 15% for Octo (a VLA model)
- 16 seconds per plan vs 4 minutes for pixel-based planners (Cosmos)

### VLA (Vision-Language-Action) Models

VLAs combine a vision encoder + language model + action decoder to create robots
that follow language instructions.

```
"Pick up the red cup" + Camera Image --> VLA --> Robot Joint Commands
```

**Key models (beginner-friendly order):**

| Model     | Size  | Who         | Key Innovation                     | Open Source |
|-----------|-------|-------------|------------------------------------|-------------|
| RT-2      | 55B   | Google      | First VLA, actions as text tokens  | No          |
| Octo      | 93M   | Berkeley    | Lightweight, diffusion actions     | Yes         |
| OpenVLA   | 7B    | Stanford    | Open-source, strong performance    | Yes         |
| pi0       | 3B    | Phys. Intel.| Flow matching, 50Hz actions       | Yes         |

**Repos to explore:**
- OpenVLA: https://github.com/openvla/openvla
- Octo: https://github.com/octo-models/octo
- pi0 (OpenPi): https://github.com/Physical-Intelligence/openpi

### JEPA + VLA Integration (2026 frontier)

Two recent papers show that adding JEPA features to VLAs improves performance:
- **JEPA-VLA**: https://arxiv.org/abs/2602.11832
- **VLA-JEPA**: https://arxiv.org/abs/2602.10098

---

## Phase 4: Safe Control for Humanoids

### Control Barrier Functions (CBFs) -- The Basics

A CBF defines a "safe zone" for a robot. Think of it as an invisible fence:
- The robot can do whatever it wants INSIDE the fence
- If it tries to cross the fence, the CBF **minimally adjusts** the command to keep it safe

```
Desired Action --> [CBF Safety Filter] --> Safe Action
                    (only modifies if unsafe)
```

Mathematically, a CBF h(x) satisfies: h(x) >= 0 means "safe". The filter solves a
small optimization problem (QP) to find the closest safe action to the desired one.

### Why CBFs for Humanoids?

Humanoid robots trained with RL can walk, run, and manipulate -- but RL gives no
formal safety guarantees. A humanoid falling over or colliding with a person is
dangerous. CBFs add a provable safety layer on top of learned policies.

### Key Approaches

**Neural CBFs** (for complex systems where hand-designing CBFs is impossible):
- BarrierNet: Differentiable CBF layer inside neural networks
  https://github.com/Weixy21/BarrierNet
- Policy Neural CBF: https://arxiv.org/abs/2310.15478

**CBFs for Humanoid Robots:**
- **SHIELD** (2025): CBF safety layer for humanoid RL locomotion on Unitree G1
  https://arxiv.org/abs/2505.11494
- **CBF-RL** (2025): Safety filtering during RL training for humanoids
  https://arxiv.org/abs/2510.14959

### The Future: JEPA + VLA + CBF

The frontier is combining all three:
1. **JEPA world model** -- understands physics from video
2. **VLA** -- follows language instructions
3. **CBF safety filter** -- guarantees safe execution

```
"Hand me the cup"
       |
   [VLA Policy] --> desired action
       |                  |
   [JEPA World Model]  [CBF Safety Filter]
   (predicts outcomes)  (ensures safety)
       |                  |
       +---> Safe, Informed Robot Action
```

---

## Recommended Study Order for Beginners

### Week 1-2: Get the intuition
- [ ] Watch 3Blue1Brown neural network series
- [ ] Read Lilian Weng's SSL blog post
- [ ] Watch any YouTube explainer on Vision Transformers
- [ ] Read LeCun's vision paper (Sections 1-4)

### Week 3-4: Understand JEPA
- [ ] Read I-JEPA paper (focus on Figures 1-3 and Section 3)
- [ ] Read V-JEPA blog post from Meta AI
- [ ] Clone and run EB-JEPA examples
- [ ] Read V-JEPA 2 paper

### Week 5-6: World models + robotics
- [ ] Understand V-JEPA 2-AC results
- [ ] Read OpenVLA paper (intro + experiments)
- [ ] Read Octo paper (intro + architecture)
- [ ] Explore pi0/OpenPi repo

### Week 7-8: VLA + JEPA integration
- [ ] Read JEPA-VLA paper
- [ ] Read VLA-JEPA paper
- [ ] Study the JEPA world model studies: https://github.com/facebookresearch/jepa-wms

### Week 9-10: Safe control
- [ ] Learn CBF basics (start with BarrierNet paper intro)
- [ ] Read SHIELD paper
- [ ] Read CBF-RL paper
- [ ] Think about how CBFs could work in JEPA's latent space

---

## Key GitHub Repos

### JEPA
| Repo | Description |
|------|-------------|
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | I-JEPA official |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | V-JEPA official |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | V-JEPA 2 + robotics |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | Educational examples (START HERE) |
| [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | World model studies |

### VLA Models
| Repo | Description |
|------|-------------|
| [openvla/openvla](https://github.com/openvla/openvla) | OpenVLA 7B |
| [octo-models/octo](https://github.com/octo-models/octo) | Octo generalist policy |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | pi0 family |

### Safety / CBF
| Repo | Description |
|------|-------------|
| [Weixy21/BarrierNet](https://github.com/Weixy21/BarrierNet) | Differentiable CBFs |
| [awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning) | Curated paper list |

---

## Glossary

| Term | Meaning |
|------|---------|
| **JEPA** | Joint Embedding Predictive Architecture -- predicts representations, not pixels |
| **I-JEPA** | Image JEPA -- applies JEPA to static images |
| **V-JEPA** | Video JEPA -- applies JEPA to video sequences |
| **VLA** | Vision-Language-Action model -- robot policy conditioned on language + vision |
| **CBF** | Control Barrier Function -- mathematical safety guarantee for control systems |
| **World Model** | Internal model that predicts future states given current state + action |
| **Latent Space** | Compressed, abstract representation space (vs raw pixel space) |
| **ViT** | Vision Transformer -- processes images as sequences of patches |
| **Flow Matching** | Generative method for smooth continuous outputs (used in pi0) |
| **Diffusion** | Iterative denoising method for generating actions/images (used in Octo) |
| **EMA** | Exponential Moving Average -- used to update target encoder in JEPA |
| **QP** | Quadratic Program -- optimization used to enforce CBF safety constraints |
| **SSL** | Self-Supervised Learning -- learning from unlabeled data |
| **RL** | Reinforcement Learning -- learning from rewards/penalties |
