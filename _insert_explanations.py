#!/usr/bin/env python3
"""Insert explanatory markdown cells after code cells in notebook 03."""
import json

with open('notebooks/03_VJEPA2_AC_WorldModel.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n')
    }

# Fix source to be a list of lines with newlines
def make_md_cell(source):
    lines = source.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": result
    }

# We insert AFTER the given original cell index.
# Process from highest index to lowest to avoid shifting.

insertions = {}

# ============================================================
# Cell [4] - Prints frame feature shapes, action shapes, state shapes
# Next cell [5] is about Frame-Causal Attention (new section, not output explanation)
# ============================================================
insertions[4] = r"""### Understanding the Output: Action & State Dimensions

**Printed tensor shapes explained:**

| Tensor | Shape | Formal Notation | Layman Explanation |
|--------|-------|----------------|-------------------|
| `frame_features` | `[2, 8, 256, 1408]` | $\mathbf{Z} \in \mathbb{R}^{B \times T \times (H \cdot W) \times D_{\text{enc}}}$ | 2 videos, each with 8 frames, each frame has 256 spatial patches, each patch is a 1408-dim vector from the frozen ViT-g encoder |
| `actions` | `[2, 7, 7]` | $\mathbf{a} \in \mathbb{R}^{B \times (T-1) \times d_a}$ | 2 videos, 7 action steps (one fewer than frames because the last frame has no "next" action), each action is 7-DoF (xyz translation + rpy rotation + gripper) |
| `states` | `[2, 8, 7]` | $\mathbf{s} \in \mathbb{R}^{B \times T \times d_a}$ | 2 videos, 8 absolute end-effector poses (one per frame), same 7-DoF format |

**Why T-1 actions but T states?**

States describe WHERE the robot is at each frame. Actions describe WHAT the robot does BETWEEN frames. With $T=8$ frames there are only $T-1=7$ transitions:

$$s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} \cdots \xrightarrow{a_6} s_7$$

**Key dimension: 256 = 16 x 16 spatial tokens**

A 256x256 pixel image with patch size 16 yields $\frac{256}{16} \times \frac{256}{16} = 16 \times 16 = 256$ spatial tokens. With 2 conditioning tokens (action + state) per frame, the total sequence length is $T \times (2 + 256) = 8 \times 258 = 2064$ tokens.

**The 7-DoF action vector:**

$$a_t = [\underbrace{\Delta x, \Delta y, \Delta z}_{\text{translation}}, \underbrace{\Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}}_{\text{rotation}}, \underbrace{g}_{\text{gripper}}] \in \mathbb{R}^7$$

*Layman terms:* Each action says "move the arm this much in x/y/z, rotate it this much around each axis, and open/close the gripper by this amount." The values are tiny (~0.01) because at 10 Hz control, each step is only 100ms of motion."""

# ============================================================
# Cell [6] - Causal attention mask plot + printed stats
# Next cell [7] is about Token Interleaving (new section)
# ============================================================
insertions[6] = r"""### Understanding the Output: Frame-Causal Attention Mask

**The plot -- how to read it:**

- **Axes:** Both axes represent token position in the flattened sequence (0 to $N-1$ where $N = T \times N_T$)
- **Blue squares:** Position $(i, j)$ is blue means token $i$ (query) CAN attend to token $j$ (key)
- **White squares:** Attention is blocked -- the attention score is set to $-\infty$ before softmax
- **Red grid lines:** Frame boundaries. Each block between red lines is one frame's worth of tokens

**The block lower-triangular pattern:**

$$\text{Frame } t \text{ tokens can attend to all tokens in frames } 0, 1, \ldots, t$$

This is a BLOCK version of the standard causal mask used in GPT-style models. The difference: in GPT, each token can see all previous tokens individually. Here, entire FRAMES are the causal unit -- all tokens within a frame can see each other freely (the diagonal blocks are fully blue).

**Printed numbers explained:**

| Output | Value | Meaning |
|--------|-------|---------|
| `Tokens per frame: 6` | $N_T = 2 + H \times W = 2 + 2 \times 2 = 6$ | 2 conditioning tokens (action + state) plus 4 spatial patches |
| `Frame 0 sees: only itself` | Row 0 block is blue, rest is white | First frame has no past context |
| `Frame 3 sees: frames 0-3` | Row 3 is fully blue across all blocks | Last frame sees entire history |
| `Real V-JEPA 2-AC: 2064 tokens` | $8 \times (2 + 256) = 2064$ | The actual sequence length in V-JEPA 2 |

**Formal mask definition:**

$$M[i, j] = \begin{cases} 1 & \text{if } \lfloor i / N_T \rfloor \geq \lfloor j / N_T \rfloor \\ 0 & \text{otherwise} \end{cases}$$

*Layman terms:* "Each frame can peek at everything that happened before (and itself), but the future is hidden." This prevents the model from cheating by copying future frames during training."""

# ============================================================
# Cell [8] - Token interleaving demo
# Next cell [9] is about ACRoPEAttention (new section)
# ============================================================
insertions[8] = r"""### Understanding the Output: Token Interleaving

**Printed shapes explained:**

| Stage | Shape | What Changed |
|-------|-------|-------------|
| Before: `frame_features` | `[1, 16, 8]` = $(B, T \times HW, D)$ | 4 frames x 4 patches = 16 spatial tokens, each 8-dim |
| Before: `actions` | `[1, 4, 7]` = $(B, T, d_a)$ | 4 action vectors (one per frame), each 7-DoF |
| Before: `states` | `[1, 4, 7]` = $(B, T, d_a)$ | 4 state vectors (one per frame), each 7-DoF |
| After: `x_flat` | `[1, 24, 8]` = $(B, T \times (2 + HW), D)$ | 4 frames x 6 tokens/frame = 24 tokens total |

**The interleaving order (critical for understanding the model):**

```
Frame 0: [action_0, state_0, patch_0_0, patch_0_1, patch_0_2, patch_0_3]  positions [0..5]
Frame 1: [action_1, state_1, patch_1_0, patch_1_1, patch_1_2, patch_1_3]  positions [6..11]
Frame 2: [action_2, state_2, patch_2_0, patch_2_1, patch_2_2, patch_2_3]  positions [12..17]
Frame 3: [action_3, state_3, patch_3_0, patch_3_1, patch_3_2, patch_3_3]  positions [18..23]
```

**Why action and state come FIRST within each frame:**

The action and state tokens are prepended so that when the causal mask allows frame $t$ to attend to frame $t-1$, the spatial patches of frame $t$ can attend to the action/state of the PREVIOUS frame. This lets the model learn "given action $a_{t-1}$ and state $s_{t-1}$, what do the patches at time $t$ look like?"

**Formal interleaving operation:**

$$\mathbf{x}_{\text{interleaved}} = \text{concat}\Big(\underbrace{W_a \cdot a_t}_{\text{action token}},\; \underbrace{W_s \cdot s_t}_{\text{state token}},\; \underbrace{x_{t,1}, \ldots, x_{t,HW}}_{\text{spatial patches}}\Big) \quad \text{for each frame } t$$

$$\text{Then flatten: } \mathbf{x}_{\text{flat}} = [\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{T-1}] \in \mathbb{R}^{T \cdot (2+HW) \times D}$$

*Layman terms:* For each video frame, we create a sequence that starts with "what action was taken" and "where the robot was," followed by all the visual patches. Then we concatenate all frames into one long sequence for the transformer."""

# ============================================================
# Cell [10] - ACRoPEAttention demo
# Next cell [11] is the TF vs AR math (a different topic)
# ============================================================
insertions[10] = r"""### Understanding the Output: Split RoPE for Action vs Spatial Tokens

**Printed position encoding assignments:**

For each token, the output shows which RoPE coordinates it receives:

| Token Type | Temporal (depth) | Spatial (height, width) | Why |
|-----------|-----------------|------------------------|-----|
| Action token | depth = $t$ (frame index) | N/A | Actions have no spatial position -- they describe the whole-frame action |
| State token | depth = $t$ (frame index) | N/A | States are whole-frame properties (end-effector pose) |
| Spatial patch $[h, w]$ | depth = $t$ (frame index) | height = $h$, width = $w$ | Patches have both temporal AND spatial positions |

**Formal RoPE application:**

For spatial tokens with 3D position $(t, h, w)$:
$$\text{RoPE}(q) = q \odot \cos(\theta_{t,h,w}) + \text{rotate}(q) \odot \sin(\theta_{t,h,w})$$

where $\theta_{t,h,w}$ encodes all three coordinates across different frequency bands.

For action/state tokens with 1D position $(t)$:
$$\text{RoPE}(q) = q \odot \cos(\theta_t) + \text{rotate}(q) \odot \sin(\theta_t)$$

**Why the split matters:**

If action tokens received spatial RoPE with arbitrary $(h, w)$ values, the attention mechanism would incorrectly treat them as belonging to a specific spatial location. By giving them ONLY temporal RoPE, the model correctly learns that actions are spatially global (they affect the entire frame) but temporally local (action at time $t$ is different from action at time $t+1$).

*Layman terms:* Visual patches know WHEN and WHERE they are in the video. Action tokens only know WHEN they happen -- they affect the whole image, not just one patch."""

# ============================================================
# Cell [13] - Rollout error plot + printed explanation
# Next cell [14] is CODE (no markdown!)
# ============================================================
insertions[13] = r"""### Understanding the Output: Rollout Error Accumulation

**The plot shows two curves over a 10-step prediction horizon:**

| Curve | Color | What It Represents | Error Growth |
|-------|-------|-------------------|-------------|
| **Teacher-Forcing** | Blue | Error when each prediction step uses ground-truth input | Flat / bounded: $\epsilon_t^{\text{TF}} \sim O(1)$ |
| **Autoregressive** | Orange/Red | Error when each step uses the model's OWN previous prediction | Growing: $\epsilon_t^{\text{AR}} \sim O(\sqrt{t})$ or worse |

**What to look for:**
- The TF curve should remain roughly flat -- errors at each step are independent
- The AR curve should grow over time -- errors compound because each prediction's mistake feeds into the next
- The GAP between them illustrates why AR training is necessary

**Printed explanation decoded:**

- "Without AR training: model is trained on perfect inputs but tested on noisy ones" -- This is the **train-test mismatch** problem. At training time with TF, the model always sees ground truth. At inference time, it sees its own (imperfect) predictions.
- "The mismatch causes errors to compound exponentially" -- Without AR training, the model has never learned to correct its own mistakes
- "AR training teaches the model to correct its own mistakes" -- By including AR rollout during training, the model learns to produce stable predictions even from imperfect inputs

**Error accumulation formula:**

$$\epsilon_t^{\text{AR}} = \|P_\phi(\hat{z}_{t-1}, a_{t-1}) - z_t^{\text{GT}}\|_1 \quad \text{where } \hat{z}_{t-1} = P_\phi(\hat{z}_{t-2}, a_{t-2})$$

The key insight: $\hat{z}_{t-1}$ itself has error, so the prediction at step $t$ starts from a noisy input, amplifying the noise.

*Layman terms:* Imagine giving directions by saying "turn left after the red house." If someone misidentifies the red house, every subsequent direction is wrong. AR training is like practicing giving directions to someone who might mishear you."""

# ============================================================
# Cell [14] - Training loop demo
# Next cell [15] is CODE (CEM visualization, no markdown!)
# ============================================================
insertions[14] = r"""### Understanding the Output: V-JEPA 2-AC Training Losses

**Printed loss values explained:**

| Loss | Value (~1.14) | Formula | What It Measures |
|------|--------------|---------|-----------------|
| $\mathcal{L}_{\text{TF}}$ | `L_tf: 1.1368` | $\frac{1}{T-1}\sum_{t=1}^{T-1}\frac{1}{HW \cdot D}\sum_{n,d}\|\hat{z}_t^{(n,d)} - z_t^{(n,d)}\|_1$ | Average L1 distance between predicted and target latents, using ground-truth inputs at each step |
| $\mathcal{L}_{\text{AR}}$ | `L_ar: 1.1401` | Same formula but $\hat{z}_t$ is computed from the model's own previous prediction, not ground truth | Same metric but with the model feeding its own outputs back as input |
| $\mathcal{L}_{\text{total}}$ | `total: ~1.14` | $\alpha \cdot \mathcal{L}_{\text{TF}} + (1 - \alpha) \cdot \mathcal{L}_{\text{AR}}$ with $\alpha = 0.5$ | Weighted combination that balances single-step accuracy (TF) with multi-step stability (AR) |

**Why TF and AR losses are similar here (~1.14):**

This is a randomly initialized model, so both losses are essentially measuring random predictions. In real training, you would see $\mathcal{L}_{\text{TF}} < \mathcal{L}_{\text{AR}}$ because autoregressive rollout compounds errors.

**Key parameter: `auto_steps = 2`**

From the V-JEPA 2-AC config: the AR loss only rolls out 2 steps into the future (not the full $T-1$ steps). This is a tradeoff:
- More AR steps = better long-horizon robustness but slower training
- Fewer AR steps = faster training but less robust rollouts at inference

**Dimensions in the training loop:**

| Quantity | Shape | Meaning |
|----------|-------|---------|
| Target representations $h$ | $(B, T \times HW, D) = (2, 128, 64)$ | Frozen encoder outputs for all frames |
| Predictions $\hat{z}$ | $(B, T \times HW, D) = (2, 128, 64)$ | Predictor outputs (same shape as targets) |
| Actions | $(B, T-1, 7) = (2, 7, 7)$ | Robot actions between frames |

*Layman terms:* The model is graded on two tests simultaneously -- one where it gets the answer key after each question (TF), and one where it has to rely on its own previous answers (AR). Both grades are averaged for the final score."""

# ============================================================
# Cell [15] - CEM energy landscape plot
# Next cell [16] explains CEM in general, not this specific plot
# ============================================================
insertions[15] = r"""### Understanding the Output: CEM Energy Landscape

**The 2x3 grid shows 5 CEM iterations (the 6th panel is the summary):**

Each panel displays a 2D slice of the action space with:
- **Background contours:** The energy function $\mathcal{E}(a_1, a_2)$ -- darker blue = lower energy = better action
- **Scattered dots:** Sampled candidate action sequences ($N = 200$)
- **Highlighted dots:** Elite candidates (top $K = 20$ lowest-energy samples)
- **Ellipse/distribution:** The current Gaussian $\mathcal{N}(\mu, \text{diag}(\sigma^2))$

**What to look for across iterations:**

| Iteration | Expected Behavior | If Something Is Wrong |
|-----------|-------------------|----------------------|
| 0 | Dots spread uniformly; ellipse covers most of the space | -- |
| 1-2 | Dots cluster around low-energy regions; ellipse shrinks | Dots stuck in local minimum = energy landscape is deceptive |
| 3-4 | Tight cluster near global minimum; ellipse is small | Ellipse not shrinking = too few elites or too much noise |

**The energy function used here:**

$$\mathcal{E}(a_1, a_2) = 0.5\left[(a_1 - 0.3)^2 + (a_2 - 0.4)^2\right] + 0.3\sin(5a_1)\cos(5a_2) + 0.2\exp\!\left(-\frac{(a_1+0.5)^2 + (a_2-0.3)^2}{0.1}\right)$$

This has multiple local minima (the sinusoidal term) and one global minimum near $(0.3, 0.4)$ (the quadratic term).

*Layman terms:* Imagine searching for the lowest point in a hilly landscape while blindfolded. You drop 200 balls, check which 20 rolled to the lowest spots, then drop 200 more balls near those spots. After 5 rounds, you have a very good estimate of the lowest point.

**Why this matters for V-JEPA 2-AC:** At inference time, the "energy landscape" is $\|\text{WorldModel}(z_t, a_{1:T}) - z_{\text{goal}}\|_1$ -- the distance between the predicted future and the goal. CEM finds the action sequence that gets closest to the goal."""

# ============================================================
# Cell [17] - CEM planning demo with printed iterations
# Next cell [18] is new section (ACBlock deep dive)
# ============================================================
insertions[17] = r"""### Understanding the Output: CEM Planning in Latent Space

**Printed iteration log explained:**

| Line | Meaning |
|------|---------|
| `Iter 0: mean energy = 0.9924, best = 0.9922` | First round: 800 random action sequences evaluated. Average energy is high (far from goal). Best candidate is only slightly better than average |
| `Iter 1-4: decreasing energies` | Each iteration, the distribution narrows around better candidates. Energy decreases as actions improve |
| `Best action sequence found` | The 5-step action sequence $[a_0, a_1, \ldots, a_4]$ with the lowest energy after all CEM iterations |

**Why energies are all ~0.99 and barely decrease:**

This uses a randomly initialized "world model" (not a trained one), so the predictions are essentially random regardless of actions. In a real V-JEPA 2-AC system, you would see energy drop from ~1.0 to ~0.1 as CEM finds actions that bring the predicted state close to the goal.

**CEM planning parameters (from the V-JEPA 2 paper):**

| Parameter | Value | Role |
|-----------|-------|------|
| `n_candidates` | 800 | Number of action sequences sampled per iteration |
| `n_elites` | 80 | Top 10% kept to update the distribution |
| `n_iterations` | 5 | Number of CEM rounds (sample -> evaluate -> update) |
| `planning_horizon` | 5 | How many future steps to plan ahead |
| `action_bound` | 0.075 | $L_1$-ball radius constraining action magnitude (~7.5cm max displacement per step) |

**The CEM objective:**

$$a_{1:T}^* = \arg\min_{a_{1:T}} \mathcal{E}(a_{1:T}) = \arg\min_{a_{1:T}} \|P_\phi(z_{\text{current}}, s_{\text{current}}, a_{1:T}) - z_{\text{goal}}\|_1$$

*Layman terms:* "Try 800 random plans. Keep the 80 best. Generate 800 new plans centered around those 80. Repeat 5 times. Execute the first action of the best plan, then re-plan from scratch."

**Dimensions:**

| Quantity | Shape | Description |
|----------|-------|------------|
| Action sequence | $(T_{\text{plan}}, d_a) = (5, 7)$ | 5 steps of 7-DoF actions |
| CEM distribution $\mu$ | $(5, 7)$ | Mean of the Gaussian over action sequences |
| CEM distribution $\sigma$ | $(5, 7)$ | Std dev of the Gaussian (initialized to 0.02) |
| Candidates | $(800, 5, 7)$ | All sampled action sequences per iteration |"""

# ============================================================
# Cell [19] - ACBlock computation trace
# Next cell [20] is about causal mask (subsection, not output explanation)
# ============================================================
insertions[19] = r"""### Understanding the Output: ACBlock Computation Trace

**The trace shows each computation step inside the Action-Conditioned Block:**

| Step | Name | Shape | What It Does |
|------|------|-------|-------------|
| 1 | `action_proj` | `[2, 4, 64]` = $(B, T \times n_{\text{act}}, D)$ | Projects raw 6D actions to 64D embedding via $W_a \in \mathbb{R}^{6 \times 64}$ |
| 2 | `interleaved` | `[2, 68, 64]` = $(B, T \times (n_{\text{act}} + HW), D)$ | Action tokens inserted at the start of each frame's token sequence |
| 3 | `layernorm1` | `[2, 68, 64]` | Pre-norm: $\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$ applied before attention |
| 4 | `qkv` | `[2, 68, 192]` = $(B, N, 3D)$ | Linear projection producing queries, keys, and values: $[Q, K, V] = x W_{qkv}$ |
| 5 | `q, k, v` | Each `[2, 4, 68, 16]` = $(B, H, N, d_h)$ | Reshaped for multi-head attention: $H=4$ heads, $d_h = D/H = 16$ |
| 6 | `attn_weights` | `[2, 4, 68, 68]` = $(B, H, N, N)$ | Attention scores: $\text{softmax}(QK^T / \sqrt{d_h} + M)$ where $M$ is the causal mask |
| 7 | `attn_output` | `[2, 68, 64]$ | Weighted sum of values: $\text{Attn} \cdot V$, then projected back |
| 8 | `residual1` | `[2, 68, 64]` | First residual connection: $x + \text{Attn}(x)$ |
| 9 | `ffn_output` | `[2, 68, 64]` | Feed-forward network: $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot \text{LN}(x))$ |
| 10 | `residual2` | `[2, 68, 64]` | Second residual connection: $x + \text{FFN}(x)$ -- final output |

**Key dimensions:**

- **Batch:** $B = 2$ (two videos processed in parallel)
- **Sequence length:** $N = 68 = 4 \times (1 + 16)$ = 4 frames, each with 1 action token + 16 spatial patches
- **Embedding dim:** $D = 64$ (simplified; real V-JEPA 2 uses $D_{\text{pred}} = 1024$)
- **Heads:** $H = 4$, head dim $d_h = 16$

**What to look for in `mean` and `std` columns:**

- `mean` near 0.0 after LayerNorm: correct -- LN centers the distribution
- `std` near 1.0 after LayerNorm: correct -- LN normalizes variance
- Attention weights: `mean` should be $\approx 1/N$ (uniform attention) for an untrained model

*Layman terms:* This trace lets you watch data flow through the transformer block step by step, like an X-ray of the neural network. Each row is one computation, and you can verify shapes and statistics are sane at every stage."""

# ============================================================
# Cell [21] - Causal mask step-by-step plot
# Next cell [22] is about attention weights (new subsection)
# ============================================================
insertions[21] = r"""### Understanding the Output: Step-by-Step Mask Construction

**The multi-panel plot shows the mask being built one frame at a time:**

| Panel | What It Shows | Pattern |
|-------|--------------|---------|
| After Frame 0 | Only the top-left $(N_T \times N_T)$ block is filled | One square on the diagonal |
| After Frame 1 | Top-left $2 \times 2$ block of frame-blocks is filled | L-shaped staircase |
| After Frame 2 | Top-left $3 \times 3$ block of frame-blocks is filled | Growing staircase |
| After Frame 3 (final) | Full lower-triangular block structure | Complete causal mask |

**Printed key property:**

- "Frame 0: can only see itself (no future leakage)" -- The first frame's predictions cannot use ANY future information
- "Frame 3: can see all frames (full context)" -- The last frame has access to the complete history
- "Action tokens at each frame boundary carry the action signal" -- The first token(s) of each frame's block are action/state tokens, which is how actions enter the prediction

**Formal construction (as implemented):**

```
for t1 in range(T):          # query frame
    for t2 in range(t1 + 1): # key frame (past + current only)
        mask[t1*N_T : (t1+1)*N_T,  t2*N_T : (t2+1)*N_T] = 1
```

Each iteration adds one more "block row" to the lower-triangular structure.

*Layman terms:* Think of it like a timeline: Frame 0 only knows the present. Frame 1 knows the present and one step back. Each subsequent frame has a longer memory, but none can see the future."""

# ============================================================
# Cell [23] - Attention weight analysis plots
# Next cell [24] is about rollout quality (new subsection)
# ============================================================
insertions[23] = r"""### Understanding the Output: Attention Weight Analysis

**Plot 1 -- Full Attention Maps (top row, one per head):**

- **Axes:** Query position (y-axis) attending to Key position (x-axis)
- **Color (hot colormap):** Bright yellow = high attention weight, dark red/black = low attention weight
- **Cyan grid lines:** Frame boundaries
- **Lower-triangular pattern:** Confirms the causal mask is working -- no attention flows to future frames

*What to look for:*
- Some heads may attend uniformly (diffuse warm colors) -- these are "averaging" heads
- Some heads may show sharp diagonal stripes -- these attend primarily to the same position in past frames
- The pattern varies per head because multi-head attention learns diverse attention strategies

**Plot 2 -- Attention TO Action Tokens (bottom row):**

Shows only the columns corresponding to action token positions. This reveals how much each token in the sequence "listens to" the action signals.

*What to look for:*
- High attention to action tokens from spatial patches = the model is using action information to condition its predictions
- Low attention to action tokens = the model may be ignoring actions (bad for an action-conditioned world model)
- In an untrained model (as shown here), attention is roughly uniform

**Plot 3 -- Bar Chart: Action vs Spatial Attention:**

Aggregates attention weight going to action tokens vs spatial tokens:

$$\text{Action attention fraction} = \frac{\sum_{j \in \text{action positions}} \alpha_{ij}}{\sum_j \alpha_{ij}}$$

*Good trained model:* Action tokens receive a meaningful fraction (5-20%) of total attention, even though they are a tiny minority of tokens ($2 / 258 \approx 0.8\%$). This indicates the model has learned to upweight action information.

*Layman terms:* These plots show WHERE the model is looking. If it ignores the action tokens, it cannot be action-conditioned. If it pays too much attention to them, it might ignore visual details. A balanced pattern is ideal."""

# ============================================================
# Cell [25] - 3D rollout quality visualization
# Next cell [26] is about action embedding (new subsection)
# ============================================================
insertions[25] = r"""### Understanding the Output: 3D Rollout Quality

**The multi-panel figure shows how prediction quality degrades with horizon:**

| Panel | What It Shows | Axes |
|-------|--------------|------|
| **3D trajectory plot** | Ground truth vs predicted trajectories in 3D space | x, y, z spatial coordinates |
| **Error vs horizon** | L1 error as a function of rollout step $t$ | x = horizon step, y = error |
| **Error distribution** | Histogram of errors across multiple runs at different horizons | x = error magnitude, y = count |

**Key relationship -- error growth with horizon:**

$$\mathbb{E}[\epsilon_t] \propto \sigma_{\text{pred}} \cdot t^{0.5}$$

where $\sigma_{\text{pred}}$ is the per-step prediction noise. Error grows as the SQUARE ROOT of the horizon (random walk behavior).

**What to look for:**

- **3D trajectories:** Predicted paths (colored) should start close to ground truth (black) and gradually diverge
- **Error curve:** Should show sub-linear growth (concave up). If it grows linearly or super-linearly, the model is not robust
- **Error distribution:** At short horizons, errors should be tightly clustered near zero. At long horizons, the distribution spreads out

*Good result:* Errors stay below 0.5 for the first 5-10 steps (the typical planning horizon for CEM).
*Bad result:* Errors exceed 1.0 within 3 steps -- the world model is too noisy for meaningful planning.

**Practical implication:** If prediction quality degrades too fast, the planning horizon must be shortened. V-JEPA 2 uses a 5-step planning horizon, which is conservative enough to maintain useful prediction accuracy.

*Layman terms:* It is like weather forecasting: tomorrow's forecast is reliable, next week's is rough, and next month's is basically a guess. The world model has the same limitation -- predictions get worse the further into the future you look."""

# ============================================================
# Cell [27] - Action embedding t-SNE/PCA
# Next cell [28] is about loss decomposition (new subsection)
# ============================================================
insertions[27] = r"""### Understanding the Output: Action Embedding Space

**The plot shows 200 action vectors projected through the ACBlock's action encoder, then visualized with PCA and t-SNE:**

| Visualization | Method | What It Reveals |
|--------------|--------|----------------|
| **PCA** | Linear projection onto top-2 principal components | Global structure: which action types are linearly separable |
| **t-SNE** | Non-linear embedding preserving local neighborhoods | Cluster structure: which actions the model treats as similar |

**Color coding (5 action types):**
- Forward ($a = [1, 0, 0, 0, 0, 0]$), Backward ($a = [-1, 0, 0, 0, 0, 0]$)
- Turn Left ($a = [0, 0, 0, 0, 0, 1]$), Turn Right ($a = [0, 0, 0, 0, 0, -1]$)
- Grasp ($a = [0, 0, -0.5, 0, 1, 0]$)

**What to look for:**

- **Distinct clusters:** Each action type should form its own cluster. This means the embedding layer has learned to separate different motor behaviors
- **Forward vs Backward separation:** These differ only in the sign of one component -- the embedding should reflect this as opposite positions
- **Grasp vs movement separation:** Grasping involves different DOFs than translation, so it should be far from movement clusters

**Printed "Key observations":**

- "Similar actions cluster together" -- The linear projection $W_a \in \mathbb{R}^{7 \times D}$ maps semantically similar actions to nearby points
- "PCA spectrum shows effective dimensionality" -- If 2-3 principal components explain most variance, the action space is low-dimensional despite having 7 DOFs

**Formal embedding:**

$$e_a = \text{LayerNorm}(W_a \cdot a + b_a) \in \mathbb{R}^D$$

*Layman terms:* The model converts a 7-number robot command into a rich 64-dimensional (or 1024-dim in production) internal representation. This plot shows that similar commands end up in similar places in that internal space, which is necessary for the model to generalize across related actions.

**Note:** With a randomly initialized model, the clusters may not be perfectly separated. After training on DROID data, the separation becomes much cleaner."""

# ============================================================
# Cell [29] - Loss decomposition plot
# Next cell [30] is a glossary section
# ============================================================
insertions[29] = r"""### Understanding the Output: Training Loss Decomposition

**The 2x3 figure shows six views of the training dynamics:**

| Panel | What It Shows | What to Look For |
|-------|--------------|-----------------|
| **Loss curves** | TF loss (blue) and AR loss (orange) over 200 epochs | TF should converge faster and lower than AR. Both should decrease smoothly |
| **Loss ratio** | $\mathcal{L}_{\text{AR}} / \mathcal{L}_{\text{TF}}$ over time | Should start > 1.0 (AR is harder) and slowly decrease toward 1.0 as the model improves at rollouts |
| **Gradient contributions** | Estimated gradient magnitude from TF vs AR terms | Shows which loss dominates training at each stage |
| **TF vs AR scatter** | $\mathcal{L}_{\text{TF}}$ vs $\mathcal{L}_{\text{AR}}$ for each epoch | Points should lie above the diagonal (AR >= TF always). Tight cluster = stable training |
| **Loss landscape** | 2D contour plot of total loss as function of TF and AR components | Shows the optimization trajectory through loss space |
| **Training weight schedule** | How the $\alpha$ weight between TF and AR changes over time | May be fixed (0.5) or follow a schedule |

**Key formulas:**

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{TF}} + (1 - \alpha) \cdot \mathcal{L}_{\text{AR}}$$

In V-JEPA 2-AC, $\alpha = 0.5$ (equal weighting). The TF loss converges faster because it always receives clean inputs. The AR loss is noisier and harder because it must learn from its own imperfect predictions.

**Convergence values (simulated):**

| Loss | Initial | Final | Interpretation |
|------|---------|-------|---------------|
| $\mathcal{L}_{\text{TF}}$ | ~0.95 | ~0.15 | Single-step prediction is accurate |
| $\mathcal{L}_{\text{AR}}$ | ~1.50 | ~0.30 | Multi-step rollout has 2x the error of single-step |
| $\mathcal{L}_{\text{total}}$ | ~1.23 | ~0.23 | Weighted average of both |

*Layman terms:* Think of the two losses as two exam scores. The TF exam (open-book) is always easier. The AR exam (closed-book) is harder but more representative of real performance. Training optimizes both simultaneously."""

# ============================================================
# Cell [31] - 4D/5D world model analysis
# Next cell [32] is math deep dive (new section)
# ============================================================
insertions[31] = r"""### Understanding the Output: 4D/5D Performance Analysis

**The multi-panel figure visualizes world model performance across four dimensions:**

1. **Horizon** (x-axis): Prediction steps into the future (1-15)
2. **Noise level** (different curves): Observation noise $\sigma \in \{0.01, 0.05, 0.1, 0.2\}$
3. **Training mode** (different panels/colors): TF-only, TF+AR balanced, TF+AR V-JEPA 2 style
4. **Error** (y-axis): Prediction L1 error

**Key findings from the printed insights:**

| Finding | What It Means |
|---------|--------------|
| "TF+AR training (V-JEPA 2 style) is best at long horizons" | The combined loss makes the model robust to error compounding |
| "CEM needs 100+ samples and 5+ iterations" | Fewer samples or iterations lead to suboptimal action selection |
| "Balanced TF/AR weight (~0.5) gives best results" | Too much TF weight ignores rollout stability; too much AR weight makes training unstable |

**Error growth model:**

$$\epsilon(h, \sigma, \text{mode}) = \sigma \cdot h^{0.5 + 0.3 \cdot f(\text{mode})}$$

where $f(\text{mode}) \in \{1.0, 0.7, 0.5\}$ for TF-only, balanced, and V-JEPA 2 style respectively. Lower exponent = slower error growth = better long-horizon predictions.

*Layman terms:* This is like comparing three students taking a multi-part exam: one who only practiced with answer keys (TF-only), one who practiced both ways equally (balanced), and one trained with V-JEPA 2's strategy. The V-JEPA 2 student handles the later questions best because they learned to work with imperfect information.

**Why this matters for your ICRA paper:** If you propose modifications to the training loss, this type of 4D analysis is how you would demonstrate the improvement -- showing that your method achieves lower error across horizons, noise levels, and training modes."""

# Now insert all cells, working from highest index to lowest
for idx in sorted(insertions.keys(), reverse=True):
    md_cell = make_md_cell(insertions[idx])
    nb['cells'].insert(idx + 1, md_cell)

with open('notebooks/03_VJEPA2_AC_WorldModel.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Inserted {len(insertions)} markdown cells")
print(f"Total cells now: {len(nb['cells'])}")
