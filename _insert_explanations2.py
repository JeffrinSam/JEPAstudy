#!/usr/bin/env python3
"""Insert explanatory markdown cells after remaining code cells in notebook 03."""
import json

with open('notebooks/03_VJEPA2_AC_WorldModel.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

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

insertions = {}

# Cell [50] - CEM planning GIF
insertions[50] = """### Reading the CEM Planning GIF

**What the GIF shows (10 frames = 10 CEM iterations):**

- **Contour lines:** Energy landscape $\\mathcal{E}(a_1, a_2)$ -- darker/lower = better action
- **Red circle:** Obstacle the robot must avoid
- **Green star:** Target position (goal)
- **Blue dots:** Sampled action candidates at this iteration
- **Yellow dots:** Elite candidates (top 20% with lowest energy)
- **Red ellipse:** Current Gaussian distribution $\\mathcal{N}(\\mu, \\text{diag}(\\sigma^2))$

**What to look for across frames:**
- **Early iterations (1-3):** Candidates spread widely; ellipse is large. The optimizer is exploring
- **Middle iterations (4-6):** Ellipse shrinks and shifts toward the target. Elites cluster near good actions
- **Late iterations (7-10):** Ellipse collapses onto the optimal action. Candidates are tightly clustered

*Good CEM run:* The ellipse converges to the target while avoiding the obstacle.
*Bad CEM run:* The ellipse gets stuck on a local minimum (near the obstacle) or never shrinks.

**CEM update equations:**

$$\\mu_{i+1} = \\frac{1}{K}\\sum_{k=1}^{K} a_k^{\\text{elite}}, \\quad \\sigma_{i+1}^2 = \\frac{1}{K}\\sum_{k=1}^{K}(a_k^{\\text{elite}} - \\mu_{i+1})^2$$

where $K$ = number of elites, $a_k^{\\text{elite}}$ = the $k$-th lowest-energy candidate.

*Layman terms:* CEM is like a search party. First everyone spreads out. Then the people who found the best spots report back, and the next search focuses around those spots. After a few rounds, everyone converges on the best location."""

# Cell [52] - Rollout degradation GIF
insertions[52] = """### Reading the Autoregressive Rollout Degradation GIF

**What to look for frame by frame:**

- **Green trajectory:** Ground truth (the actual path the robot arm follows)
- **Red trajectory:** Autoregressive prediction (the world model's rollout using its own predictions)
- **Growing gap:** The shaded area between green and red showing accumulated error

**Frame-by-frame progression:**
- **Frames 1-5:** Red closely tracks green. Prediction error is small ($\\epsilon \\approx 0.15$)
- **Frames 6-10:** Red starts diverging. Error grows as $\\sim O(\\sqrt{t})$ due to noise compounding
- **Frames 11-20:** Red may diverge significantly. This is WHY planning horizons are kept short (5-10 steps)

**The error accumulation model:**

$$\\epsilon_t = \\epsilon_{t-1} + \\delta_t, \\quad \\delta_t \\sim \\mathcal{N}(0, \\sigma_{\\text{pred}}^2)$$

$$\\mathbb{E}[\\epsilon_t^2] = t \\cdot \\sigma_{\\text{pred}}^2 \\implies \\text{RMS error} \\propto \\sqrt{t}$$

*Layman terms:* Each prediction adds a small random error. Over many steps, these errors accumulate like a random walk -- the prediction drifts further and further from reality.

**Why this matters:** This degradation is the fundamental limitation of autoregressive world models. It directly motivates: (1) short planning horizons in CEM, (2) the "predict $K$, execute 1, replan" strategy, and (3) the AR training loss that teaches the model to be robust to its own errors."""

# Cell [54] - Robot decision loop visualization
insertions[54] = """### Reading the Robot Decision Loop Output

**The 2x3 figure shows the complete inference pipeline (6 steps):**

| Panel | Step | What It Shows | Key Dimension |
|-------|------|--------------|---------------|
| **Top-left** | OBSERVE | Camera image encoded by frozen V-JEPA 2 encoder into $z_t$ | $(3, 256, 256) \\to z_t \\in \\mathbb{R}^{T \\times HW \\times D}$ |
| **Top-center** | CEM SAMPLE | 500 candidate action sequences from $\\mathcal{N}(\\mu, \\sigma^2)$ | Each: $(T_{\\text{plan}}, 7)$ |
| **Top-right** | ROLLOUT | World model predicts future states for each candidate | $z_{t+k} = P_\\phi(z_t, a_{1:k})$ |
| **Bottom-left** | SCORE | Energy for each candidate: $\\mathcal{E} = \\|z_{t+T} - z_{\\text{goal}}\\|_1$ | Scalar per candidate |
| **Bottom-center** | SELECT | Top-$K$ elites update $\\mu, \\sigma$ | $K = 80$ of 800 candidates |
| **Bottom-right** | EXECUTE | First action $a_1^*$ sent to robot | $a_1^* \\in \\mathbb{R}^7$ |

*This entire loop runs at 10 Hz (100ms per cycle).* The world model rollout is the bottleneck.

**The complete planning objective:**

$$a_{1:T}^* = \\arg\\min_{a_{1:T}} \\|P_\\phi(z_t, s_t, a_{1:T}) - z_{\\text{goal}}\\|_1$$

Only $a_1^*$ (the first action) is executed. Then the robot re-observes, re-encodes, and re-plans. This "model-predictive control" (MPC) loop is robust because it constantly corrects for prediction errors.

*Layman terms:* The robot looks at the scene, imagines 500 possible action sequences, mentally simulates each one, picks the best, takes ONE step, then replans from scratch."""

# Cell [56] - TF vs AR GIF
insertions[56] = """### Reading the Teacher-Forcing vs Autoregressive GIF

**What the GIF shows at each step $t$ (1 through 20):**

| Trajectory | Color | Input at Step $t$ | Error Pattern |
|-----------|-------|-------------------|---------------|
| **Teacher-Forcing** | Blue | Ground-truth state $z_t^{\\text{GT}}$ | Bounded: $\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)$, independent per step |
| **Autoregressive** | Red | Own previous prediction $\\hat{z}_{t-1}$ | Cumulative: $\\epsilon_t \\approx \\epsilon_{t-1} + \\delta_t$ (random walk) |

**What to look for:**
- **Early frames (t < 5):** Both trajectories track ground truth (green) closely
- **Middle frames (5 < t < 15):** AR (red) drifts; TF (blue) stays accurate
- **Late frames (t > 15):** AR may diverge significantly; TF still tracks

**Error formulas:**

$$\\epsilon_t^{\\text{TF}} = \\|P_\\phi(z_t^{\\text{GT}}, a_t) - z_{t+1}^{\\text{GT}}\\|_1 \\quad \\text{(bounded, i.i.d.)}$$

$$\\epsilon_t^{\\text{AR}} \\leq \\sum_{k=1}^{t} \\|\\delta_k\\| \\quad \\text{(cumulative, grows as } O(\\sqrt{t})\\text{)}$$

*Layman terms:* Teacher-forcing is like a student who gets the correct answer after each exam question. Autoregressive is like a student who must use their own (possibly wrong) answer as input to the next question -- mistakes snowball.

**Why both are needed in training:** TF provides stable gradients for learning accurate single-step predictions. AR prevents the train-test mismatch by forcing the model to practice with its own imperfect outputs."""

# Cell [60] - 7-DoF action space
insertions[60] = """### Reading the 7-DoF Action Space Output

**Plot 1 -- Raw vs Normalized Action Vector (bar chart):**

| Component | Physical Meaning | Typical Range | Units |
|-----------|-----------------|---------------|-------|
| $a_0$ (x) | End-effector translation along x | [-0.05, 0.05] | meters |
| $a_1$ (y) | End-effector translation along y | [-0.05, 0.05] | meters |
| $a_2$ (z) | End-effector translation along z | [-0.05, 0.05] | meters |
| $a_3$ (roll) | Rotation around x-axis | [-0.15, 0.15] | radians |
| $a_4$ (pitch) | Rotation around y-axis | [-0.15, 0.15] | radians |
| $a_5$ (yaw) | Rotation around z-axis | [-0.15, 0.15] | radians |
| $a_6$ (gripper) | Open (0) / closed (1) | [0, 1] | scalar |

- **Left bars (raw):** Physical units as recorded by the robot at 10 Hz
- **Right bars (normalized):** Scaled to $[-1, 1]$ via $a_{\\text{norm}} = 2 \\cdot \\frac{a - a_{\\min}}{a_{\\max} - a_{\\min}} - 1$

*What to look for:* Raw values are tiny (millimeters per 100ms timestep) but normalized values fill $[-1, 1]$, which is what the neural network sees.

**Plot 2 -- Action Embedding Projection:**

Shows how $W_a \\in \\mathbb{R}^{7 \\times D_{\\text{pred}}}$ projects 7D actions into predictor embedding space.

**Plot 3 -- Action Chunking:**

Shows the "predict $K$, execute 1, replan" paradigm. V-JEPA 2-AC predicts multiple future actions but only executes the first, then replans for temporal smoothness and reactivity.

**Why this matters:** The 7-DoF action space is the interface between neural network and physical robot. Understanding normalization and embedding is essential for debugging action prediction."""

# Cell [62] - Causal attention deep dive
insertions[62] = """### Reading the Causal Attention Deep Dive Output

**Plot 1 -- Frame-Causal Mask Heatmap:**

- **Axes:** Token positions (0 to 23 for 24 tokens = 4 frames x 6 tokens/frame)
- **Blue = allowed attention, White = blocked**
- **Red lines = frame boundaries**

The mask has a **block lower-triangular** structure:

$$M[i, j] = \\begin{cases} 1 & \\text{if } \\lfloor i / N_T \\rfloor \\geq \\lfloor j / N_T \\rfloor \\\\ 0 & \\text{otherwise} \\end{cases}$$

**Printed numbers:**
- `Blocked entries: 216 / 576` -- 37.5% of attention pairs are masked (the future)
- `Allowed entries: 360 / 576` -- 62.5% are allowed (past + present)

**Plot 2 -- Causal vs Non-Causal Training Comparison:**

- **With causal mask:** Higher test error but the model GENUINELY PREDICTS
- **Without causal mask:** Lower training error but the model COPIES future frames (information leakage). At test time, future frames are unavailable, so it fails

*Key insight:* A non-causal model appears to train better but generalizes worse -- a classic case of "cheating during training."

**GIF -- Mask Construction Animation:**

Watch the mask build up frame-by-frame. Each new frame opens attention to all previous frames while keeping future frames hidden.

*Layman terms:* The causal mask is like taking an exam where each question can only reference previous questions, never future ones. This forces the model to learn real prediction, not just copying."""

# Cell [64] - Training monitoring dashboard
insertions[64] = """### Reading the Training Dashboard Output

**Scenario 1 -- Healthy Training (6 subplots):**

| Subplot | Metric | Healthy Range | Formula |
|---------|--------|--------------|---------|
| TF Loss | $\\mathcal{L}_{\\text{TF}}$ | Decreasing, final < 0.15 | $\\frac{1}{T}\\sum_t \\|\\hat{z}_t - z_t\\|_1$ |
| AR Loss | $\\mathcal{L}_{\\text{AR}}$ | Higher than TF, decreasing | Same L1 but with AR rollout inputs |
| Gradient Norm | $\\|\\nabla_\\theta \\mathcal{L}\\|_2$ | [0.01, 10.0] | L2 norm across all parameters |
| Cosine Similarity | $\\cos(\\hat{z}_t, z_t)$ | Rising 0.3 to 0.9 | Semantic alignment measure |
| Rollout Error | Cumulative L1 over AR steps | Sub-linear growth, < 0.5 | Error at horizon $h$ |
| EMA Momentum | $\\tau$ | Cosine schedule 0.996 to 1.0 | $\\bar\\theta \\leftarrow \\tau\\bar\\theta + (1-\\tau)\\theta$ |

**Scenario 2 -- Unhealthy Training (same 6 subplots but pathological):**

| Warning Sign | What It Means | Fix |
|-------------|--------------|-----|
| Gradient Norm > 100 | Gradient explosion | Reduce LR, enable gradient clipping |
| Cosine Sim near 1.0 | Representation collapse | Check augmentation, EMA schedule |
| Rising TF Loss | Model forgetting | Check LR schedule, data pipeline |
| Rollout Error > 1.0 | Predictions diverge | Increase AR training weight |

**The Health Report** gives concrete pass/fail at step 500 with thresholds.

*Layman terms:* This dashboard is your "vital signs monitor" during training. Green metrics mean the model is learning; red warnings mean you need to intervene before wasting more GPU hours."""

# Cell [66] - Action conditioning vs unconditioned
insertions[66] = """### Reading the Action-Conditioned vs Unconditioned Output

**Plot 1 -- Three Prediction Panels:**

| Panel | Mode | Prediction Shape | Key Observation |
|-------|------|-----------------|----------------|
| Unconditioned | No action given | Large uncertainty ellipse centered on start | Model hedges by predicting the mean of all possible futures |
| "Move Right" | $a = [1, 0, \\ldots]$ | Small ellipse shifted right | Model knows the specific outcome |
| "Move Toward Object" | $a = \\text{dir}(\\text{cup})$ | Small ellipse toward cup | Model predicts the specific trajectory |

**Plot 2 -- Multi-Step Trajectory:**

- **Green = ground truth**, **Blue = action-conditioned**, **Red = unconditioned**
- Blue should track green closely; red should drift toward the center

**GIF -- Action Conditioning Over Time:**

Frame-by-frame visualization of how conditioned predictions stay on track while unconditioned ones blur.

**Formal comparison:**

$$\\text{Unconditioned: } \\hat{z}_{t+1} = P_\\phi(z_t) \\approx \\mathbb{E}_{a}[z_{t+1} \\mid z_t]$$

$$\\text{Conditioned: } \\hat{z}_{t+1} = P_\\phi(z_t, a_t) \\quad \\text{(specific to the given action)}$$

*Layman terms:* Without actions, the model says "the robot could go anywhere, so I'll predict the average." With actions, it says "you're moving right, so I predict you'll be over there."

**Why this matters:** CEM planning evaluates many action sequences through the world model. If the model cannot distinguish between action outcomes, all candidates score the same, and planning fails. Action conditioning is what makes V-JEPA 2-AC useful for robotics."""

# Insert all, highest index first
for idx in sorted(insertions.keys(), reverse=True):
    md_cell = make_md_cell(insertions[idx])
    nb['cells'].insert(idx + 1, md_cell)

with open('notebooks/03_VJEPA2_AC_WorldModel.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Inserted {len(insertions)} markdown cells")
print(f"Total cells now: {len(nb['cells'])}")
