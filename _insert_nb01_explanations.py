"""
Insert explanation markdown cells after every code cell that produces output
in notebook 01_JEPA_Fundamentals.ipynb.

Works from bottom to top to avoid index shifting issues.
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('D:/Work/JEPAstudy/notebooks/01_JEPA_Fundamentals.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def make_md_cell(source_text):
    """Create a markdown cell dict."""
    lines = source_text.split("\n")
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + "\n")
        else:
            source.append(line)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

# Define all explanation cells to insert.
# Key = cell index (0-based) AFTER which to insert.
# Value = markdown source text.
# We process in reverse order of keys to avoid index shifting.

explanations = {}

# Cell 91: Training Monitoring Dashboard (last cell, no explanation after)
explanations[91] = r"""### Reading the Output Above

**Printed Metrics (200-step simulation)**:
- *Loss at each step*: $\mathcal{L}_t$ -- the L1 prediction error at training step $t$. Starts around 2.5, should decrease to ~0.15.
- *Moving average*: Smoothed loss over a window of ~10--20 steps. Reveals the true trend beneath noisy per-step values.
- *Gradient norm*: $\|\nabla_\theta \mathcal{L}\|_2$ -- magnitude of the gradient vector. Healthy range: 0.01--10. If it drops to ~0, learning has stopped. If it spikes above 100, training is unstable.
- *Representation diversity*: Standard deviation of pairwise cosine similarities between encoder outputs. Should increase from ~0.3 to ~0.7. If it drops to 0, all outputs are identical (collapse).
- *EMA gap*: $\|\theta - \bar{\theta}\|$ -- Euclidean distance between online and target encoder parameters. Should be small but nonzero.

**Plot -- Training Monitoring Dashboard (6 panels)**:
- *Top-left (Loss Curve)*: X-axis = training step (0--200), Y-axis = L1 loss. Blue line = raw loss, orange = moving average. Look for: smooth downward trend. Worry if: flat after 50+ steps, or sudden spikes that never recover.
- *Top-center (Gradient Norm)*: X-axis = step, Y-axis = $\|\nabla\|_2$. Should stay in [0.01, 10]. Worry if: drops below 0.001 (vanishing gradients) or exceeds 100 (exploding gradients).
- *Top-right (Representation Diversity)*: X-axis = step, Y-axis = std of cosine similarities. Should increase over training. Worry if: drops toward 0 (representation collapse).
- *Bottom-left (EMA Gap)*: X-axis = step, Y-axis = parameter distance. Should be small (~0.01--0.1) and roughly stable. Large or growing gap means the target encoder is diverging.
- *Bottom-center (Learning Rate Schedule)*: X-axis = step, Y-axis = LR value. Shows warmup + cosine decay pattern.
- *Bottom-right (Loss Components or Summary)*: May show further breakdown.
- *Dimensions*: All metrics are scalar values (one number per step). No tensor dimensions involved -- these are aggregated statistics.
- *Good run*: Loss curves downward, gradients bounded, diversity grows, EMA gap small.
- *Bad run*: Loss flat or rising, gradients vanish/explode, diversity collapses to 0."""

# Cell 89: Failure simulation (3 scenarios)
explanations[89] = r"""### Reading the Output Above

**Printed Metrics -- Three Training Scenarios**:

Each scenario runs for 100 simulated training steps and prints loss, diversity, and gradient statistics.

1. **Healthy Training**:
   - *Loss*: Decreases smoothly from ~2.0 to ~0.1. *Formal*: $\mathcal{L}_t \approx 2.0 \cdot e^{-t/25} + 0.1$
   - *Diversity*: Increases from ~0.3 to ~0.7. This means encoder outputs are becoming more distinct from each other.
   - *Gradient norm*: Stays in range 0.1--1.0. Stable gradients mean every layer is learning.

2. **Representation Collapse** (no stop-gradient):
   - *Loss*: Drops fast to ~0, but this is **bad** -- the model found a trivial shortcut (all outputs identical).
   - *Diversity*: Drops to ~0. All 200 images produce the same representation vector.
   - *Gradient norm*: May vanish because there is nothing left to learn (trivial solution).

3. **Gradient Explosion** (learning rate too high):
   - *Loss*: Oscillates wildly or increases to NaN/Inf.
   - *Diversity*: Erratic, no clear trend.
   - *Gradient norm*: Spikes above 100 or reaches Inf.

**Plot -- Side-by-Side Comparison (3x3 grid)**:
- *Each column* = one scenario (healthy, collapse, explosion).
- *Row 1 (Loss)*: X-axis = step, Y-axis = loss. Healthy shows smooth decay. Collapse shows fast drop to 0. Explosion shows erratic oscillation.
- *Row 2 (Diversity)*: X-axis = step, Y-axis = representation spread. Healthy increases. Collapse goes to 0. Explosion is chaotic.
- *Row 3 (Gradient Norm)*: X-axis = step, Y-axis = $\|\nabla\|_2$. Healthy is bounded. Collapse vanishes. Explosion grows unboundedly.
- *Why this matters*: These are the three failure modes you will encounter in real JEPA training. Recognizing them early saves days of wasted compute."""

# Cell 87: Diagnostic toolkit
explanations[87] = r"""### Reading the Output Above

**Health Check Report**:

The `jepa_health_check()` function runs a series of diagnostic tests on the encoder and reports:

- **Output variance**: $\text{Var}(z)$ where $z = f_\theta(x)$ across $n$ samples. *Healthy*: 0.1--2.0. *Collapsed*: < 0.01 (all outputs nearly identical). *Dimensions*: scalar, computed over all elements of output tensor $[B, N, D]$.
- **Cosine similarity matrix**: Pairwise $\cos(z_i, z_j)$ for $i \neq j$. *Healthy*: mean ~0.3--0.7 with spread. *Collapsed*: mean ~1.0 (all pairs identical). *Dimensions*: $[n, n]$ symmetric matrix.
- **Gradient magnitude**: $\|\nabla_\theta \mathcal{L}\|$ per layer. *Healthy*: 0.001--10. *Vanishing*: < 0.0001. *Exploding*: > 100.
- **Rank of representation matrix**: Effective rank of the $[n \times D]$ matrix of outputs. *Healthy*: close to $\min(n, D)$. *Collapsed*: 1 (all outputs on a single line).

**Interpreting the printed diagnostics**:
- "HEALTHY" next to a metric means it falls in the expected range.
- "WARNING" means the metric is borderline -- training may be struggling.
- "CRITICAL" means a failure mode is active (collapse, vanishing gradients, etc.).
- *Action*: If you see CRITICAL, check: (1) is stop-gradient applied? (2) is EMA momentum too high/low? (3) is learning rate appropriate?"""

# Cell 85: Hand-computed JEPA forward pass
explanations[85] = r"""### Reading the Output Above

**Step-by-Step Numerical Trace**:

This cell traces every single number through a tiny JEPA with 2 patches and embed_dim=4. Each printed block shows one computation:

- **Input patches**: $x \in \mathbb{R}^{[2, 2, 4]}$ -- 2 samples, 2 patches each, 4-dimensional. The actual numbers are shown so you can follow along.
- **Encoder Linear layer**: $z = xW + b$ where $W \in \mathbb{R}^{4 \times 4}$, $b \in \mathbb{R}^4$. Each row of the output is one patch's new representation.
- **GELU activation**: $\text{GELU}(z) = z \cdot \Phi(z)$ where $\Phi$ is the Gaussian CDF. Negative values are shrunk toward 0 (not hard-zeroed like ReLU).
- **LayerNorm**: For each patch independently, subtract mean and divide by std: $\hat{z}_i = (z_i - \mu) / \sigma$ where $\mu, \sigma$ are computed over the $D=4$ dimensions.
- **Target vs Prediction comparison**: The final printed values show $\hat{z}_{\text{pred}}$ vs $z_{\text{target}}$ side by side. The L1 loss is $\frac{1}{D}\sum_d |\hat{z}_d - z_d|$.

**How to read the matrices**:
- Each row = one patch (or one sample, depending on the dimension being shown).
- Each column = one dimension of the embedding.
- *Dimensions*: All intermediate tensors have shape $[B, N, D] = [2, 2, 4]$ or $[B, N_{\text{masked}}, D]$ for masked subsets.

**Why this matters**: By seeing every number, you can verify that your understanding of the math matches reality. If any step surprises you, revisit the corresponding section above."""

# Cell 82: LogSumExp trick
explanations[82] = r"""### Reading the Output Above

**Printed Step-by-Step Computation**:

- **Input logits**: $x = [90, 100, 85, 95]$ -- these are large values where naive $e^{x_i}$ overflows float32 (max ~$e^{88}$).
- **Naive approach**: Computing $e^{100}$ directly gives `Inf` in float32, making $\log(\sum e^{x_i})$ undefined.
- **LogSumExp trick**: Subtract $c = \max(x) = 100$ first, then compute $c + \log\sum_i e^{x_i - c}$. Now the largest exponent is $e^0 = 1$, which is safe.
  - *Formal*: $\text{LSE}(x) = c + \log\sum_i e^{x_i - c}$ where $c = \max_i x_i$.
  - *Layman*: Shift all numbers down so the biggest one becomes 0, do the math safely, then shift back up.
- **Result**: The printed LSE value should be ~100.048 (close to 100 because the max term dominates).

**Plot -- Numerical Stability Comparison**:
- *X-axis*: Scale factor applied to logits (how big the numbers get).
- *Y-axis*: Computed log-sum-exp value.
- *Blue line (naive)*: Works for small inputs but produces NaN/Inf for large inputs.
- *Red line (LogSumExp trick)*: Produces correct values at all scales.
- *Dimensions*: Input $x \in \mathbb{R}^4$ (4 logits), output is a scalar.
- *Why it matters*: JEPA uses softmax in its attention layers. Without this trick, attention scores with large magnitudes would cause NaN during training."""

# Cell 80: Softmax numerical stability
explanations[80] = r"""### Reading the Output Above

**Printed Examples**:

1. **Small logits** $x = [1, 2, 3, 4]$:
   - $e^x = [2.72, 7.39, 20.09, 54.60]$ -- all representable in float32.
   - $\text{softmax}(x) = e^{x_i} / \sum_j e^{x_j}$ -- probabilities that sum to 1.0.
   - *Layman*: Each value gets turned into a probability. Bigger input = bigger probability.

2. **Large logits** $x = [90, 100, 85, 95]$:
   - $e^{100} \approx 2.69 \times 10^{43}$ -- overflows float32 max ($\approx 3.4 \times 10^{38}$).
   - Naive softmax returns `NaN` or `Inf`.
   - With the max-subtraction trick: subtract $\max(x) = 100$, compute $\text{softmax}([{-10}, 0, {-15}, {-5}])$ -- now all exponents are safe.
   - *Formal*: $\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$ (mathematically identical, numerically stable).

3. **Very negative logits**: Similar overflow issue in the denominator can cause underflow (all $e^{x_i} \approx 0$).

**Plot -- Stability Comparison**:
- Shows softmax output for increasingly extreme inputs.
- *Stable version* (red) always produces valid probabilities; *naive version* (blue) breaks.
- *Dimensions*: Input $x \in \mathbb{R}^D$ (vector of logits), output $\in \mathbb{R}^D$ (probability vector summing to 1).
- *Why it matters*: Every attention layer in JEPA's ViT encoder computes softmax. If logits grow large (common in deep networks), unstable softmax produces NaN, killing the entire training run."""

# Cell 77: Loss landscape GIF
explanations[77] = r"""### Reading the Output Above

**GIF -- Loss Landscape Evolution During Training**:

This animation shows how the loss surface changes as the JEPA model trains over multiple epochs.

- **Each frame** = the loss landscape at a different training epoch.
- **X-axis and Y-axis**: Two principal directions in parameter space (found via random perturbation of model weights). These are not individual parameters but linear combinations that capture the most variation.
- **Color/Height**: Loss value $\mathcal{L}(\theta + \alpha v_1 + \beta v_2)$ where $v_1, v_2$ are random directions and $\alpha, \beta$ are the axis coordinates.
- **What to look for**:
  - *Early training*: Rough, chaotic surface with many local minima. The model hasn't found a good region yet.
  - *Mid training*: Surface becomes smoother. A clear basin (valley) forms around the current parameters.
  - *Late training*: Smooth, wide basin. The model is near a good minimum. Wider basins = better generalization.
- **What good looks like**: The landscape evolves from rough to smooth, with the minimum (darkest color) deepening over time.
- **What bad looks like**: If the landscape stays rough or the minimum keeps jumping around, the learning rate may be too high.
- *Dimensions*: The GIF evaluates loss on a 2D grid of parameter perturbations. Each frame is a 2D heatmap. The model parameters $\theta \in \mathbb{R}^P$ are projected onto 2 directions."""

# Cell 76: Masking strategy GIF
explanations[76] = r"""### Reading the Output Above

**GIF -- Masking Strategy Animation**:

This animation visualizes different masking strategies applied to a grid of patches, showing how they affect what the encoder sees vs. what the predictor must predict.

- **Each frame** = one random mask applied to the patch grid.
- **Green patches**: Visible to the encoder ($\sim$10--25% of all patches). The encoder only processes these.
- **Red/dark patches**: Masked (hidden). The predictor must predict their latent representations from context alone.
- **What to look for**:
  - *Block masking* (V-JEPA 2 style): Large contiguous rectangular regions are masked. This forces the model to understand spatial structure, not just fill in random gaps.
  - *Random masking* (simpler): Individual patches are randomly dropped. Easier to predict because nearby visible patches provide direct clues.
  - *Temporal consistency*: In video masking, the same spatial block is masked across all time steps (tubelets), forcing temporal reasoning.
- **Mask ratio**: ~75--90% of patches are masked. Higher ratios = harder prediction task = stronger learned representations.
- *Dimensions*: The grid shown is $[H_{\text{tok}}, W_{\text{tok}}]$ (e.g., $16 \times 16 = 256$ spatial patches). In video, there is an additional temporal dimension $T_{\text{tok}}$."""

# Cell 73: Tensor dimension tracker
explanations[73] = r"""### Reading the Output Above

**Plot -- Complete Tensor Shape Flow Through V-JEPA 2**:

This diagram shows every tensor shape as data flows through the full V-JEPA 2 pipeline from raw video input to final loss.

- **Each box** represents one computational stage (e.g., patch embedding, attention, predictor).
- **The numbers in brackets** are tensor dimensions: $[B, C, T, H, W]$ for video, $[B, N, D]$ for patch sequences, etc.
- **Arrows** show data flow direction (top to bottom or left to right).
- **Color coding**: Different colors for encoder path, predictor path, and target encoder path.

**Key dimension transitions to understand**:
1. $[B, 3, 16, 224, 224] \to [B, 384, 8, 14, 14]$: Patch embedding (Conv3D) reduces spatial/temporal resolution while increasing channel dimension.
2. $[B, 384, 8, 14, 14] \to [B, 1568, 384]$: Flatten spatial-temporal grid into a sequence of $N = 8 \times 14 \times 14 = 1568$ tokens.
3. $[B, 1568, 384] \to [B, 157, 384]$: Masking keeps only ~10% of tokens (157 out of 1568).
4. $[B, 157, 384] \to [B, 1411, 384]$: Predictor produces representations for the 1411 masked positions.

- *Why it matters*: Tracking dimensions prevents shape mismatch bugs (the most common error in transformer implementations) and builds intuition for memory/compute requirements."""

# Cell 71: Backprop GIF
explanations[71] = r"""### Reading the Output Above

**GIF -- Gradient Flow Through JEPA During Backpropagation**:

This animation shows how gradients propagate backward through the encoder layers during one training step.

- **Each frame** = gradients at a different layer, moving from the loss backward to the input.
- **Layer bars/heatmap**: Shows gradient magnitude $|\partial \mathcal{L} / \partial W_l|$ at each layer $l$. Taller/brighter = larger gradient.
- **Color scale**: Blue = small gradients (potentially vanishing), Red = large gradients (potentially exploding), Green = healthy range.

**What to look for**:
- *Healthy gradient flow*: All layers have roughly similar gradient magnitudes (within 1--2 orders of magnitude). This means all layers are learning at a similar rate.
- *Vanishing gradients*: Early layers (near input) have much smaller gradients than later layers (near loss). Fix: use residual connections, LayerNorm, or careful initialization.
- *Exploding gradients*: Gradient magnitudes grow exponentially from output to input. Fix: gradient clipping, lower learning rate.

**Mathematical connection**:
- *Formal*: For layer $l$ with weight $W_l$, the gradient is $\frac{\partial \mathcal{L}}{\partial W_l} = \frac{\partial \mathcal{L}}{\partial z_L} \prod_{k=l+1}^{L} \frac{\partial z_k}{\partial z_{k-1}} \cdot \frac{\partial z_l}{\partial W_l}$
- *Layman*: Each layer multiplies the gradient by its own Jacobian. If these multipliers are consistently < 1, gradients vanish. If > 1, they explode.
- *Why LayerNorm + residual connections matter*: They keep these multipliers near 1, enabling gradient flow through 40+ layers (as in V-JEPA 2's ViT-giant)."""

# Cell 69: LayerNorm vs BatchNorm
explanations[69] = r"""### Reading the Output Above

**Plot -- LayerNorm vs BatchNorm Comparison (2x4 grid)**:

This visualization shows how the two normalization methods operate on the same data, highlighting why transformers use LayerNorm.

- **Row 1 (BatchNorm)**: Normalizes across the *batch* dimension for each feature independently.
  - *Formal*: $\hat{x}_{b,i} = \frac{x_{b,i} - \mu_i}{\sigma_i}$ where $\mu_i = \frac{1}{B}\sum_b x_{b,i}$, computed across all $B$ samples for feature $i$.
  - *Layman*: For each feature column, compute the average across all samples in the batch, then subtract it and scale.
  - *Problem for transformers*: Requires large, consistent batch sizes. Fails with variable-length sequences.

- **Row 2 (LayerNorm)**: Normalizes across the *feature* dimension for each sample independently.
  - *Formal*: $\hat{x}_{b,i} = \frac{x_{b,i} - \mu_b}{\sigma_b}$ where $\mu_b = \frac{1}{D}\sum_i x_{b,i}$, computed across all $D$ features for sample $b$.
  - *Layman*: For each sample (each patch, each token), normalize its feature vector to have mean 0 and std 1.
  - *Why transformers use it*: Works with any batch size, any sequence length. Each token is normalized independently.

- **Panels show**: Before normalization (raw activations) vs after normalization. After LayerNorm, each row (sample) has mean~0 and std~1. After BatchNorm, each column (feature) has mean~0 and std~1.
- *Dimensions*: Input tensor $[B, N, D]$ where $B$ = batch, $N$ = sequence length, $D$ = embedding dim. LayerNorm normalizes over $D$; BatchNorm normalizes over $B$."""

# Cell 67: Softmax temperature
explanations[67] = r"""### Reading the Output Above

**Plot -- Softmax Temperature Visualization (3 panels)**:

Shows how the temperature parameter $\tau$ in $\text{softmax}(x/\tau)$ controls the "sharpness" of attention distributions.

- **X-axis** (all panels): Token index (the 5 tokens being attended to, labeled by relevance).
- **Y-axis**: Attention probability $\alpha_i = \frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}}$ where $s_i$ are raw attention scores.

- **Panel 1 ($\tau = 0.5$, low temperature)**: Distribution is very "peaked" -- nearly all attention goes to the highest-scoring token. Like a hard argmax.
  - *Layman*: The model looks at only one token and ignores everything else.
  - *Problem*: Loses context from other tokens. Gradients are near-zero for low-attention tokens.

- **Panel 2 ($\tau = 1.0$, standard)**: Moderate distribution. Most attention on relevant tokens, some on others.
  - *Layman*: The model primarily looks at the most relevant token but also considers others.
  - *In practice*: This is what $\text{softmax}(QK^T / \sqrt{d_k})$ produces, where $\sqrt{d_k}$ acts as the temperature.

- **Panel 3 ($\tau = 3.0$, high temperature)**: Nearly uniform distribution. All tokens get similar attention.
  - *Layman*: The model looks at everything equally, regardless of relevance.
  - *Problem*: No selectivity. The attention mechanism isn't doing anything useful.

- **Why $\sqrt{d_k}$ scaling matters in JEPA**: Without dividing by $\sqrt{d_k}$, dot products $QK^T$ grow with $d_k$, pushing softmax toward Panel 1 (too peaked). The scaling keeps the distribution in the healthy middle range (Panel 2).
- *Dimensions*: Scores $s \in \mathbb{R}^N$, output $\alpha \in \mathbb{R}^N$ with $\sum_i \alpha_i = 1$."""

# Cell 65: Self-attention by hand
explanations[65] = r"""### Reading the Output Above

**Printed Step-by-Step Self-Attention Computation**:

This cell computes attention with 3 tokens and 4 dimensions, showing every intermediate value.

1. **Input $X \in \mathbb{R}^{3 \times 4}$**: Three tokens representing "cat ear", "cat nose", "sky background". Each is a 4-dimensional vector.

2. **QKV Projections**: $Q = XW_Q$, $K = XW_K$, $V = XW_V$ where $W_Q, W_K, W_V \in \mathbb{R}^{4 \times 4}$.
   - *Dimensions*: $Q, K, V$ each $\in \mathbb{R}^{3 \times 4}$ (3 tokens, 4 dims).
   - *Layman*: Each token gets transformed into three roles: what it's looking for (Q), what it offers (K), and what information it carries (V).

3. **Attention scores**: $S = QK^T / \sqrt{d_k}$ where $d_k = 4$.
   - *Dimensions*: $S \in \mathbb{R}^{3 \times 3}$ (every token scores against every other token).
   - $S_{ij}$ = how much token $i$ should attend to token $j$.
   - *Layman*: A 3x3 grid where each entry says "how relevant is token $j$ to token $i$?"

4. **Softmax**: $\alpha = \text{softmax}(S)$, applied row-wise.
   - Each row sums to 1.0 (probability distribution over which tokens to attend to).
   - *What to look for*: "cat ear" and "cat nose" should attend to each other (similar semantics) more than to "sky".

5. **Output**: $\text{Attn}(X) = \alpha V$ -- weighted sum of value vectors.
   - *Dimensions*: Output $\in \mathbb{R}^{3 \times 4}$ (same shape as input).
   - *Layman*: Each token's new representation is a blend of all tokens' values, weighted by relevance.

**Why this matters for JEPA**: The encoder uses self-attention to let visible patches share information. If patch 3 is masked, patches 1 and 2 must use attention to "communicate" and help the predictor reconstruct patch 3's representation."""

# Cell 63: Patch lifecycle visualization
explanations[63] = r"""### Reading the Output Above

**Plot -- The Complete Lifecycle of a Patch in JEPA (2x4 grid, 8 stages)**:

Each subplot shows one stage of what happens to a single patch as it passes through the JEPA pipeline.

1. **Raw Pixels** (top-left): A $16 \times 16$ pixel patch from the original image. Shown as a small image/heatmap.
2. **Patch Embedding**: The $16 \times 16 \times 3 = 768$ pixel values are linearly projected to a $D$-dimensional vector (e.g., $D = 384$). Shown as a 1D bar chart of embedding values.
3. **Positional Encoding Added**: Positional information is added so the model knows *where* this patch is in the image. The bars shift slightly.
4. **Masking Decision**: This patch is either kept (visible to encoder) or masked. If masked, its position gets a learnable mask token $\Delta_y$ instead.
5. **Encoder Output** (if visible): After passing through transformer blocks, the patch has a new representation that encodes both local content and global context from other visible patches.
6. **Target Encoder Output**: The EMA-updated target encoder processes this patch (regardless of masking) to produce the ground-truth target representation.
7. **Predictor Output** (if masked): The predictor generates a prediction $\hat{z}$ for this patch's representation using context from visible patches.
8. **Loss Computation**: $|\hat{z} - z_{\text{target}}|$ is computed element-wise. Shown as a bar chart of per-dimension errors.

- *Dimensions*: Raw pixels $[16, 16, 3]$, embedded patch $[D]$, encoder output $[D]$, loss is scalar per patch.
- *What to look for*: The embedding should look structured (not random noise). The prediction should be close to the target (small error bars in stage 8)."""

# Cell 61: What-if experiments
explanations[61] = r"""### Reading the Output Above

**Printed Experimental Results -- "What If..." Scenarios**:

Each scenario modifies one component of JEPA and reports the effect on training. Results are printed as final loss values and qualitative observations.

1. **mask_ratio=0 (no masking)**:
   - *What happens*: The predictor sees everything, so there is nothing to predict. Loss is trivially low but the model learns nothing useful.
   - *Expected loss*: ~0 (meaningless).
   - *Layman*: Like giving a student the answer sheet during the test -- they get 100% but learn nothing.

2. **mask_ratio=0.99 (almost everything masked)**:
   - *What happens*: Only ~1% of patches are visible. The predictor has almost no context to work with.
   - *Expected loss*: High (> 1.0). The task is too hard with so little information.
   - *Layman*: Like asking someone to describe a whole painting from seeing only one tiny corner.

3. **No stop-gradient**:
   - *What happens*: Both encoders are updated by gradients. They collude to find a trivial solution (all outputs = same constant).
   - *Expected loss*: Drops to ~0 quickly, but representations are useless (all identical).
   - *Key indicator*: Representation diversity drops to 0.

4. **EMA momentum = 0 (instant copy)**:
   - *What happens*: Target encoder is always an exact copy of the online encoder. The "slowly moving target" benefit is lost.
   - *Expected loss*: May train but produces worse representations.

5. **EMA momentum = 1 (frozen target)**:
   - *What happens*: Target encoder never updates. It provides fixed, random targets.
   - *Expected loss*: Loss plateaus at a high value because targets are meaningless.

- *Dimensions*: Each experiment trains for ~50 epochs on the toy dataset. Loss values are scalars. "Diversity" is computed as std of pairwise cosine similarities between encoded images."""

# Cell 59: Step-by-step walkthrough
explanations[59] = r"""### Reading the Output Above

**Printed Step-by-Step: One JEPA Training Iteration**:

This traces a single forward + backward pass with tiny dimensions ($B=2$, $N=8$ patches, $D=4$) so every number is visible.

- **Step 1 -- Input**: $x \in \mathbb{R}^{[2, 8, 4]}$ -- 2 samples, 8 patches each, 4-dimensional embeddings. The actual tensor values are printed.
- **Step 2 -- Masking**: With mask_ratio=0.75, 2 patches are visible and 6 are masked per sample. The indices (e.g., `[1, 5]` visible, `[0, 2, 3, 4, 6, 7]` masked) are printed.
- **Step 3 -- Encoder**: Visible patches $x_{\text{vis}} \in \mathbb{R}^{[2, 2, 4]}$ pass through the encoder. Output $z_{\text{vis}} \in \mathbb{R}^{[2, 2, D_{\text{enc}}]}$.
- **Step 4 -- Target Encoder**: ALL patches $x \in \mathbb{R}^{[2, 8, 4]}$ pass through the target encoder (with `torch.no_grad()`). Output $h \in \mathbb{R}^{[2, 8, D_{\text{enc}}]}$.
- **Step 5 -- Target Extraction**: Gather target representations at masked positions: $h_{\text{mask}} \in \mathbb{R}^{[2, 6, D_{\text{enc}}]}$.
- **Step 6 -- Predictor**: Takes $z_{\text{vis}}$ and predicts $\hat{z}_{\text{mask}} \in \mathbb{R}^{[2, 6, D_{\text{enc}}]}$.
- **Step 7 -- Loss**: $\mathcal{L} = \frac{1}{B \cdot N_{\text{mask}} \cdot D} \sum |\hat{z} - h_{\text{mask}}|$. A single scalar value.
- **Step 8 -- Backward**: Gradients computed for encoder + predictor parameters (not target encoder).
- **Step 9 -- EMA Update**: $\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta$.

- *Why this matters*: Seeing the concrete numbers at each step makes abstract equations tangible. You can manually verify any step with pen and paper."""

# Cell 57: Collapse GIF
explanations[57] = r"""### Reading the Output Above

**GIF -- Representation Collapse Without Stop-Gradient**:

This animation shows a side-by-side comparison of JEPA training with and without the stop-gradient operation on the target encoder.

- **Left panel (Healthy -- with stop-gradient)**: Representations in 2D (via PCA or t-SNE) spread out and form clusters. Different inputs produce different representations.
- **Right panel (Collapsed -- without stop-gradient)**: All representations converge to a single point. Every input produces the same output vector.

**Frame-by-frame progression**:
- *Early frames*: Both sides look similar -- representations are scattered randomly (untrained network).
- *Mid frames*: Left side shows clusters forming. Right side shows points starting to converge.
- *Late frames*: Left has clear structure. Right has all points collapsed to one location.

**Why collapse happens (mathematically)**:
- Without stop-gradient, both encoders receive gradients from the loss $\mathcal{L} = |\hat{z} - z_{\text{target}}|$.
- The easiest way to minimize this: make both encoders output the same constant vector $c$ for all inputs. Then $|c - c| = 0$ -- perfect loss, useless representations.
- *Formal*: $f_\theta(x) = c, \; \forall x$ is a global minimum of $\mathcal{L}$ when both sides are trained jointly.
- Stop-gradient prevents this by making the target encoder a *fixed* (slowly-moving) target that the online encoder must chase.

- *Dimensions*: Each dot in the animation represents one image's global representation (mean-pooled over patches), projected to 2D."""

# Cell 55: EMA GIF
explanations[55] = r"""### Reading the Output Above

**GIF -- EMA Momentum Comparison**:

This animation shows multiple JEPA training runs side-by-side, each with a different EMA momentum value $\tau$, demonstrating its effect on training stability.

- **Each panel**: One training run with a specific $\tau$ value (e.g., 0.9, 0.99, 0.999, 0.9999).
- **What is shown**: Training loss curve and/or parameter trajectories over time.

**Reading each panel**:
- **$\tau = 0.9$ (too low)**: Target encoder updates too fast, closely tracking the online encoder. The loss may oscillate or fail to converge because the target is not stable enough.
  - *Layman*: The teacher changes their answer key every minute, so the student can never learn.
- **$\tau = 0.99$**: Better stability but still relatively fast updates. May work for simple tasks.
- **$\tau = 0.999$ (good range)**: Target encoder provides a smooth, slowly-moving target. Loss decreases steadily.
  - *Layman*: The teacher updates the answer key once a day -- slow enough to be consistent, fast enough to improve.
- **$\tau = 0.99925$ (V-JEPA 2 value)**: The sweet spot found through extensive hyperparameter search. Very slow updates.
- **$\tau = 0.9999$ (too high)**: Target encoder barely changes. It provides nearly frozen targets, which limits what the model can learn.

**Mathematical connection**: EMA update rule is $\bar{\theta}_t = \tau \bar{\theta}_{t-1} + (1 - \tau) \theta_t$. The effective "memory" is $\sim 1/(1-\tau)$ steps. For $\tau = 0.99925$, this is ~1333 steps."""

# Cell 49: 4D animated visualization
explanations[49] = r"""### Reading the Output Above

**Plot -- 4D Animated Training Dynamics (Multiple Snapshots)**:

This visualization uses multiple 3D scatter plots at different training epochs to show how representations evolve over time (the 4th dimension).

- **Each subplot** = one training epoch (e.g., epoch 0, 20, 40, 60, 80, 100).
- **X, Y, Z axes**: First 3 PCA components of the learned representations.
- **Dot colors**: Different colors represent different data clusters/classes.
- **Dot size or opacity**: May encode additional information (e.g., loss contribution).

**What to look for across frames**:
- *Epoch 0*: Points are randomly scattered (untrained network produces random representations).
- *Early epochs*: Points begin to form loose groupings.
- *Mid epochs*: Clear clusters emerge -- similar inputs are mapped nearby.
- *Late epochs*: Tight, well-separated clusters. The representation space has meaningful structure.

**Why this matters**: The encoder's job is to produce representations where semantically similar inputs are close together and different inputs are far apart. This visualization directly shows whether that goal is being achieved.
- *Dimensions*: Each dot represents one image's representation $z \in \mathbb{R}^D$ projected to 3D via PCA. The original $D = 64$ in the toy model."""

# Cell 48: 5D visualization
explanations[48] = r"""### Reading the Output Above

**Plot -- 5D Visualization: Mask Ratio x Embed Dim x EMA Momentum x Loss x Convergence Speed**:

Displaying 5 dimensions on a 2D plot requires encoding extra dimensions through visual properties.

- **X-axis**: Mask ratio (0.5 to 0.9). Higher = more patches hidden from the encoder.
- **Y-axis**: Embedding dimension (16 to 128). Larger = more expressive representations but slower training.
- **Dot color**: Final training loss. Blue/cool = low loss (good), Red/warm = high loss (bad).
- **Dot size**: Convergence speed (epochs to reach a threshold loss). Larger dots = slower convergence.
- **Shape or opacity**: May encode EMA momentum $\tau$.

**How to read it**:
- Look for the region with the *smallest, coolest-colored dots* -- that is the optimal hyperparameter combination (low loss, fast convergence).
- *Best region*: Typically mask_ratio ~0.75, embed_dim ~64, $\tau$ ~0.996 for this toy model.
- *Worst region*: Extreme mask ratios (0.0 or 0.99) combined with tiny embedding dims.

**Interpreting the 5th dimension**: Convergence speed adds practical information beyond just "did it work?" -- it tells you which configurations are feasible under compute budgets.
- *Dimensions*: Each dot represents one complete training run. The grid samples $4 \times 4 \times 4 = 64$ hyperparameter combinations."""

# Cell 47: 4D visualization
explanations[47] = r"""### Reading the Output Above

**Plot -- 4D Visualization: Mask Ratio x Embed Dim x EMA Momentum x Loss**:

This 3D scatter plot encodes 4 variables: three on spatial axes and one via color.

- **X-axis**: Mask ratio (fraction of patches hidden from encoder, range 0.5--0.9).
- **Y-axis**: Embedding dimension (size of representation vectors, 16--128).
- **Z-axis**: EMA momentum $\tau$ (how slowly the target encoder updates, 0.99--0.999).
- **Color**: Final training loss $\mathcal{L}$. Cool colors (blue/purple) = low loss, warm colors (red/yellow) = high loss.

**How to read it**:
- Rotate the 3D plot mentally (or interactively in a notebook) to find clusters of blue dots -- these are the good hyperparameter combinations.
- *Key insight*: The loss landscape is not random -- there is a "sweet spot" region where mask_ratio $\approx 0.75$, embed_dim $\geq 32$, and $\tau \approx 0.996$.
- Extreme values in any dimension tend to produce higher loss (red dots).

**Why this ablation matters**: In real JEPA research, hyperparameter sensitivity determines whether a method is practical. If the good region is tiny, the method is fragile. If it is large, the method is robust.
- *Dimensions*: 64 data points from the $4 \times 4 \times 4$ grid of hyperparameter combinations. Each point summarizes one full training run."""

# Cell 44: Masking visualization
explanations[44] = r"""### Reading the Output Above

**Plot -- V-JEPA 2 Multiblock Spatiotemporal Masking**:

This visualization shows V-JEPA 2's actual masking strategy applied to a simulated $16 \times 16 \times 8$ (H x W x T) patch grid.

- **Grid cells**: Each cell represents one spatiotemporal patch (a "tubelet" covering a $16 \times 16$ pixel region across 2 video frames).
- **White/light cells**: Visible patches (encoder sees these). Only ~10% of all patches.
- **Dark/colored blocks**: Masked regions. These are contiguous rectangular blocks, not random individual patches.

**Key masking properties**:
- *Block structure*: V-JEPA 2 masks 8 large spatiotemporal blocks, each covering ~15% of the spatial area and 35--100% of the temporal extent.
- *Aspect ratio*: Blocks have random aspect ratios (0.75 to 1.5) to avoid bias toward horizontal or vertical structures.
- *Temporal consistency*: Each spatial block extends across multiple consecutive frames. This forces the model to learn temporal dynamics, not just spatial patterns.
- *Overlap*: Blocks may overlap, so the effective mask ratio can vary.

**Why block masking > random masking**:
- Random masking is easy: nearby visible patches provide direct interpolation clues.
- Block masking is hard: the model must understand *structure* (what objects look like, how they move) to fill in large missing regions.
- *Formal*: $M = \{(t, h, w) : (t, h, w) \in \bigcup_{k=1}^{K} B_k\}$ where each $B_k$ is a rectangular block with random position and size.
- *Dimensions*: Mask $M \in \{0, 1\}^{T_{\text{tok}} \times H_{\text{tok}} \times W_{\text{tok}}}$. Total patches $N = 8 \times 16 \times 16 = 2048$."""

# Cell 43: Gather operation visualization
explanations[43] = r"""### Reading the Output Above

**Plot -- Gather Operation + Information Theory of Masking (2 panels)**:

**Left Panel -- `torch.gather` Operation**:
- Shows a concrete example of how masking works in code.
- *Top heatmap*: Full sequence of 8 patches, each with 4 features. All values visible (the "full" representation).
- *Bottom heatmap*: After `torch.gather(x, dim=1, index=mask_expanded)`, only patches at indices $[1, 3, 6]$ are kept. The other 5 patches are discarded.
- *Formal*: `gathered[b, i, d] = x[b, mask[b, i], d]` for batch $b$, position $i$, dimension $d$.
- *Layman*: `gather` is like picking specific rows from a table by their row numbers.
- *Dimensions*: Input $x \in \mathbb{R}^{[1, 8, 4]}$, mask indices $\in \mathbb{Z}^{[1, 3]}$, output $\in \mathbb{R}^{[1, 3, 4]}$.

**Right Panel -- Information Theory of Masking**:
- Shows the relationship between mask ratio and prediction difficulty.
- If patches are independent: $I(x_{\text{vis}}; x_{\text{mask}}) = 0$ and prediction is impossible.
- If patches are perfectly correlated: $I$ is maximal and prediction is trivial.
- Real images fall in between: nearby patches are correlated (spatial structure), so ~25% visible patches carry substantial information about the ~75% masked patches.
- *Why this matters*: JEPA works because natural images have spatial redundancy. The masking ratio must be high enough to make the task challenging but low enough that the task is solvable."""

# Cell 40: L1 vs L2 loss
explanations[40] = r"""### Reading the Output Above

**Plot -- L1 vs L2 Loss Comparison (2 panels)**:

**Left Panel -- Loss Functions**:
- *X-axis*: Prediction error $e = \hat{z}_i - z_i$ (difference between predicted and target value), range $[-3, 3]$.
- *Y-axis*: Loss value.
- *Blue line (L1)*: $|e|$ -- absolute value. Forms a V-shape. Grows linearly with error.
- *Red line (L2)*: $e^2$ -- squared error. Forms a U-shape (parabola). Grows quadratically with error.
- *Key difference*: For large errors ($|e| > 1$), L2 penalizes much more heavily. For small errors ($|e| < 1$), L2 penalizes less than L1.

**Right Panel -- Gradients**:
- *X-axis*: Same prediction error $e$.
- *Y-axis*: Gradient $\partial \mathcal{L} / \partial e$ (the "push" applied during backprop).
- *Blue line (L1 gradient)*: $\text{sign}(e) = \pm 1$. Constant magnitude regardless of error size.
  - *Layman*: Every wrong prediction gets the same strength correction, whether it is off by 0.01 or 10.
- *Red line (L2 gradient)*: $2e$. Proportional to error size.
  - *Layman*: Big mistakes get big corrections; small mistakes get small corrections.

**Why V-JEPA 2 uses L1**:
- L1's constant gradient avoids "overcorrecting" for outlier predictions (robust to noise).
- L2's quadratic penalty causes the model to obsess over outliers, potentially wasting capacity.
- *Formal*: V-JEPA 2 uses `loss_exp = 1.0`, meaning $\mathcal{L} = \frac{1}{|M| \cdot D} \sum_{i \in M} \sum_{d=1}^{D} |\hat{z}_{i,d} - z_{i,d}|^1$.
- *Dimensions*: Loss is a scalar computed over $|M|$ masked patches $\times$ $D$ embedding dimensions."""

# Cell 37: Similarity matrix plot
explanations[37] = r"""### Reading the Output Above

**Plot -- Pairwise Cosine Similarity Matrix**:

- *Axes*: Both X and Y axes represent image indices (first 50 of the 200 training images).
- *Color*: Cosine similarity $\cos(z_i, z_j) = \frac{z_i \cdot z_j}{\|z_i\| \|z_j\|}$ between global representations of image $i$ and image $j$. Range: $[-1, 1]$.
  - Yellow/bright ($\approx 1.0$): Images have very similar representations.
  - Dark/purple ($\approx 0$ or negative): Images have dissimilar representations.
- *Diagonal*: Always 1.0 (every image is identical to itself).

**What to look for**:
- *Block structure*: If images are generated with different "base patterns" (as in our toy data), similar images should form bright blocks along the diagonal. This means the encoder learned to group similar inputs.
- *Uniform brightness (bad)*: If the entire matrix is the same color, all images have the same representation -- this indicates collapse.
- *Random noise pattern (bad)*: If the matrix looks like TV static with no structure, the encoder hasn't learned meaningful features.

**Dimensions**:
- Each image's global representation: $z_i = \frac{1}{N}\sum_{n=1}^{N} f_\theta(x_i)_n \in \mathbb{R}^{64}$ (mean-pooled over 16 patches, 64-dim embedding).
- Similarity matrix: $\in \mathbb{R}^{50 \times 50}$ (showing first 50 images).
- *Why this matters*: This is the primary way to verify that JEPA learned useful representations. Good representations = structured similarity matrix."""

# Cell 36: Representation collapse experiment
explanations[36] = r"""### Reading the Output Above

**Printed Training Logs -- Collapse Experiment (No Stop-Gradient)**:

- **Loss values per epoch**: The loss drops rapidly to near 0 within the first few epochs.
  - *Formal*: $\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} |\hat{z}_i - z_i| \to 0$ because both encoders converge to the same constant function.
  - *Layman*: The loss says "perfect predictions!" but this is misleading -- the model is cheating.

- **Why loss~0 is BAD here**: When both the online and target encoders can both be updated by gradients, they discover a trivial solution: output the same constant vector $c$ for every input. Then $\hat{z}_i = c$ and $z_i = c$, so $|c - c| = 0$.

- **Collapse indicators to watch**:
  - Loss drops to near 0 suspiciously fast (within 5--10 epochs instead of 50+).
  - If you computed representation diversity, it would be ~0.
  - All images produce the same encoder output regardless of their content.

**Comparison with healthy training (cell 13)**:
  - Healthy: loss decreases gradually over 50 epochs to ~0.3--0.5.
  - Collapsed: loss drops to ~0.01 within 5 epochs.
  - *Key difference*: The stop-gradient on the target encoder prevents the trivial solution.

- *Dimensions*: Same as the healthy training loop: loss is scalar, computed over $[B, N_{\text{mask}}, D] = [32, 12, 64]$ elements per batch."""

# Cell 33: Mutual information
explanations[33] = r"""### Reading the Output Above

**Plot -- Mutual Information Between Layers**:

- *X-axis*: Layer pair (e.g., "Input-Layer1", "Layer1-Layer2", "Layer2-Output").
- *Y-axis*: Estimated mutual information $\hat{I}(Z_l; Z_{l+1})$ in bits (or nats).
  - *Formal*: $I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$, estimated via histogram binning with $n = 20$ bins.
  - *Layman*: How much knowing the output of one layer tells you about the output of the next layer. Higher = more information preserved.

**What to look for**:
- *Decreasing MI from input to output*: Expected. Each layer discards some noise while preserving relevant information (information bottleneck principle).
- *Sudden drop in MI*: A layer that drastically reduces MI may be acting as a bottleneck -- it discards too much information.
- *MI near 0 at any point*: The layer is not transmitting useful information. This is a sign of a dead layer or representational collapse at that stage.
- *MI roughly constant*: The layer is preserving most information (residual connections enable this).

**Caveats**:
- Histogram-based MI estimation is approximate, especially in high dimensions. The values are relative, not absolute.
- *Dimensions*: MI is a scalar per layer pair. Computed from activations $Z_l \in \mathbb{R}^{[B \cdot N, D]}$ flattened to 1D for binning.
- *Why it matters*: The information bottleneck theory suggests good representations compress irrelevant details while preserving task-relevant information. MI analysis reveals whether each layer contributes to this goal."""

# Cell 31: Attention/correlation pattern evolution
explanations[31] = r"""### Reading the Output Above

**Plot -- Correlation Pattern Evolution During Training (6 snapshots)**:

Each subplot shows a $16 \times 16$ heatmap of patch-to-patch correlations at a different training epoch.

- *Axes*: Both X and Y represent patch indices (0--15 for our 16-patch grid).
- *Color*: Pearson correlation $r_{ij} = \text{corr}(z_i, z_j)$ between the encoder's representation of patch $i$ and patch $j$, computed across all training images.
  - Bright/warm: High positive correlation (these patches tend to produce similar representations).
  - Dark/cool: Low or negative correlation (these patches produce different representations).
- *Diagonal*: Always 1.0 (each patch is perfectly correlated with itself).

**What to look for across snapshots**:
- *Epoch 0 (untrained)*: Nearly uniform correlations -- the model treats all patches similarly because it hasn't learned anything.
- *Early epochs*: Some structure begins to emerge. Nearby patches (e.g., patches 0-3 in the same row of the 4x4 grid) may show higher correlation.
- *Late epochs*: Clear block structure. Patches that share semantic content (e.g., all "foreground" patches) correlate highly, while patches from different regions (e.g., "foreground" vs "background") decorrelate.

**In real transformers**: These correspond to attention patterns. Self-attention learns to connect semantically related patches regardless of spatial distance.
- *Dimensions*: Correlation matrix $\in \mathbb{R}^{[16, 16]}$, computed from representations $Z \in \mathbb{R}^{[200, 16, D]}$ (200 images, 16 patches, $D$-dim embeddings)."""

# Cell 29: Ablation study
explanations[29] = r"""### Reading the Output Above

**Plot -- Ablation Study: Hyperparameter Sensitivity (multi-panel grid)**:

This systematically varies three hyperparameters and shows training curves for each combination.

**Panel layout**: Each subplot shows loss curves (Y-axis = L1 loss, X-axis = training epoch) for different settings of one hyperparameter while holding others fixed.

**Key hyperparameters tested**:

1. **Mask ratio** (0.5, 0.625, 0.75, 0.875):
   - *Formal*: Fraction $\rho$ of patches in $M$ (the masked set). $|M| = \rho \cdot N$.
   - *Too low ($\rho < 0.5$)*: Task is too easy -- the predictor has abundant context. Final loss is low but representations are weak.
   - *Sweet spot ($\rho \approx 0.75$)*: Challenging but solvable. Produces the best representations.
   - *Too high ($\rho > 0.9$)*: Task is too hard with so little context. Loss stays high.

2. **Embedding dimension** (16, 32, 64, 128):
   - *Formal*: $D$ in $z \in \mathbb{R}^D$.
   - *Larger $D$*: More expressive but more parameters and slower training. May overfit on toy data.
   - *Smaller $D$*: Less expressive but faster. May underfit.

3. **EMA momentum** (0.99, 0.996, 0.999, 0.9999):
   - *Formal*: $\tau$ in $\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta$.
   - *$\tau$ too low*: Target encoder changes too fast, providing unstable targets.
   - *$\tau$ too high*: Target encoder barely updates, providing stale targets.
   - *V-JEPA 2 uses*: $\tau = 0.99925$.

- *Dimensions*: Each training run produces a loss curve (50 scalar values). The grid shows $4 \times 4 \times 4 = 64$ total runs.
- *What to look for*: Find the combination with the lowest final loss AND smooth convergence. The "best" combination should match V-JEPA 2's recommendations."""

# Cell 27: 3D representation space
explanations[27] = r"""### Reading the Output Above

**Plot -- 3D Representation Space (PCA and t-SNE, 3 panels)**:

Three 3D scatter plots showing the learned representations projected to 3 dimensions.

- **Panel 1 (PCA 3D)**: Linear projection onto the 3 directions of maximum variance.
  - *Axes*: PC1, PC2, PC3 (principal components). PC1 captures the most variance.
  - *Formal*: $z_{\text{3D}} = U_3^T (z - \bar{z})$ where $U_3$ contains the top 3 eigenvectors of the covariance matrix.
  - *Preserves*: Global structure and distances. Good for seeing overall spread.

- **Panel 2 (t-SNE 3D)**: Non-linear projection that preserves local neighborhood structure.
  - *Axes*: t-SNE dimensions (arbitrary units, not interpretable individually).
  - *Preserves*: Which points are near each other. Good for seeing clusters.

- **Panel 3** (may show UMAP or additional view): Another non-linear projection for comparison.

**Dot colors**: Each color represents a different "class" of synthetic data (images with similar base patterns).

**What to look for**:
- *Well-separated clusters*: Different colors form distinct groups. This means the encoder learned to distinguish different types of inputs.
- *Compact clusters*: Points of the same color are tightly grouped. This means similar inputs get similar representations.
- *Random scatter (bad)*: No visible clusters. The encoder hasn't learned meaningful structure.
- *Single blob (bad)*: All points collapse to one location regardless of color. This is representation collapse.

- *Dimensions*: Each dot is one image's global representation: $z \in \mathbb{R}^{64}$ projected to $\mathbb{R}^3$. 200 dots total (one per training image)."""

# Cell 25: PCA/t-SNE 2D
explanations[25] = r"""### Reading the Output Above

**Plot -- 2D Representation Space (PCA, t-SNE, and optionally UMAP)**:

These 2D scatter plots show the encoder's learned representations projected onto 2 dimensions for visualization.

- **PCA plot**: $z_{\text{2D}} = U_2^T(z - \bar{z})$. Linear projection onto the 2 directions of maximum variance.
  - *X-axis*: First principal component (direction of greatest variance).
  - *Y-axis*: Second principal component.
  - *Explained variance ratio* (printed or shown): How much of the total variance these 2 dimensions capture. If > 80%, the 2D view is a good summary.

- **t-SNE plot**: Non-linear dimensionality reduction that preserves local neighborhoods.
  - *Axes*: Arbitrary (not directly interpretable as features).
  - *Key parameter*: perplexity (typically 30) controls the effective number of neighbors.

- **Dot colors**: Each color represents a different synthetic "class" (images generated with similar base patterns).

**What to look for**:
- *Clear clusters by color*: The encoder groups similar images together in representation space. This is the goal of self-supervised learning.
- *Overlap between clusters*: Some overlap is normal. Too much overlap means the encoder hasn't learned to distinguish the classes.
- *PCA vs t-SNE disagreement*: If PCA shows no structure but t-SNE shows clusters, the structure is non-linear. If both show clusters, the structure is robust.

- *Dimensions*: Input representations $z \in \mathbb{R}^{[200, 64]}$ (200 images, 64-dim global representation via mean pooling). Output: 2D coordinates for each image."""

# Cell 23: Loss landscape 3D
explanations[23] = r"""### Reading the Output Above

**Plot -- 3D Loss Landscape Visualization**:

A surface plot showing how the loss function changes when model parameters are perturbed in two random directions.

- **X-axis**: Perturbation magnitude along random direction $v_1$ (range: typically $[-1, 1]$).
- **Y-axis**: Perturbation magnitude along random direction $v_2$.
- **Z-axis (height) and Color**: Loss value $\mathcal{L}(\theta + \alpha v_1 + \beta v_2)$.
  - *Formal*: $v_1, v_2 \in \mathbb{R}^P$ are random unit vectors in parameter space ($P$ = total number of model parameters).
  - *Layman*: Imagine the model's "position" in a high-dimensional space. We slice through two directions and plot the loss as a mountain landscape.

**What to look for**:
- *Smooth, bowl-shaped valley*: The loss landscape is well-conditioned. Optimization (gradient descent) will easily find the minimum.
- *Rough, chaotic surface*: Many local minima and sharp ridges. Optimization may get stuck.
- *Flat plateau*: The loss doesn't change much in some directions. These are "degenerate" dimensions where parameters don't matter.
- *Sharp, narrow valley*: The model is sensitive to small parameter changes. Training may be unstable.

- *Dimensions*: The grid evaluates loss at $n \times n$ points (e.g., $25 \times 25 = 625$ forward passes). Each evaluation uses the full dataset. $v_1, v_2 \in \mathbb{R}^P$ where $P$ is the total parameter count.
- *Why it matters*: Flat, wide minima tend to generalize better than sharp, narrow ones. This visualization helps diagnose optimization difficulties."""

# Cell 21: Gradient flow analysis
explanations[21] = r"""### Reading the Output Above

**Plot -- Gradient Flow Analysis**:

Shows the magnitude of gradients at each layer of the encoder during one backward pass.

- *X-axis*: Layer name (e.g., "block_0.weight", "block_1.weight", ..., "block_3.weight").
- *Y-axis*: Gradient magnitude, typically shown as mean $|\partial \mathcal{L} / \partial W_l|$ and/or max gradient per layer.
  - *Formal*: For parameter tensor $W_l$ at layer $l$, the gradient is $g_l = \frac{\partial \mathcal{L}}{\partial W_l}$. We plot $\text{mean}(|g_l|)$.
  - *Layman*: How strong the "learning signal" is at each layer. Stronger = that layer updates more during training.

**What to look for**:
- *Roughly uniform across layers*: Healthy. All layers receive similar learning signals and update at similar rates.
- *Exponential decrease from right to left*: Vanishing gradients. Early layers (near input) barely update. Fix: add residual connections, use LayerNorm, or increase learning rate.
- *Exponential increase from right to left*: Exploding gradients. Fix: gradient clipping, reduce learning rate, or add normalization.
- *Some layers at exactly 0*: Dead layers -- their parameters aren't being updated at all. Check for ReLU "dying" or disconnected computation paths.

- *Dimensions*: Each gradient tensor has the same shape as its corresponding weight tensor (e.g., $W_l \in \mathbb{R}^{[D_{\text{in}}, D_{\text{out}}]}$). The plotted values are scalar summaries (mean or max) of each gradient tensor.
- *Why it matters*: Gradient flow determines which layers actually learn. In V-JEPA 2's 40-layer ViT, maintaining gradient flow through all 40 layers is critical."""

# Cell 19: Activation statistics
explanations[19] = r"""### Reading the Output Above

**Printed Statistics and Plot -- Activation Statistics Per Layer**:

For each computational stage in the encoder, four statistics are computed and displayed:

- **Mean** ($\mu$): Average activation value. *Healthy*: near 0 after LayerNorm. *Warning*: large positive or negative bias.
- **Variance** ($\sigma^2$): Spread of activation values. *Healthy*: ~1.0 after LayerNorm. *Collapsed*: near 0 (all activations identical). *Exploding*: >> 10.
- **Kurtosis** ($\kappa$): "Peakedness" of the distribution. $\kappa = 3$ for Gaussian. $\kappa > 3$: heavy tails (some extreme activations). $\kappa < 3$: light tails. *Formal*: $\kappa = \frac{E[(x - \mu)^4]}{\sigma^4}$.
- **Max** ($\max |x|$): Largest absolute activation. *Warning*: if >> 100, risk of numerical overflow in subsequent layers.

**Plot (bar charts or table)**:
- *X-axis*: Layer name (e.g., "linear_1", "gelu", "linear_2", "layernorm").
- *Y-axis*: Statistic value.
- **What to look for**:
  - Variance should stay roughly constant (between 0.5 and 2.0) across layers. This is what LayerNorm ensures.
  - GELU activation reduces variance slightly (because it suppresses negative values).
  - If variance increases exponentially layer by layer, activations are exploding.
  - If variance drops toward 0, representational information is being lost.

- *Dimensions*: Each statistic is computed over all elements of the activation tensor $Z_l \in \mathbb{R}^{[B, N, D]}$ (flattened to 1D). The statistics themselves are scalars.
- *Why it matters*: Monitoring activation statistics catches problems (collapse, explosion) before they manifest as NaN losses or failed training."""

# Cell 17: Layer-by-layer activation analysis
explanations[17] = r"""### Reading the Output Above

**Printed Activation Trace -- Layer-by-Layer Through the JEPA Forward Pass**:

This cell instruments the encoder with hooks to record every intermediate tensor. For each layer, it prints:

- **Layer name**: Which operation produced this tensor (e.g., "linear_input", "gelu_activation", "linear_hidden", "layernorm_output").
- **Shape**: The tensor dimensions at this stage. Key shapes:
  - Input: $[B, N_{\text{vis}}, D_{\text{in}}]$ where $B$ = batch, $N_{\text{vis}}$ = visible patches, $D_{\text{in}}$ = input dim (32).
  - After Linear: $[B, N_{\text{vis}}, D_{\text{hidden}}]$ where $D_{\text{hidden}} = 64$.
  - After GELU: Same shape, but values are non-linearly transformed ($\text{GELU}(x) = x \cdot \Phi(x)$).
  - After LayerNorm: Same shape, but each feature vector is normalized to mean=0, std=1.
- **Value statistics**: Mean, std, min, max of the tensor values.
  - *After LayerNorm*: mean $\approx 0$, std $\approx 1$ (by construction).
  - *After GELU*: mean shifts positive (because GELU suppresses negatives).

**Why this matters**: Understanding the exact data flow helps debug shape mismatches, identify collapsing activations, and verify that normalization layers work as expected. In V-JEPA 2's ViT-giant, these same patterns repeat across 40 transformer blocks.
- *Dimensions*: All intermediate tensors maintain the sequence dimension $N_{\text{vis}}$. Only the feature dimension $D$ changes between layers."""

# Cell 15: Training loss plot
explanations[15] = r"""### Reading the Output Above

**Plot -- JEPA Training Loss Curve**:

- *X-axis*: Epoch number (0 to 49). Each epoch processes all 200 training images once.
- *Y-axis*: Average L1 loss per epoch. $\mathcal{L} = \frac{1}{|M| \cdot D} \sum_{i \in M} \sum_{d=1}^{D} |\hat{z}_{i,d} - z_{i,d}|$ where $M$ is the set of masked patches.
  - *Layman*: On average, how far off is each predicted number from the correct target value.
- *Blue line*: Raw loss per epoch.

**How to read it**:
- *Downward trend*: The model is learning to predict masked patch representations. This is good.
- *Flat line (bad)*: The model isn't learning. Check learning rate, architecture, or data.
- *Oscillation*: Some noise is normal (different random masks each epoch). Large oscillations suggest learning rate is too high.
- *Final value*: For our toy model, expect ~0.3--0.5 after 50 epochs. This means each dimension of the predicted representation is off by ~0.3--0.5 on average.
- *Would not reach 0*: Even a perfect model has some prediction error because masking removes information. The irreducible error depends on how correlated the patches are.

**Dimensions**:
- Loss is a scalar (single number per epoch).
- Computed over: $B = 32$ images per batch, $|M| = 12$ masked patches per image (75% of 16), $D = 64$ embedding dimensions.
- Total values averaged per batch: $32 \times 12 \times 64 = 24{,}576$.

**Note**: The title says "Predicting Representations, NOT Pixels" -- this is the core JEPA insight. The loss measures error in latent space ($\mathbb{R}^{64}$), not in pixel space ($\mathbb{R}^{16 \times 16 \times 3}$). Predicting in latent space avoids modeling irrelevant low-level details (exact pixel values, textures) and focuses on semantic content."""

# Cell 14: EMA convergence visualization
explanations[14] = r"""### Reading the Output Above

**Plot -- EMA Convergence Visualization (3 panels)**:

**Panel 1 (Full Trajectory)**:
- *X-axis*: Training step (0 to 1000).
- *Y-axis*: Parameter value (simulated single parameter from the encoder).
- *Blue line*: Online encoder parameter $\theta_t$ -- changes every step via gradient descent. Noisy with upward drift.
- *Red line*: Target encoder parameter $\bar{\theta}_t$ -- updated via EMA. Smooth, lagging behind the blue line.
- *Formal*: $\bar{\theta}_t = \tau \bar{\theta}_{t-1} + (1 - \tau) \theta_t$ with $\tau = 0.99925$.
- *Layman*: The red line is a smoothed version of the blue line. It follows the same trend but ignores short-term noise.

**Panel 2 (Different $\tau$ Values)**:
- Multiple target encoder trajectories with different momentum values overlaid.
- *Lower $\tau$ (e.g., 0.99)*: Red line follows blue more closely, reacting faster to changes but also more jittery.
- *Higher $\tau$ (e.g., 0.9999)*: Red line is very smooth but lags far behind. It takes thousands of steps to catch up.
- *V-JEPA 2's $\tau = 0.99925$*: Balances responsiveness and stability.

**Panel 3 (Lag / Gap)**:
- Shows the gap $|\theta_t - \bar{\theta}_t|$ over time for different $\tau$ values.
- *Layman*: How "out of date" the target encoder is compared to the current online encoder.
- *Key insight*: The target encoder should be slightly behind (providing stable targets) but not so far behind that it gives outdated targets.

**Dimensions**: These are scalar parameter values (one number per step). In practice, EMA is applied independently to every parameter in the network (millions of parameters)."""

# Cell 13: Training loop
explanations[13] = r"""### Reading the Output Above

**Printed Training Logs**:

Every 10 epochs, the cell prints: `Epoch XX/50 | Loss: X.XXXX`

- **Loss value at each checkpoint**:
  - *Formal*: $\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} |\hat{z}_i - z_i|$ averaged over all batches in the epoch, where $\hat{z}_i \in \mathbb{R}^{D}$ is the predictor output and $z_i$ is the target encoder output for masked patch $i$.
  - *Layman*: The average absolute difference between what the predictor guessed and what the target encoder says the correct representation should be.
  - *Dimensions*: Loss is scalar. Per batch, it is computed over $[B, N_{\text{mask}}, D] = [32, 12, 64]$ -- 32 images, 12 masked patches each, 64 dimensions per patch.

- **Expected progression**: Loss should decrease roughly monotonically:
  - Epoch 10: ~1.0--1.5 (model beginning to learn)
  - Epoch 30: ~0.5--0.8 (making progress)
  - Epoch 50: ~0.3--0.5 (converging)

- **Final loss printed**: `Final loss: X.XXXX`
  - *Good range*: 0.2--0.6 for this toy model.
  - *If > 1.0 after 50 epochs*: Model hasn't learned. Check hyperparameters.
  - *If < 0.05*: Suspiciously low. May indicate collapse (verify with the similarity matrix in later cells).

**What happens each epoch** (not printed but happening internally):
1. Data is shuffled (random permutation of 200 images).
2. Processed in batches of 32 ($\lfloor 200/32 \rfloor = 6$ batches per epoch).
3. Each batch: mask $\to$ encode $\to$ predict $\to$ loss $\to$ backprop $\to$ EMA update.
4. Epoch loss = average of 6 batch losses."""

# Cell 11: Model instantiation
explanations[11] = r"""### Reading the Output Above

**Printed Parameter Counts**:

- **Encoder params: ~6,336**: Total trainable parameters in the online encoder.
  - *Breakdown*: Linear(32 $\to$ 64) = $32 \times 64 + 64 = 2{,}112$ params. GELU has 0 params. Linear(64 $\to$ 64) = $64 \times 64 + 64 = 4{,}160$ params. LayerNorm(64) = $64 + 64 = 128$ params (scale + shift).
  - *Layman*: About 6,300 numbers that the optimizer will adjust during training.
  - *V-JEPA 2 comparison*: ViT-giant has ~1.1 billion parameters (175,000x larger).

- **Predictor params: ~5,728**: Total trainable parameters in the predictor.
  - Includes: embedding projection, mask token ($1 \times 1 \times 32 = 32$ params), positional embeddings ($1 \times 16 \times 32 = 512$ params), MLP layers, output projection.
  - *Design principle*: Predictor is intentionally smaller than encoder (32-dim vs 64-dim). This prevents the predictor from memorizing and forces the encoder to learn good representations.

- **Target encoder params: ~6,336 (frozen)**: Same architecture as the online encoder, initialized as an exact copy.
  - "Frozen" means `requires_grad = False` -- no gradients flow through it.
  - Updated only via EMA: $\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau) \theta$.
  - *Layman*: This is the "teacher" whose answers the predictor tries to match. It slowly evolves to track the "student" (online encoder).

**Dimensions reference**:
- Encoder: input $[B, N, 32] \to$ output $[B, N, 64]$
- Predictor: input $[B, N_{\text{vis}}, 64] \to$ output $[B, N_{\text{mask}}, 64]$
- Target encoder: input $[B, N, 32] \to$ output $[B, N, 64]$"""

# Cell 9: Masking demo
explanations[9] = r"""### Reading the Output Above

**Printed Masking Example**:

- **Visible patches**: e.g., `[2, 7, 11, 14]` -- the 4 patch indices (out of 16) that the encoder gets to see.
  - *Formal*: $V \subset \{0, 1, ..., N-1\}$ with $|V| = \lfloor(1 - \rho) \cdot N\rfloor = \lfloor 0.25 \times 16 \rfloor = 4$.
  - *Layman*: Out of 16 patches in the image, the encoder only sees these 4. The rest are hidden.

- **Masked patches**: e.g., `[0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15]` -- the 12 patch indices the predictor must predict.
  - *Formal*: $M = \{0, ..., N-1\} \setminus V$ with $|M| = \rho \cdot N = 0.75 \times 16 = 12$.

- **Encoder sees 4/16 = 25%**: This is $1 - \rho = 0.25$ of the patches. In V-JEPA 2, only ~10% are visible ($\rho = 0.90$).
- **Predictor must predict 12/16 = 75%**: The predictor must reconstruct representations for 75% of the image from only 25% context.

**Dimensions**:
- `visible_idx`: $[B, N_{\text{vis}}] = [2, 4]$ -- batch of 2 samples, 4 visible patches each.
- `masked_idx`: $[B, N_{\text{mask}}] = [2, 12]$ -- batch of 2 samples, 12 masked patches each.
- Note: Indices are sorted within each sample (`.sort()[0]`) for consistent processing.

**Why the mask ratio matters**: Higher masking forces the model to learn deeper structure. If only 1 patch is masked, the model can trivially interpolate from neighbors. With 75-90% masked, the model must understand the *semantics* of the image to make accurate predictions."""

# Cell 7: Data creation
explanations[7] = r"""### Reading the Output Above

**Printed Dataset Info**:

- **Dataset shape: [200, 16, 32]**: A tensor containing 200 synthetic "images".
  - *Formal*: $X \in \mathbb{R}^{[N_{\text{images}}, N_{\text{patches}}, D_{\text{patch}}]}$
  - *Dimension breakdown*:
    - $200$ = number of images (our tiny training set)
    - $16$ = number of patches per image (like a $4 \times 4$ grid from a $64 \times 64$ image with patch_size $= 16$)
    - $32$ = patch dimension (each patch is a 32-dimensional vector)
  - *Layman*: 200 images, each cut into 16 pieces, each piece described by 32 numbers.

- **Data generation process**: Each image has a random "base pattern" ($\mathbb{R}^{32}$, scaled by 2) that is shared across all 16 patches, plus per-patch noise (scaled by 0.5). This simulates real images where nearby patches are correlated (they share the same object, lighting, etc.).
  - *Formal*: $x_{i,n} = b_i + \epsilon_{i,n}$ where $b_i \sim \mathcal{N}(0, 4I)$ and $\epsilon_{i,n} \sim \mathcal{N}(0, 0.25I)$.
  - *Layman*: Each image has a "theme" (the base pattern) and each patch is a slightly different version of that theme.

**Why structured data matters**: If patches were independent random noise, there would be no correlation between visible and masked patches -- the prediction task would be impossible. The shared base pattern ensures that seeing some patches tells you something about the others, making JEPA's prediction task meaningful."""

# Cell 5: Imports
explanations[5] = r"""### Reading the Output Above

**Printed Environment Info**:

- **PyTorch version**: The version of PyTorch installed (e.g., `2.x.x`). Any version >= 2.0 works for this notebook.
- **Device: CPU**: We are running on CPU (not GPU). This is fine for the toy model -- training takes seconds, not hours.
  - In production V-JEPA 2 training, you would use multiple GPUs (e.g., 64 A100s for 4 days).
  - Our toy model has ~6K parameters vs V-JEPA 2's ~1.1B, so CPU is sufficient.

**Libraries loaded**:
- `torch`: Core tensor operations and autograd (automatic differentiation).
- `torch.nn`: Neural network layers (Linear, LayerNorm, GELU, etc.).
- `torch.nn.functional as F`: Functional versions of operations (e.g., `F.layer_norm`).
- `matplotlib.pyplot`: Plotting library for all visualizations in this notebook.
- `copy`: For `copy.deepcopy()` -- creating the target encoder as an exact copy of the online encoder.

**`torch.manual_seed(42)`**: Sets the random number generator to a fixed seed. This ensures that every time you run this notebook, you get exactly the same results. Without this, random initialization and masking would produce different numbers each run."""

# Cell 3: Matrix dimension tracking
explanations[3] = r"""### Reading the Output Above

**Printed Table -- JEPA Forward Pass Dimension Tracking**:

This table traces every tensor shape through V-JEPA 2's full pipeline using real architecture values (ViT-giant).

**Key dimensions** ($B=2$, $D=1408$, $N_{\text{heads}}=22$):

1. **Input Video** $[2, 3, 16, 256, 256]$: Batch of 2 RGB videos, 16 frames, $256 \times 256$ pixels.
2. **PatchEmbed3D** $[2, 1408, 8, 16, 16]$: Conv3D with kernel $(2, 16, 16)$ converts pixels to $D=1408$-dim patch tokens. Temporal resolution halved ($16 \to 8$), spatial by $16\times$ ($256 \to 16$).
3. **Flatten** $[2, 2048, 1408]$: $N = 8 \times 16 \times 16 = 2048$ total patches, each $1408$-dim.
4. **Masking (90%)** $[2, 205, 1408]$ visible, $[2, 1843, 1408]$ masked: Only 205 patches (10%) go to the encoder. The other 1843 are hidden.
5. **QKV projection**: Each Q, K, V $\in \mathbb{R}^{[2, 22, 205, 64]}$ -- 22 attention heads, each with $d_k = 1408/22 = 64$.
6. **Attention matrix** $[2, 22, 205, 205]$: Each of the 22 heads computes a $205 \times 205$ attention map (which visible patch attends to which).
7. **Predictor**: Projects from $D=1408$ to $D_{\text{pred}}=384$, processes all $N=2048$ tokens (visible + mask tokens), then projects predictions back to $D=1408$.
8. **Loss**: Scalar value computed over $[2, 1843, 1408] = 5{,}192{,}608$ individual comparisons per batch.

**Why these numbers matter**:
- Memory scales as $O(N^2 \cdot H)$ for attention: $205^2 \times 22 \approx 925{,}000$ entries per sample.
- The predictor uses smaller dimension ($384$ vs $1408$) to save compute.
- 90% masking reduces encoder cost by $10\times$ (only 205 tokens instead of 2048)."""

print(f"All explanation cells defined. Total: {len(explanations)}")
print(f"Cell indices: {sorted(explanations.keys())}")

# Now insert them from bottom to top
for idx in sorted(explanations.keys(), reverse=True):
    md_cell = make_md_cell(explanations[idx])
    nb['cells'].insert(idx + 1, md_cell)

# Save
with open('D:/Work/JEPAstudy/notebooks/01_JEPA_Fundamentals.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done! New total cells: {len(nb['cells'])}")
