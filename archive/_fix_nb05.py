"""Fix gaps in NB05 (05_ICRA_Research_Gaps.ipynb) by editing notebook JSON directly."""

import json
import re
import sys

NB_PATH = r"D:\Work\JEPAstudy\notebooks\05_ICRA_Research_Gaps.ipynb"

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

def cell_source(cell):
    """Return cell source as a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src

def replace_in_cell(cell, old, new, label=""):
    """Replace old with new in cell source lines. Returns True if replaced."""
    src = cell.get("source", [])
    if isinstance(src, str):
        src = [src]
    joined = "".join(src)
    if old not in joined:
        return False
    joined = joined.replace(old, new)
    cell["source"] = joined.splitlines(keepends=True)
    # Ensure last line has content
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        pass  # last line without newline is fine
    print(f"  [REPLACED] {label}")
    print(f"    OLD: {old[:120]}...")
    print(f"    NEW: {new[:120]}...")
    return True

def append_to_cell(cell, text, label=""):
    """Append text to end of cell source."""
    src = cell.get("source", [])
    if isinstance(src, str):
        src = [src]
    joined = "".join(src)
    if not joined.endswith("\n"):
        joined += "\n"
    joined += text
    cell["source"] = joined.splitlines(keepends=True)
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        pass
    print(f"  [APPENDED] {label}")
    print(f"    TEXT: {text[:150]}...")
    return True

def insert_cell_after(nb, idx, cell_type, source, label=""):
    """Insert a new cell after index idx."""
    new_cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }
    if cell_type == "code":
        new_cell["execution_count"] = None
        new_cell["outputs"] = []
    nb["cells"].insert(idx + 1, new_cell)
    print(f"  [INSERTED] New {cell_type} cell after index {idx}: {label}")
    return True

changes = 0

nb = load_nb(NB_PATH)
cells = nb["cells"]

# ============================================================
# 1. Fix wrong architecture description
# ============================================================
print("\n=== Fix 1: Wrong MLP architecture ===")
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "3-layer MLP (1408 -> 256 -> 128 -> 1)" in src:
        replace_in_cell(
            cell,
            "3-layer MLP (1408 -> 256 -> 128 -> 1)",
            "3-layer MLP with latent_dim=2 in demo (2 -> 256 -> 256 -> 1). In real use: 1408 -> 256 -> 256 -> 1",
            "Fix MLP architecture to match code"
        )
        changes += 1
        break

# ============================================================
# 2. Fix "4 plots" -> "3 subplots"
# ============================================================
print("\n=== Fix 2: 4 plots -> 3 subplots ===")
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "What the 4 plots show:" in src:
        replace_in_cell(
            cell,
            "What the 4 plots show:",
            "What the 3 subplots show:",
            "Fix plot count"
        )
        changes += 1
        break

# ============================================================
# 3. Fix CBF formula: ||x-o||^2 - r^2 -> ||x-o|| - r - margin
#    (where the code uses L2 norm, not squared)
#    Note: There are multiple CBF formulas. The navigation demo code
#    uses np.linalg.norm - r - margin, but the markdown claims ||x-o||^2 - r^2.
#    Fix the three markdown formulas that mismatch the navigation code.
# ============================================================
print("\n=== Fix 3: CBF formula to match navigation code ===")
cbf_formula_fixes = [
    (
        r"$h(z) = \|z - z_{\text{obstacle}}\|^2 - r^2$ (stay away from obstacle representation)",
        r"$h(z) = \|z - z_{\text{obstacle}}\| - r - \text{margin}$ (stay away from obstacle representation). Note: uses L2 norm (not squared) to match the navigation demo code",
    ),
    (
        r"2. **Define safe set in latent space:** $h(z) = \|z - z_{\text{obstacle}}\|^2 - r^2$",
        r"2. **Define safe set in latent space:** $h(z) = \|z - z_{\text{obstacle}}\| - r - \text{margin}$ (L2 norm, not squared, matching the navigation demo code)",
    ),
    (
        r"where $h_i(x) = \|x - o_i\|^2 - r_i^2$ is the CBF for obstacle $i$ (positive = outside obstacle, negative = inside).",
        r"where $h_i(x) = \|x - o_i\| - r_i - \text{margin}$ is the CBF for obstacle $i$ (positive = outside obstacle, negative = inside). This uses L2 norm (not squared) to match the `cbf_h()` code above.",
    ),
]
for old_f, new_f in cbf_formula_fixes:
    for i, cell in enumerate(cells):
        src = cell_source(cell)
        if old_f in src:
            replace_in_cell(cell, old_f, new_f, "Fix CBF formula")
            changes += 1
            break

# ============================================================
# 4. Add alpha selection guide near alpha usage
# ============================================================
print("\n=== Fix 4: Alpha selection guide ===")
alpha_guide = (
    "\n> **Alpha selection guide:** `alpha=0.1` = gentle safety (far from obstacles), "
    "`alpha=1.0` = moderate, `alpha=3.0` = aggressive (close to obstacles). "
    "Higher alpha = more conservative but jerkier motion.\n"
)
# Add after the first markdown cell that explains alpha
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "how aggressively to enforce safety (higher = more conservative)" in src and cell["cell_type"] == "markdown":
        if "Alpha selection guide" not in src:
            append_to_cell(cell, alpha_guide, "Add alpha selection guide")
            changes += 1
        break

# ============================================================
# 5. Add definitions for key terms
# ============================================================
print("\n=== Fix 5: Add definitions cell ===")
definitions_text = """## Key Definitions (Quick Reference)

| Term | Definition |
|------|-----------|
| **CBF** (Control Barrier Function) | A scalar function $h(x)$ that defines a safe set ($h(x) \\geq 0$ = safe). The CBF constraint ensures the system never leaves this safe set. |
| **QP** (Quadratic Program) | An optimization problem with a quadratic objective and linear constraints. Used here to find the closest safe action to the desired action. |
| **Class-$\\mathcal{K}$ function** | A continuous, strictly increasing function $\\alpha: [0,\\infty) \\to [0,\\infty)$ with $\\alpha(0)=0$. Controls how aggressively safety is enforced. Linear example: $\\alpha(h) = c \\cdot h$. |
| **Lie derivative** | $L_f h(x) = \\nabla h(x) \\cdot f(x)$: the rate of change of $h$ along the vector field $f$. Tells you how the safety value changes under the system dynamics. |
| **p-value** | The probability of observing results at least as extreme as yours, assuming the null hypothesis (no real difference) is true. $p < 0.05$ is conventionally "statistically significant." |
| **Welch's t-test** | A variant of the t-test that does NOT assume equal variances between groups. More robust than the standard t-test for comparing two methods with different variability. |
| **LIBERO** | A benchmark suite for evaluating robot manipulation policies. Has 4 suites: Spatial (10 tasks), Object (10), Goal (10), Long (10 multi-step tasks). Standard in VLA papers. |
| **Gradient checkpointing** | A memory-saving technique that trades compute for VRAM: instead of storing all intermediate activations, recompute them during the backward pass. Reduces memory by ~60% but slows training ~20%. |
| **ROCm** | AMD's open-source GPU compute platform (equivalent to NVIDIA's CUDA). Required for running PyTorch on AMD GPUs like the RX 6700S. |
| **Frozen encoder** | A pretrained model whose weights are NOT updated during training. Saves compute and memory. V-JEPA 2 encoder is typically frozen when used as a feature extractor. |
"""
# Find the first cell with "Landscape Map" and insert definitions after the 5W+H section
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "0.5 The 5W+H of JEPA" in src:
        # Insert after this cell
        insert_cell_after(nb, i, "markdown", definitions_text, "Key definitions table")
        cells = nb["cells"]  # refresh reference
        changes += 1
        break

# ============================================================
# 6. Add units note to CBF demo cells
# ============================================================
print("\n=== Fix 6: Add units note ===")
units_note = "\n> **Units note:** All positions in these demos are in arbitrary units (a.u.). For real robots, these would be in meters.\n"
# Add to the CBF navigation demo explanation cell
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "CBF-filtered trajectory" in src and "smoothly curves around obstacles" in src:
        if "arbitrary units" not in src:
            append_to_cell(cell, units_note, "Add units note")
            changes += 1
        break

# ============================================================
# 7. Add fabricated results warning
# ============================================================
print("\n=== Fix 7: Fabricated results warnings ===")

# Find cells with simulated experimental data
fabricated_patterns = [
    "Simulated ablation results",
    "Simulated success rates",
    "simulated success rates",
    "Simulated based on literature trends",
]
warning_text = "\n> **Warning:** These values are simulated for illustration. Your actual experimental results will differ.\n"

warned_cells = set()
for i, cell in enumerate(cells):
    src = cell_source(cell)
    for pat in fabricated_patterns:
        if pat in src and i not in warned_cells and "These values are simulated for illustration" not in src:
            append_to_cell(cell, warning_text, f"Add fabricated results warning (pattern: '{pat}')")
            warned_cells.add(i)
            changes += 1
            break

# Also find the ablation table with specific numbers
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "73.4%" in src and "Full Model (Ours)" in src and i not in warned_cells:
        if "These values are simulated for illustration" not in src:
            append_to_cell(cell, warning_text, "Add fabricated results warning (ablation table)")
            warned_cells.add(i)
            changes += 1

# Find "success rate" simulation cells
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "base_rates" in src and "Simulated" in src and i not in warned_cells:
        if "These values are simulated for illustration" not in src:
            append_to_cell(cell, warning_text, "Add fabricated results warning (base_rates)")
            warned_cells.add(i)
            changes += 1

# ============================================================
# 8. Add Bonferroni correction note
# ============================================================
print("\n=== Fix 8: Bonferroni correction note ===")
# The notebook doesn't mention Bonferroni yet, so add it near statistical testing
bonferroni_text = (
    "\n> **Bonferroni correction:** When running N statistical tests, divide your significance "
    "threshold by N (e.g., use p < 0.05/3 = 0.017 when running 3 comparisons). "
    "This prevents inflating your false positive rate when making multiple comparisons.\n"
)
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "Is that statistically significant?" in src:
        if "Bonferroni" not in src:
            append_to_cell(cell, bonferroni_text, "Add Bonferroni correction note")
            changes += 1
        break

# ============================================================
# 9. Add GPU hours caveat
# ============================================================
print("\n=== Fix 9: GPU hours caveat ===")
gpu_caveat = "\n> **Note:** These GPU hour estimates are rough approximations. Actual times depend on hardware, batch size, and implementation efficiency.\n"
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "gpu_hours" in src and "estimated GPU hours" in src:
        if "rough approximations" not in src:
            append_to_cell(cell, gpu_caveat, "Add GPU hours caveat")
            changes += 1
        break

# ============================================================
# 10. Add torch.manual_seed explanation
# ============================================================
print("\n=== Fix 10: torch.manual_seed explanation ===")
seed_explanation = "  # Fixes random number generator for reproducible results. 42 is an arbitrary conventional choice.\n"
seed_count = 0
for i, cell in enumerate(cells):
    src = cell_source(cell)
    if "torch.manual_seed(42)" in src and cell["cell_type"] == "code":
        if "reproducible results" not in src:
            replace_in_cell(
                cell,
                "torch.manual_seed(42)\n",
                "torch.manual_seed(42)" + seed_explanation,
                f"Add seed explanation (cell {i})"
            )
            seed_count += 1
            changes += 1

print(f"\n  Added seed explanation to {seed_count} cells")

# ============================================================
# Save
# ============================================================
save_nb(nb, NB_PATH)
print(f"\n{'='*60}")
print(f"DONE. Total changes: {changes}")
print(f"Saved to: {NB_PATH}")
