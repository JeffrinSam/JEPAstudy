"""
Regenerate 3 GIFs with smoother animations (more frames, better durations)
and copy all PNG files from notebooks/gifs/ to notebooks/plots/.
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from io import BytesIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIF_DIR = os.path.join(SCRIPT_DIR, "gifs")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")


def fig_to_pil(fig, dpi=120):
    """Render a matplotlib figure to a PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img


def save_gif(frames, path, duration):
    """Save a list of PIL images as an animated GIF."""
    # Convert RGBA to RGB with white background for GIF compatibility
    rgb_frames = []
    for f in frames:
        bg = Image.new("RGB", f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3])
        rgb_frames.append(bg)
    rgb_frames[0].save(
        path, save_all=True, append_images=rgb_frames[1:],
        duration=duration, loop=0
    )
    print(f"  Saved {path}  ({len(rgb_frames)} frames, {duration}ms each)")


# ─────────────────────────────────────────────────────────────────────────────
# 1) 01_gradient_backprop.gif  (15 frames, 300ms)
#    Show 4 layers; gradient signal flows from output back to input.
#    Smooth transition: gradient "fills" each layer progressively.
# ─────────────────────────────────────────────────────────────────────────────
def make_gradient_backprop():
    print("Generating 01_gradient_backprop.gif ...")
    n_layers = 4
    layer_labels = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
    n_frames = 15

    # Positions for layers (left to right)
    x_positions = np.linspace(0.12, 0.88, n_layers)
    layer_width = 0.12
    layer_height = 0.55

    frames = []
    for frame_idx in range(n_frames):
        progress = frame_idx / (n_frames - 1)  # 0 -> 1

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Backpropagation: Gradient Flow", fontsize=14, fontweight="bold", pad=10)

        # Gradient flows right-to-left (from output to input)
        # progress=0: nothing lit; progress=1: all layers lit
        # Each layer lights up in sequence: output first, then hidden2, etc.
        for i, (x, label) in enumerate(zip(x_positions, layer_labels)):
            rev_i = n_layers - 1 - i  # reversed index (output=0, input=3)
            # How "reached" is this layer by the gradient?
            layer_progress = np.clip((progress * n_layers) - rev_i, 0, 1)

            # Base color: light gray; Gradient color: red-orange gradient
            if layer_progress <= 0:
                color = "#E0E0E0"
                edge_color = "#999999"
                text_color = "#555555"
            else:
                # Interpolate from light orange to deep red
                r = 1.0
                g = 0.85 * (1 - layer_progress) + 0.15 * layer_progress
                b = 0.7 * (1 - layer_progress)
                color = (r, g, b)
                edge_color = "#CC3300"
                text_color = "#222222"

            rect = mpatches.FancyBboxPatch(
                (x - layer_width / 2, 0.5 - layer_height / 2),
                layer_width, layer_height,
                boxstyle="round,pad=0.015",
                facecolor=color, edgecolor=edge_color, linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x, 0.5, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

            # Draw gradient magnitude below the layer
            if layer_progress > 0:
                bar_w = layer_width * 0.8 * layer_progress
                bar_rect = mpatches.FancyBboxPatch(
                    (x - bar_w / 2, 0.08), bar_w, 0.06,
                    boxstyle="round,pad=0.005",
                    facecolor=(1.0, 0.3, 0.1, 0.7), edgecolor="none"
                )
                ax.add_patch(bar_rect)
                ax.text(x, 0.04, f"|∇|={layer_progress:.1f}", ha="center",
                        va="center", fontsize=7, color="#CC3300")

        # Draw arrows between layers (forward pass: gray; backward: colored)
        for i in range(n_layers - 1):
            x_start = x_positions[i] + layer_width / 2 + 0.01
            x_end = x_positions[i + 1] - layer_width / 2 - 0.01

            # Forward arrow (top, always visible, gray)
            ax.annotate("", xy=(x_end, 0.62), xytext=(x_start, 0.62),
                        arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.5))

            # Backward arrow (bottom, colored based on gradient progress)
            rev_i = n_layers - 2 - i  # which backward arrow
            arrow_progress = np.clip((progress * n_layers) - rev_i - 0.5, 0, 1)
            if arrow_progress > 0:
                alpha = min(arrow_progress * 1.5, 1.0)
                ax.annotate("", xy=(x_start, 0.38), xytext=(x_end, 0.38),
                            arrowprops=dict(arrowstyle="->,head_width=0.4",
                                            color=(0.9, 0.2, 0.1, alpha), lw=2.5))
                # Small "∇L" label on the arrow
                mid_x = (x_start + x_end) / 2
                ax.text(mid_x, 0.32, "∇L", ha="center", va="center",
                        fontsize=8, color=(0.8, 0.1, 0.0, alpha), fontstyle="italic")

        # Loss label at output
        loss_x = x_positions[-1] + layer_width / 2 + 0.04
        ax.text(loss_x, 0.5, "L", fontsize=16, fontweight="bold",
                color="#CC0000", ha="left", va="center")

        # Progress label
        ax.text(0.5, 0.95, f"Backward pass: {progress*100:.0f}%",
                ha="center", va="top", fontsize=10, color="#666666",
                transform=ax.transAxes)

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    save_gif(frames, os.path.join(GIF_DIR, "01_gradient_backprop.gif"), duration=300)


# ─────────────────────────────────────────────────────────────────────────────
# 2) 02_weight_init.gif  (15 frames, 400ms)
#    Show activations propagating through 12 layers for 4 init methods.
#    Each frame adds one more layer of activation distributions.
# ─────────────────────────────────────────────────────────────────────────────
def make_weight_init():
    print("Generating 02_weight_init.gif ...")
    np.random.seed(42)

    n_layers = 12
    n_frames = 15
    width = 256  # neurons per layer
    methods = {
        "Too Small (σ=0.01)": 0.01,
        "Too Large (σ=1.0)": 1.0,
        "Xavier Init": "xavier",
        "Kaiming (He) Init": "kaiming",
    }

    # Pre-compute activations for all methods and all layers
    all_activations = {}
    for name, init in methods.items():
        acts = [np.random.randn(1000)]  # input
        for layer in range(n_layers):
            x = acts[-1]
            if init == "xavier":
                scale = np.sqrt(2.0 / (width + width))
            elif init == "kaiming":
                scale = np.sqrt(2.0 / width)
            else:
                scale = init
            W = np.random.randn(width, width) * scale
            # Simulate: take random projection of the 1000 samples
            x_proj = x @ np.random.randn(len(x[0]) if x.ndim > 1 else 1, 1).flatten() \
                if x.ndim > 1 else x
            # Simpler: just multiply std and apply ReLU
            new_std = np.std(x_proj) * scale * np.sqrt(width)
            new_acts = np.random.randn(1000) * new_std
            new_acts = np.maximum(new_acts, 0)  # ReLU
            if np.std(new_acts) < 1e-10:
                new_acts = np.zeros(1000)
            acts.append(new_acts)
        all_activations[name] = acts

    # Pre-compute layer schedule: ensure each frame shows a unique n_show
    # Use linspace to get 15 distinct fractional values, each frame is unique
    layer_schedule = np.linspace(0, n_layers, n_frames + 1)[1:]  # 15 values from ~0.8 to 12

    frames = []
    for frame_idx in range(n_frames):
        # Continuous progress for smooth animation
        progress = (frame_idx + 1) / n_frames  # 0.067 to 1.0
        n_show = max(1, int(np.ceil(layer_schedule[frame_idx])))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.set_facecolor("white")
        fig.suptitle("Weight Initialization: Activation Propagation",
                     fontsize=14, fontweight="bold", y=1.02)

        colors_map = {
            "Too Small (σ=0.01)": "#3498DB",
            "Too Large (σ=1.0)": "#E74C3C",
            "Xavier Init": "#2ECC71",
            "Kaiming (He) Init": "#9B59B6",
        }

        for ax, (name, acts) in zip(axes, all_activations.items()):
            color = colors_map[name]
            stds = [np.std(a) if np.std(a) > 0 else 1e-12 for a in acts[:n_show + 1]]

            # Animate the last bar growing in (partial fill for smooth transition)
            bar_alphas = [0.7] * len(stds)
            frac = layer_schedule[frame_idx] - int(layer_schedule[frame_idx])
            if frac > 0 and len(stds) > 1:
                bar_alphas[-1] = max(0.2, frac * 0.7)

            ax.bar(range(len(stds)), stds, color=color, alpha=0.7, edgecolor=color)

            # Shade future layers as empty
            if n_show < n_layers:
                for j in range(n_show + 1, n_layers + 1):
                    ax.bar(j, 0, color="#EEEEEE", edgecolor="#CCCCCC", linewidth=0.5)

            ax.set_title(name, fontsize=9, fontweight="bold", color=color)
            ax.set_xlabel("Layer", fontsize=8)
            ax.set_ylabel("Activation Std", fontsize=8)
            ax.set_xlim(-0.5, n_layers + 0.5)
            ax.set_xticks(range(0, n_layers + 1, 3))
            ax.tick_params(labelsize=7)

            # Set consistent y-axis per method
            if "Small" in name:
                ax.set_ylim(0, max(1.5, max(stds) * 1.2))
            elif "Large" in name:
                ax.set_ylim(0, max(stds) * 1.2 if max(stds) > 0 else 5)
            else:
                ax.set_ylim(0, max(3, max(stds) * 1.3))

            # Status annotation
            final_std = stds[-1]
            if final_std < 0.01:
                status = "Vanishing!"
                sc = "#E74C3C"
            elif final_std > 10:
                status = "Exploding!"
                sc = "#E74C3C"
            else:
                status = "Healthy"
                sc = "#2ECC71"

            if frame_idx >= n_frames // 2:
                ax.text(0.95, 0.95, status, transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, fontweight="bold",
                        color=sc, bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor="white", edgecolor=sc, alpha=0.8))

        fig.text(0.5, -0.02, f"Layer {n_show} / {n_layers}  (frame {frame_idx+1}/{n_frames})",
                 ha="center", fontsize=10, color="#666666")
        plt.tight_layout()
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    save_gif(frames, os.path.join(GIF_DIR, "02_weight_init.gif"), duration=400)


# ─────────────────────────────────────────────────────────────────────────────
# 3) 03_causal_mask.gif  (12 frames, 400ms)
#    Build the causal attention mask token by token for T=4 tokens.
#    Show the 4x4 matrix being filled in step by step.
# ─────────────────────────────────────────────────────────────────────────────
def make_causal_mask():
    print("Generating 03_causal_mask.gif ...")
    T = 4
    n_frames = 12
    token_labels = [f"t{i}" for i in range(T)]

    # Animation plan:
    # Frames 0-2:   row 0 builds (t0 can attend to t0)
    # Frames 3-5:   row 1 builds (t1 can attend to t0, t1)
    # Frames 6-8:   row 2 builds (t2 can attend to t0, t1, t2)
    # Frames 9-11:  row 3 builds (t3 can attend to t0, t1, t2, t3)
    # Within each row group:
    #   sub-frame 0: highlight the query token
    #   sub-frame 1: fill in allowed cells
    #   sub-frame 2: mark blocked cells with X

    frames = []
    # Track which cells are "revealed"
    revealed = np.zeros((T, T), dtype=int)  # 0=hidden, 1=allowed, -1=blocked

    for frame_idx in range(n_frames):
        row = frame_idx // 3
        sub = frame_idx % 3

        # Update revealed state
        if sub == 0:
            pass  # just highlighting
        elif sub == 1:
            for col in range(row + 1):
                revealed[row, col] = 1
        elif sub == 2:
            for col in range(row + 1, T):
                revealed[row, col] = -1

        fig, (ax_mask, ax_info) = plt.subplots(1, 2, figsize=(9, 5),
                                                gridspec_kw={"width_ratios": [1.2, 1]})
        fig.set_facecolor("white")
        fig.suptitle("Causal (Autoregressive) Attention Mask",
                     fontsize=14, fontweight="bold")

        # Draw the mask matrix
        for r in range(T):
            for c in range(T):
                x, y = c, T - 1 - r  # flip y so row 0 is at top

                if revealed[r, c] == 1:
                    color = "#2ECC71"  # green = allowed
                    text = "1"
                    tc = "white"
                elif revealed[r, c] == -1:
                    color = "#E74C3C"  # red = blocked
                    text = "0"
                    tc = "white"
                else:
                    color = "#F0F0F0"  # gray = not yet revealed
                    text = "?"
                    tc = "#AAAAAA"

                # Highlight current row being built
                if r == row and sub == 0 and revealed[r, c] == 0:
                    color = "#FFF3CD"  # light yellow highlight

                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="#666666", linewidth=1
                )
                ax_mask.add_patch(rect)
                ax_mask.text(x + 0.5, y + 0.5, text,
                             ha="center", va="center", fontsize=16,
                             fontweight="bold", color=tc)

        # Labels
        for i in range(T):
            ax_mask.text(i + 0.5, T + 0.15, token_labels[i],
                         ha="center", va="bottom", fontsize=11, fontweight="bold",
                         color="#3498DB" if i == row and sub == 0 else "#333333")
            ax_mask.text(-0.2, T - 1 - i + 0.5, token_labels[i],
                         ha="right", va="center", fontsize=11, fontweight="bold",
                         color="#E67E22" if i == row else "#333333")

        ax_mask.set_xlim(-0.5, T + 0.3)
        ax_mask.set_ylim(-0.3, T + 0.5)
        ax_mask.set_aspect("equal")
        ax_mask.axis("off")
        ax_mask.text(T / 2, -0.5, "Key (columns)", ha="center", fontsize=10, color="#3498DB")
        ax_mask.text(-0.6, T / 2, "Query\n(rows)", ha="center", va="center",
                     fontsize=10, color="#E67E22", rotation=90)

        # Info panel
        ax_info.axis("off")
        info_lines = []
        info_lines.append(("Causal Mask Rule:", "#333333", 13, "bold"))
        info_lines.append(("Token i can only attend to", "#555555", 10, "normal"))
        info_lines.append(("tokens j where j ≤ i", "#555555", 10, "normal"))
        info_lines.append(("", "#555555", 6, "normal"))

        if sub == 0:
            info_lines.append((f"Step: Examining query {token_labels[row]}",
                               "#E67E22", 11, "bold"))
            info_lines.append((f"{token_labels[row]} needs to decide which", "#555555", 9, "normal"))
            info_lines.append(("tokens it can attend to.", "#555555", 9, "normal"))
        elif sub == 1:
            attn = ", ".join(token_labels[:row + 1])
            info_lines.append((f"Step: {token_labels[row]} → attends to [{attn}]",
                               "#2ECC71", 11, "bold"))
            info_lines.append((f"{token_labels[row]} can see past & present.", "#555555", 9, "normal"))
        else:
            blocked = ", ".join(token_labels[row + 1:]) if row + 1 < T else "none"
            info_lines.append((f"Step: {token_labels[row]} blocks [{blocked}]",
                               "#E74C3C", 11, "bold"))
            info_lines.append(("Future tokens are masked out.", "#555555", 9, "normal"))

        info_lines.append(("", "#555555", 6, "normal"))

        # Progress
        completed_rows = row if sub < 2 else row + 1
        info_lines.append((f"Progress: {completed_rows}/{T} rows complete",
                           "#666666", 9, "normal"))

        # Legend
        info_lines.append(("", "#555555", 8, "normal"))
        info_lines.append(("Legend:", "#333333", 10, "bold"))

        y_pos = 0.92
        for text, color, size, weight in info_lines:
            ax_info.text(0.05, y_pos, text, transform=ax_info.transAxes,
                         fontsize=size, color=color, fontweight=weight, va="top")
            y_pos -= 0.07 if size >= 10 else 0.05

        # Legend boxes
        for label, color, yoff in [("Allowed (1)", "#2ECC71", 0),
                                    ("Blocked (0)", "#E74C3C", -0.06),
                                    ("Unrevealed", "#F0F0F0", -0.12)]:
            rect = mpatches.FancyBboxPatch(
                (0.05, y_pos + yoff - 0.01), 0.06, 0.04,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="#666666", linewidth=0.5,
                transform=ax_info.transAxes
            )
            ax_info.add_patch(rect)
            ax_info.text(0.15, y_pos + yoff + 0.01, label,
                         transform=ax_info.transAxes, fontsize=8,
                         color="#333333", va="center")

        plt.tight_layout()
        frames.append(fig_to_pil(fig))
        plt.close(fig)

    save_gif(frames, os.path.join(GIF_DIR, "03_causal_mask.gif"), duration=400)


# ─────────────────────────────────────────────────────────────────────────────
# Copy PNGs from gifs/ to plots/
# ─────────────────────────────────────────────────────────────────────────────
def copy_pngs():
    print("Copying PNG files from gifs/ to plots/ ...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    count = 0
    for f in os.listdir(GIF_DIR):
        if f.lower().endswith(".png"):
            src = os.path.join(GIF_DIR, f)
            dst = os.path.join(PLOT_DIR, f)
            shutil.copy2(src, dst)
            count += 1
            print(f"  Copied {f}")
    print(f"  Total: {count} PNG files copied.")


if __name__ == "__main__":
    make_gradient_backprop()
    make_weight_init()
    make_causal_mask()
    copy_pngs()
    print("\nDone!")
