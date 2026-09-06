"""Centralised visual style constants for all plots in this project.

Import constants directly; call ``apply_latex_style()`` explicitly in plot
scripts to avoid matplotlib side-effects when this module is imported during
training.
"""
import re

# ── Model palette ─────────────────────────────────────────────────────────────
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "rnn": "RNN",
    "cnn": "CNN",
    "transformer": "Transformer",
    "mlp": "MLP",
    "hypermodel": "Hypermodel",
}

MODEL_COLORS: dict[str, str] = {
    "rnn": "#5DA5DA",         # Muted blue   — simple / sequential
    "cnn": "#C49A6C",         # Warm brown   — structured / spatial
    "transformer": "#B276B2", # Purple       — complex / expressive
    "mlp": "#9D9D9D",         # Grey         — baseline
    "hypermodel": "#2ECC71",  # Bright green — hypermodel
}

# ── Task category labels ───────────────────────────────────────────────────────
TASK_CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "1d_move_1p": "Move 1 Pixel",
    "1d_move_2p": "Move 2 Pixels",
    "1d_move_3p": "Move 3 Pixels",
    "1d_move_dp": "Move Dynamic",
    "1d_move_2p_dp": "Move 2 Pixels Towards",
    "1d_fill": "Fill",
    "1d_padded_fill": "Padded Fill",
    "1d_hollow": "Hollow",
    "1d_flip": "Flip",
    "1d_mirror": "Mirror",
    "1d_denoising_1c": "Denoise",
    "1d_denoising_mc": "Denoise Multicolor",
    "1d_pcopy_1c": "Pattern Copy",
    "1d_pcopy_mc": "Pattern Copy Multicolor",
    "1d_recolor_oe": "Recolor by Odd Even",
    "1d_recolor_cnt": "Recolor by Size",
    "1d_recolor_cmp": "Recolor by Size Comparison",
    "1d_scale_dp": "Scaling",
    # Compositional generalization held-out categories (data_modules/arc1d_compositional.py):
    # each chains two of the single-task rules above in sequence (stage 1 then stage 2, left to
    # right). "$\circ$" (mathtext, not the plain unicode "∘") marks that composition -- the bare
    # unicode glyph isn't in Nimbus Roman and was silently dropped by matplotlib; routing it
    # through mathtext (still available even with font.serif overridden to Nimbus Roman, since
    # mathtext.fontset stays "cm") renders it correctly.
    "1d_comp_denoise1c_shift3": r"Denoise $\circ$ Shift 3",
    "1d_comp_fill_mirror": r"Fill $\circ$ Mirror",
    "1d_comp_fill_shift3": r"Fill $\circ$ Shift 3",
    "1d_comp_fill_movedynamic": r"Fill $\circ$ Move Dynamic",
    "1d_comp_hollow_shift3": r"Hollow $\circ$ Shift 3",
    "1d_comp_denoisemc_copy": r"Denoise Multicolor $\circ$ Pattern Copy",
    "1d_comp_denoisemc_denoise1c": r"Denoise Multicolor $\circ$ Denoise",
    "1d_comp_movedynamic_hollow": r"Move Dynamic $\circ$ Hollow",
    "1d_comp_shift3_copy": r"Shift 3 $\circ$ Pattern Copy",
    "1d_comp_denoisemc_mirror": r"Denoise Multicolor $\circ$ Mirror",
}

def normalize_task_category(category: str) -> str:
    """Return a canonical task-category key used by plot label lookups."""
    # Strip any variant prefix (e.g. mc_, var_mc_small_, var_mc_medium_, var_mc_large_)
    return re.sub(r"^(?:var_mc_(?:small|medium|large)_|mc_)", "", category)


def format_task_category(category: str) -> str:
    """Return a human-readable label for a task category key."""
    normalized = normalize_task_category(category)
    return TASK_CATEGORY_DISPLAY_NAMES.get(normalized, normalized)


# ── Font sizes (single-column LaTeX figure, ~3.5 in wide) ─────────────────────
FONT_SIZES: dict[str, int] = {
    "title": 11,
    "label": 10,
    "tick": 9,
    "legend": 9,
    "annotation": 8,
    "panel_label": 8,
}


# ── LaTeX-ready rcParams (explicit opt-in, not set on import) ─────────────────
def apply_latex_style() -> None:
    """Apply LaTeX-compatible matplotlib rcParams.

    Call this at the top of ``main()`` in plot scripts. Do NOT call it from
    library code — importing this module must have zero side-effects on the
    matplotlib state.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
