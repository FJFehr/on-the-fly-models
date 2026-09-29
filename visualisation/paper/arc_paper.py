"""Paper-quality ARC1D rendering: rounded cells, muted rainbow palette.

Kept separate from arc.py, whose ``draw_sequence``/``render_*`` functions are
shared with W&B logging during training and use the harsh, high-contrast
standard ARC-AGI palette on purpose (fast to read on a dashboard). This
module targets LaTeX figures instead: softer, dustier colours and flush,
rounded "pill" cells meant to sit directly on a white page.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from visualisation.core.style import format_task_category


def shorten_task_label(label: str) -> str:
    """Drop the word 'Pixel(s)' and abbreviate 'Multicolor' to 'MC'.

    Kept in sync by hand with the identically-named helper in
    experiments/01_multitask_capacity/plot_per_task.py so task titles
    read the same way across the paper's figures.
    """
    label = label.replace(" Pixels", "").replace(" Pixel", "")
    return label.replace("Multicolor", "MC")


# Marks a composition in a compositional-task title (see visualisation.core.style's
# TASK_CATEGORY_DISPLAY_NAMES) -- mathtext, not the plain unicode "∘" (missing
# from Nimbus Roman, silently dropped by matplotlib).
COMPOSITION_MARK = r"$\circ$"
# How much bigger than the surrounding title text the composition mark reads
# -- plain mathtext has no size commands (\Large/\displaystyle both raise
# ParseFatalException on this matplotlib version), so getting it bigger than
# the title's own fontsize means rendering it as its own Text artist instead
# of embedding it in the title string. 1.3x reads as a deliberate accent
# without ballooning into a stray "O" (\bigcirc, tried and rejected -- looks
# like a letter, not an operator).
COMPOSITION_MARK_SCALE = 1.3


def draw_task_title(fig: plt.Figure, title: str, fontsize: int, color: str, y: float = 0.98) -> None:
    """Draw a task-figure title, enlarging the "$\\circ$" composition mark
    (if present) relative to the rest of the text.

    Falls back to a plain ``fig.suptitle`` when the title has no composition
    mark -- i.e. every one of the original 14 single-step task categories,
    unchanged from before this function existed. Only the 10 compositional
    categories (data_modules/arc1d_compositional.py) hit the multi-artist
    path below.
    """
    if COMPOSITION_MARK not in title:
        fig.suptitle(title, fontsize=fontsize, fontweight="bold", color=color, y=y)
        return

    left_text, _, right_text = title.partition(COMPOSITION_MARK)
    left_text = left_text.rstrip()
    right_text = right_text.lstrip()
    parts = [
        (left_text, fontsize),
        (COMPOSITION_MARK, fontsize * COMPOSITION_MARK_SCALE),
        (right_text, fontsize),
    ]

    # No mathtext size commands available, so measure each fragment's
    # rendered width (as a fraction of figure width) with invisible probe
    # artists, then lay the three real artists out centred as one unit --
    # the same visual effect as a single suptitle, just piecewise-sized.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = []
    for text, size in parts:
        probe = fig.text(0, 0, text, fontsize=size, fontweight="bold", alpha=0)
        fig.canvas.draw()
        widths.append(probe.get_window_extent(renderer=renderer).width / fig.bbox.width)
        probe.remove()

    # Top-aligning all three artists at the same y left the larger circle
    # hanging noticeably lower than "Fill"/"Mirror" -- its taller glyph box
    # extends further below y, while the top edge stays pinned. Fix: find the
    # vertical centre of the (unscaled) left_text's own box at this y, then
    # centre every artist -- circle included -- on that same line, so the
    # circle sits optically level with the surrounding text no matter its
    # scale, while the block's top edge still lands at y like a plain suptitle.
    probe = fig.text(0.5, y, left_text, fontsize=fontsize, fontweight="bold", alpha=0, va="top")
    fig.canvas.draw()
    bbox = probe.get_window_extent(renderer=renderer)
    probe.remove()
    center_y = (bbox.y0 + bbox.y1) / 2 / fig.bbox.height

    gap = 0.008
    x = 0.5 - (sum(widths) + gap * (len(parts) - 1)) / 2
    for (text, size), width in zip(parts, widths, strict=True):
        fig.text(x + width / 2, center_y, text, fontsize=size, fontweight="bold", color=color, ha="center", va="center")
        x += width + gap


def lighten(hex_color: str, amount: float = 0.45) -> str:
    """Blend a hex color toward white.

    Kept in sync by hand with the identically-named helper in
    experiments/01_multitask_capacity/plot_per_task.py, which uses it
    for the same outline-colour / lighter-fill bar styling this module
    mirrors for pill cells.
    """
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# Dull, mysterious rainbow: 9 muted hues swept red -> orange -> gold -> olive
# -> teal -> slate blue -> indigo -> purple -> plum. A neutral khaki/tan is
# the null / background value (0), sitting on a white/transparent canvas.
# Values are hardcoded (rather than generated at import time) so the palette
# is stable to read off a colour picker and easy to nudge by hand. Each cell
# is drawn outline-in-full-colour / lighter-fill-inside, the same treatment
# as the per-task bar plot's COLORS/FILL_COLORS -- PAPER_FILL_COLORS below is
# the lightened counterpart. The null value goes through the same
# outline/fill treatment as every other value, just with a desaturated hue,
# so it reads as "no colour" rather than another rainbow entry.
PAPER_COLORS: dict[int, str] = {
    0: "#CDBFA3",  # light khaki/tan -- null / background value
    1: "#B2524A",  # brick red
    2: "#C97A3D",  # burnt orange
    3: "#C9A227",  # gold
    4: "#7C8F4A",  # olive
    5: "#2F8F82",  # teal
    6: "#4C6E96",  # slate blue
    7: "#4B4B80",  # indigo
    8: "#6B4C93",  # purple
    9: "#8C4A73",  # plum
}

PAPER_FILL_COLORS: dict[int, str] = {value: lighten(hex_color) for value, hex_color in PAPER_COLORS.items()}

CELL_LINEWIDTH = 1.6

PAPER_BG = "#FFFFFF"  # canvas background -- white (saved as transparent)
PAPER_LABEL_COLOR = "#3A3730"
CELL_GAP = 0.05  # a small gap between cells -- distinct pills, not fused runs
ROUNDING = 0.3  # absolute radius (cell size is 1x1) -- pill-like end caps

# Panel/title text sizes. Deliberately much larger than visualisation.core.style's
# FONT_SIZES: these figures render at a large native size (PANEL_WIDTH is
# fixed in inches, independent of a LaTeX column width) and get shrunk a lot
# on the page, so labels need to be oversized to still read at print size.
PAPER_FONT_SIZES = {
    "title": 34 * 4,
    "panel_label": 26 * 4,
    "mask_glyph": 40,  # not scaled with the labels (was too big at 4x), just bumped up
}

# Fixed per-panel width (inches), independent of sequence length -- ARC1D
# sequence lengths range from ~7 to ~93 depending on task category, and a
# width that scaled with length used to make every rendered figure a
# different total size. Fixing it means every example, regardless of task
# category, drops into a LaTeX grid at the same declared width; cell size
# adapts to fit instead.
#
# CELL_TARGET is the actual lever for "how big are the pixels": each row is
# an equal-aspect box, so a cell's size is min(PANEL_WIDTH / n, row height).
# Sizing PANEL_WIDTH off CELL_TARGET at REFERENCE_LENGTH and then sizing the
# row height to (just barely more than) CELL_TARGET means every sequence up
# to REFERENCE_LENGTH cells long renders at the *same* large cell size --
# height-bound, capped at CELL_TARGET -- instead of the row height merely
# adding blank padding around a smaller, width-bound cell. Longer sequences
# still shrink to fit PANEL_WIDTH, same as before. Simply scaling PANEL_WIDTH
# up on its own doesn't make pixels look any larger once the whole canvas is
# normalized to a display/column width -- the fix has to come from shrinking
# the *unused* space around each cell.
#
# REFERENCE_LENGTH is set to 33: the longest sequence among the 14 task
# categories the multitask experiment (and hence the paper) actually uses --
# see PAPER_TASK_CATEGORIES in plot_paper_tasks.py. Not 32 (the modal
# length): with only these 14 in play there's no reason to leave any of them
# width-bound, so the reference is set to whatever the longest one needs.
CELL_TARGET = 1.1  # inches, at the reference length
REFERENCE_LENGTH = 33
PANEL_WIDTH = REFERENCE_LENGTH * CELL_TARGET

# The arrow's own gridspec column, sized off CELL_TARGET rather than a bare
# constant so it stays proportionate if cell size changes. It used to be a
# bare 0.4in, which left barely any room for the arrowhead itself -- growing
# mutation_scale alone couldn't make the arrow read as bigger once it was
# already filling that column.
ARROW_COLUMN_WIDTH = CELL_TARGET * 1.4


def _finish_axes(ax: plt.Axes, n: int, anchor: str) -> None:
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    # A sequence shorter than the panel's reference length doesn't fill the
    # panel's equal-aspect box; the default anchor ('C') would then center
    # it, leaving a bigger gap to the arrow for short sequences than long
    # ones. Anchoring "In" panels East and "Out" panels West instead keeps
    # the cell-to-arrow gap constant regardless of sequence length.
    ax.set_anchor(anchor)
    ax.set_facecolor(PAPER_BG)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_cells(ax: plt.Axes, sequence: list[int], anchor: str) -> None:
    half = (1.0 - CELL_GAP) / 2
    for index, value in enumerate(sequence):
        ax.add_patch(
            FancyBboxPatch(
                (index - half, -half),
                2 * half,
                2 * half,
                boxstyle=f"round,pad=0,rounding_size={ROUNDING}",
                linewidth=CELL_LINEWIDTH,
                edgecolor=PAPER_COLORS[value],
                facecolor=PAPER_FILL_COLORS[value],
            )
        )
    _finish_axes(ax, len(sequence), anchor)


def draw_sequence_rounded(
    ax: plt.Axes, sequence: list[int], label: str | None = None, anchor: str = "C"
) -> None:
    """Draw one ARC1D sequence as a flush row of rounded, pill-like cells."""
    _draw_cells(ax, sequence, anchor)
    if label:
        ax.set_title(
            label,
            fontsize=PAPER_FONT_SIZES["panel_label"],
            color=PAPER_LABEL_COLOR,
            pad=40,
            # Latin Modern (a Computer Modern clone) bold italic, via the
            # "custom" mathtext fontset set up by use_label_fonts().
            math_fontfamily="custom",
        )


def draw_masked_sequence_rounded(
    ax: plt.Axes, sequence_length: int, label: str | None = None, anchor: str = "C"
) -> None:
    """Draw a row of empty rounded cells with a '?' -- the masked query output."""
    half = (1.0 - CELL_GAP) / 2
    for index in range(sequence_length):
        ax.add_patch(
            FancyBboxPatch(
                (index - half, -half),
                2 * half,
                2 * half,
                boxstyle=f"round,pad=0,rounding_size={ROUNDING}",
                linewidth=CELL_LINEWIDTH,
                edgecolor="#000000",
                facecolor=PAPER_BG,
            )
        )
        ax.text(
            index, 0, "?",
            ha="center", va="center",
            fontsize=PAPER_FONT_SIZES["mask_glyph"], fontweight="bold",
            color="#000000",
        )
    _finish_axes(ax, sequence_length, anchor)
    if label:
        ax.set_title(
            label,
            fontsize=PAPER_FONT_SIZES["panel_label"],
            color=PAPER_LABEL_COLOR,
            pad=40,
            # Latin Modern (a Computer Modern clone) bold italic, via the
            # "custom" mathtext fontset set up by use_label_fonts().
            math_fontfamily="custom",
        )


def draw_io_arrow_rounded(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.annotate(
        "",
        xy=(0.86, 0.5),
        xytext=(0.14, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 70,
            "linewidth": 5.5,
            "color": "#000000",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


FONTS_DIR = Path(__file__).parent / "fonts"


def use_label_fonts() -> None:
    """Point the "custom" mathtext fontset at the bundled Latin Modern fonts.

    The x/y panel labels opt into this fontset per-text, so everything else
    keeps whatever mathtext.fontset the caller configured. Call from a
    script's main(), not at import, like apply_latex_style().
    """
    for name in ("lmroman10-regular.otf", "lmroman10-bolditalic.otf"):
        font_manager.fontManager.addfont(FONTS_DIR / name)
    plt.rcParams.update(
        {
            "mathtext.rm": "Latin Modern Roman",
            "mathtext.it": "Latin Modern Roman:bold:italic",
            "mathtext.bf": "Latin Modern Roman:bold:italic",
        }
    )


def render_task_figure_paper(
    task: dict, num_support: int | None = None, show_title: bool = True
) -> plt.Figure:
    """Render a task's support examples + masked query as rounded, muted panels.

    ``num_support`` optionally truncates the number of support rows shown
    (useful for a compact paper figure); defaults to all support pairs.
    ``show_title=False`` drops the task-name title and the space reserved for it.
    """
    support_inputs = task["support_inputs"]
    support_outputs = task["support_outputs"]
    query_input = task["query_input"]
    if num_support is not None:
        support_inputs = support_inputs[:num_support]
        support_outputs = support_outputs[:num_support]
    sequence_length = len(query_input)

    num_support_rows = len(support_inputs)
    grid_rows = num_support_rows + 1
    # Row height is just barely more than CELL_TARGET -- tight enough that
    # there's no dead vertical space around the cells inside each row's box
    # (that dead space is what made earlier, width-only scaling look like it
    # did nothing). The label text itself draws above the axes box via its
    # title pad, into the hspace gap, not into this height.
    row_height = CELL_TARGET * 1.08
    hspace = 2.0  # just enough room for the panel label between rows
    # Without a title, only the first row's panel label needs room up top.
    top_margin, bottom_margin = (0.72 if show_title else 0.87), 0.03
    # subplots_adjust below squeezes the grid into [bottom_margin, top_margin]
    # of the figure (to leave room for the suptitle) -- so a fig_height sized
    # for row_height at 100% usage actually gives each row only
    # row_height * (top_margin - bottom_margin) once that squeeze is
    # applied, leaving cells height-bound well short of CELL_TARGET and a
    # wide blank margin down both sides (anchor E/W then pushes the
    # undersized cells to one edge). Divide it back out so the *post-margin*
    # row height is the one that actually equals row_height.
    fig_height = row_height * (grid_rows + (grid_rows - 1) * hspace) / (top_margin - bottom_margin)
    fig = plt.figure(
        figsize=(PANEL_WIDTH * 2 + ARROW_COLUMN_WIDTH, fig_height),
        facecolor=PAPER_BG,
    )
    grid = fig.add_gridspec(
        grid_rows,
        3,
        width_ratios=[PANEL_WIDTH, ARROW_COLUMN_WIDTH, PANEL_WIDTH],
        height_ratios=[1] * num_support_rows + [1],
        hspace=hspace,
        wspace=0.02,
    )

    for row_index, (support_input, support_output) in enumerate(
        zip(support_inputs, support_outputs, strict=True)
    ):
        ax_in = fig.add_subplot(grid[row_index, 0])
        ax_arrow = fig.add_subplot(grid[row_index, 1])
        ax_out = fig.add_subplot(grid[row_index, 2])
        draw_sequence_rounded(ax_in, support_input, rf"$\mathit{{x}}_{{\mathrm{{{row_index + 1}}}}}$", anchor="E")
        draw_sequence_rounded(ax_out, support_output, rf"$\mathit{{y}}_{{\mathrm{{{row_index + 1}}}}}$", anchor="W")
        draw_io_arrow_rounded(ax_arrow)

    ax_in = fig.add_subplot(grid[-1, 0])
    ax_arrow = fig.add_subplot(grid[-1, 1])
    ax_out = fig.add_subplot(grid[-1, 2])
    draw_sequence_rounded(ax_in, query_input, r"$\mathit{x}'$", anchor="E")
    draw_masked_sequence_rounded(ax_out, sequence_length, r"$\hat{\mathit{y}}'$", anchor="W")
    draw_io_arrow_rounded(ax_arrow)

    if show_title:
        task_title = shorten_task_label(format_task_category(task["task_category"]))
        draw_task_title(fig, task_title, PAPER_FONT_SIZES["title"], PAPER_LABEL_COLOR, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.99, top=top_margin, bottom=bottom_margin)
    return fig
