"""Attention visualisation helpers for task-conditioned hypermodels."""

import matplotlib.pyplot as plt
import torch

from visualisation.style import FONT_SIZES, format_task_category


def resolve_attention_matrix(
    attention_by_head: torch.Tensor,
    reduction: str = "mean",
    head_index: int | None = None,
) -> tuple[torch.Tensor, str]:
    """Reduce or select a single attention head for plotting."""
    if attention_by_head.ndim != 3:
        msg = "Expected attention tensor with shape [heads, query_tokens, key_tokens]."
        raise ValueError(msg)

    if head_index is not None:
        if head_index < 0 or head_index >= attention_by_head.shape[0]:
            msg = (
                f"head_index must be in [0, {attention_by_head.shape[0] - 1}] "
                f"but got {head_index}."
            )
            raise ValueError(msg)
        return attention_by_head[head_index], f"head_{head_index}"

    if reduction == "mean":
        return attention_by_head.mean(dim=0), "mean"
    if reduction == "max":
        return attention_by_head.max(dim=0).values, "max"

    msg = f"Unsupported attention reduction {reduction!r}."
    raise ValueError(msg)


def build_segment_boundaries(
    token_metadata: list[dict],
) -> tuple[list[int], list[str], list[float]]:
    """Return boundary indices and center points for consecutive task segments."""
    boundaries = []
    segment_labels = []
    segment_centers = []

    start_index = 0
    current_segment = token_metadata[0]["segment_display_label"]
    for token_index, token_info in enumerate(token_metadata[1:], start=1):
        if token_info["segment_display_label"] == current_segment:
            continue
        boundaries.append(token_index)
        segment_labels.append(current_segment)
        segment_centers.append((start_index + token_index) / 2)
        start_index = token_index
        current_segment = token_info["segment_display_label"]

    segment_labels.append(current_segment)
    segment_centers.append((start_index + len(token_metadata)) / 2)
    return boundaries, segment_labels, segment_centers


def render_task_attention_figure(
    attention_matrix: torch.Tensor,
    token_labels: list[str],
    token_metadata: list[dict],
    task_category: str,
    task_id: int,
    layer_index: int,
    head_label: str,
    query_exact_match: bool | None = None,
    query_accuracy: float | None = None,
) -> plt.Figure:
    """Render a token-level self-attention heatmap."""
    attention_matrix = attention_matrix.detach().float().cpu()
    token_vmin = float(attention_matrix.min().item())
    token_vmax = float(attention_matrix.max().item())
    if abs(token_vmax - token_vmin) < 1e-8:
        token_vmax = token_vmin + 1e-8
    boundaries, segment_labels, segment_centers = build_segment_boundaries(token_metadata)

    figure, token_axis = plt.subplots(1, 1, figsize=(14, 10))
    token_image = token_axis.imshow(
        attention_matrix,
        cmap="viridis",
        vmin=token_vmin,
        vmax=token_vmax,
        interpolation="nearest",
        aspect="auto",
    )
    token_colorbar = figure.colorbar(token_image, ax=token_axis, fraction=0.046, pad=0.04)
    token_colorbar.set_label("Attention weight")

    tick_stride = max(1, len(token_labels) // 36)
    tick_positions = list(range(0, len(token_labels), tick_stride))
    if tick_positions[-1] != len(token_labels) - 1:
        tick_positions.append(len(token_labels) - 1)
    tick_locations = [position + 0.5 for position in tick_positions]
    tick_labels = [token_labels[position] for position in tick_positions]
    token_axis.set_xticks(tick_locations)
    token_axis.set_xticklabels(tick_labels, rotation=90, fontsize=FONT_SIZES["tick"])
    token_axis.set_yticks(tick_locations)
    token_axis.set_yticklabels(tick_labels, rotation=0, fontsize=FONT_SIZES["tick"])

    for boundary in boundaries:
        token_axis.axhline(boundary, color="white", linewidth=0.6)
        token_axis.axvline(boundary, color="white", linewidth=0.6)

    token_axis_top = token_axis.secondary_xaxis("top")
    token_axis_top.set_xticks(segment_centers)
    token_axis_top.set_xticklabels(
        segment_labels,
        rotation=45,
        ha="left",
        fontsize=FONT_SIZES["annotation"],
    )
    token_axis_top.tick_params(length=0, pad=6)

    token_axis_right = token_axis.secondary_yaxis("right")
    token_axis_right.set_yticks(segment_centers)
    token_axis_right.set_yticklabels(segment_labels, rotation=0, fontsize=FONT_SIZES["annotation"])
    token_axis_right.tick_params(length=0, pad=6)

    token_axis.set_xlabel("Key tokens")
    token_axis.set_ylabel("Query tokens")
    token_axis.set_title("Token-level attention")

    title = (
        f"{format_task_category(task_category)} | task_id={task_id} | "
        f"layer={layer_index} | head={head_label}"
    )
    if query_exact_match is not None:
        title += f" | query_exact_match={query_exact_match}"
    if query_accuracy is not None:
        title += f" | query_acc={query_accuracy:.2f}"

    figure.suptitle(title, fontsize=FONT_SIZES["title"], fontweight="bold", y=0.98)
    figure.subplots_adjust(top=0.9, bottom=0.24)
    return figure
