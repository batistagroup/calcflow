"""
Plotting style definitions and utilities for publication-quality figures.
"""

from typing import Any

import plotly.colors
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Style parameters
# -----------------------------------------------------------------------------


# Universal font settings
FONT_FAMILY = "Helvetica"
FONT_COLOR = "#333333"

# Font sizes for different elements
FONT_SIZES: dict[str, int] = {
    "title": 20,
    "axis_title": 16,
    "tick_label": 16,
    "subtitle": 11,
    "legend": 12,
    "subplot_title": 14,
}

# Universal axis style settings
AXIS_STYLE: dict[str, Any] = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": "#E7E7E7",
    "zeroline": False,
    "linewidth": 2,
    "linecolor": "#333333",
}

# Universal layout settings
LAYOUT_STYLE: dict[str, Any] = {
    "plot_bgcolor": "#FBFCFF",
    "paper_bgcolor": "#FBFCFF",
    "margin": dict(t=40, b=40, r=40),
}

# Development style settings
DEVELOPMENT_STYLE: dict[str, Any] = {
    "template": "plotly_dark",
    "plot_bgcolor": "black",
    "paper_bgcolor": "black",
    "font": dict(color="white"),
}

# -----------------------------------------------------------------------------
# Color generation utilities
# -----------------------------------------------------------------------------


def get_plotly_colorscale(label: str, zero: str | None = None) -> str | list[str]:
    """Get a Plotly colorscale with optional zero color override.

    Args:
        label: Name of the Plotly sequential colorscale
        zero: Optional color to override the first color in the scale

    Returns:
        Either the colorscale name or a modified list of colors
    """
    if zero is None:
        return label
    _colorscale = getattr(plotly.colors.sequential, label)
    colorscale = _colorscale.copy()
    colorscale[0] = zero
    return list(map(str, colorscale))


# -----------------------------------------------------------------------------
# Styling functions
# -----------------------------------------------------------------------------


def get_font_dict(size: int, bold: bool = False) -> dict[str, Any]:
    """Helper function to create consistent font dictionaries.

    Args:
        size: Font size to use
        bold: Whether to use bold font weight

    Returns:
        Dictionary with font settings
    """
    return dict(
        family=FONT_FAMILY,
        size=size,
        color=FONT_COLOR,
        weight="bold" if bold else None,
    )


def apply_publication_fonts(fig: go.Figure) -> None:
    """Apply publication-quality font settings to a figure.

    Args:
        fig: A plotly figure
    """
    # Update global font
    fig.update_layout(font=get_font_dict(FONT_SIZES["tick_label"]))

    # Update title font if title exists
    if fig.layout.title is not None:
        fig.layout.title.update(font=get_font_dict(FONT_SIZES["title"], bold=True))

    # Update subplot titles if they exist
    if fig.layout.annotations:
        for annotation in fig.layout.annotations:
            if "<b>" in str(annotation.text):  # This is a subplot title
                annotation.update(font=get_font_dict(FONT_SIZES["subplot_title"], bold=True))


def update_axis(axis: go.layout.XAxis | go.layout.YAxis, axis_style: dict[str, Any]) -> None:
    """Helper function to update a single axis with publication styling.

    Args:
        axis: Axis to update
        axis_style: Style parameters to apply
    """
    axis.update(
        axis_style,
        title_font=get_font_dict(FONT_SIZES["axis_title"], bold=True),
        tickfont=get_font_dict(FONT_SIZES["tick_label"]),
    )


def apply_axis_style(fig: go.Figure, row: int | None = None, col: int | None = None, **kwargs: Any) -> None:
    """Apply publication-quality axis styling to a figure.

    Args:
        fig: A plotly figure
        row: Optional row index for subplots
        col: Optional column index for subplots
        **kwargs: Additional axis style parameters to override defaults
    """
    axis_style = AXIS_STYLE.copy()
    axis_style.update(kwargs)

    if row is not None and col is not None:
        update_axis(fig.get_xaxes()[row - 1], axis_style)
        update_axis(fig.get_yaxes()[col - 1], axis_style)
    else:
        update_axis(fig.layout.xaxis, axis_style)
        update_axis(fig.layout.yaxis, axis_style)


def apply_publication_style(fig: go.Figure, **kwargs: Any) -> None:
    """Apply all publication-quality styling to a figure.

    Args:
        fig: A plotly figure
        show_legend: Whether to show and style the legend
        **kwargs: Additional layout parameters to override defaults
    """
    # Apply fonts
    apply_publication_fonts(fig)

    # Apply axis style to all axes
    # Handle both single plot and subplot cases by looking for axis objects in layout
    for key in fig.layout:
        if key.startswith("xaxis") or key.startswith("yaxis"):
            update_axis(getattr(fig.layout, key), AXIS_STYLE)

    layout_style: dict[str, Any] = LAYOUT_STYLE.copy()
    layout_style.update(kwargs)
    fig.update_layout(layout_style, legend=dict(font=get_font_dict(FONT_SIZES["legend"])))


def apply_development_style(fig: go.Figure) -> None:
    """Apply dark theme development styling to a figure.

    This applies a dark theme with black background, suitable for development
    and debugging visualizations.

    Args:
        fig: A plotly figure
    """
    fig.update_layout(**DEVELOPMENT_STYLE)
