"""Plotly defaults + palette shared across every chart module."""
from __future__ import annotations

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7",
    "#ce93d8", "#9ecbff", "#ffa726", "#26a69a", "#ef5350",
]


def dark_layout(title: str, height: int = 420) -> dict:
    """Common plotly_dark layout used everywhere."""
    return dict(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        margin=dict(l=50, r=160, t=50, b=40),
        height=height,
        legend=dict(
            orientation="v", x=1.02, y=1, xanchor="left",
            bgcolor="rgba(22,27,34,0.85)", font=dict(size=10),
        ),
    )


def grid_axes_update(fig) -> None:
    fig.update_xaxes(gridcolor="#21262d", zeroline=False)
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
