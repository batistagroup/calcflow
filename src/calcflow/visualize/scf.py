from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from calcflow.parsers.orca.typing import ScfIteration
from calcflow.visualize.colors import blue_20, coral_20, green_20, orange_20, purple_20, red_20
from calcflow.visualize.style import apply_development_style


def plot_scf_convergence(
    scf_iterations: Sequence[Sequence[ScfIteration]], title: str = "SCF Convergence Monitor"
) -> go.Figure:
    """
    Plots SCF convergence for a list of SCF iteration histories.

    Args:
        histories (list[list[ScfIteration]]): A list of SCF iteration histories,
            where each history is a list of ScfIteration objects.
        title (str, optional): Title of the plot. Defaults to "SCF Convergence Monitor".

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    # fmt:off
    fig = make_subplots(rows=5, cols=1, subplot_titles=[
            "SCF Energy (Eh)", "Energy Change (delta_e_eh)",
            "RMS Density Matrix Residual (rmsdp)", "Max Density Matrix Residual (maxdp)",
            "Iteration Time (sec)"])
    n = len(scf_iterations)
    colors = [blue_20.sample_n_hex(n), red_20.sample_n_hex(n), purple_20.sample_n_hex(n), coral_20.sample_n_hex(n), green_20.sample_n_hex(n), orange_20.sample_n_hex(n)]
    for i, history in enumerate(scf_iterations):
        iterations = [h.iteration for h in history]
        energy = [(h.energy)  for h in history]
        delta_e_eh = [h.delta_e_eh  for h in history]
        rmsdp = [h.rmsdp for h in history]
        maxdp = [h.maxdp for h in history]
        time_sec = [h.time_sec for h in history]

        trace_name_suffix = f" (Cycle {i + 1})" if len(scf_iterations) > 1 else ""
        fig.add_trace(go.Scatter(x=iterations, y=energy, mode="lines+markers", line=dict(color=colors[0][i]), name="Energy" + trace_name_suffix), row=1, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=delta_e_eh, mode="lines+markers", line=dict(color=colors[1][i]), name="Delta E" + trace_name_suffix), row=2, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=rmsdp, mode="lines+markers", line=dict(color=colors[2][i]), name="RMSDP" + trace_name_suffix), row=3, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=maxdp, mode="lines+markers", line=dict(color=colors[3][i]), name="MAXDP" + trace_name_suffix), row=4, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=time_sec, mode="lines+markers", line=dict(color=colors[4][i]), name="Time (sec)" + trace_name_suffix), row=5, col=1)
        # fmt:on

    fig.update_layout(title=title, height=1000, showlegend=n>1, legend=dict(orientation="h", entrywidth=150 ))
    for i in range(1, 6):
        fig.update_xaxes(title_text="Iteration", row=i, col=1)

    fig.update_yaxes(type="log", dtick=1, row=3, col=1)
    fig.update_yaxes(type="log", dtick=1, row=4, col=1)

    apply_development_style(fig)
    return fig
