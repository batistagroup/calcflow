from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from calcflow.parsers.orca.opt import OptimizationData
from calcflow.parsers.orca.typing import Atom
from calcflow.visualize.colors import blue_20, purple_20, red_20
from calcflow.visualize.style import apply_development_style


def get_interatomic_distances(atoms: Sequence[Atom]) -> tuple[list[float], list[str]]:
    """Calculate all pairwise interatomic distances.

    Args:
        atoms: List of atoms with coordinates

    Returns:
        Tuple of (distances, labels) where distances is a list of interatomic distances
        and labels describes each distance
    """
    dists: list[float] = []
    labels: list[str] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            # Calculate distance using Pythagorean theorem
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z - atoms[j].z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            dists.append(dist)
            # Create label from atom symbols
            label = f"{i + 1}{atoms[i].symbol}-{j + 1}{atoms[j].symbol}"
            labels.append(label)
    return dists, labels


def plot_optimization_progress(opt_data: OptimizationData) -> Figure:
    """Create a figure showing optimization progress metrics.

    Args:
        opt_data: Parsed optimization data from ORCA output

    Returns:
        A plotly Figure with 4 subplots showing energy, gradient, step size and geometry changes
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Energy Convergence", "Gradient Convergence", "Step Size Convergence", "Geometry Changes"),
    )

    # Extract data from optimization cycles
    cycles = list(range(1, len(opt_data.cycles) + 1))
    energies = [cycle.energy_eh for cycle in opt_data.cycles]
    rms_grads = [cycle.relaxation_step.rms_gradient for cycle in opt_data.cycles if cycle.relaxation_step]
    max_grads = [cycle.relaxation_step.max_gradient for cycle in opt_data.cycles if cycle.relaxation_step]
    rms_steps = [cycle.relaxation_step.rms_step for cycle in opt_data.cycles if cycle.relaxation_step]
    max_steps = [cycle.relaxation_step.max_step for cycle in opt_data.cycles if cycle.relaxation_step]

    # Get color lists
    blue_colors = blue_20.sample_n_hex(1)
    red_colors = red_20.sample_n_hex(2)
    purple_colors = purple_20.sample_n_hex(2)

    # fmt:off
    # Energy convergence
    fig.add_trace(go.Scatter(x=cycles, y=energies, mode="lines+markers", name="Total Energy", line=dict(color=blue_colors[0])), row=1, col=1)

    # Gradient convergence
    fig.add_trace(go.Scatter(x=cycles, y=rms_grads, mode="lines+markers", name="RMS Gradient", line=dict(color=red_colors[0])), row=1, col=2)
    fig.add_trace(go.Scatter(x=cycles, y=max_grads, mode="lines+markers", name="Max Gradient", line=dict(color=red_colors[1])), row=1, col=2)

    # Step size convergence
    fig.add_trace(go.Scatter(x=cycles, y=rms_steps, mode="lines+markers", name="RMS Step", line=dict(color=purple_colors[0])), row=2, col=1)
    fig.add_trace(go.Scatter(x=cycles, y=max_steps, mode="lines+markers", name="Max Step", line=dict(color=purple_colors[1])), row=2, col=1)

    # Get distances and labels for each cycle
    distances_and_labels = [get_interatomic_distances(cycle.geometry) for cycle in opt_data.cycles if cycle.geometry]
    distances = [d[0] for d in distances_and_labels]
    labels = distances_and_labels[0][1]  # Labels will be same for all cycles
    n_distances = len(distances[0])
    distance_colors = blue_20.sample_n_hex(n_distances)

    for i in range(n_distances):
        dist_i = [d[i] for d in distances]
        fig.add_trace(go.Scatter(x=cycles, y=dist_i, mode="lines+markers", name=labels[i], line=dict(color=distance_colors[i])), row=2, col=2)

    # Update layout
    fig.update_layout(height=800, width=1200, showlegend=True, title_text="Geometry Optimization Progress")

    # Update axes labels
    fig.update_xaxes(title_text="Optimization Cycle", row=2, col=1)
    fig.update_xaxes(title_text="Optimization Cycle", row=2, col=2)
    fig.update_yaxes(title_text="Energy (Eh)", row=1, col=1)
    fig.update_yaxes(title_text="Gradient (a.u.)", row=1, col=2)
    fig.update_yaxes(title_text="Step Size (a.u.)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (Å)", row=2, col=2)
    apply_development_style(fig)

    return fig
