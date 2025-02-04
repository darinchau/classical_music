from typing import TYPE_CHECKING, List, Sequence

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy import ndarray

def _plot_pianoroll(
    ax: Axes,
    pianoroll: ndarray,
    is_drum: bool = False,
    resolution: int | None = None,
    beats: ndarray | None = None,
    downbeats: ndarray | None = None,
    preset: str = "full",
    cmap: str = "Blues",
    xtick: str = "auto",
    ytick: str = "octave",
    xticklabel: bool = True,
    yticklabel: str = "auto",
    tick_loc: Sequence[str] = ("bottom", "left"),
    tick_direction: str = "in",
    label: str = "both",
    grid_axis: str = "both",
    grid_color: str = "gray",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
    **kwargs,
):
    """
    Plot a piano roll.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to plot the piano roll on.
    pianoroll : ndarray, shape=(?, 128), (?, 128, 3) or (?, 128, 4)
        Piano roll to plot. For a 3D piano-roll array, the last axis can
        be either RGB or RGBA.
    is_drum : bool, default: False
        Whether it is a percussion track.
    resolution : int
        Time steps per quarter note. Required if `xtick` is 'beat'.
    beats : ndarray, dtype=int, shape=(?, 1),
        A boolean array that indicates the time steps that contain a
        beat.
    downbeats : ndarray, dtype=int, shape=(?, 1),
        A boolean array that indicates the time steps that contain a
        downbeat (i.e., the first time step of a bar).
    preset : {'full', 'frame', 'plain'}, default: 'full'
        Preset theme. For 'full' preset, ticks, grid and labels are on.
        For 'frame' preset, ticks and grid are both off. For 'plain'
        preset, the x- and y-axis are both off.
    cmap : str or :class:`matplotlib.colors.Colormap`, default: 'Blues'
        Colormap. Will be passed to :func:`matplotlib.pyplot.imshow`.
        Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        Tick format for the x-axis. For 'auto' mode, set to 'beat' if
        `beats` is given, otherwise set to 'step'. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}, default: 'octave'
        Tick format for the y-axis.
    xticklabel : bool
        Whether to add tick labels along the x-axis.
    yticklabel : {'auto', 'name', 'number', 'off'}, default: 'auto'
        Tick label format for the y-axis. For 'name' mode, use pitch
        name as tick labels. For 'number' mode, use pitch number. For
        'auto' mode, set to 'name' if `ytick` is 'octave' and 'number'
        if `ytick` is 'pitch'.
    tick_loc : sequence of {'bottom', 'top', 'left', 'right'}
        Tick locations. Defaults to `('bottom', 'left')`.
    tick_direction : {'in', 'out', 'inout'}, default: 'in'
        Tick direction.
    label : {'x', 'y', 'both', 'off'}, default: 'both'
        Whether to add labels to x- and y-axes.
    grid_axis : {'x', 'y', 'both', 'off'}, default: 'both'
        Whether to add grids to the x- and y-axes.
    grid_color : str, default: 'gray'
        Grid color. Will be passed to :meth:`matplotlib.axes.Axes.grid`.
    grid_linestyle : str, default: '-'
        Grid line style. Will be passed to
        :meth:`matplotlib.axes.Axes.grid`.
    grid_linewidth : float, default: 0.5
        Grid line width. Will be passed to
        :meth:`matplotlib.axes.Axes.grid`.
    **kwargs
        Keyword arguments to be passed to
        :meth:`matplotlib.axes.Axes.imshow`.

    """
    # Plot the piano roll
    if pianoroll.ndim == 2:
        transposed = pianoroll.T
    elif pianoroll.ndim == 3:
        transposed = pianoroll.transpose(1, 0, 2)
    else:
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")

    img = ax.imshow(
        transposed,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1 if pianoroll.dtype == np.bool_ else 127,
        origin="lower",
        interpolation="none",
        **kwargs,
    )

    # Format ticks and labels
    if xtick == "auto":
        xtick = "beat" if beats is not None else "step"
    elif xtick not in ("beat", "step", "off"):
        raise ValueError(
            "`xtick` must be one of 'auto', 'beat', 'step' or 'off', not "
            f"{xtick}."
        )
    if yticklabel == "auto":
        yticklabel = "name" if ytick == "octave" else "number"
    elif yticklabel not in ("name", "number", "off"):
        raise ValueError(
            "`yticklabel` must be one of 'auto', 'name', 'number' or 'off', "
            f"{yticklabel}."
        )

    if preset == "full":
        ax.tick_params(
            direction=tick_direction,
            bottom=("bottom" in tick_loc),
            top=("top" in tick_loc),
            left=("left" in tick_loc),
            right=("right" in tick_loc),
            labelbottom=xticklabel,
            labelleft=(yticklabel != "off"),
            labeltop=False,
            labelright=False,
        )
    elif preset == "frame":
        ax.tick_params(
            direction=tick_direction,
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
    elif preset == "plain":
        ax.axis("off")
    else:
        raise ValueError(
            f"`preset` must be one of 'full', 'frame' or 'plain', not {preset}"
        )

    # Format x-axis
    if xtick == "beat" and preset != "frame":
        if beats is None:
            raise RuntimeError(
                "Beats must be given when using `beat` for ticks on the "
                "x-axis."
            )
        if len(beats) < 2:
            raise RuntimeError(
                "There muse be at least two beats given when using `beat` for "
                "ticks on the x-axis."
            )
        beats_arr = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
        ax.set_xticks(beats_arr[:-1] - 0.5)
        ax.set_xticklabels("")
        ax.set_xticks((beats_arr[1:] + beats_arr[:-1]) / 2 - 0.5, minor=True)
        ax.set_xticklabels(np.arange(1, len(beats) + 1), minor=True)
        ax.tick_params(axis="x", which="minor", width=0)

    # Format y-axis
    if ytick == "octave":
        ax.set_yticks(np.arange(0, 128, 12))
        if yticklabel == "name":
            ax.set_yticklabels([f"C{i - 2}" for i in range(11)])
    elif ytick == "step":
        ax.set_yticks(np.arange(0, 128))
        if yticklabel == "name":
            if is_drum:
                ax.set_yticklabels(
                    [note_number_to_drum_name(i) for i in range(128)]
                )
            else:
                ax.set_yticklabels(
                    [note_number_to_name(i) for i in range(128)]
                )
    elif ytick != "off":
        raise ValueError(
            f"`ytick` must be one of 'octave', 'pitch' or 'off', not {ytick}."
        )

    # Format axis labels
    if label not in ("x", "y", "both", "off"):
        raise ValueError(
            f"`label` must be one of 'x', 'y', 'both' or 'off', not {label}."
        )

    if label in ("x", "both"):
        if xtick == "step" or not xticklabel:
            ax.set_xlabel("time (step)")
        else:
            ax.set_xlabel("time (beat)")

    if label in ("y", "both"):
        if is_drum:
            ax.set_ylabel("key name")
        else:
            ax.set_ylabel("pitch")

    # Plot the grid
    if grid_axis not in ("x", "y", "both", "off"):
        raise ValueError(
            "`grid` must be one of 'x', 'y', 'both' or 'off', not "
            f"{grid_axis}."
        )
    if grid_axis != "off":
        ax.grid(
            axis=grid_axis,
            color=grid_color,
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
        )

    # Plot downbeat boundaries
    if downbeats is not None:
        for downbeat in downbeats:
            ax.axvline(x=downbeat, color="k", linewidth=1)

    return img
