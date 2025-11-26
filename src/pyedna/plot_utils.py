import numpy as np
import matplotlib.pyplot as plt
import math

# get colors for donor and acceptor
def getColors(role, n = 6):
    assert role in ["D", "A"], "Specify Donor (D) or Acceptor (A) for color scheme"

    # colormaps for donor and acceptor
    if role == "D":
        cmap_name = 'PuBu'
    elif role == "A":
        cmap_name = 'RdPu'
    
    cmap = plt.get_cmap(cmap_name)

    # sample n colors evenly across the colormap
    if n < 1:
        n = 1
    xs = [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
    palette = [cmap(x) for x in xs]

    # pick some convenient representatives
    light = cmap(0.2)               # quite light
    main  = cmap(0.6)               # mid–strong
    dark  = cmap(0.9)               # darkest

    color_scheme = dict()

    color_scheme = {
        "name": role,
        "cmap_name": cmap_name,
        "cmap": cmap,
        "palette": palette,
        "main": main,
        "light": light,
        "dark": dark,
    }
    return color_scheme

# plot Excited States Spectrum (from TDDFT)
def plotExcitedEnergies(exc, exc_std = None, color='limegreen'):
    energies = np.asarray(exc)
    x_pos = np.ones_like(exc)

    fig, ax = plt.subplots(figsize=(2, 6), dpi=300)

    for i, (x, y) in enumerate(zip(x_pos, energies)):
        ax.plot([x - 0.3, x + 0.3], [y, y], color=color, linewidth=2, label='Excited States' if i == 0 else "")
        if exc_std is not None:
            ax.errorbar(x, y, yerr=exc_std[i], color=color, capsize=2, elinewidth=0.8, linewidth=0, alpha=0.7)

    ax.set_xticks([1])
    ax.set_xticklabels(['Excited States'])
    ax.set_ylabel(r'Energy ($\mathrm{cm}^{-1}$)')
    ax.set_title(f'Excited-State Energy Spectrum')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    return fig, ax

# plot Oscilaltor Strengths (from TDDFT)
def plotOscillatorStrengths(osc, osc_std = None, alpha=0.8, color='limegreen'):
    no_states = len(osc)
    states = [f"state {i}" for i in range(no_states)]
    x = np.arange(no_states)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    if osc_std is not None:
        osc_std = np.asarray(osc_std, dtype=float)
        ax.bar(x, osc, yerr=osc_std, capsize=4, color=color, edgecolor='black')
    else:
        ax.bar(x, osc, color=color, alpha=alpha, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.set_ylabel("Oscillator Strength")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Excited State")
    ax.set_title("Oscillator Strengths (per State)")
    ax.margins(x=0.05)
    fig.tight_layout()
    return fig, ax

# plot spectrum
def plotSpectrum(x, y, color=None, alpha=0.8, edgecolor="black", title=None):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D arrays of the same length")

    # infer bar width from bin centers so that bars "touch"
    if len(x) > 1:
        dx = np.diff(x)
        # use median spacing to be robust
        width = np.median(dx)
    else:
        # single bin: arbitrary width
        width = 1.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.bar(x,y,width=width,align="center",color=color,alpha=alpha,edgecolor=edgecolor,)
    ax.set_xlabel(r"Energy ($\mathrm{cm}^{-1}$)")
    ax.set_ylabel("Intensity")
    if title is not None:
        ax.set_title(title)
    ax.margins(x=0.01)

    fig.tight_layout()
    return fig, ax


# plot OPA results
def plotOPA(opa, cmap='viridis'):

    num_matrices = opa.shape[0]

    # 2 rows, enough columns to fit all matrices
    nrows = 2
    ncols = math.ceil(num_matrices / nrows)
    figsize_scaler = 1.2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_scaler, nrows * figsize_scaler),
        dpi=300,
        constrained_layout=True
    )

    # ensure axes is always 2D array
    axes = np.atleast_2d(axes)

    vmin, vmax = 0, 1  # or opa.min(), opa.max()

    for i, ax in enumerate(axes.flat):
        if i >= num_matrices:
            # turn off unused subplots if num_matrices is odd
            ax.axis('off')
            continue

        im = ax.imshow(opa[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"State {i}", fontsize=8)

        for (y, x), val in np.ndenumerate(opa[i]):
            ax.text(
                x, y, f"{val:.2f}",
                ha='center', va='center',
                color='white' if val > 0.5 else 'black',
                fontsize=7
            )

    # add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Population", fontsize=10)

    return fig, axes

# plot Mulliken Fragment Analysis results
def plotMulliken(mulliken, fragment_names, colors=None):

    # load number of states and labels
    no_states = len(mulliken)               
    states = [f"state {i}" for i in range(no_states)]

    data = np.array(mulliken).T             # shape: (fragments, states)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    bottom = np.zeros(no_states)
    for i, frag in enumerate(fragment_names):
        color = colors[i] if colors is not None else None
        ax.bar(states, data[i], bottom=bottom, label=frag, color=color)
        bottom += data[i]

    ax.set_ylabel("Normalized Mulliken Charge")
    ax.set_title(f"Mulliken Charges per Excited State")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45)
    ax.axhline(0.5, color='white', linestyle = 'dashed')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig, ax