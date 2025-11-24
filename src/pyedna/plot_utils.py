import numpy as np
import matplotlib.pyplot as plt
import math


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
    plt.tight_layout()
    plt.show()


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

    plt.show()