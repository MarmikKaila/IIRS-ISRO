"""Visualization helpers: image grids, comparison grids, animated GIFs."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def _to_uint8(img, vmin=None, vmax=None):
    img = np.asarray(img, dtype=np.float64)
    vmin = float(img.min()) if vmin is None else float(vmin)
    vmax = float(img.max()) if vmax is None else float(vmax)
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def save_image_grid(images, path, titles=None, ncols=8, cmap="gray",
                    vmin=None, vmax=None):
    """Save a grid of single-channel images."""
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows),
                             squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(pred, gt, path, titles=None, cmap="gray",
                         vmin=None, vmax=None, row_labels=("predicted", "ground truth")):
    """Save a 2-row grid: predicted on top, ground truth below."""
    n = len(pred)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.2), squeeze=False)
    for col in range(n):
        for row, data in ((0, pred), (1, gt)):
            ax = axes[row][col]
            ax.imshow(data[col], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
        if titles is not None:
            axes[0][col].set_title(str(titles[col]), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_gif(images, path, fps=4, vmin=None, vmax=None):
    """Save a sequence of single-channel images as an animated GIF."""
    frames = [_to_uint8(im, vmin, vmax) for im in images]
    imageio.mimsave(path, frames, duration=1.0 / max(fps, 1))
