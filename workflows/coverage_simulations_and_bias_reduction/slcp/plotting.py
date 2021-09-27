import hypothesis as h
import matplotlib.pyplot as plt
import torch

from hypothesis.plot import make_square
from hypothesis.stat import highest_density_level
from matplotlib.colors import LogNorm
from ratio_estimation import compute_log_posterior
from ratio_estimation import extent


@torch.no_grad()
def plot_posterior(ax, r, observable, nominal=None, resolution=250):
    nominal = nominal.squeeze()
    observable = observable.squeeze()
    pdf = compute_log_posterior(r, observable, resolution=resolution).exp().numpy()
    im = ax.imshow(pdf.T + 1, norm=LogNorm(), alpha=.75, interpolation="bilinear", extent=extent, origin="lower", cmap=h.plot.colormap.cold_r)
    ax.scatter(nominal[0], nominal[1], s=400, marker='*', c="C0", alpha=1.0, zorder=10)
    ax.set_xlabel(r"$\vartheta_1$")
    ax.set_ylabel(r"$\vartheta_1$")
    h.plot.make_square(ax)


@torch.no_grad()
def plot_contours(ax, r, observable, cls=[0.95], labels=[r"95\%"], resolution=250):
    if labels is None:
        labels = [None] * len(cls)
    observable = observable.squeeze()
    epsilon = 0.00001
    p1 = torch.linspace(extent[0], extent[1] - epsilon, resolution)  # Account for half-open interval of uniform prior
    p2 = torch.linspace(extent[2], extent[3] - epsilon, resolution)  # Account for half-open interval of uniform prior
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    g1 = g1.cpu().numpy()
    g2 = g2.cpu().numpy()
    pdf = compute_log_posterior(r, observable, resolution=resolution).exp().numpy()
    fmt = {}
    for cl, label in zip(cls, labels):
        alpha = 1.0 - cl
        level = highest_density_level(pdf, alpha=alpha)
        c = ax.contour(g1, g2, pdf, [level], colors="C0")
        if label is not None:
            fmt[c.levels[0]] = label
            ax.clabel(c, c.levels, inline=True, fontsize=20, fmt=fmt)
    h.plot.make_square(ax)
