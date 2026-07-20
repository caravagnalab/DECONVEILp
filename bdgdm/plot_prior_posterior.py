"""
Faceted prior-versus-posterior plots for two BDGDM subtypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchicalNormalPrior:
    """Prior for a global mean plus centered subtype offsets."""

    loc: float
    global_scale: float
    offset_scale: float


BDGDM_GENE_DOSAGE_PRIORS: dict[str, HierarchicalNormalPrior] = {
    "b0": HierarchicalNormalPrior(
        loc=5.0,
        global_scale=2.0,
        offset_scale=0.5,
    ),
    "b_scaling": HierarchicalNormalPrior(
        loc=0.4,
        global_scale=0.5,
        offset_scale=0.3,
    ),
    "b_deviation": HierarchicalNormalPrior(
        loc=0.0,
        global_scale=0.3,
        offset_scale=0.2,
    ),
}


def _resolve_raw_fit(fit: Any) -> Any:
    """Return the raw CmdStan fit from a raw fit or BDGDMFit wrapper."""
    return getattr(fit, "fit", fit)


def _extract_subtype_draws(
    fit: Any,
    parameter: str,
) -> np.ndarray:
    """
    Extract a subtype-indexed Stan variable.

    Returns
    -------
    ndarray
        Shape ``(draws, subtypes)``.
    """
    raw_fit = _resolve_raw_fit(fit)
    stan_variable = getattr(raw_fit, "stan_variable", None)

    if not callable(stan_variable):
        raise TypeError(
            "The fit must expose stan_variable(), either directly "
            "or through fit.fit."
        )

    try:
        values = np.asarray(
            stan_variable(parameter),
            dtype=float,
        )
    except Exception as exc:
        raise KeyError(
            f"Posterior variable {parameter!r} was not found."
        ) from exc

    if values.ndim != 2:
        raise ValueError(
            f"{parameter!r} must have shape (draws, subtypes); "
            f"received {values.shape}. This function is for the "
            "subtype model, not the scalar single-group model."
        )

    if values.shape[1] != 2:
        raise ValueError(
            "This plotting function requires exactly two subtypes; "
            f"the posterior contains {values.shape[1]}."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{parameter!r} contains non-finite posterior draws."
        )

    return values


def _infer_subtype_labels(
    fit: Any,
    subtype_labels: Sequence[str] | None,
) -> list[str]:
    """Infer two subtype labels from BDGDMFit metadata when possible."""
    if subtype_labels is not None:
        labels = [str(label) for label in subtype_labels]
    else:
        metadata = getattr(fit, "metadata", None)
        labels = []

        if isinstance(metadata, Mapping):
            levels = metadata.get("subtype_levels")
            if levels is not None:
                labels = [str(level) for level in levels]

        if not labels:
            levels = getattr(fit, "subtype_levels", None)
            if levels is not None:
                labels = [str(level) for level in levels]

    if len(labels) != 2:
        raise ValueError(
            "Provide subtype_labels with exactly two labels, or use a "
            "BDGDMFit whose metadata contains two subtype_levels."
        )

    return labels


def _normal_density(
    x: np.ndarray,
    *,
    loc: float,
    scale: float,
) -> np.ndarray:
    """Evaluate a Normal density without SciPy."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Prior scale must be positive and finite.")

    z = (x - loc) / scale

    return (
        np.exp(-0.5 * z**2)
        / (scale * np.sqrt(2.0 * np.pi))
    )


def plot_prior_vs_posterior(
    fit: Any,
    parameter: str,
    *,
    subtype_labels: Sequence[str] | None = None,
    gene: str | None = None,
    prior_specs: Mapping[
        str,
        HierarchicalNormalPrior,
    ] = BDGDM_GENE_DOSAGE_PRIORS,
    bins: int | str = 35,
    credible_interval: float = 0.95,
    prior_color: str = "0.80",
    posterior_colors: Sequence[str] = (
        "steelblue",
        "darkorange",
    ),
    median_color: str = "firebrick",
    alpha_prior: float = 0.60,
    alpha_posterior: float = 0.72,
    share_x: bool = True,
    share_y: bool = True,
    figsize: tuple[float, float] = (11.0, 4.8),
    xlabel: str | None = None,
):
    """
    Plot prior and posterior distributions in two subtype facets.

    Parameters
    ----------
    fit
        Raw CmdStan fit or ``BDGDMFit`` from the subtype model.

    parameter
        One of ``b0``, ``b_scaling``, or ``b_deviation``.

    subtype_labels
        Two labels in the same order used by Stan. When omitted, the
        function tries ``fit.metadata["subtype_levels"]``.

    gene
        Optional gene name used in the figure title.

    Returns
    -------
    fig, axes, summary
        ``summary`` is a DataFrame with one row per subtype.
    """
    import matplotlib.pyplot as plt

    if parameter not in prior_specs:
        raise ValueError(
            f"Unknown parameter {parameter!r}. "
            f"Available parameters: {sorted(prior_specs)}."
        )

    if not 0 < credible_interval < 1:
        raise ValueError(
            "credible_interval must lie between 0 and 1."
        )

    colors = list(posterior_colors)

    if len(colors) != 2:
        raise ValueError(
            "posterior_colors must contain exactly two colors."
        )

    labels = _infer_subtype_labels(
        fit,
        subtype_labels,
    )
    draws = _extract_subtype_draws(
        fit,
        parameter,
    )

    prior = prior_specs[parameter]
    n_subtypes = draws.shape[1]

    # Marginal prior of one centered subtype coefficient:
    # Var(global mean) + Var(raw offset - mean(raw offsets)).
    prior_sd = float(
        np.sqrt(
            prior.global_scale**2
            + prior.offset_scale**2
            * (1.0 - 1.0 / n_subtypes)
        )
    )

    alpha = (1.0 - credible_interval) / 2.0

    posterior_limits = np.quantile(
        draws,
        [0.001, 0.999],
        axis=0,
    )

    prior_low = prior.loc - 4.0 * prior_sd
    prior_high = prior.loc + 4.0 * prior_sd

    if share_x:
        x_low = float(
            min(
                prior_low,
                posterior_limits[0, 0],
                posterior_limits[0, 1],
            )
        )
        x_high = float(
            max(
                prior_high,
                posterior_limits[1, 0],
                posterior_limits[1, 1],
            )
        )
        x_ranges = [(x_low, x_high), (x_low, x_high)]
    else:
        x_ranges = [
            (
                float(min(prior_low, posterior_limits[0, index])),
                float(max(prior_high, posterior_limits[1, index])),
            )
            for index in range(2)
        ]

    padded_ranges: list[tuple[float, float]] = []

    for x_low, x_high in x_ranges:
        width = x_high - x_low

        if width <= 0:
            padding = max(abs(x_low) * 0.05, 1e-6)
        else:
            padding = 0.04 * width

        padded_ranges.append(
            (x_low - padding, x_high + padding)
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=share_x,
        sharey=share_y,
    )

    summary_rows: list[dict[str, float | str | int]] = []

    for subtype_index, ax in enumerate(axes):
        posterior = draws[:, subtype_index]
        q_low, median, q_high = np.quantile(
            posterior,
            [
                alpha,
                0.5,
                1.0 - alpha,
            ],
        )

        x_low, x_high = padded_ranges[subtype_index]
        x_grid = np.linspace(
            x_low,
            x_high,
            600,
        )
        prior_density = _normal_density(
            x_grid,
            loc=prior.loc,
            scale=prior_sd,
        )

        ax.fill_between(
            x_grid,
            0.0,
            prior_density,
            color=prior_color,
            alpha=alpha_prior,
            linewidth=0,
            label="Marginal prior",
        )
        ax.plot(
            x_grid,
            prior_density,
            color=prior_color,
            linewidth=1.4,
        )

        ax.hist(
            posterior,
            bins=bins,
            density=True,
            color=colors[subtype_index],
            alpha=alpha_posterior,
            edgecolor="none",
            label="Posterior",
        )

        ax.axvspan(
            q_low,
            q_high,
            color=median_color,
            alpha=0.08,
            linewidth=0,
            label=f"{credible_interval:.0%} interval",
        )

        ax.axvline(
            median,
            color=median_color,
            linestyle=(0, (6, 4)),
            linewidth=2.3,
            label=f"Median = {median:.3g}",
        )

        ax.set_xlim(x_low, x_high)
        ax.set_title(labels[subtype_index])
        ax.set_xlabel(
            xlabel if xlabel is not None else parameter
        )
        ax.grid(alpha=0.22)

        summary_rows.append(
            {
                "parameter": parameter,
                "subtype_index": subtype_index + 1,
                "subtype_label": labels[subtype_index],
                "prior_mean": float(prior.loc),
                "prior_sd": prior_sd,
                "posterior_median": float(median),
                "posterior_q_low": float(q_low),
                "posterior_q_high": float(q_high),
                "credible_interval": float(credible_interval),
                "n_draws": int(posterior.size),
            }
        )

    axes[0].set_ylabel("Density")

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )

    title = f"Prior vs posterior: {parameter}"

    if gene is not None:
        title = f"{gene} — {title}"

    fig.suptitle(
        title,
        y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(
        bottom=0.22,
        top=0.86,
    )

    summary = pd.DataFrame(summary_rows)

    return fig, axes, summary
