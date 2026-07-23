from __future__ import annotations

import math
from typing import Any, Sequence
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes


"""
Visualization functions for fitted BDGDM models.

"""

__all__ = [
    "plot_copy_number_fit",
    "plot_ppc_mean",
    "plot_ppc_variance",
    "plot_ppc_density"
]


def plot_ppc_mean(
    fit,
    y_observed: Sequence[float] | np.ndarray | None = None,
    *,
    gene: str | None = None,
    histogram_color: str = "C0",
    observed_color: str = "C1",
    bins: int | str = 30,
    alpha: float = 0.75,
    credible_interval: float = 0.90,
    show_interval: bool = True,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes, dict]:
    """
    Plot a posterior predictive check for the mean RNA-seq count.

    Parameters
    ----------
    fit
        BDGDMFit object returned by ``fit_one_gene_bdgdm``.

    y_observed
        Observed counts used to fit the model. If omitted, the function
        attempts to obtain ``expr`` from ``fit.processed_data``.

    gene
        Gene name displayed in the title. If omitted, ``fit.gene`` is used.

    histogram_color
        Color of the replicated-mean histogram.

    observed_color
        Color of the observed-mean reference line.

    bins
        Number of histogram bins or a NumPy histogram binning rule,
        such as ``"auto"`` or ``"fd"``.

    alpha
        Histogram transparency.

    credible_interval
        Central posterior predictive interval shown as a shaded region.

    show_interval
        Whether to show the posterior predictive interval.

    figsize
        Figure size when ``ax`` is not supplied.

    title
        Optional custom title.

    ax
        Existing Matplotlib axis. When omitted, a new figure is created.

    show
        Whether to call ``plt.show()``.

    Returns
    -------
    Figure
        Matplotlib figure.

    Axes
        Matplotlib axis.

    dict
        Numerical PPC summary.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in the interval (0, 1].")

    if not 0 < credible_interval < 1:
        raise ValueError(
            "credible_interval must lie between 0 and 1."
        )

    # Obtain observed counts.
    if y_observed is None:
        processed_data = getattr(
            fit,
            "processed_data",
            None,
        )

        if (
            processed_data is None
            or "expr" not in processed_data.columns
        ):
            raise ValueError(
                "y_observed was not provided and fit.processed_data "
                "does not contain an 'expr' column."
            )

        y_observed = processed_data["expr"].to_numpy()

    y_observed = np.asarray(
        y_observed,
        dtype=float,
    ).reshape(-1)

    if y_observed.size == 0:
        raise ValueError("y_observed is empty.")

    if not np.all(np.isfinite(y_observed)):
        raise ValueError(
            "y_observed contains non-finite values."
        )

    if np.any(y_observed < 0):
        raise ValueError(
            "y_observed contains negative counts."
        )

    # Extract posterior predictive replicated counts.
    try:
        y_rep = np.asarray(
            fit.fit.stan_variable("y_rep"),
            dtype=float,
        )
    except Exception as exc:
        raise KeyError(
            "The fitted Stan model does not contain 'y_rep'. "
            "Posterior predictive quantities must be generated "
            "by the Stan model."
        ) from exc

    if y_rep.ndim == 1 and y_observed.size == 1:
        y_rep = y_rep[:, None]

    if y_rep.ndim != 2:
        raise ValueError(
            "y_rep must have shape "
            "(posterior draws, observations). "
            f"Received shape {y_rep.shape}."
        )

    if y_rep.shape[1] != y_observed.size:
        raise ValueError(
            "The number of observed values does not match y_rep: "
            f"{y_observed.size} observed values versus "
            f"{y_rep.shape[1]} replicated observations."
        )

    replicated_means = np.mean(y_rep, axis=1)

    observed_mean = float(np.mean(y_observed))

    lower_probability = (1.0 - credible_interval) / 2.0

    upper_probability = (1.0 + credible_interval) / 2.0

    interval_lower = float(np.quantile(replicated_means, lower_probability))

    interval_upper = float(np.quantile(replicated_means, upper_probability))

    # Two-sided posterior predictive p-value.
    probability_below = float(np.mean(replicated_means <= observed_mean))

    ppc_pvalue = float(min(1.0, 2.0 * min(probability_below, 1.0 - probability_below)))

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
        )
    else:
        fig = ax.figure

    ax.hist(
        replicated_means,
        bins=bins,
        color=histogram_color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.6,
        label="Replicated means",
    )

    if show_interval:
        ax.axvspan(
            interval_lower,
            interval_upper,
            color=histogram_color,
            alpha=0.15,
            label=(
                f"{credible_interval:.0%} predictive interval"
            ),
        )

    ax.axvline(
        observed_mean,
        color=observed_color,
        linestyle="--",
        linewidth=2.2,
        label=f"Observed mean = {observed_mean:.1f}",
    )

    gene_name = (
        gene
        if gene is not None
        else getattr(fit, "gene", None)
    )

    if title is None:
        if gene_name:
            title = (
                f"{gene_name}: posterior predictive check "
                "for mean expression"
            )
        else:
            title = (
                "PPC for mean expression"
            )

    ax.set_title(title)
    ax.set_xlabel("Replicated mean RNA-seq count")
    ax.set_ylabel("Posterior draws")
    ax.legend(frameon=False)

    fig.tight_layout()

    summary = {
        "gene": gene_name,
        "observed_mean": observed_mean,
        "replicated_mean_median": float(
            np.median(replicated_means)
        ),
        "replicated_mean_mean": float(
            np.mean(replicated_means)
        ),
        "predictive_interval_lower": interval_lower,
        "predictive_interval_upper": interval_upper,
        "credible_interval": credible_interval,
        "ppc_pvalue_two_sided": ppc_pvalue,
        "n_posterior_draws": int(
            replicated_means.size
        ),
        "n_observations": int(
            y_observed.size
        ),
    }

    if show:
        plt.show()

    return fig, ax, summary


def plot_ppc_variance(
    fit,
    y_observed: Sequence[float] | np.ndarray | None = None,
    *,
    gene: str | None = None,
    histogram_color: str = "C0",
    observed_color: str = "C1",
    bins: int | str = 30,
    alpha: float = 0.75,
    credible_interval: float = 0.90,
    show_interval: bool = True,
    log_scale: bool = True,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes, dict]:
    """
    Plot a posterior predictive check for RNA-seq count variance.

    Parameters
    ----------
    fit
        BDGDMFit object returned by ``fit_one_gene_bdgdm``.

    y_observed
        Observed counts used to fit the model. If omitted, the function
        attempts to obtain the ``expr`` column from
        ``fit.processed_data``.

    gene
        Gene name displayed in the title. If omitted, ``fit.gene`` is
        used.

    histogram_color
        Color of the replicated-variance histogram.

    observed_color
        Color of the observed-variance reference line.

    bins
        Number of histogram bins or a NumPy histogram rule such as
        ``"auto"`` or ``"fd"``.

    alpha
        Histogram transparency.

    credible_interval
        Width of the central posterior predictive interval.

    show_interval
        Whether to display the posterior predictive interval.

    log_scale
        If True, plot ``log(1 + variance)``. This is generally preferable
        for RNA-seq counts because variances may span a large range.

    figsize
        Figure size when ``ax`` is not provided.

    title
        Optional custom title.

    ax
        Existing Matplotlib axis. If omitted, a new figure is created.

    show
        Whether to call ``plt.show()``.

    Returns
    -------
    Figure
        Matplotlib figure.

    Axes
        Matplotlib axis.

    dict
        Numerical posterior predictive summary.
    """
    if not 0 < alpha <= 1:
        raise ValueError(
            "alpha must lie in the interval (0, 1]."
        )

    if not 0 < credible_interval < 1:
        raise ValueError(
            "credible_interval must lie between 0 and 1."
        )

    # Obtain observed counts.
    if y_observed is None:
        processed_data = getattr(fit, "processed_data", None)
        
        if (
            processed_data is None
            or "expr" not in processed_data.columns
        ):
            raise ValueError(
                "y_observed was not provided and "
                "fit.processed_data does not contain an 'expr' column."
            )

        y_observed = processed_data["expr"].to_numpy()

    y_observed = np.asarray(y_observed, dtype=float).reshape(-1)

    if y_observed.size < 2:
        raise ValueError(
            "At least two observed values are required to calculate "
            "sample variance."
        )

    if not np.all(np.isfinite(y_observed)):
        raise ValueError(
            "y_observed contains non-finite values."
        )

    if np.any(y_observed < 0):
        raise ValueError(
            "y_observed contains negative counts."
        )

    # Extract replicated counts.
    try:
        y_rep = np.asarray(
            fit.fit.stan_variable("y_rep"),
            dtype=float,
        )
    except Exception as exc:
        raise KeyError(
            "The fitted Stan model does not contain 'y_rep'. "
            "Posterior predictive quantities must be generated "
            "by the Stan model."
        ) from exc

    if y_rep.ndim == 1 and y_observed.size == 1:
        y_rep = y_rep[:, None]

    if y_rep.ndim != 2:
        raise ValueError(
            "y_rep must have shape "
            "(posterior draws, observations). "
            f"Received shape {y_rep.shape}."
        )

    if y_rep.shape[1] != y_observed.size:
        raise ValueError(
            "The number of observed values does not match y_rep: "
            f"{y_observed.size} observed values versus "
            f"{y_rep.shape[1]} replicated observations."
        )

    if y_rep.shape[1] < 2:
        raise ValueError(
            "At least two replicated observations per posterior draw "
            "are required to calculate sample variance."
        )

    if not np.all(np.isfinite(y_rep)):
        raise ValueError(
            "y_rep contains non-finite values."
        )

    # Sample variance for each posterior predictive draw.
    replicated_variances = np.var(y_rep, axis=1, ddof=1)

    observed_variance = float(np.var(y_observed, ddof=1))

    lower_probability = (1.0 - credible_interval) / 2.0

    upper_probability = (1.0 + credible_interval) / 2.0

    interval_lower = float(np.quantile(replicated_variances, lower_probability))

    interval_upper = float(np.quantile(replicated_variances, upper_probability))

    replicated_median = float(np.median(replicated_variances))

    replicated_mean = float(np.mean(replicated_variances))

    # Two-sided posterior predictive p-value.
    probability_below = float(np.mean(replicated_variances <= observed_variance))

    ppc_pvalue = float(min(1.0, 2.0 * min(probability_below, 1.0 - probability_below)))

    # Apply transformation only for visualization.
    if log_scale:
        plotted_variances = np.log1p(
            replicated_variances
        )

        plotted_observed = float(
            np.log1p(observed_variance)
        )

        plotted_lower = float(
            np.log1p(interval_lower)
        )

        plotted_upper = float(
            np.log1p(interval_upper)
        )

        x_label = "log(1 + variance)"
    else:
        plotted_variances = replicated_variances
        plotted_observed = observed_variance
        plotted_lower = interval_lower
        plotted_upper = interval_upper
        x_label = "Variance of RNA-seq counts"

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
        )
    else:
        fig = ax.figure

    ax.hist(
        plotted_variances,
        bins=bins,
        color=histogram_color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.6,
        label="Replicated variances",
    )

    if show_interval:
        ax.axvspan(
            plotted_lower,
            plotted_upper,
            color=histogram_color,
            alpha=0.15,
            label=(
                f"{credible_interval:.0%} predictive interval"
            ),
        )

    ax.axvline(
        plotted_observed,
        color=observed_color,
        linestyle="--",
        linewidth=2.2,
        label=(
            f"Observed variance = "
            f"{observed_variance:.2f}"
        ),
    )

    gene_name = (
        gene
        if gene is not None
        else getattr(fit, "gene", None)
    )

    if title is None:
        if gene_name:
            title = (
                f"{gene_name}: posterior predictive check "
                "for variance"
            )
        else:
            title = (
                "PPC for variance"
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Posterior draws")
    ax.legend(frameon=False)

    fig.tight_layout()

    summary = {
        "gene": gene_name,
        "observed_variance": observed_variance,
        "replicated_variance_median": replicated_median,
        "replicated_variance_mean": replicated_mean,
        "predictive_interval_lower": interval_lower,
        "predictive_interval_upper": interval_upper,
        "credible_interval": credible_interval,
        "ppc_pvalue_two_sided": ppc_pvalue,
        "log_scale": bool(log_scale),
        "n_posterior_draws": int(
            replicated_variances.size
        ),
        "n_observations": int(
            y_observed.size
        ),
    }

    if show:
        plt.show()

    return fig, ax, summary


def _resolve_raw_fit(fit: Any) -> Any:
    """Return the raw CmdStan fit from a raw fit or BDGDMFit wrapper."""
    return getattr(fit, "fit", fit)


def _extract_y_rep(
    fit: Any,
    *,
    variable: str = "y_rep",
) -> np.ndarray:
    """Extract posterior-predictive draws with shape (draws, observations)."""
    raw_fit = _resolve_raw_fit(fit)
    stan_variable = getattr(raw_fit, "stan_variable", None)

    if not callable(stan_variable):
        raise TypeError(
            "fit must expose stan_variable(), directly or through fit.fit."
        )

    try:
        y_rep = np.asarray(stan_variable(variable), dtype=float)
    except Exception as exc:
        raise KeyError(
            f"Posterior-predictive variable {variable!r} was not found."
        ) from exc

    if y_rep.ndim == 1:
        y_rep = y_rep[:, None]

    if y_rep.ndim != 2:
        raise ValueError(
            f"{variable!r} must have shape (draws, observations); "
            f"received {y_rep.shape}."
        )

    if y_rep.size == 0 or not np.isfinite(y_rep).all():
        raise ValueError(
            f"{variable!r} must contain finite posterior-predictive draws."
        )

    return y_rep


def _resolve_transform(
    transform: str | Callable[[np.ndarray], np.ndarray],
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    if callable(transform):
        return transform, ""

    transforms = {
        "identity": (lambda values: values, ""),
        "log1p": (np.log1p, "log1p "),
        "sqrt": (np.sqrt, "sqrt "),
    }

    if transform not in transforms:
        raise ValueError(
            "transform must be 'identity', 'log1p', 'sqrt', or a callable."
        )

    return transforms[transform]


def _make_x_edges(
    x: np.ndarray,
    bins: int | str | Sequence[float],
) -> np.ndarray:
    if isinstance(bins, str) and bins == "integer":
        lower = np.floor(np.min(x)) - 0.5
        upper = np.ceil(np.max(x)) + 0.5
        return np.arange(lower, upper + 1.0, 1.0)

    edges = np.histogram_bin_edges(x, bins=bins)

    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Could not construct valid x-axis bins.")

    return edges


def _make_y_edges(
    y_observed: np.ndarray,
    y_rep: np.ndarray,
    bins: int | str | Sequence[float],
    predictive_quantiles: tuple[float, float],
) -> np.ndarray:
    q_low, q_high = predictive_quantiles

    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError(
            "predictive_quantiles must satisfy 0 <= lower < upper <= 1."
        )

    lower_rep, upper_rep = np.quantile(y_rep, [q_low, q_high])
    lower = float(min(lower_rep, np.min(y_observed)))
    upper = float(max(upper_rep, np.max(y_observed)))

    width = upper - lower
    padding = 0.04 * width if width > 0 else max(abs(lower) * 0.05, 0.5)
    lower -= padding
    upper += padding

    if isinstance(bins, Sequence) and not isinstance(bins, str):
        edges = np.asarray(bins, dtype=float)
    else:
        values = y_rep[(y_rep >= lower) & (y_rep <= upper)]
        edges = np.histogram_bin_edges(
            np.concatenate([values, y_observed]),
            bins=bins,
            range=(lower, upper),
        )

    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Could not construct valid y-axis bins.")

    return edges


def _scale_histogram(
    counts: np.ndarray,
    *,
    density_mode: str,
    n_draws: int,
) -> np.ndarray:
    if density_mode == "probability":
        total = counts.sum()
        return counts / total if total > 0 else counts

    if density_mode == "expected_count":
        return counts / float(n_draws)

    if density_mode == "count":
        return counts

    raise ValueError(
        "density_mode must be 'probability', 'expected_count', or 'count'."
    )


def _two_sided_ppc(
    replicated: np.ndarray,
    observed: float,
) -> float:
    p_lower = np.mean(replicated <= observed)
    p_upper = np.mean(replicated >= observed)
    return float(min(1.0, 2.0 * min(p_lower, p_upper)))


def plot_ppc_density(
    fit: Any,
    x_observed: Sequence[float] | np.ndarray,
    y_observed: Sequence[float] | np.ndarray,
    *,
    subtype_observed: Sequence[str] | np.ndarray | None = None,
    subtype_order: Sequence[str] | None = None,
    variable: str = "y_rep",
    gene: str | None = None,
    subtype: str | None = None,
    x_label: str = "Copy number",
    y_label: str = "Expression",
    transform: str | Callable[[np.ndarray], np.ndarray] = "identity",
    x_bins: int | str | Sequence[float] = 30,
    y_bins: int | str | Sequence[float] = 45,
    density_mode: str = "probability",
    max_draws: int | None = 2000,
    seed: int = 123,
    predictive_quantiles: tuple[float, float] = (0.001, 0.999),
    cmap: str = "Blues",
    observed_color: str = "#E84A36",
    observed_size: float = 30.0,
    observed_alpha: float = 0.90,
    selected_index: int | None = None,
    crosshair_color: str = "#E84A36",
    show_colorbar: bool = True,
    share_x: bool = True,
    share_y: bool = True,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
):
    """
    Plot posterior-predictive density, optionally faceted by subtype.

    Parameters
    ----------
    fit
        Raw CmdStan fit or a BDGDMFit wrapper whose ``fit`` attribute
        contains the raw CmdStan fit.

    x_observed
        Predictor values in the same row order used for fitting, usually
        ``processed["copies"]``.

    y_observed
        Observed outcomes in the same row order used for fitting, usually
        ``stan_data["y"]``.

    subtype_observed
        One subtype label per fitted row. When omitted, a single pooled
        panel is produced.

    subtype_order
        Facet order. Prefer the exact order stored by preprocessing, such
        as ``fit.metadata["subtype_levels"]``.

    density_mode
        ``"probability"`` normalizes each facet to sum to one and is the
        best default for comparing predictive shapes across subtypes.

        ``"expected_count"`` displays the expected number of replicated
        observations per posterior draw in each bin.

        ``"count"`` displays raw accumulated replicated observations.

    selected_index
        Optional zero-based row index to emphasize using dashed crosshairs.

    Returns
    -------
    fig, axes, summary
        ``axes`` is a one-dimensional NumPy array of visible facet axes.
        ``summary`` is a DataFrame with subtype-specific mean and variance
        posterior-predictive diagnostics.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    def resolve_raw_fit(value: Any) -> Any:
        return getattr(value, "fit", value)

    def extract_y_rep(value: Any) -> np.ndarray:
        raw_fit = resolve_raw_fit(value)
        stan_variable = getattr(raw_fit, "stan_variable", None)

        if not callable(stan_variable):
            raise TypeError(
                "fit must expose stan_variable(), directly or through fit.fit."
            )

        try:
            draws = np.asarray(
                stan_variable(variable),
                dtype=float,
            )
        except Exception as exc:
            raise KeyError(
                f"Posterior-predictive variable {variable!r} was not found."
            ) from exc

        if draws.ndim == 1:
            draws = draws[:, None]

        if draws.ndim != 2:
            raise ValueError(
                f"{variable!r} must have shape (draws, observations); "
                f"received {draws.shape}."
            )

        if draws.size == 0 or not np.isfinite(draws).all():
            raise ValueError(
                f"{variable!r} must contain finite posterior-predictive draws."
            )

        return draws

    def resolve_transform(
        value: str | Callable[[np.ndarray], np.ndarray],
    ) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        if callable(value):
            return value, ""

        transform_map = {
            "identity": (lambda array: array, ""),
            "log1p": (np.log1p, "log1p "),
            "sqrt": (np.sqrt, "sqrt "),
        }

        if value not in transform_map:
            raise ValueError(
                "transform must be 'identity', 'log1p', 'sqrt', or a callable."
            )

        return transform_map[value]

    def make_x_edges(
        values: np.ndarray,
        bins: int | str | Sequence[float],
    ) -> np.ndarray:
        if isinstance(bins, str):
            if bins != "integer":
                return np.histogram_bin_edges(values, bins=bins)

            lower = np.floor(np.min(values)) - 0.5
            upper = np.ceil(np.max(values)) + 0.5

            if upper <= lower:
                upper = lower + 1.0

            return np.arange(
                lower,
                upper + 1.0,
                1.0,
                dtype=float,
            )

        edges = np.histogram_bin_edges(
            values,
            bins=bins,
        )

        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(
                "Could not construct valid x-axis bins."
            )

        return edges

    def make_y_edges(
        observed: np.ndarray,
        replicated: np.ndarray,
        bins: int | str | Sequence[float],
    ) -> np.ndarray:
        q_low, q_high = predictive_quantiles

        if not 0.0 <= q_low < q_high <= 1.0:
            raise ValueError(
                "predictive_quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )

        predictive_low, predictive_high = np.quantile(
            replicated,
            [q_low, q_high],
        )

        lower = float(
            min(
                predictive_low,
                np.min(observed),
            )
        )
        upper = float(
            max(
                predictive_high,
                np.max(observed),
            )
        )

        width = upper - lower
        padding = (
            0.04 * width
            if width > 0
            else max(abs(lower) * 0.05, 0.5)
        )

        lower -= padding
        upper += padding

        if (
            isinstance(bins, Sequence)
            and not isinstance(bins, str)
        ):
            edges = np.asarray(
                bins,
                dtype=float,
            )
        else:
            in_range = replicated[
                (replicated >= lower)
                & (replicated <= upper)
            ]

            edges = np.histogram_bin_edges(
                np.concatenate(
                    [
                        in_range,
                        observed,
                    ]
                ),
                bins=bins,
                range=(lower, upper),
            )

        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(
                "Could not construct valid y-axis bins."
            )

        return edges

    def scale_histogram(
        counts: np.ndarray,
        *,
        number_of_draws: int,
    ) -> np.ndarray:
        if density_mode == "probability":
            total = counts.sum()
            return (
                counts / total
                if total > 0
                else counts
            )

        if density_mode == "expected_count":
            return counts / float(number_of_draws)

        if density_mode == "count":
            return counts

        raise ValueError(
            "density_mode must be 'probability', "
            "'expected_count', or 'count'."
        )

    def two_sided_ppc(
        replicated: np.ndarray,
        observed: float,
    ) -> float:
        p_lower = float(
            np.mean(replicated <= observed)
        )
        p_upper = float(
            np.mean(replicated >= observed)
        )

        return float(
            min(
                1.0,
                2.0 * min(p_lower, p_upper),
            )
        )

    # Validate observed data and extract predictive draws
  
    x = np.asarray(
        x_observed,
        dtype=float,
    ).reshape(-1)

    y = np.asarray(
        y_observed,
        dtype=float,
    ).reshape(-1)

    if x.size == 0:
        raise ValueError(
            "Observed arrays must not be empty."
        )

    if x.shape != y.shape:
        raise ValueError(
            "x_observed and y_observed must have the same length."
        )

    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "Observed values must all be finite."
        )

    y_rep = extract_y_rep(fit)

    if y_rep.shape[1] != y.size:
        raise ValueError(
            "Observed rows do not match posterior-predictive columns: "
            f"{y.size} observed rows versus {y_rep.shape[1]} columns. "
            "Use the exact processed rows used to fit the model."
        )

    # Thin predictive draws for plotting speed
  
    if max_draws is not None:
        if (
            not isinstance(max_draws, (int, np.integer))
            or max_draws <= 0
        ):
            raise ValueError(
                "max_draws must be a positive integer or None."
            )

        if y_rep.shape[0] > max_draws:
            rng = np.random.default_rng(seed)
            draw_indices = rng.choice(
                y_rep.shape[0],
                size=int(max_draws),
                replace=False,
            )
            y_rep = y_rep[draw_indices]

    # Transform the response
   
    transform_function, transform_prefix = resolve_transform(
        transform
    )

    with np.errstate(
        invalid="raise",
        divide="raise",
        over="raise",
    ):
        try:
            y_plot = np.asarray(
                transform_function(y),
                dtype=float,
            )
            y_rep_plot = np.asarray(
                transform_function(y_rep),
                dtype=float,
            )
        except FloatingPointError as exc:
            raise ValueError(
                "The selected response transformation is invalid "
                "for some observed or replicated values."
            ) from exc

    if (
        y_plot.shape != y.shape
        or y_rep_plot.shape != y_rep.shape
        or not np.isfinite(y_plot).all()
        or not np.isfinite(y_rep_plot).all()
    ):
        raise ValueError(
            "The response transformation must preserve shape and "
            "produce finite values."
        )

    # Resolve subtype facets
   
    if subtype_observed is None:
        pooled_label = (
            str(subtype)
            if subtype is not None
            else "All samples"
        )
        facet_values = np.repeat(
            pooled_label,
            y.size,
        )
        facet_order = [pooled_label]
    else:
        subtype_array = np.asarray(
            subtype_observed,
        ).reshape(-1)

        if subtype_array.size != y.size:
            raise ValueError(
                "subtype_observed must contain one label per fitted row."
            )

        if pd.isna(subtype_array).any():
            raise ValueError(
                "subtype_observed contains missing labels."
            )

        facet_values = subtype_array.astype(str)
        observed_levels = [
            str(level)
            for level in pd.unique(facet_values)
        ]

        if subtype_order is None:
            metadata = getattr(
                fit,
                "metadata",
                None,
            )
            stored_levels = None

            if isinstance(metadata, Mapping):
                stored_levels = metadata.get(
                    "subtype_levels"
                )

            facet_order = (
                [
                    str(level)
                    for level in stored_levels
                ]
                if stored_levels is not None
                else observed_levels
            )
        else:
            facet_order = [
                str(level)
                for level in subtype_order
            ]

        if len(facet_order) != len(set(facet_order)):
            raise ValueError(
                "subtype_order contains duplicate labels."
            )

        if set(facet_order) != set(observed_levels):
            raise ValueError(
                "subtype_order must contain exactly the observed "
                f"subtypes. Observed: {observed_levels}; "
                f"supplied: {facet_order}."
            )

    if selected_index is not None:
        if (
            not isinstance(selected_index, (int, np.integer))
            or not 0 <= int(selected_index) < y.size
        ):
            raise IndexError(
                "selected_index must be a valid zero-based row index."
            )

        selected_index = int(selected_index)

    # Shared bins and panel histograms
  
    x_edges = make_x_edges(
        x,
        x_bins,
    )
    y_edges = make_y_edges(
        y_plot,
        y_rep_plot,
        y_bins,
    )

    masks: list[np.ndarray] = []
    histograms: list[np.ndarray] = []

    for level in facet_order:
        mask = facet_values == level

        if not np.any(mask):
            raise ValueError(
                f"Subtype {level!r} contains no observations."
            )

        x_replicated = np.tile(
            x[mask],
            y_rep_plot.shape[0],
        )
        y_replicated = y_rep_plot[:, mask].reshape(-1)

        counts, _, _ = np.histogram2d(
            x_replicated,
            y_replicated,
            bins=(x_edges, y_edges),
        )

        masks.append(mask)
        histograms.append(
            scale_histogram(
                counts,
                number_of_draws=y_rep_plot.shape[0],
            )
        )

    maximum_density = max(
        float(np.max(histogram))
        for histogram in histograms
    )

    if (
        not np.isfinite(maximum_density)
        or maximum_density <= 0
    ):
        raise ValueError(
            "The posterior-predictive histogram is empty."
        )

    # Figure layout with a dedicated colorbar column
    
    number_of_facets = len(facet_order)

    if ncols is None:
        number_of_columns = min(
            number_of_facets,
            2,
        )
    else:
        if (
            not isinstance(ncols, (int, np.integer))
            or ncols <= 0
        ):
            raise ValueError(
                "ncols must be a positive integer."
            )

        number_of_columns = min(
            int(ncols),
            number_of_facets,
        )

    number_of_rows = int(
        np.ceil(
            number_of_facets
            / number_of_columns
        )
    )

    if figsize is None:
        figure_width = (
            5.3 * number_of_columns
            + (0.8 if show_colorbar else 0.0)
        )
        figure_height = (
            4.6 * number_of_rows
            + 0.8
        )
        figsize = (
            figure_width,
            figure_height,
        )

    fig = plt.figure(
        figsize=figsize,
    )

    width_ratios = [1.0] * number_of_columns

    if show_colorbar:
        width_ratios.append(0.045)

    grid = fig.add_gridspec(
        nrows=number_of_rows,
        ncols=(
            number_of_columns + 1
            if show_colorbar
            else number_of_columns
        ),
        width_ratios=width_ratios,
        left=0.08,
        right=0.93,
        bottom=0.18,
        top=0.84,
        wspace=0.10,
        hspace=0.28,
    )

    axes_list: list[Any] = []
    reference_axis = None

    for facet_index in range(number_of_facets):
        row = facet_index // number_of_columns
        column = facet_index % number_of_columns

        panel = fig.add_subplot(
            grid[row, column],
            sharex=(
                reference_axis
                if share_x
                else None
            ),
            sharey=(
                reference_axis
                if share_y
                else None
            ),
        )

        if reference_axis is None:
            reference_axis = panel

        axes_list.append(panel)

    colorbar_axis = (
        fig.add_subplot(
            grid[:, -1]
        )
        if show_colorbar
        else None
    )

    axes = np.asarray(
        axes_list,
        dtype=object,
    )

    normalization = Normalize(
        vmin=0.0,
        vmax=maximum_density,
    )

    summary_rows: list[dict[str, Any]] = []
    observed_artist = None
    mappable = None

    # Draw facets
   
    for facet_index, (
        level,
        mask,
        histogram,
    ) in enumerate(
        zip(
            facet_order,
            masks,
            histograms,
        )
    ):
        panel = axes[facet_index]

        mappable = panel.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_less_equal(
                histogram.T,
                0.0,
            ),
            cmap=cmap,
            norm=normalization,
            shading="auto",
            rasterized=True,
        )

        observed_artist = panel.scatter(
            x[mask],
            y_plot[mask],
            s=observed_size,
            color=observed_color,
            alpha=observed_alpha,
            edgecolors="white",
            linewidths=0.45,
            zorder=4,
            label="Observed",
        )

        if (
            selected_index is not None
            and mask[selected_index]
        ):
            selected_x = x[selected_index]
            selected_y = y_plot[selected_index]

            panel.axvline(
                selected_x,
                color=crosshair_color,
                linestyle=(0, (5, 4)),
                linewidth=1.8,
                zorder=3,
            )
            panel.axhline(
                selected_y,
                color=crosshair_color,
                linestyle=(0, (5, 4)),
                linewidth=1.8,
                zorder=3,
            )
            panel.scatter(
                [selected_x],
                [selected_y],
                s=observed_size * 1.8,
                color=observed_color,
                edgecolors="none",
                zorder=5,
            )

        observed_subset = y_plot[mask]
        replicated_subset = y_rep_plot[:, mask]

        replicated_means = replicated_subset.mean(
            axis=1
        )
        observed_mean = float(
            observed_subset.mean()
        )

        if observed_subset.size >= 2:
            replicated_variances = replicated_subset.var(
                axis=1,
                ddof=1,
            )
            observed_variance = float(
                observed_subset.var(
                    ddof=1
                )
            )
            variance_ppc = two_sided_ppc(
                replicated_variances,
                observed_variance,
            )
        else:
            observed_variance = np.nan
            variance_ppc = np.nan

        panel.set_title(
            f"{level} (n={int(mask.sum())})"
        )
        panel.set_xlabel(
            x_label
        )
        panel.grid(
            alpha=0.16
        )

        column = facet_index % number_of_columns

        if column == 0:
            panel.set_ylabel(
                f"{transform_prefix}{y_label}".strip()
            )

        summary_rows.append(
            {
                "subtype": str(level),
                "n_observations": int(mask.sum()),
                "n_predictive_draws": int(
                    y_rep_plot.shape[0]
                ),
                "observed_mean": observed_mean,
                "predictive_mean_median": float(
                    np.median(
                        replicated_means
                    )
                ),
                "mean_ppc_two_sided": two_sided_ppc(
                    replicated_means,
                    observed_mean,
                ),
                "observed_variance": float(
                    observed_variance
                ),
                "variance_ppc_two_sided": float(
                    variance_ppc
                ),
                "density_mode": density_mode,
            }
        )

    # Shared colorbar and observed-data legend

    if (
        show_colorbar
        and colorbar_axis is not None
        and mappable is not None
    ):
        colorbar_labels = {
            "probability": (
                "Posterior predictive probability per bin"
            ),
            "expected_count": (
                "Expected replicated observations per draw"
            ),
            "count": (
                "Accumulated replicated observations"
            ),
        }

        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_axis,
        )
        colorbar.set_label(
            colorbar_labels[density_mode],
            rotation=270,
            labelpad=22,
        )

    if observed_artist is not None:
        fig.legend(
            handles=[observed_artist],
            labels=["Observed"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            frameon=False,
            ncol=1,
        )

    title = "Posterior predictive check"

    if gene is not None:
        title += f" — {gene}"

    fig.suptitle(
        title,
        y=0.95,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        fig,
        axes,
        summary,
    )


def plot_copy_number_fit(
    fit,
    processed_df: pd.DataFrame,
    *,
    subtype_col: str = "subtype",
    cn_col: str = "copies",
    expr_col: str = "expr",
    posterior_var: str | None = None,
    cn_states: list[float] | None = None,
    round_cn: bool = True,
    credible_interval: float = 0.90,
    jitter: float = 0.06,
    point_alpha: float = 0.25,
    point_size: float = 20,
    max_columns: int = 3,
    random_seed: int = 123,
    show_sample_sizes: bool = True,
) -> tuple[Figure, pd.DataFrame]:
    """
    Plot the observed and fitted copy-number–expression relationship.

    The function shows:

    - jittered observed log1p RNA-seq counts;
    - observed mean expression at each copy-number state;
    - posterior fitted or predictive mean;
    - posterior credible intervals;
    - one panel per subtype.

    Parameters
    ----------
    fit
        BDGDMFit object returned by ``fit_one_gene_bdgdm``.

    processed_df
        Data frame returned by ``prepare_gene_data``. Its row order must
        be identical to the data used for model fitting.

    subtype_col
        Column containing subtype labels.

    cn_col
        Column containing copy-number values.

    expr_col
        Column containing observed RNA-seq counts.

    posterior_var
        Stan variable used for fitted values. When omitted, the function
        first tries ``mu_rep`` and then ``y_rep``.

        ``mu_rep`` represents posterior expected expression.
        ``y_rep`` represents posterior predictive replicated counts.

    cn_states
        Copy-number states to plot. When omitted, states are inferred
        from the processed data.

    round_cn
        Round copy-number values to the nearest integer before grouping.

    credible_interval
        Width of the central posterior credible interval.

    jitter
        Horizontal jitter applied to observed data points.

    point_alpha
        Transparency of observed points.

    point_size
        Size of observed points.

    max_columns
        Maximum number of subtype panels per row.

    random_seed
        Seed used for reproducible jitter.

    show_sample_sizes
        Display the number of samples under each copy-number state.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.

    pandas.DataFrame
        Summary values used to construct the figure.
    """
    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError("processed_df must be a pandas DataFrame.")

    if processed_df.empty:
        raise ValueError("processed_df is empty.")

    required_columns = {
        subtype_col,
        cn_col,
        expr_col,
    }

    missing_columns = required_columns.difference(
        processed_df.columns
    )

    if missing_columns:
        raise ValueError(
            "processed_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < credible_interval < 1:
        raise ValueError(
            "credible_interval must lie between 0 and 1."
        )

    if jitter < 0:
        raise ValueError("jitter cannot be negative.")

    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    df = processed_df.copy().reset_index(drop=True)

    for column in [subtype_col, cn_col, expr_col]:
        if df[column].isna().any():
            raise ValueError(
                f"Column {column!r} contains missing values. "
                "Use exactly the processed data used during fitting."
            )

    expression = df[expr_col].to_numpy(dtype=float)
    copy_number = df[cn_col].to_numpy(dtype=float)

    if np.any(expression < 0):
        raise ValueError(
            f"Column {expr_col!r} contains negative values."
        )

    if not np.all(np.isfinite(expression)):
        raise ValueError(
            f"Column {expr_col!r} contains non-finite values."
        )

    if not np.all(np.isfinite(copy_number)):
        raise ValueError(
            f"Column {cn_col!r} contains non-finite values."
        )

    # Preserve the row correspondence with the Stan output.
    df["_fit_row"] = np.arange(len(df))

    if round_cn:
        df["_cn_state"] = np.rint(copy_number).astype(int)
    else:
        df["_cn_state"] = copy_number

    # Select the posterior fitted-value variable.
    if posterior_var is None:
        candidate_variables = ["mu_rep", "y_rep"]
    else:
        candidate_variables = [posterior_var]

    posterior_draws = None
    selected_variable = None

    for candidate in candidate_variables:
        try:
            posterior_draws = np.asarray(
                fit.fit.stan_variable(candidate),
                dtype=float,
            )
            selected_variable = candidate
            break
        except Exception:
            continue

    if posterior_draws is None or selected_variable is None:
        try:
            available_variables = sorted(
                fit.fit.stan_variables().keys()
            )
        except Exception:
            available_variables = []

        raise KeyError(
            "Could not find a fitted-value Stan variable. "
            "Expected 'mu_rep' or 'y_rep'. "
            f"Available variables: {available_variables}"
        )

    if posterior_draws.ndim == 1 and len(df) == 1:
        posterior_draws = posterior_draws[:, None]

    if posterior_draws.ndim != 2:
        raise ValueError(
            f"{selected_variable!r} must have shape "
            "(posterior draws, samples), but its shape is "
            f"{posterior_draws.shape}."
        )

    if posterior_draws.shape[1] != len(df):
        raise ValueError(
            "The fitted-value array and processed_df do not have the "
            "same number of samples. "
            f"{selected_variable} has {posterior_draws.shape[1]} samples, "
            f"while processed_df has {len(df)} rows. "
            "Use the exact processed_df returned before fitting."
        )

    posterior_draws = np.clip(
        posterior_draws,
        a_min=0,
        a_max=None,
    )

    # Determine subtype order, preferring the order saved in fit metadata.
    observed_subtypes = list(
        pd.unique(df[subtype_col])
    )

    metadata_order = []

    if hasattr(fit, "metadata") and isinstance(
        fit.metadata,
        dict,
    ):
        metadata_order = fit.metadata.get(
            "subtype_order",
            fit.metadata.get("subtype_levels", []),
        )

    metadata_order = list(metadata_order or [])

    subtypes = [
        subtype
        for subtype in metadata_order
        if subtype in observed_subtypes
    ]

    subtypes.extend(
        subtype
        for subtype in observed_subtypes
        if subtype not in subtypes
    )

    if not subtypes:
        raise ValueError(
            f"No subtype labels were found in {subtype_col!r}."
        )

    # Infer the displayed CN states.
    observed_states = sorted(
        pd.unique(df["_cn_state"])
    )

    if cn_states is None:
        selected_states = observed_states
    else:
        selected_states = list(cn_states)

    if not selected_states:
        raise ValueError("No copy-number states are available to plot.")

    # Figure layout.
    n_panels = len(subtypes)
    n_columns = min(max_columns, n_panels)
    n_rows = math.ceil(n_panels / n_columns)

    figure_width = max(5.0, 4.8 * n_columns)
    figure_height = max(4.5, 4.1 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(figure_width, figure_height),
        sharey=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    palette = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get(
        "color",
        ["C0"],
    )

    subtype_colors = {
        subtype: palette[index % len(palette)]
        for index, subtype in enumerate(subtypes)
    }

    lower_probability = (1.0 - credible_interval) / 2.0

    upper_probability = (1.0 + credible_interval) / 2.0

    rng = np.random.default_rng(random_seed)
    summary_rows: list[dict] = []

    for panel_index, subtype in enumerate(subtypes):
        ax = axes_flat[panel_index]
        color = subtype_colors[subtype]

        subtype_mask = (
            df[subtype_col].to_numpy() == subtype
        )

        subtype_rows = np.flatnonzero(subtype_mask)

        subtype_df = df.iloc[subtype_rows]

        raw_x = subtype_df["_cn_state"].to_numpy(
            dtype=float
        )

        raw_y = np.log1p(
            subtype_df[expr_col].to_numpy(
                dtype=float
            )
        )

        jittered_x = raw_x + rng.uniform(
            low=-jitter,
            high=jitter,
            size=len(raw_x),
        )

        ax.scatter(
            jittered_x,
            raw_y,
            s=point_size,
            alpha=point_alpha,
            color=color,
            edgecolors="none",
            label="Observed samples",
            zorder=1,
        )

        fitted_states = []
        observed_means = []
        fitted_medians = []
        fitted_lowers = []
        fitted_uppers = []

        for state in selected_states:
            state_mask = (
                subtype_df["_cn_state"].to_numpy()
                == state
            )

            state_rows_local = np.flatnonzero(
                state_mask
            )

            if state_rows_local.size == 0:
                continue

            state_rows_global = subtype_rows[
                state_rows_local
            ]

            observed_values = np.log1p(
                df.iloc[state_rows_global][
                    expr_col
                ].to_numpy(dtype=float)
            )

            # Transform each sample-level posterior fitted value and
            # calculate the mean within each posterior draw.
            posterior_values = np.log1p(
                posterior_draws[
                    :,
                    state_rows_global,
                ]
            )

            posterior_state_means = np.mean(
                posterior_values,
                axis=1,
            )

            observed_mean = float(
                np.mean(observed_values)
            )

            fitted_median = float(
                np.median(posterior_state_means)
            )

            fitted_lower = float(
                np.quantile(
                    posterior_state_means,
                    lower_probability,
                )
            )

            fitted_upper = float(
                np.quantile(
                    posterior_state_means,
                    upper_probability,
                )
            )

            fitted_states.append(state)
            observed_means.append(observed_mean)
            fitted_medians.append(fitted_median)
            fitted_lowers.append(fitted_lower)
            fitted_uppers.append(fitted_upper)

            summary_rows.append(
                {
                    "subtype": subtype,
                    "cn_state": state,
                    "n_samples": int(
                        state_rows_local.size
                    ),
                    "observed_mean_log1p": observed_mean,
                    "fitted_median_log1p": fitted_median,
                    "fitted_lower_log1p": fitted_lower,
                    "fitted_upper_log1p": fitted_upper,
                    "credible_interval": credible_interval,
                    "posterior_variable": selected_variable,
                }
            )

        fitted_states_array = np.asarray(
            fitted_states,
            dtype=float,
        )

        observed_means_array = np.asarray(
            observed_means,
            dtype=float,
        )

        fitted_medians_array = np.asarray(
            fitted_medians,
            dtype=float,
        )

        fitted_lowers_array = np.asarray(
            fitted_lowers,
            dtype=float,
        )

        fitted_uppers_array = np.asarray(
            fitted_uppers,
            dtype=float,
        )

        if fitted_states_array.size > 0:
            if fitted_states_array.size > 1:
                ax.fill_between(
                    fitted_states_array,
                    fitted_lowers_array,
                    fitted_uppers_array,
                    color=color,
                    alpha=0.18,
                    label=(
                        f"{credible_interval:.0%} "
                        "credible interval"
                    ),
                    zorder=2,
                )
            else:
                ax.vlines(
                    fitted_states_array,
                    fitted_lowers_array,
                    fitted_uppers_array,
                    color=color,
                    linewidth=5,
                    alpha=0.25,
                    label=(
                        f"{credible_interval:.0%} "
                        "credible interval"
                    ),
                    zorder=2,
                )

            fitted_label = (
                "Posterior predictive mean"
                if selected_variable == "y_rep"
                else "Posterior fitted mean"
            )

            ax.plot(
                fitted_states_array,
                fitted_medians_array,
                marker="o",
                linewidth=2.2,
                color=color,
                label=fitted_label,
                zorder=3,
            )

            ax.plot(
                fitted_states_array,
                observed_means_array,
                marker="s",
                linestyle="--",
                linewidth=1.5,
                color="black",
                label="Observed mean",
                zorder=4,
            )

        if show_sample_sizes:
            for state in fitted_states:
                sample_count = int(
                    np.sum(
                        subtype_df[
                            "_cn_state"
                        ].to_numpy() == state
                    )
                )

                ax.text(
                    state,
                    0.02,
                    f"n={sample_count}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.axvline(
            2,
            linestyle=":",
            linewidth=1,
            color="gray",
            label="Diploid CN = 2",
            zorder=0,
        )

        ax.set_title(str(subtype))
        ax.set_xlabel("Copy-number state")
        ax.set_xticks(selected_states)

        state_min = float(min(selected_states))
        state_max = float(max(selected_states))

        ax.set_xlim(
            state_min - 0.35,
            state_max + 0.35,
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

    # Hide unused panels.
    for unused_index in range(
        n_panels,
        len(axes_flat),
    ):
        axes_flat[unused_index].set_visible(False)

    for row_index in range(n_rows):
        first_axis_index = row_index * n_columns

        if first_axis_index < n_panels:
            axes_flat[first_axis_index].set_ylabel(
                "log(1 + RNA-seq count)"
            )

    gene_name = getattr(
        fit,
        "gene",
        "Gene",
    )

    fig.suptitle(
        f"{gene_name} CN–expression relationship",
        fontsize=15,
    )

    # Construct one figure-level legend without duplicate labels.
    legend_handles = []
    legend_labels = []

    for ax in axes_flat[:n_panels]:
        handles, labels = ax.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(legend_labels)),
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    summary_df = pd.DataFrame(summary_rows)

    return fig, summary_df