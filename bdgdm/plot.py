from __future__ import annotations

import math
from typing import Any, Sequence

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