from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "GENE_DOSAGE_CLASS_ORDER",
    "GENE_DOSAGE_CLASS_PALETTE",
    "canonicalize_gene_dosage_class",
    "resolve_gene_dosage_palette",
    "gene_dosage_class_color",
    "classification_wide_to_long",
    "plot_class_distribution",
    "plot_classification_parameter_map"
]

GENE_DOSAGE_CLASS_ORDER: tuple[str, ...] = (
    "DSG",
    "DCG",
    "HYPER",
    "Mixed",
    "DIG",
    "UNC",
)

GENE_DOSAGE_CLASS_PALETTE: dict[str, str] = {
    "DSG": "#009E73",
    "DCG": "#0072B2",
    "HYPER": "#D55E00",
    "Mixed": "#CC79A7",
    "DIG": "#E69F00",
    "UNC": "#7A7A7A",
    "Missing": "#BDBDBD",
}

_CLASS_ALIASES: dict[str, str] = {
    "dsg": "DSG",
    "dcg": "DCG",
    "hyper": "HYPER",
    "mixed": "Mixed",
    "dig": "DIG",
    "unc": "UNC",
    "unknown": "UNC",
    "unclassified": "UNC",
    "missing": "Missing",
    "nan": "Missing",
    "none": "Missing",
}

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "gene": (
        "gene",
        "gene_name",
    ),
    "subtype": (
        "subtype",
        "subtype_label",
        "group",
        "analysis_group",
    ),
    "classification": (
        "response_class",
        "classification",
        "class",
        "gene_dosage_class",
        "dosage_class",
        "label",
    ),
    "confidence": (
        "classification_confidence",
        "confidence",
        "class_probability",
        "max_probability",
        "posterior_confidence",
    ),
    "scaling_median": (
        "b_scaling_median",
        "b_scaling_med",
        "median_b_scaling",
        "scaling_median",
        "beta_scaling_median",
    ),
    "deviation_median": (
        "b_deviation_median",
        "b_dev_median",
        "median_b_deviation",
        "deviation_median",
        "beta_deviation_median",
    ),
    "scaling_positive": (
        "p_scaling_positive",
        "p_scaling_pos",
        "p_b_scaling_positive",
        "p_bscaling_positive",
    ),
    "scaling_rope": (
        "p_rope_scaling",
        "p_scaling_rope",
        "p_b_scaling_rope",
    ),
    "scaling_negative": (
        "p_scaling_negative",
        "p_scaling_neg",
        "p_b_scaling_negative",
        "p_bscaling_negative",
    ),
    "deviation_positive": (
        "p_dev_positive",
        "p_deviation_positive",
        "p_b_deviation_positive",
        "p_bdev_positive",
    ),
    "deviation_rope": (
        "p_rope_dev",
        "p_rope_deviation",
        "p_deviation_rope",
        "p_b_deviation_rope",
    ),
    "deviation_negative": (
        "p_dev_negative",
        "p_deviation_negative",
        "p_b_deviation_negative",
        "p_bdev_negative",
    ),
}


def canonicalize_gene_dosage_class(
    value: Any,
) -> str:
    """Return a canonical gene-dosage class label."""
    if value is None or pd.isna(value):
        return "Missing"

    text = str(value).strip()

    if not text:
        return "Missing"

    return _CLASS_ALIASES.get(
        text.casefold(),
        text,
    )

def resolve_gene_dosage_palette(
    palette: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Return the default palette, optionally updated with user colors.

    A partial custom palette is allowed.
    """
    resolved = dict(
        GENE_DOSAGE_CLASS_PALETTE
    )

    if palette is not None:
        resolved.update(
            {
                canonicalize_gene_dosage_class(key): value
                for key, value in palette.items()
            }
        )

    return resolved


def gene_dosage_class_color(
    value: Any,
    *,
    palette: Mapping[str, str] | None = None,
    fallback: str = "#9E9E9E",
) -> str:
    """Return the plotting color for one class."""
    resolved = resolve_gene_dosage_palette(
        palette
    )
    canonical = canonicalize_gene_dosage_class(
        value
    )

    return resolved.get(
        canonical,
        fallback,
    )


def _resolve_column(
    frame: pd.DataFrame,
    role: str,
    explicit: str | None = None,
    *,
    required: bool = True,
) -> str | None:
    """Resolve a column by explicit name or common aliases."""
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(
                f"Column {explicit!r} was not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return explicit

    for candidate in _COLUMN_ALIASES.get(role, ()):
        if candidate in frame.columns:
            return candidate

    if required:
        raise KeyError(
            f"Could not identify a column for {role!r}. "
            f"Supply it explicitly. Available columns: "
            f"{list(frame.columns)}"
        )

    return None


def _as_dataframe(
    data: pd.DataFrame | pd.Series | Mapping[str, Any],
) -> pd.DataFrame:
    """Normalize supported tabular inputs to a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, pd.Series):
        return data.to_frame().T

    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])

    raise TypeError(
        "data must be a pandas DataFrame, pandas Series, or mapping."
    )


def _finite_numeric(
    values: pd.Series,
    *,
    name: str,
) -> np.ndarray:
    """Convert a Series to a finite float array."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"{name} contains missing or non-finite values."
        )

    return numeric


def _normalize_probability_triplet(
    negative: float,
    rope: float,
    positive: float,
    *,
    label: str,
) -> np.ndarray:
    """Validate and normalize a negative/ROPE/positive triplet."""
    values = np.asarray(
        [negative, rope, positive],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} probabilities must be finite."
        )

    if np.any(values < 0):
        raise ValueError(
            f"{label} probabilities must be nonnegative."
        )

    total = float(values.sum())

    if total <= 0:
        raise ValueError(
            f"{label} probabilities sum to zero."
        )

    return values / total


def classification_wide_to_long(
    classified_df: pd.DataFrame,
    *,
    subtype_slots: Sequence[str] = ("s1", "s2"),
    drop_empty_slots: bool = True,
) -> pd.DataFrame:
    """
    Convert BDGDM wide subtype output to one row per gene and subtype.

    Columns ending in ``_s1`` and ``_s2`` are stripped of their suffixes.
    Unsuffixed columns are retained as common gene-level or rewiring-level
    metadata.

    Parameters
    ----------
    classified_df
        Standard wide BDGDM classification output.

    subtype_slots
        Suffix slots to convert. The defaults are ``("s1", "s2")``.

    drop_empty_slots
        Drop rows whose converted ``subtype`` value is missing.

    Returns
    -------
    pandas.DataFrame
        Long table with a ``subtype_slot`` column.
    """
    if not isinstance(classified_df, pd.DataFrame):
        raise TypeError(
            "classified_df must be a pandas DataFrame."
        )

    slots = [
        str(slot).removeprefix("_")
        for slot in subtype_slots
    ]

    suffixed_columns: set[str] = set()

    for slot in slots:
        suffix = f"_{slot}"
        suffixed_columns.update(
            column
            for column in classified_df.columns
            if column.endswith(suffix)
        )

    if not suffixed_columns:
        raise ValueError(
            "No subtype-specific columns ending in the requested "
            f"suffixes were found: {slots}."
        )

    common_columns = [
        column
        for column in classified_df.columns
        if column not in suffixed_columns
    ]

    parts: list[pd.DataFrame] = []

    for slot in slots:
        suffix = f"_{slot}"
        slot_columns = [
            column
            for column in classified_df.columns
            if column.endswith(suffix)
        ]

        if not slot_columns:
            continue

        rename_map = {
            column: column.removesuffix(suffix)
            for column in slot_columns
        }

        part = pd.concat(
            [
                classified_df[common_columns].copy(),
                classified_df[slot_columns]
                .rename(columns=rename_map)
                .copy(),
            ],
            axis=1,
        )

        part["subtype_slot"] = slot

        if (
            drop_empty_slots
            and "subtype" in part.columns
        ):
            part = part.loc[
                part["subtype"].notna()
            ].copy()

        parts.append(part)

    if not parts:
        raise ValueError(
            "No subtype slots could be converted."
        )

    return pd.concat(
        parts,
        ignore_index=True,
        sort=False,
    )


def plot_class_distribution(
    classifications: pd.DataFrame,
    *,
    classification_col: str | None = None,
    group_col: str | None = None,
    class_order: Sequence[str] = GENE_DOSAGE_CLASS_ORDER,
    palette: Mapping[str, str] | None = None,
    normalize: bool = False,
    include_missing: bool = False,
    title: str | None = None,
    ax: Any | None = None,
):
    """
    Plot classification counts or proportions.

    When ``group_col`` is supplied, each group receives one stacked bar.
    For example, after converting to long format, use
    ``group_col="subtype"``.
    """
    frame = _as_dataframe(classifications)
    class_column = _resolve_column(
        frame,
        "classification",
        classification_col,
    )

    if group_col is not None and group_col not in frame.columns:
        raise KeyError(
            f"Group column {group_col!r} was not found."
        )

    working = frame.copy()

    if include_missing:
        working[class_column] = working[
            class_column
        ].map(canonicalize_gene_dosage_class)
    else:
        working = working.loc[
            working[class_column].notna()
        ].copy()
        working[class_column] = working[
            class_column
        ].map(canonicalize_gene_dosage_class)
        working = working.loc[
            working[class_column] != "Missing"
        ].copy()

    if working.empty:
        raise ValueError(
            "No classification rows are available for plotting."
        )

    class_palette = resolve_gene_dosage_palette(
        palette
    )

    observed_classes = [
        canonicalize_gene_dosage_class(value)
        for value in pd.unique(
            working[class_column]
        )
    ]
    requested_order = [
        canonicalize_gene_dosage_class(value)
        for value in class_order
    ]

    final_order = (
        requested_order
        + [
            value
            for value in observed_classes
            if value not in requested_order
        ]
    )

    if include_missing and "Missing" not in final_order:
        final_order.append("Missing")

    if group_col is None:
        counts = (
            working[class_column]
            .value_counts()
            .reindex(
                final_order,
                fill_value=0,
            )
        )

        if normalize:
            denominator = int(counts.sum())
            plot_values = (
                counts / denominator
                if denominator > 0
                else counts.astype(float)
            )
        else:
            plot_values = counts.astype(float)

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(5.0, 5.0)
            )
        else:
            fig = ax.figure

        bar_colors = [
            class_palette.get(
                class_label,
                "#9E9E9E",
            )
            for class_label in plot_values.index
        ]

        bars = ax.bar(
            plot_values.index,
            plot_values.to_numpy(),
            color=bar_colors,
        )

        for bar, value in zip(
            bars,
            plot_values.to_numpy(),
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                (
                    f"{value:.1%}"
                    if normalize
                    else str(int(value))
                ),
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Classification")
        ax.set_ylabel(
            "Proportion"
            if normalize
            else "Number of genes"
        )

        total = max(int(counts.sum()), 1)
        summary = pd.DataFrame(
            {
                "classification": counts.index,
                "count": counts.to_numpy(dtype=int),
                "proportion": (
                    counts.to_numpy(dtype=float)
                    / total
                ),
            }
        )

    else:
        working[group_col] = working[
            group_col
        ].astype(str)

        table = pd.crosstab(
            working[group_col],
            working[class_column],
        ).reindex(
            columns=final_order,
            fill_value=0,
        )

        if normalize:
            plot_table = table.div(
                table.sum(axis=1).replace(
                    0,
                    np.nan,
                ),
                axis=0,
            ).fillna(0.0)
        else:
            plot_table = table.astype(float)

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(5.0, 5.0)
            )
        else:
            fig = ax.figure

        bottom = np.zeros(
            len(plot_table),
            dtype=float,
        )

        for class_label in plot_table.columns:
            values = plot_table[
                class_label
            ].to_numpy(dtype=float)

            ax.bar(
                plot_table.index,
                values,
                bottom=bottom,
                label=class_label,
                color=class_palette.get(
                    class_label,
                    "#9E9E9E",
                ),
            )
            bottom += values

        ax.set_xlabel(group_col)
        ax.set_ylabel(
            "Proportion"
            if normalize
            else "Number of genes"
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(
                4,
                max(
                    1,
                    len(plot_table.columns),
                ),
            ),
            frameon=False,
        )

        summary = (
            table
            .stack(future_stack=True)
            .rename("count")
            .reset_index()
            .rename(
                columns={
                    group_col: "group",
                    class_column: "classification",
                }
            )
        )
        summary["proportion"] = summary.groupby(
            "group"
        )["count"].transform(
            lambda values: (
                values / values.sum()
                if values.sum() > 0
                else 0.0
            )
        )

    if title is None:
        title = (
            "Gene-dosage classification composition"
            if group_col is None
            else f"Gene-dosage classifications by {group_col}"
        )

    ax.set_title(title)
    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    if group_col is not None:
        fig.subplots_adjust(bottom=0.23)

    return fig, ax, summary


def plot_classification_parameter_map(
    classifications: pd.DataFrame,
    *,
    scaling_col: str | None = None,
    deviation_col: str | None = None,
    classification_col: str | None = None,
    confidence_col: str | None = None,
    gene_col: str | None = None,
    subtype_col: str | None = None,
    annotate: bool = False,
    point_size: float = 55.0,
    confidence_size_range: tuple[float, float] = (
        35.0,
        150.0,
    ),
    class_order: Sequence[str] = GENE_DOSAGE_CLASS_ORDER,
    palette: Mapping[str, str] | None = None,
    x_reference: float = 0.0,
    y_reference: float = 0.0,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "Gene-dosage classification map",
    ax: Any | None = None,
):
    """
    Plot scaling versus deviation statistics colored by response class.

    Point size represents confidence when a confidence column is supplied.
    """
    import matplotlib.pyplot as plt

    frame = _as_dataframe(classifications)

    scaling_column = _resolve_column(
        frame,
        "scaling_median",
        scaling_col,
    )
    deviation_column = _resolve_column(
        frame,
        "deviation_median",
        deviation_col,
    )
    class_column = _resolve_column(
        frame,
        "classification",
        classification_col,
    )
    confidence_column = _resolve_column(
        frame,
        "confidence",
        confidence_col,
        required=False,
    )
    gene_column = _resolve_column(
        frame,
        "gene",
        gene_col,
        required=False,
    )
    subtype_column = _resolve_column(
        frame,
        "subtype",
        subtype_col,
        required=False,
    )

    working = frame.loc[
        frame[
            [
                scaling_column,
                deviation_column,
                class_column,
            ]
        ].notna().all(axis=1)
    ].copy()

    if working.empty:
        raise ValueError(
            "No complete rows are available for the parameter map."
        )

    working[class_column] = working[
        class_column
    ].map(canonicalize_gene_dosage_class)

    x = _finite_numeric(
        working[scaling_column],
        name=scaling_column,
    )
    y = _finite_numeric(
        working[deviation_column],
        name=deviation_column,
    )

    if confidence_column is not None:
        confidence = np.clip(
            _finite_numeric(
                working[confidence_column],
                name=confidence_column,
            ),
            0.0,
            1.0,
        )
        size_low, size_high = (
            float(confidence_size_range[0]),
            float(confidence_size_range[1]),
        )

        if size_high < size_low or size_low <= 0:
            raise ValueError(
                "confidence_size_range must contain positive "
                "increasing values."
            )

        sizes = (
            size_low
            + confidence
            * (size_high - size_low)
        )
    else:
        sizes = np.repeat(
            float(point_size),
            len(working),
        )

    class_palette = resolve_gene_dosage_palette(
        palette
    )
    ordered_classes = _ordered_observed_classes(
        working[class_column],
        class_order,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(5.0, 5.0)
        )
    else:
        fig = ax.figure

    class_values = working[
        class_column
    ].to_numpy(dtype=str)

    for class_label in ordered_classes:
        mask = class_values == class_label

        ax.scatter(
            x[mask],
            y[mask],
            s=sizes[mask],
            color=class_palette.get(
                class_label,
                "#9E9E9E",
            ),
            alpha=0.78,
            label=class_label,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.axvline(
        float(x_reference),
        linestyle="--",
        linewidth=1.2,
    )
    ax.axhline(
        float(y_reference),
        linestyle="--",
        linewidth=1.2,
    )

    if annotate:
        for row_position, (_, row) in enumerate(
            working.iterrows()
        ):
            label_parts: list[str] = []

            if gene_column is not None:
                label_parts.append(
                    str(row[gene_column])
                )

            if subtype_column is not None:
                label_parts.append(
                    str(row[subtype_column])
                )

            label = " | ".join(label_parts)

            if label:
                ax.annotate(
                    label,
                    (
                        x[row_position],
                        y[row_position],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                )

    ax.set_xlabel(
        x_label
        if x_label is not None
        else scaling_column
    )
    ax.set_ylabel(
        y_label
        if y_label is not None
        else deviation_column
    )
    ax.set_title(title)
    ax.grid(alpha=0.20)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(
            4,
            max(
                1,
                len(ordered_classes),
            ),
        ),
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    summary_columns = [
        class_column,
        scaling_column,
        deviation_column,
    ]

    for optional_column in (
        gene_column,
        subtype_column,
        confidence_column,
    ):
        if (
            optional_column is not None
            and optional_column not in summary_columns
        ):
            summary_columns.append(
                optional_column
            )

    return (
        fig,
        ax,
        working[
            summary_columns
        ].reset_index(drop=True),
    )