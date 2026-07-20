from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

"""

Utilities for preparing Stan BDGDM inputs.

"""
# Metadata object

@dataclass(frozen=True)
class BDGDMMetadata:
    """Metadata describing the processed one-gene dataset."""

    gene: str | None
    analysis_mode: str
    subtype_levels: list[str]
    S: int
    N: int
    n_aneup: int
    cna: str


def prepare_gene_data(
    gene_df: pd.DataFrame,
    gene: str | None = None,
    *,
    subtype_col: str = "subtype",
    cna: str = "all",
    et: float = 0.15,
    min_aneup: int = 5,
    min_unique_counts: int = 5,
    min_cn_abs_sum: float = 1.0,
    subtype_order: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], BDGDMMetadata]:
    """
    Validate one-gene data and construct input for the appropriate Stan model.

    The returned Stan dictionary depends on the inferred analysis mode:

    ``single_group``
        Returns only variables declared by ``bdgdm_single.stan``:
        ``N``, ``y``, ``sf``, ``purity``, ``dose_log``, and ``dev``.

    ``subtype_comparison``
        Also returns ``S`` and ``subtype`` for
        ``bdgdm_subtype.stan``.

    Notes
    -----
    ``cna="amp"`` retains diploid/reference and gain samples by removing clear losses. 
    ``cna="del"`` retains diploid/reference and loss samples by removing clear gains.
    """
    if not isinstance(gene_df, pd.DataFrame):
        raise TypeError("gene_df must be a pandas DataFrame.")

    if not 0 <= et < 1:
        raise ValueError("et must satisfy 0 <= et < 1.")

    if min_aneup < 0:
        raise ValueError("min_aneup cannot be negative.")

    if min_unique_counts < 1:
        raise ValueError("min_unique_counts must be at least 1.")

    if min_cn_abs_sum < 0:
        raise ValueError("min_cn_abs_sum cannot be negative.")

    df = gene_df.copy()


    # Select exactly one gene.

    if "gene" in df.columns:
        if gene is not None:
            df = df.loc[df["gene"].astype(str) == str(gene)].copy()
        else:
            available_genes = (
                df["gene"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            if len(available_genes) != 1:
                raise ValueError(
                    "gene was not supplied and gene_df contains "
                    f"{len(available_genes)} genes. Select exactly one gene."
                )

            gene = available_genes[0]
            df = df.loc[df["gene"].astype(str) == gene].copy()

    if df.empty:
        raise ValueError("No rows remain after gene filtering.")

    
    # Required columns and missing values.

    required = [
        "expr",
        "copies",
        "purity",
        "sf",
        subtype_col,
    ]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("All rows were removed after NA filtering.")

    # Explicit numeric conversion prevents silent object/string handling.
    for column in ["expr", "copies", "purity", "sf"]:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column {column!r} must contain numeric values."
            ) from exc

    numeric_values = df[["expr", "copies", "purity", "sf"]].to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "expr, copies, purity, and sf must contain only finite values."
        )

    # Stan's count outcome is integer-valued. Do not silently truncate.
    expr_values = df["expr"].to_numpy(dtype=float)

    if (expr_values < 0).any():
        raise ValueError("Expression counts must be non-negative.")

    if not np.allclose(expr_values, np.rint(expr_values)):
        raise ValueError(
            "Expression counts must be integer-valued; "
            "non-integer values would otherwise be truncated."
        )

    if (df["copies"] < 0).any():
        raise ValueError("Copy-number values must be non-negative.")

    if not df["purity"].between(0, 1, inclusive="both").all():
        raise ValueError("Purity must lie in [0, 1].")

    if not (df["sf"] > 0).all():
        raise ValueError("Size factors must be strictly positive.")

    # ------------------------------------------------------------------
    # CNA-focused filtering.
    #
    # These definitions retain the diploid reference group:
    #   amp -> remove clear losses
    #   del -> remove clear gains
    # ------------------------------------------------------------------
    if cna == "amp":
        df = df.loc[df["copies"] > (2.0 - et)].copy()
    elif cna == "del":
        df = df.loc[df["copies"] < (2.0 + et)].copy()
    elif cna != "all":
        raise ValueError("cna must be 'all', 'amp', or 'del'.")

    if df.empty:
        raise ValueError("No samples remain after CNA filtering.")

    df = df.reset_index(drop=True)

    if df["expr"].nunique() < min_unique_counts:
        raise ValueError(
            "Too few unique expression counts after filtering: "
            f"{df['expr'].nunique()} < {min_unique_counts}."
        )

    if (df["expr"] == 0).all():
        raise ValueError("All expression counts are zero.")

    # Copy-number support.
    
    aneuploid_mask = np.abs(df["copies"] - 2.0) > (1.0 - et)
    n_aneup = int(aneuploid_mask.sum())

    if n_aneup < min_aneup:
        raise ValueError(
            f"Only {n_aneup} aneuploid samples; "
            f"at least {min_aneup} are required."
        )

    dev = (df["copies"] - 2.0) / 2.0

    if cna == "all" and float(np.abs(dev).sum()) < min_cn_abs_sum:
        raise ValueError(
            "Insufficient copy-number variation: "
            f"sum(abs(dev))={float(np.abs(dev).sum()):.3f} "
            f"< {min_cn_abs_sum}."
        )

    # Encode subtype.
    
    df[subtype_col] = df[subtype_col].astype(str)
    observed_levels = df[subtype_col].drop_duplicates().tolist()

    if subtype_order is None:
        subtype_levels = sorted(observed_levels)
    else:
        subtype_levels = [str(value) for value in subtype_order]

        if len(subtype_levels) != len(set(subtype_levels)):
            raise ValueError("subtype_order contains duplicate levels.")

        observed_set = set(observed_levels)
        ordered_set = set(subtype_levels)

        missing_from_order = sorted(observed_set - ordered_set)
        unused_in_order = sorted(ordered_set - observed_set)

        if missing_from_order:
            raise ValueError(
                "subtype_order is missing observed subtype(s): "
                f"{missing_from_order}"
            )

        if unused_in_order:
            raise ValueError(
                "subtype_order contains subtype(s) absent after filtering: "
                f"{unused_in_order}"
            )

    categorical = pd.Categorical(
        df[subtype_col],
        categories=subtype_levels,
        ordered=True,
    )

    if categorical.isna().any():
        bad = sorted(
            df.loc[categorical.isna(), subtype_col]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Unknown subtype(s): {bad}")

    subtype_codes = categorical.codes.astype(np.int64) + 1
    number_of_subtypes = len(subtype_levels)

    if number_of_subtypes < 1:
        raise ValueError("At least one subtype/group is required.")

    analysis_mode = (
        "single_group"
        if number_of_subtypes == 1
        else "subtype_comparison"
    )

    # ------------------------------------------------------------------
    # Model covariates.
    #
    # CN values below 1 are floored only inside the logarithmic term to
    # avoid log(0). The linear deviation term retains the original CN.
    # ------------------------------------------------------------------
    effective_cn = df["copies"].clip(lower=1.0)
    df["dose_log"] = np.log(effective_cn / 2.0)
    df["dev"] = dev.to_numpy(dtype=float)

    common_stan_data: dict[str, Any] = {
        "N": int(len(df)),
        "y": np.rint(df["expr"]).astype(np.int64).to_numpy(),
        "sf": df["sf"].astype(float).to_numpy(),
        "purity": df["purity"].astype(float).to_numpy(),
        "dose_log": df["dose_log"].astype(float).to_numpy(),
        "dev": df["dev"].astype(float).to_numpy(),
    }

    if analysis_mode == "single_group":
        # bdgdm_single.stan no longer declares S or subtype.
        stan_data = common_stan_data
    else:
        # bdgdm_subtype.stan requires S and one-based subtype indices.
        stan_data = {
            **common_stan_data,
            "S": int(number_of_subtypes),
            "subtype": subtype_codes,
        }

    metadata = BDGDMMetadata(
        gene=None if gene is None else str(gene),
        analysis_mode=analysis_mode,
        subtype_levels=subtype_levels,
        S=int(number_of_subtypes),
        N=int(len(df)),
        n_aneup=n_aneup,
        cna=cna,
    )

    return df, stan_data, metadata
