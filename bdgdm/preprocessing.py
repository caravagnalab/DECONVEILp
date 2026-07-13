from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

"""

Utilities for preparing Stan BDGDM inputs.

"""
# Metadata object

@dataclass
class BDGDMMetadata:
    gene: str
    analysis_mode: str
    subtype_levels: list[str]
    S: int
    N: int
    n_aneup: int
    cna: str


# Main preprocessing function

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
):
    """
    Validate one-gene input table and construct Stan input.

    Required columns
    ----------------
    expr
    copies
    purity
    sf
    subtype

    Returns
    -------
    df
        Processed dataframe.

    stan_data
        Dictionary passed directly to CmdStan.

    metadata
        BDGDMMetadata object.
    """

    df = gene_df.copy()

    # Subset gene
    
    if gene is not None and "gene" in df.columns:
        df = df.loc[df["gene"] == gene].copy()

    if df.empty:
        raise ValueError("No rows remaining after gene filtering.")

    if gene is None and "gene" in df.columns:
        gene = str(df["gene"].iloc[0])

    # Required columns
    
    required = {
        "expr",
        "copies",
        "purity",
        "sf",
        subtype_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # CNA filtering
    
    if cna == "amp":
        df = df[df["copies"] > (2 - et)]

    elif cna == "del":
        df = df[df["copies"] < (2 + et)]

    elif cna == "all":
        pass

    else:
        raise ValueError(
            "cna must be 'all', 'amp' or 'del'."
        )

    if df.empty:
        raise ValueError(
            "No samples remaining after CNA filtering."
        )

    # Remove missing values
    
    df = df.dropna(subset=list(required))
    
    if df.empty:
        raise ValueError(
            "All rows removed after NA filtering."
        )
        
    # Input validation
    
    if (df["expr"] < 0).any():
        raise ValueError("Expression counts must be non-negative.")

    if not df["purity"].between(0, 1).all():
        raise ValueError("Purity must lie in [0,1].")

    if not (df["sf"] > 0).all():
        raise ValueError("Size factors must be positive.")

    if df["expr"].nunique() < min_unique_counts:
        raise ValueError(
            "Too few unique expression counts."
        )

    # CNA support
    
    n_aneup = int(
        (np.abs(df["copies"] - 2) > (1 - et)).sum()
    )

    if n_aneup < min_aneup:
        raise ValueError(
            f"Only {n_aneup} aneuploid samples."
        )

    if (df["expr"] == 0).all():
        raise ValueError(
            "All expression counts are zero."
        )

    dev_tmp = (df["copies"] - 2) / 2

    if (
        cna == "all"
        and np.abs(dev_tmp).sum() < min_cn_abs_sum
    ):
        raise ValueError(
            "Insufficient CN variation."
        )

    # Encode subtype
    
    if subtype_order is None:

        subtype_levels = sorted(
            pd.unique(df[subtype_col]).tolist()
        )

    else:

        subtype_levels = subtype_order

    cat = pd.Categorical(
        df[subtype_col],
        categories=subtype_levels,
        ordered=True,
    )

    if cat.isna().any():

        bad = df.loc[
            cat.isna(),
            subtype_col,
        ].unique()

        raise ValueError(
            f"Unknown subtype(s): {bad}"
        )

    subtype = cat.codes + 1

    S = len(subtype_levels)

    analysis_mode = (
        "single_group"
        if S == 1
        else "subtype_comparison"
    )

    # CN covariates
    
    CN_eff = df["copies"].clip(lower=1.0)
    df["dose_log"] = np.log(CN_eff / 2)
    df["dev"] = (df["copies"] - 2) / 2

    # Stan data
    
    stan_data = {
        "N": len(df),
        "y": df["expr"].astype(int).to_numpy(),
        "S": S,
        "subtype": subtype.astype(int),
        "sf": df["sf"].to_numpy(float),
        "purity": df["purity"].to_numpy(float),
        "dose_log": df["dose_log"].to_numpy(float),
        "dev": df["dev"].to_numpy(float),
    }

    metadata = BDGDMMetadata(
        gene=gene,
        analysis_mode=analysis_mode,
        subtype_levels=subtype_levels,
        S=S,
        N=len(df),
        n_aneup=n_aneup,
        cna=cna,
    )

    return df, stan_data, metadata
