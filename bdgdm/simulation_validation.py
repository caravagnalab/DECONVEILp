from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

"""
Simulation utilities for validating ``bdgdm_single.stan``.
The expected bulk-tumour count is
    mu_n =
        sf_n * purity_n
        * exp(b0 + dose_log_n * b_scaling + dev_n * b_deviation)
        + sf_n * (1 - purity_n) * exp(b_noncancer_log)

and counts follow NB(mu, phi), where

    Var(Y_n) = mu_n + mu_n**2 / phi.
"""


SCENARIOS: tuple[str, ...] = (
    "null",
    "scaling",
    "deviation_pos",
    "deviation_neg",
    "mixed",
)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def _probabilities(
    values: Sequence[float],
    *,
    expected_length: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.shape != (expected_length,):
        raise ValueError(
            f"{name} must contain {expected_length} values."
        )

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")

    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative.")

    total = float(array.sum())

    if total <= 0:
        raise ValueError(f"{name} must have a positive sum.")

    return array / total


# Sample-level covariates

def simulate_covariates(
    N: int = 200,
    subtype_label: str = "all_samples",
    seed: int = 123,
    copy_probs: Sequence[float] = (
        0.15,
        0.50,
        0.15,
        0.10,
        0.10,
    ),
    purity_a: float = 5.0,
    purity_b: float = 2.0,
    sf_meanlog: float = 0.0,
    sf_sdlog: float = 0.2,
) -> dict[str, Any]:
    """
    Simulate covariates shared by all genes.

    Exactly one subtype is generated. ``subtype_idx`` is zero-based and
    contains only zero; ``subtype_code`` is one-based and contains only one.
    Neither index is required by the single-group Stan model.
    """
    N = _positive_int(N, "N")
    subtype_label = str(subtype_label).strip()

    if not subtype_label:
        raise ValueError("subtype_label must be non-empty.")

    if purity_a <= 0 or purity_b <= 0:
        raise ValueError("purity_a and purity_b must be positive.")

    if sf_sdlog < 0:
        raise ValueError("sf_sdlog must be nonnegative.")

    copy_probabilities = _probabilities(
        copy_probs,
        expected_length=5,
        name="copy_probs",
    )

    rng = np.random.default_rng(seed)

    sf = rng.lognormal(
        mean=sf_meanlog,
        sigma=sf_sdlog,
        size=N,
    )
    # Geometric mean one stabilizes the interpretation of b0.
    sf /= np.exp(np.mean(np.log(sf)))

    purity = rng.beta(
        a=purity_a,
        b=purity_b,
        size=N,
    )

    copies = rng.choice(
        np.asarray([1, 2, 3, 4, 5], dtype=float),
        size=N,
        replace=True,
        p=copy_probabilities,
    )

    dose_log = np.log(
        np.clip(copies, 1.0, None) / 2.0
    )

    # Kept identical to the original validation script.
    dev = (copies - 2.0) / 2.0

    covars = {
        "N": N,
        "S": 1,
        "analysis_mode": "single_group",
        "subtype_levels": (subtype_label,),
        "subtype_idx": np.zeros(N, dtype=int),
        "subtype_code": np.ones(N, dtype=int),
        "subtype_labels": np.repeat(subtype_label, N),
        "sf": sf,
        "purity": purity,
        "copies": copies,
        "dose_log": dose_log,
        "dev": dev,
    }

    _validate_covariates(covars)
    return covars


def _validate_covariates(covars: Mapping[str, Any]) -> None:
    required = {
        "N",
        "S",
        "subtype_levels",
        "subtype_idx",
        "subtype_code",
        "subtype_labels",
        "sf",
        "purity",
        "copies",
        "dose_log",
        "dev",
    }
    missing = required - set(covars)

    if missing:
        raise KeyError(
            "Missing covariates: " + ", ".join(sorted(missing))
        )

    N = int(covars["N"])

    if int(covars["S"]) != 1:
        raise ValueError("Single-subtype simulation requires S == 1.")

    if len(tuple(covars["subtype_levels"])) != 1:
        raise ValueError("Exactly one subtype level is required.")

    for column in (
        "subtype_idx",
        "subtype_code",
        "subtype_labels",
        "sf",
        "purity",
        "copies",
        "dose_log",
        "dev",
    ):
        array = np.asarray(covars[column])

        if array.shape != (N,):
            raise ValueError(
                f"{column} must have shape ({N},), got {array.shape}."
            )

    if not np.all(np.asarray(covars["subtype_idx"]) == 0):
        raise ValueError("subtype_idx must contain only zero.")

    if not np.all(np.asarray(covars["subtype_code"]) == 1):
        raise ValueError("subtype_code must contain only one.")

    sf = np.asarray(covars["sf"], dtype=float)
    purity = np.asarray(covars["purity"], dtype=float)
    copies = np.asarray(covars["copies"], dtype=float)

    if not np.isfinite(sf).all() or np.any(sf <= 0):
        raise ValueError("Size factors must be finite and positive.")

    if (
        not np.isfinite(purity).all()
        or np.any(purity <= 0)
        or np.any(purity >= 1)
    ):
        raise ValueError(
            "Purity must lie strictly between zero and one."
        )

    if not np.isfinite(copies).all() or np.any(copies < 0):
        raise ValueError(
            "Copy numbers must be finite and nonnegative."
        )


# True parameter generation

def simulate_gene_params(
    covars: Mapping[str, Any],
    rng: np.random.Generator,
    gene_id: str,
    scenario: str = "mixed",
    b0_loc: float = 5.0,
    b0_sd: float = 0.7,
    b_noncancer_loc: float = float(np.log(5.0)),
    b_noncancer_sd: float = 0.5,
    phi_logmean: float = float(np.log(10.0)),
    phi_logsd: float = 0.25,
) -> dict[str, Any]:
    """
    Simulate scalar parameters for one gene.

    Scenarios
    ---------
    null
        Near-zero scaling and deviation.

    scaling
        Positive scaling and near-zero deviation.

    deviation_pos
        Near-zero scaling and positive deviation.

    deviation_neg
        Near-zero scaling and negative deviation.

    mixed
        Positive scaling and negative deviation.
    """
    _validate_covariates(covars)

    if scenario not in SCENARIOS:
        raise ValueError(
            "scenario must be one of: " + ", ".join(SCENARIOS)
        )

    gene_id = str(gene_id).strip()

    if not gene_id:
        raise ValueError("gene_id must be non-empty.")

    b0 = float(rng.normal(b0_loc, b0_sd))

    if scenario == "null":
        b_scaling = float(rng.normal(0.0, 0.05))
        b_deviation = float(rng.normal(0.0, 0.05))

    elif scenario == "scaling":
        b_scaling = float(rng.normal(0.60, 0.10))
        b_deviation = float(rng.normal(0.0, 0.05))

    elif scenario == "deviation_pos":
        b_scaling = float(rng.normal(0.0, 0.05))
        b_deviation = float(rng.normal(0.40, 0.10))

    elif scenario == "deviation_neg":
        b_scaling = float(rng.normal(0.0, 0.05))
        b_deviation = float(rng.normal(-0.40, 0.10))

    else:
        b_scaling = float(rng.normal(0.50, 0.10))
        b_deviation = float(rng.normal(-0.25, 0.10))

    b_noncancer_log = float(
        rng.normal(b_noncancer_loc, b_noncancer_sd)
    )
    phi = float(
        np.exp(rng.normal(phi_logmean, phi_logsd))
    )

    generated = np.asarray(
        [
            b0,
            b_scaling,
            b_deviation,
            b_noncancer_log,
            phi,
        ],
        dtype=float,
    )

    if not np.isfinite(generated).all() or phi <= 0:
        raise ValueError("Generated parameters are invalid.")

    return {
        "gene": gene_id,
        "scenario": scenario,
        "analysis_mode": "single_group",
        "b0": b0,
        "b_scaling": b_scaling,
        "b_deviation": b_deviation,
        "b_noncancer_log": b_noncancer_log,
        "phi": phi,

        # Compatibility aliases for older summary code.
        "b0_mean": b0,
        "b_scaling_mean": b_scaling,
        "b_dev_mean": b_deviation,
    }


# Count generation

def simulate_counts_for_gene(
    covars: Mapping[str, Any],
    true_params: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate NB counts under the scalar single-group mean model.
    """
    _validate_covariates(covars)

    sf = np.asarray(covars["sf"], dtype=float)
    purity = np.asarray(covars["purity"], dtype=float)
    dose_log = np.asarray(covars["dose_log"], dtype=float)
    dev = np.asarray(covars["dev"], dtype=float)

    b0 = float(true_params["b0"])
    b_scaling = float(true_params["b_scaling"])
    b_deviation = float(true_params["b_deviation"])
    b_noncancer_log = float(
        true_params["b_noncancer_log"]
    )
    phi = float(true_params["phi"])

    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("phi must be finite and positive.")

    tumour_linear_predictor = (
        b0
        + dose_log * b_scaling
        + dev * b_deviation
    )

    tumour_mu = (
        sf
        * purity
        * np.exp(tumour_linear_predictor)
    )
    noncancer_mu = (
        sf
        * (1.0 - purity)
        * np.exp(b_noncancer_log)
    )
    mu = tumour_mu + noncancer_mu

    if not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError(
            "Generated expected counts must be finite and positive."
        )

    # NumPy's negative-binomial parameterization:
    # E[Y] = r(1-p)/p = mu and Var(Y) = mu + mu^2/r.
    probability = phi / (phi + mu)
    probability = np.clip(
        probability,
        1e-12,
        1.0 - 1e-12,
    )

    expression = rng.negative_binomial(
        n=phi,
        p=probability,
        size=int(covars["N"]),
    ).astype(int)

    return expression, mu


# Scenario assignment

def assign_gene_scenarios(
    G: int,
    scenario_probs: Mapping[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Assign a scenario to each gene.
    """
    G = _positive_int(G, "G")

    if rng is None:
        rng = np.random.default_rng(123)

    if scenario_probs is None:
        scenario_probs = {
            "null": 0.30,
            "scaling": 0.30,
            "deviation_pos": 0.10,
            "deviation_neg": 0.10,
            "mixed": 0.20,
        }

    unknown = set(scenario_probs) - set(SCENARIOS)

    if unknown:
        raise ValueError(
            "Unknown scenarios: " + ", ".join(sorted(unknown))
        )

    scenarios = list(scenario_probs)
    probabilities = _probabilities(
        list(scenario_probs.values()),
        expected_length=len(scenarios),
        name="scenario_probs",
    )

    return rng.choice(
        np.asarray(scenarios, dtype=object),
        size=G,
        replace=True,
        p=probabilities,
    )


# Multi-gene dataset

def simulate_dataset_multi_gene(
    G: int = 100,
    N: int = 200,
    subtype_label: str = "all_samples",
    seed: int = 123,
    scenario_probs: Mapping[str, float] | None = None,
    copy_probs: Sequence[float] = (
        0.15,
        0.50,
        0.15,
        0.10,
        0.10,
    ),
    purity_a: float = 5.0,
    purity_b: float = 2.0,
    sf_meanlog: float = 0.0,
    sf_sdlog: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Simulate one subtype across multiple genes.

    Returns
    -------
    sim_df
        Long observed table with G * N rows.

    truth_df
        One scalar-parameter truth row per gene.

    covars
        Shared sample-level covariates.
    """
    G = _positive_int(G, "G")
    N = _positive_int(N, "N")
    rng = np.random.default_rng(seed)

    # A separate seed avoids restarting the same random sequence used
    # later for gene-level parameters.
    covariate_seed = int(
        rng.integers(0, np.iinfo(np.int32).max)
    )

    covars = simulate_covariates(
        N=N,
        subtype_label=subtype_label,
        seed=covariate_seed,
        copy_probs=copy_probs,
        purity_a=purity_a,
        purity_b=purity_b,
        sf_meanlog=sf_meanlog,
        sf_sdlog=sf_sdlog,
    )

    scenario_vector = assign_gene_scenarios(
        G=G,
        scenario_probs=scenario_probs,
        rng=rng,
    )

    sample_ids = np.asarray(
        [f"S{index + 1}" for index in range(N)],
        dtype=object,
    )
    observed_frames: list[pd.DataFrame] = []
    truth_rows: list[dict[str, Any]] = []

    for gene_index in range(G):
        gene_id = f"G{gene_index + 1}"
        scenario = str(scenario_vector[gene_index])

        true_parameters = simulate_gene_params(
            covars=covars,
            rng=rng,
            gene_id=gene_id,
            scenario=scenario,
        )
        expression, mu = simulate_counts_for_gene(
            covars,
            true_parameters,
            rng,
        )

        observed_frames.append(
            pd.DataFrame(
                {
                    "gene": gene_id,
                    "scenario": scenario,
                    "analysis_mode": "single_group",
                    "sample_id": sample_ids,
                    "expr": expression,
                    "copies": covars["copies"],
                    "purity": covars["purity"],
                    "sf": covars["sf"],
                    "subtype": covars["subtype_labels"],

                    # Compatibility only. The single Stan model ignores it.
                    "subtype_idx": covars["subtype_code"],

                    "dose_log": covars["dose_log"],
                    "dev": covars["dev"],
                    "mu_true": mu,
                }
            )
        )

        truth_rows.append(
            {
                "gene": gene_id,
                "scenario": scenario,
                "analysis_mode": "single_group",
                "subtype_s1": covars["subtype_levels"][0],
                "phi_true": true_parameters["phi"],
                "b_noncancer_log_true": (
                    true_parameters["b_noncancer_log"]
                ),
                "b0_true": true_parameters["b0"],
                "b_scaling_true": true_parameters["b_scaling"],
                "b_deviation_true": (
                    true_parameters["b_deviation"]
                ),

                # Compatibility aliases from the previous simulator.
                "b0_mean_true": true_parameters["b0"],
                "b_scaling_mean_true": (
                    true_parameters["b_scaling"]
                ),
                "b_dev_mean_true": (
                    true_parameters["b_deviation"]
                ),
                "b0_s1_true": true_parameters["b0"],
                "b_scaling_s1_true": (
                    true_parameters["b_scaling"]
                ),
                "b_deviation_s1_true": (
                    true_parameters["b_deviation"]
                ),
            }
        )

    sim_df = pd.concat(
        observed_frames,
        ignore_index=True,
    )
    truth_df = pd.DataFrame(truth_rows)

    validate_single_subtype_dataset(
        sim_df,
        truth_df,
        covars,
    )

    return sim_df, truth_df, covars


# Fixed-scenario dataset

def simulate_dataset_one_scenario(
    G: int = 100,
    N: int = 200,
    scenario: str = "null",
    subtype_label: str = "all_samples",
    seed: int = 123,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Simulate all genes under one scenario and one subtype.
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            "scenario must be one of: " + ", ".join(SCENARIOS)
        )

    return simulate_dataset_multi_gene(
        G=G,
        N=N,
        subtype_label=subtype_label,
        seed=seed,
        scenario_probs={scenario: 1.0},
        **kwargs,
    )


# Single-group stan data

def make_single_group_stan_data(
    gene_df: pd.DataFrame,
    *,
    expression_col: str = "expr",
    size_factor_col: str = "sf",
    purity_col: str = "purity",
    dose_log_col: str = "dose_log",
    deviation_col: str = "dev",
) -> dict[str, Any]:
    """
    Construct the data dictionary expected by ``bdgdm_single.stan``.
    """
    if not isinstance(gene_df, pd.DataFrame):
        raise TypeError("gene_df must be a pandas DataFrame.")

    if gene_df.empty:
        raise ValueError("gene_df must not be empty.")

    required = {
        expression_col,
        size_factor_col,
        purity_col,
        dose_log_col,
        deviation_col,
    }
    missing = required - set(gene_df.columns)

    if missing:
        raise KeyError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    if (
        "gene" in gene_df.columns
        and gene_df["gene"].nunique(dropna=False) != 1
    ):
        raise ValueError(
            "gene_df must contain observations for exactly one gene."
        )

    y = pd.to_numeric(
        gene_df[expression_col],
        errors="coerce",
    ).to_numpy()
    sf = pd.to_numeric(
        gene_df[size_factor_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    purity = pd.to_numeric(
        gene_df[purity_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    dose_log = pd.to_numeric(
        gene_df[dose_log_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    dev = pd.to_numeric(
        gene_df[deviation_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(y).all()
        or np.any(y < 0)
        or not np.equal(y, np.floor(y)).all()
    ):
        raise ValueError(
            "Expression must contain finite nonnegative integer counts."
        )

    for name, values in (
        ("sf", sf),
        ("purity", purity),
        ("dose_log", dose_log),
        ("dev", dev),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")

    if np.any(sf <= 0):
        raise ValueError("Size factors must be positive.")

    if np.any(purity <= 0) or np.any(purity >= 1):
        raise ValueError(
            "Purity must lie strictly between zero and one."
        )

    return {
        "N": int(len(gene_df)),
        "y": y.astype(int),
        "sf": sf,
        "purity": purity,
        "dose_log": dose_log,
        "dev": dev,
    }


def validate_single_subtype_dataset(
    sim_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    covars: Mapping[str, Any],
) -> None:
    """
    Validate that no subtype-comparison structure remains.
    """
    _validate_covariates(covars)

    if sim_df.empty or truth_df.empty:
        raise ValueError(
            "sim_df and truth_df must both be non-empty."
        )

    required_observed = {
        "gene",
        "scenario",
        "analysis_mode",
        "sample_id",
        "expr",
        "copies",
        "purity",
        "sf",
        "subtype",
        "subtype_idx",
        "dose_log",
        "dev",
        "mu_true",
    }
    missing = required_observed - set(sim_df.columns)

    if missing:
        raise KeyError(
            "sim_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    if sim_df["subtype"].nunique(dropna=False) != 1:
        raise ValueError(
            "sim_df must contain exactly one subtype."
        )

    if not sim_df["subtype_idx"].eq(1).all():
        raise ValueError(
            "Long-format subtype_idx must equal one for every row."
        )

    if not sim_df["analysis_mode"].eq("single_group").all():
        raise ValueError(
            "Every observed row must use analysis_mode='single_group'."
        )

    expected_rows = len(truth_df) * int(covars["N"])

    if len(sim_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(sim_df)}."
        )

    observations_per_gene = sim_df.groupby("gene").size()

    if not observations_per_gene.eq(int(covars["N"])).all():
        raise ValueError(
            "Every gene must contain exactly N observations."
        )

    if set(sim_df["gene"]) != set(truth_df["gene"]):
        raise ValueError(
            "Gene identifiers differ between observed and truth tables."
        )

    if truth_df["subtype_s1"].nunique(dropna=False) != 1:
        raise ValueError(
            "truth_df must contain exactly one subtype_s1 label."
        )

    if not truth_df["analysis_mode"].eq("single_group").all():
        raise ValueError(
            "Every truth row must use analysis_mode='single_group'."
        )


# Summary helpers

def summarize_simulated_truth(
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "b0_true",
        "b_scaling_true",
        "b_deviation_true",
        "phi_true",
    ]
    missing = set(columns) - set(truth_df.columns)

    if missing:
        raise KeyError(
            "truth_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    return truth_df.groupby("scenario")[columns].agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max",
        ]
    )


def summarize_simulated_counts(
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "scenario",
        "gene",
        "expr",
        "mu_true",
    }
    missing = required - set(sim_df.columns)

    if missing:
        raise KeyError(
            "sim_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    return (
        sim_df.groupby(
            ["scenario", "gene"],
            as_index=False,
        )
        .agg(
            n=("expr", "size"),
            mean_expr=("expr", "mean"),
            var_expr=("expr", "var"),
            zero_frac=(
                "expr",
                lambda values: float(
                    np.mean(np.asarray(values) == 0)
                ),
            ),
            mean_mu_true=("mu_true", "mean"),
            var_mu_true=("mu_true", "var"),
        )
    )
