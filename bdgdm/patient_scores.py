"""Patient-level dosage-response scoring for BDGDM.

This module converts gene-level BDGDM posterior draws into patient-level
phenotypes. For patient ``i`` and gene ``g``, the tumour-cell log effect is

    delta_ig = b_scaling_g * log(CN_ig / 2)
               + b_deviation_g * (CN_ig - 2) / 2

Copy numbers below one are floored to one only inside the logarithmic term,
matching BDGDM preprocessing. Tumour purity is not applied again because it
has already been used to estimate the tumour-cell response parameters.

Chromosome-arm-balanced scores are intentionally not implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd

from .posterior import extract_posterior_draws


InferenceEngine = Literal["nuts", "vi_meanfield", "vi_fullrank"]
AnalysisMode = Literal["single_group", "subtype_comparison"]

__all__ = [
    "DosageDraws",
    "PatientScoreConfig",
    "PatientScoreResult",
    "build_dosage_covariates",
    "build_cn_support",
    "extract_dosage_draws",
    "extract_dosage_draws_from_fits",
    "compute_patient_scores",
    "save_patient_score_result",
]


@dataclass(frozen=True)
class DosageDraws:
    """Compact posterior draws required for patient-level scoring."""

    gene: str
    analysis_mode: AnalysisMode
    subtype_levels: tuple[str, ...]
    b_scaling: np.ndarray
    b_deviation: np.ndarray
    cn_support: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        gene = str(self.gene)
        levels = tuple(str(x) for x in self.subtype_levels)
        mode = str(self.analysis_mode)

        if not gene:
            raise ValueError("gene cannot be empty.")
        if not levels:
            raise ValueError("subtype_levels cannot be empty.")
        if len(levels) != len(set(levels)):
            raise ValueError("subtype_levels contains duplicates.")
        if mode not in {"single_group", "subtype_comparison"}:
            raise ValueError("Unknown analysis_mode.")
        if mode == "single_group" and len(levels) != 1:
            raise ValueError("single_group requires exactly one subtype.")
        if mode == "subtype_comparison" and len(levels) < 2:
            raise ValueError("subtype_comparison requires at least two subtypes.")

        scaling = _as_draw_matrix(
            self.b_scaling,
            name="b_scaling",
            n_subtypes=len(levels),
        )
        deviation = _as_draw_matrix(
            self.b_deviation,
            name="b_deviation",
            n_subtypes=len(levels),
        )
        if scaling.shape != deviation.shape:
            raise ValueError("b_scaling and b_deviation must have identical shapes.")

        support: dict[str, np.ndarray] | None = None
        if self.cn_support is not None:
            unknown = set(self.cn_support).difference(levels)
            if unknown:
                raise ValueError(
                    "cn_support contains unknown subtype labels: "
                    f"{sorted(unknown)}."
                )
            support = {}
            for subtype, values in self.cn_support.items():
                array = np.asarray(values, dtype=float).reshape(-1)
                support[str(subtype)] = array[np.isfinite(array)].copy()

        object.__setattr__(self, "gene", gene)
        object.__setattr__(self, "analysis_mode", mode)
        object.__setattr__(self, "subtype_levels", levels)
        object.__setattr__(self, "b_scaling", scaling)
        object.__setattr__(self, "b_deviation", deviation)
        object.__setattr__(self, "cn_support", support)

    @property
    def n_draws(self) -> int:
        """Number of posterior draws."""

        return int(self.b_scaling.shape[0])

    @property
    def n_subtypes(self) -> int:
        """Number of fitted subtype levels."""

        return int(self.b_scaling.shape[1])


@dataclass(frozen=True)
class PatientScoreConfig:
    """Configuration for patient-level dosage-response scoring."""

    sample_col: str = "sampleID"
    gene_col: str = "gene"
    cn_col: str = "copies"
    subtype_col: str = "subtype"

    neutral_cn: float = 2.0
    cn_log_floor: float = 1.0
    et: float = 0.15
    hyper_multiplier: float = 1.50

    require_cn_support: bool = False
    cn_support_radius: float = 0.25
    min_cn_support_samples: int = 3

    credible_interval: tuple[float, float] = (0.025, 0.975)
    return_gene_contributions: bool = False

    def __post_init__(self) -> None:
        for name in ("sample_col", "gene_col", "cn_col", "subtype_col"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not np.isfinite(self.neutral_cn) or self.neutral_cn <= 0:
            raise ValueError("neutral_cn must be finite and positive.")
        if not np.isfinite(self.cn_log_floor) or self.cn_log_floor <= 0:
            raise ValueError("cn_log_floor must be finite and positive.")
        if not 0 <= self.et < 1:
            raise ValueError("et must lie in [0, 1).")
        if not np.isfinite(self.hyper_multiplier) or self.hyper_multiplier < 1:
            raise ValueError("hyper_multiplier must be finite and at least one.")
        if not np.isfinite(self.cn_support_radius) or self.cn_support_radius < 0:
            raise ValueError("cn_support_radius must be finite and non-negative.")
        if (
            not isinstance(self.min_cn_support_samples, (int, np.integer))
            or self.min_cn_support_samples < 1
        ):
            raise ValueError("min_cn_support_samples must be a positive integer.")
        q_low, q_high = self.credible_interval
        if not 0 <= q_low < 0.5 < q_high <= 1:
            raise ValueError(
                "credible_interval must satisfy 0 <= lower < 0.5 < upper <= 1."
            )


@dataclass
class PatientScoreResult:
    """Output of :func:`compute_patient_scores`."""

    summary: pd.DataFrame
    wide: pd.DataFrame
    qc: pd.DataFrame
    score_draws: dict[str, np.ndarray]
    sample_order: tuple[str, ...]
    gene_contributions: pd.DataFrame | None
    config: PatientScoreConfig


def _as_draw_matrix(
    values: np.ndarray,
    *,
    name: str,
    n_subtypes: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != n_subtypes:
        raise ValueError(
            f"{name} must have shape (draws, {n_subtypes}); received {array.shape}."
        )
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} contains no valid finite posterior draws.")
    return array.copy()


def build_dosage_covariates(
    copies: Sequence[float] | np.ndarray,
    *,
    neutral_cn: float = 2.0,
    cn_log_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the BDGDM logarithmic and linear CN covariates."""

    cn = np.asarray(copies, dtype=float)
    if not np.isfinite(cn).all():
        raise ValueError("copies contains non-finite values.")
    if (cn < 0).any():
        raise ValueError("copies cannot contain negative values.")
    if neutral_cn <= 0 or cn_log_floor <= 0:
        raise ValueError("neutral_cn and cn_log_floor must be positive.")

    dose_log = np.log(np.maximum(cn, cn_log_floor) / neutral_cn)
    deviation = (cn - neutral_cn) / neutral_cn
    return dose_log, deviation


def build_cn_support(
    training_data: pd.DataFrame,
    *,
    subtype_levels: Sequence[str],
    cn_col: str = "copies",
    subtype_col: str = "subtype",
) -> dict[str, np.ndarray]:
    """Extract finite training CN values for each fitted subtype."""

    missing = {cn_col, subtype_col}.difference(training_data.columns)
    if missing:
        raise ValueError(f"training_data is missing columns: {sorted(missing)}.")

    data = training_data[[subtype_col, cn_col]].copy()
    data[subtype_col] = data[subtype_col].astype(str)
    data[cn_col] = pd.to_numeric(data[cn_col], errors="coerce")

    support: dict[str, np.ndarray] = {}
    for subtype in map(str, subtype_levels):
        values = data.loc[data[subtype_col].eq(subtype), cn_col].to_numpy(float)
        support[subtype] = values[np.isfinite(values)]
    return support


def _mapping_value(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _resolve_fit_metadata(
    fit: Any,
    *,
    gene: str | None,
    subtype_levels: Sequence[str] | None,
    analysis_mode: str | None,
    engine: str | None,
) -> tuple[str, tuple[str, ...], AnalysisMode, InferenceEngine, Any]:
    posterior = getattr(fit, "posterior", None)
    metadata = getattr(fit, "metadata", None)
    posterior_map = posterior if isinstance(posterior, Mapping) else None
    metadata_map = metadata if isinstance(metadata, Mapping) else None

    resolved_gene = gene or getattr(fit, "gene", None)
    resolved_gene = resolved_gene or _mapping_value(metadata_map, "gene")
    if resolved_gene is None:
        raise ValueError("gene could not be inferred; provide it explicitly.")

    levels = subtype_levels
    if levels is None:
        levels = _mapping_value(metadata_map, "subtype_levels", "subtype_order")
    if levels is None:
        levels = _mapping_value(posterior_map, "subtype_levels")
    if levels is None:
        raise ValueError("subtype_levels could not be inferred.")
    resolved_levels = tuple(str(x) for x in levels)

    mode = analysis_mode or getattr(fit, "analysis_mode", None)
    mode = mode or _mapping_value(metadata_map, "analysis_mode")
    mode = mode or _mapping_value(posterior_map, "analysis_mode")
    mode = mode or ("single_group" if len(resolved_levels) == 1 else "subtype_comparison")
    if mode not in {"single_group", "subtype_comparison"}:
        raise ValueError(f"Unknown analysis_mode: {mode!r}.")

    resolved_engine = engine or _mapping_value(posterior_map, "engine")
    if resolved_engine is None:
        config = _mapping_value(metadata_map, "config")
        if isinstance(config, Mapping):
            resolved_engine = config.get("engine")

    raw_fit = getattr(fit, "fit", fit)
    if resolved_engine is None and callable(getattr(raw_fit, "stan_variable", None)):
        resolved_engine = "nuts"
    if resolved_engine not in {"nuts", "vi_meanfield", "vi_fullrank"}:
        raise ValueError("engine could not be inferred.")

    return (
        str(resolved_gene),
        resolved_levels,
        mode,
        resolved_engine,
        raw_fit,
    )


def extract_dosage_draws(
    fit: Any,
    *,
    gene: str | None = None,
    subtype_levels: Sequence[str] | None = None,
    analysis_mode: AnalysisMode | None = None,
    engine: InferenceEngine | None = None,
    n_draws: int | None = 500,
    random_state: int | np.random.Generator | None = 123,
    training_data: pd.DataFrame | None = None,
    cn_col: str = "copies",
    subtype_col: str = "subtype",
    require_converged: bool = True,
) -> DosageDraws:
    """Extract compact scoring draws from a BDGDMFit or raw CmdStan fit."""

    if n_draws is not None and (
        not isinstance(n_draws, (int, np.integer)) or n_draws < 1
    ):
        raise ValueError("n_draws must be a positive integer or None.")

    diagnostics = getattr(fit, "diagnostics", None)
    if (
        require_converged
        and isinstance(diagnostics, Mapping)
        and diagnostics.get("converged") is False
    ):
        raise RuntimeError("The fit did not pass convergence diagnostics.")

    resolved_gene, levels, mode, resolved_engine, raw_fit = _resolve_fit_metadata(
        fit,
        gene=gene,
        subtype_levels=subtype_levels,
        analysis_mode=analysis_mode,
        engine=engine,
    )

    draws = extract_posterior_draws(
        raw_fit,
        engine=resolved_engine,
        analysis_mode=mode,
        n_subtypes=len(levels),
    )
    scaling = _as_draw_matrix(
        np.asarray(draws["b_scaling"]),
        name="b_scaling",
        n_subtypes=len(levels),
    )
    deviation = _as_draw_matrix(
        np.asarray(draws["b_deviation"]),
        name="b_deviation",
        n_subtypes=len(levels),
    )
    if scaling.shape != deviation.shape:
        raise ValueError("Extracted scaling and deviation draws are incompatible.")

    if n_draws is not None and n_draws < scaling.shape[0]:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        selected = np.sort(
            rng.choice(scaling.shape[0], size=int(n_draws), replace=False)
        )
        scaling = scaling[selected]
        deviation = deviation[selected]

    support = None
    if training_data is not None:
        training = training_data.copy()
        if "gene" in training.columns:
            training = training.loc[
                training["gene"].astype(str).eq(resolved_gene)
            ].copy()
        if training.empty:
            raise ValueError(f"No training rows found for gene {resolved_gene!r}.")
        support = build_cn_support(
            training,
            subtype_levels=levels,
            cn_col=cn_col,
            subtype_col=subtype_col,
        )

    return DosageDraws(
        gene=resolved_gene,
        analysis_mode=mode,
        subtype_levels=levels,
        b_scaling=scaling,
        b_deviation=deviation,
        cn_support=support,
    )


def extract_dosage_draws_from_fits(
    fits: Mapping[str, Any],
    *,
    training_data: pd.DataFrame | None = None,
    gene_col: str = "gene",
    subtype_col: str = "subtype",
    cn_col: str = "copies",
    n_draws: int | None = 500,
    random_state: int | None = 123,
    require_converged: bool = True,
    skip_failed: bool = False,
) -> dict[str, DosageDraws]:
    """Extract compact draws from a gene-to-fit mapping."""

    normalized_fits = {str(gene): fit for gene, fit in fits.items()}
    if not normalized_fits:
        raise ValueError("fits cannot be empty.")

    genes = sorted(normalized_fits)
    child_seeds = np.random.SeedSequence(random_state).spawn(len(genes))
    output: dict[str, DosageDraws] = {}
    failures: dict[str, str] = {}

    for gene, child_seed in zip(genes, child_seeds, strict=True):
        gene_training = None
        if training_data is not None:
            if gene_col not in training_data.columns:
                raise ValueError(f"training_data is missing {gene_col!r}.")
            gene_training = training_data.loc[
                training_data[gene_col].astype(str).eq(gene)
            ].copy()
        try:
            output[gene] = extract_dosage_draws(
                normalized_fits[gene],
                gene=gene,
                n_draws=n_draws,
                random_state=np.random.default_rng(child_seed),
                training_data=gene_training,
                cn_col=cn_col,
                subtype_col=subtype_col,
                require_converged=require_converged,
            )
        except Exception as exc:
            if not skip_failed:
                raise
            failures[gene] = f"{type(exc).__name__}: {exc}"

    if not output:
        raise RuntimeError(f"No valid gene fits were available. Failures: {failures}")
    counts = {draws.n_draws for draws in output.values()}
    if len(counts) != 1:
        raise ValueError(
            "Genes contain different posterior draw counts. Choose n_draws "
            "that is available for every fit."
        )
    return output


def _validate_patient_data(
    data: pd.DataFrame,
    config: PatientScoreConfig,
) -> pd.DataFrame:
    required = {
        config.sample_col,
        config.gene_col,
        config.cn_col,
        config.subtype_col,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data is missing columns: {sorted(missing)}.")

    selected = data[
        [config.sample_col, config.gene_col, config.cn_col, config.subtype_col]
    ].copy()
    for column in (config.sample_col, config.gene_col, config.subtype_col):
        selected[column] = selected[column].astype(str)
        if selected[column].eq("").any():
            raise ValueError(f"{column!r} cannot contain empty values.")
    selected[config.cn_col] = pd.to_numeric(selected[config.cn_col], errors="coerce")

    duplicates = selected.duplicated(
        [config.sample_col, config.gene_col], keep=False
    )
    if duplicates.any():
        examples = selected.loc[
            duplicates, [config.sample_col, config.gene_col]
        ].head(5).to_dict("records")
        raise ValueError(
            "data must contain at most one row per sample and gene. "
            f"Examples: {examples}"
        )

    cn = selected[config.cn_col].to_numpy(float)
    if (np.isfinite(cn) & (cn < 0)).any():
        raise ValueError("copy-number values cannot be negative.")

    subtype_counts = selected.groupby(config.sample_col)[config.subtype_col].nunique()
    inconsistent = subtype_counts[subtype_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each sample must have one subtype label. Inconsistent samples: "
            f"{inconsistent.index[:5].tolist()}."
        )
    return selected


def _cn_is_supported(
    patient_cn: np.ndarray,
    training_cn: np.ndarray | None,
    *,
    radius: float,
    minimum_samples: int,
) -> tuple[np.ndarray, bool]:
    if training_cn is None:
        return np.ones(len(patient_cn), dtype=bool), False
    training = np.asarray(training_cn, dtype=float).reshape(-1)
    training = training[np.isfinite(training)]
    if training.size == 0:
        return np.zeros(len(patient_cn), dtype=bool), True
    supported = (
        np.abs(patient_cn[:, None] - training[None, :]) <= radius
    ).sum(axis=1) >= minimum_samples
    return supported, True


def _summarize_draw_matrix(
    values: np.ndarray,
    *,
    q_low: float,
    q_high: float,
) -> dict[str, np.ndarray]:
    with warnings.catch_warnings(), np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "mean": np.nanmean(values, axis=0),
            "sd": np.nanstd(values, axis=0, ddof=1),
            "q_low": np.nanquantile(values, q_low, axis=0),
            "median": np.nanquantile(values, 0.5, axis=0),
            "q_high": np.nanquantile(values, q_high, axis=0),
        }

def compute_patient_scores(
    data: pd.DataFrame,
    dosage_draws: Mapping[str, DosageDraws],
    *,
    config: PatientScoreConfig | None = None,
) -> PatientScoreResult:
    """Compute posterior patient-level dosage-response scores.

    Scores
    ------
    altered_fraction
        Altered scored genes divided by all scored genes.
    dosage_burden
        Sum of absolute altered-gene effects divided by all scored genes.
    dosage_intensity
        Mean absolute effect conditional on a gene being altered.
    gain_burden
        Positive effects among gained genes divided by all scored genes.
    loss_burden
        Magnitudes of negative effects among deleted genes divided by all
        scored genes.
    hyper_excess
        Same-direction effect above ``hyper_multiplier`` times the canonical
        proportional effect, divided by all scored genes.
    compensation_burden
        Missing same-direction effect relative to proportional scaling,
        divided by all scored genes.
    reverse_burden
        Effect magnitude opposite to the canonical CN direction, divided by
        all scored genes.
    """

    config = config or PatientScoreConfig()
    if not dosage_draws:
        raise ValueError("dosage_draws cannot be empty.")

    draws_map = {str(gene): value for gene, value in dosage_draws.items()}
    for gene, value in draws_map.items():
        if not isinstance(value, DosageDraws):
            raise TypeError(f"dosage_draws[{gene!r}] must be DosageDraws.")
        if value.gene != gene:
            raise ValueError(
                f"Mapping key {gene!r} does not match DosageDraws.gene "
                f"{value.gene!r}."
            )
    draw_counts = {value.n_draws for value in draws_map.values()}
    if len(draw_counts) != 1:
        raise ValueError("Every gene must contain the same number of draws.")
    n_draws = draw_counts.pop()

    scoring_data = _validate_patient_data(data, config)
    samples = pd.Index(
        scoring_data[config.sample_col].drop_duplicates(),
        name=config.sample_col,
    )
    sample_order = tuple(samples.astype(str))
    n_samples = len(samples)
    sample_position = pd.Series(np.arange(n_samples), index=samples)
    subtype_by_sample = (
        scoring_data.drop_duplicates(config.sample_col)
        .set_index(config.sample_col)[config.subtype_col]
        .reindex(samples)
        .astype(str)
    )

    numerator_names = [
        "dosage_burden",
        "gain_burden",
        "loss_burden",
        "hyper_excess",
        "compensation_burden",
        "reverse_burden",
    ]
    accumulators = {
        name: np.zeros((n_draws, n_samples), dtype=float)
        for name in numerator_names
    }

    n_input_genes = (
        scoring_data.groupby(config.sample_col)[config.gene_col]
        .nunique()
        .reindex(samples)
        .fillna(0)
        .to_numpy(np.int64)
    )
    n_modelled_genes = np.zeros(n_samples, dtype=np.int64)
    n_scored_genes = np.zeros(n_samples, dtype=np.int64)
    n_altered_modelled = np.zeros(n_samples, dtype=np.int64)
    n_altered_genes = np.zeros(n_samples, dtype=np.int64)
    n_gain_genes = np.zeros(n_samples, dtype=np.int64)
    n_loss_genes = np.zeros(n_samples, dtype=np.int64)
    n_support_evaluable = np.zeros(n_samples, dtype=np.int64)
    n_supported_genes = np.zeros(n_samples, dtype=np.int64)
    n_extrapolated_altered = np.zeros(n_samples, dtype=np.int64)
    n_missing_cn = np.zeros(n_samples, dtype=np.int64)
    n_unknown_subtype = np.zeros(n_samples, dtype=np.int64)

    contribution_rows: list[dict[str, Any]] = []
    altered_distance = 1.0 - config.et

    for gene, gene_df in scoring_data.groupby(config.gene_col, sort=False):
        gene_draws = draws_map.get(str(gene))
        if gene_draws is None:
            continue

        all_positions = sample_position.loc[
            gene_df[config.sample_col]
        ].to_numpy(np.int64)
        n_modelled_genes[all_positions] += 1
        subtype_lookup = {
            subtype: index
            for index, subtype in enumerate(gene_draws.subtype_levels)
        }

        for subtype, subtype_df in gene_df.groupby(config.subtype_col, sort=False):
            positions_all = sample_position.loc[
                subtype_df[config.sample_col]
            ].to_numpy(np.int64)
            cn_all = subtype_df[config.cn_col].to_numpy(float)
            finite = np.isfinite(cn_all)
            if (~finite).any():
                n_missing_cn[positions_all[~finite]] += 1
            if not finite.any():
                continue

            finite_df = subtype_df.loc[finite]
            positions = positions_all[finite]
            cn = cn_all[finite]
            subtype = str(subtype)
            subtype_index = subtype_lookup.get(subtype)
            if subtype_index is None:
                n_unknown_subtype[positions] += 1
                continue

            support_values = None
            if gene_draws.cn_support is not None:
                support_values = gene_draws.cn_support.get(subtype)
            supported, support_evaluable = _cn_is_supported(
                cn,
                support_values,
                radius=config.cn_support_radius,
                minimum_samples=config.min_cn_support_samples,
            )
            if support_evaluable:
                n_support_evaluable[positions] += 1
                n_supported_genes[positions] += supported.astype(np.int64)

            if config.require_cn_support:
                score_mask = (
                    supported
                    if support_evaluable
                    else np.zeros(len(cn), dtype=bool)
                )
            else:
                score_mask = np.ones(len(cn), dtype=bool)

            dose_log, deviation = build_dosage_covariates(
                cn,
                neutral_cn=config.neutral_cn,
                cn_log_floor=config.cn_log_floor,
            )
            scaling = gene_draws.b_scaling[:, subtype_index]
            dev_coef = gene_draws.b_deviation[:, subtype_index]
            delta = (
                scaling[:, None] * dose_log[None, :]
                + dev_coef[:, None] * deviation[None, :]
            )

            proportional = dose_log
            altered = np.abs(cn - config.neutral_cn) > altered_distance
            gain = cn > config.neutral_cn + altered_distance
            loss = cn < config.neutral_cn - altered_distance
            scored_altered = altered & score_mask
            scored_gain = gain & score_mask
            scored_loss = loss & score_mask

            n_altered_modelled[positions] += altered.astype(np.int64)
            if support_evaluable:
                n_extrapolated_altered[positions] += (
                    altered & ~supported
                ).astype(np.int64)
            n_scored_genes[positions] += score_mask.astype(np.int64)
            n_altered_genes[positions] += scored_altered.astype(np.int64)
            n_gain_genes[positions] += scored_gain.astype(np.int64)
            n_loss_genes[positions] += scored_loss.astype(np.int64)

            magnitude = np.abs(delta)
            prop_magnitude = np.abs(proportional)
            direction_product = delta * proportional[None, :]
            same_direction = direction_product >= 0
            reverse_direction = direction_product < 0

            accumulators["dosage_burden"][:, positions] += (
                magnitude * scored_altered[None, :]
            )
            accumulators["gain_burden"][:, positions] += (
                np.maximum(delta, 0) * scored_gain[None, :]
            )
            accumulators["loss_burden"][:, positions] += (
                np.maximum(-delta, 0) * scored_loss[None, :]
            )
            accumulators["hyper_excess"][:, positions] += (
                np.where(
                    same_direction,
                    np.maximum(
                        magnitude
                        - config.hyper_multiplier * prop_magnitude[None, :],
                        0,
                    ),
                    0,
                )
                * scored_altered[None, :]
            )
            accumulators["compensation_burden"][:, positions] += (
                np.where(
                    same_direction,
                    np.maximum(prop_magnitude[None, :] - magnitude, 0),
                    0,
                )
                * scored_altered[None, :]
            )
            accumulators["reverse_burden"][:, positions] += (
                np.where(reverse_direction, magnitude, 0)
                * scored_altered[None, :]
            )

            if config.return_gene_contributions:
                stats = _summarize_draw_matrix(
                    delta,
                    q_low=config.credible_interval[0],
                    q_high=config.credible_interval[1],
                )
                for j, (_, row) in enumerate(finite_df.iterrows()):
                    contribution_rows.append(
                        {
                            config.sample_col: str(row[config.sample_col]),
                            config.gene_col: str(gene),
                            config.subtype_col: subtype,
                            config.cn_col: float(cn[j]),
                            "altered": bool(altered[j]),
                            "gain": bool(gain[j]),
                            "loss": bool(loss[j]),
                            "cn_support_evaluable": bool(support_evaluable),
                            "cn_supported": (
                                bool(supported[j]) if support_evaluable else None
                            ),
                            "included_in_scores": bool(score_mask[j]),
                            "proportional_log_effect": float(proportional[j]),
                            "delta_mean": float(stats["mean"][j]),
                            "delta_sd": float(stats["sd"][j]),
                            "delta_q025": float(stats["q_low"][j]),
                            "delta_median": float(stats["median"][j]),
                            "delta_q975": float(stats["q_high"][j]),
                        }
                    )

    denominator = np.maximum(n_scored_genes, 1)[None, :]
    for name in numerator_names:
        accumulators[name] /= denominator
        accumulators[name][:, n_scored_genes == 0] = np.nan

    dosage_intensity = np.full((n_draws, n_samples), np.nan)
    has_alterations = n_altered_genes > 0
    dosage_intensity[:, has_alterations] = (
        accumulators["dosage_burden"][:, has_alterations]
        * n_scored_genes[has_alterations][None, :]
        / n_altered_genes[has_alterations][None, :]
    )
    altered_fraction = np.divide(
        n_altered_genes,
        n_scored_genes,
        out=np.full(n_samples, np.nan),
        where=n_scored_genes > 0,
    )

    score_draws = {
        "altered_fraction": np.broadcast_to(
            altered_fraction[None, :], (n_draws, n_samples)
        ).copy(),
        "dosage_burden": accumulators["dosage_burden"],
        "dosage_intensity": dosage_intensity,
        "gain_burden": accumulators["gain_burden"],
        "loss_burden": accumulators["loss_burden"],
        "hyper_excess": accumulators["hyper_excess"],
        "compensation_burden": accumulators["compensation_burden"],
        "reverse_burden": accumulators["reverse_burden"],
    }

    q_low, q_high = config.credible_interval
    summary_parts: list[pd.DataFrame] = []
    for score, matrix in score_draws.items():
        stats = _summarize_draw_matrix(matrix, q_low=q_low, q_high=q_high)
        summary_parts.append(
            pd.DataFrame(
                {
                    config.sample_col: sample_order,
                    config.subtype_col: subtype_by_sample.to_numpy(),
                    "score": score,
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "q025": stats["q_low"],
                    "median": stats["median"],
                    "q975": stats["q_high"],
                }
            )
        )
    summary = pd.concat(summary_parts, ignore_index=True)

    support_fraction = np.divide(
        n_supported_genes,
        n_support_evaluable,
        out=np.full(n_samples, np.nan),
        where=n_support_evaluable > 0,
    )
    extrapolated_fraction = np.divide(
        n_extrapolated_altered,
        n_altered_modelled,
        out=np.full(n_samples, np.nan),
        where=n_altered_modelled > 0,
    )

    qc = pd.DataFrame(
        {
            config.sample_col: sample_order,
            config.subtype_col: subtype_by_sample.to_numpy(),
            "n_input_genes": n_input_genes,
            "n_modelled_genes": n_modelled_genes,
            "n_scored_genes": n_scored_genes,
            "n_altered_modelled": n_altered_modelled,
            "n_altered_genes": n_altered_genes,
            "n_gain_genes": n_gain_genes,
            "n_loss_genes": n_loss_genes,
            "n_support_evaluable": n_support_evaluable,
            "n_supported_genes": n_supported_genes,
            "support_fraction": support_fraction,
            "n_extrapolated_altered": n_extrapolated_altered,
            "extrapolated_altered_fraction": extrapolated_fraction,
            "n_missing_cn": n_missing_cn,
            "n_unknown_subtype": n_unknown_subtype,
            "modelled_gene_fraction": np.divide(
                n_modelled_genes,
                n_input_genes,
                out=np.full(n_samples, np.nan),
                where=n_input_genes > 0,
            ),
            "scored_gene_fraction": np.divide(
                n_scored_genes,
                n_input_genes,
                out=np.full(n_samples, np.nan),
                where=n_input_genes > 0,
            ),
        }
    )

    index_cols = [config.sample_col, config.subtype_col]
    wide_parts = []
    for statistic in ("median", "q025", "q975"):
        part = summary.pivot(
            index=index_cols,
            columns="score",
            values=statistic,
        )
        part.columns = [f"{column}_{statistic}" for column in part.columns]
        wide_parts.append(part)
    wide = wide_parts[0].join(wide_parts[1:]).reset_index()
    wide = wide.merge(qc, on=index_cols, how="left", validate="one_to_one")

    contributions = (
        pd.DataFrame(contribution_rows)
        if config.return_gene_contributions
        else None
    )
    return PatientScoreResult(
        summary=summary,
        wide=wide,
        qc=qc,
        score_draws=score_draws,
        sample_order=sample_order,
        gene_contributions=contributions,
        config=config,
    )


def save_patient_score_result(
    result: PatientScoreResult,
    output_dir: str | Path,
    *,
    prefix: str = "patient_scores",
    save_draws: bool = False,
) -> dict[str, Path]:
    """Save patient-score tables and optional compressed posterior draws."""

    if not isinstance(result, PatientScoreResult):
        raise TypeError("result must be PatientScoreResult.")
    if not prefix:
        raise ValueError("prefix cannot be empty.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output / f"{prefix}_summary.csv",
        "wide": output / f"{prefix}_wide.csv",
        "qc": output / f"{prefix}_qc.csv",
    }
    result.summary.to_csv(paths["summary"], index=False)
    result.wide.to_csv(paths["wide"], index=False)
    result.qc.to_csv(paths["qc"], index=False)

    if result.gene_contributions is not None:
        path = output / f"{prefix}_gene_contributions.csv"
        result.gene_contributions.to_csv(path, index=False)
        paths["gene_contributions"] = path

    if save_draws:
        path = output / f"{prefix}_draws.npz"
        np.savez_compressed(
            path,
            sample_order=np.asarray(result.sample_order, dtype=str),
            **result.score_draws,
        )
        paths["score_draws"] = path
    return paths
