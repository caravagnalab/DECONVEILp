from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd



"""
Posterior extraction and summary utilities for BDGDM.
"""

AnalysisMode = Literal["single_group", "subtype_comparison"]
InferenceEngine = Literal["nuts", "vi_meanfield", "vi_fullrank"]

def posterior_quantiles(
    values: np.ndarray,
    probabilities: tuple[float, float, float] = (
        0.025,
        0.5,
        0.975,
    ),
) -> tuple[float, float, float]:
    probabilities_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        probabilities_array.shape != (3,)
        or np.any(probabilities_array < 0)
        or np.any(probabilities_array > 1)
        or np.any(np.diff(probabilities_array) < 0)
    ):
        raise ValueError(
            "probabilities must contain three ordered values "
            "between 0 and 1."
        )

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    q025, median, q975 = np.quantile(
        values,
        probabilities_array,
    )

    return float(q025), float(median), float(q975)


def summarize_draws(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    q025, median, q975 = posterior_quantiles(values)

    return {
        f"{prefix}_median": median,
        f"{prefix}_q025": q025,
        f"{prefix}_q975": q975,
    }


def probability_of_direction(
    values: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(
        max(
            np.mean(values > 0),
            np.mean(values < 0),
        )
    )


def probability_in_rope(
    values: np.ndarray,
    epsilon: float,
) -> float:
    if epsilon < 0:
        raise ValueError("epsilon cannot be negative.")

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.mean(np.abs(values) <= epsilon))


def directional_probabilities(
    values: np.ndarray,
    epsilon: float = 0.0,
) -> dict[str, float]:
    if epsilon < 0:
        raise ValueError("epsilon cannot be negative.")

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "p_positive": np.nan,
            "p_negative": np.nan,
        }

    return {
        "p_positive": float(np.mean(values > epsilon)),
        "p_negative": float(np.mean(values < -epsilon)),
    }


def _vi_draws_dataframe(fit: Any) -> pd.DataFrame:
    if not hasattr(fit, "variational_sample"):
        raise TypeError(
            "The fit does not contain variational samples."
        )

    if not hasattr(fit, "column_names"):
        raise TypeError(
            "The variational fit does not contain column names."
        )

    samples = np.asarray(
        fit.variational_sample,
        dtype=float,
    )
    column_names = list(fit.column_names)

    if samples.ndim != 2:
        raise ValueError(
            "variational_sample must have shape "
            "(draws, parameters)."
        )

    if samples.shape[1] != len(column_names):
        raise ValueError(
            "variational_sample columns do not match "
            "column_names."
        )

    return pd.DataFrame(samples, columns=column_names)


def _extract_mcmc(
    fit: Any,
    name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    try:
        return np.asarray(
            fit.stan_variable(name),
            dtype=float,
        )
    except Exception as exc:
        if required:
            raise KeyError(
                f"Posterior variable {name!r} was not found."
            ) from exc
        return None


def _extract_vi_scalar(
    draws_df: pd.DataFrame,
    name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    if name in draws_df.columns:
        return draws_df[name].to_numpy(dtype=float)

    if required:
        raise KeyError(
            f"Variational parameter {name!r} was not found."
        )

    return None


def _extract_vi_vector(
    draws_df: pd.DataFrame,
    name: str,
    length: int,
    *,
    required: bool = True,
) -> np.ndarray | None:
    columns = [
        f"{name}[{index}]"
        for index in range(1, length + 1)
    ]

    if all(column in draws_df.columns for column in columns):
        return draws_df[columns].to_numpy(dtype=float)

    if required:
        missing = [
            column
            for column in columns
            if column not in draws_df.columns
        ]
        raise KeyError(
            f"Missing columns for {name!r}: {missing}"
        )

    return None


def _normalize_indexed_draws(
    values: np.ndarray | None,
    *,
    name: str,
    analysis_mode: AnalysisMode,
    n_subtypes: int,
) -> np.ndarray | None:
    """
    Normalize single-model scalars and subtype-model vectors to
    shape (draws, subtypes).
    """
    if values is None:
        return None

    values = np.asarray(values, dtype=float)

    if analysis_mode == "single_group":
        if values.ndim == 1:
            return values[:, None]

        if values.ndim == 2 and values.shape[1] == 1:
            return values

        raise ValueError(
            f"Single-group variable {name!r} must have "
            f"shape (draws,) or (draws, 1), not {values.shape}."
        )

    if values.ndim != 2 or values.shape[1] != n_subtypes:
        raise ValueError(
            f"Subtype variable {name!r} must have shape "
            f"(draws, {n_subtypes}), not {values.shape}."
        )

    return values


def extract_posterior_draws(
    fit: Any,
    *,
    engine: InferenceEngine,
    analysis_mode: AnalysisMode,
    n_subtypes: int,
) -> dict[str, np.ndarray | None]:
    engine = engine.lower()

    if engine not in {
        "nuts",
        "vi_meanfield",
        "vi_fullrank",
    }:
        raise ValueError(f"Unknown engine: {engine!r}.")

    if analysis_mode == "single_group":
        if n_subtypes != 1:
            raise ValueError(
                "single_group requires exactly one subtype level."
            )
    elif analysis_mode == "subtype_comparison":
        if n_subtypes < 2:
            raise ValueError(
                "subtype_comparison requires at least two subtypes."
            )
    else:
        raise ValueError(
            f"Unknown analysis_mode: {analysis_mode!r}."
        )

    required_indexed = [
        "b0",
        "b_scaling",
        "b_deviation",
        "lp_2to1",
        "lp_2to3",
        "lp_2to4",
    ]

    optional_indexed = [
        "lp_scaling_2to1",
        "lp_dev_2to1",
        "lp_scaling_2to3",
        "lp_dev_2to3",
        "lp_scaling_2to4",
        "lp_dev_2to4",
        "cancel_index_2to1",
        "cancel_index_2to3",
        "cancel_index_2to4",
    ]

    draws: dict[str, np.ndarray | None] = {}

    if engine == "nuts":
        if not callable(getattr(fit, "stan_variable", None)):
            raise TypeError(
                "engine='nuts' requires a CmdStanMCMC-like object."
            )

        draws["phi"] = _extract_mcmc(fit, "phi")
        draws["b_noncancer_log"] = _extract_mcmc(
            fit,
            "b_noncancer_log",
        )

        for name in required_indexed:
            draws[name] = _normalize_indexed_draws(
                _extract_mcmc(fit, name),
                name=name,
                analysis_mode=analysis_mode,
                n_subtypes=n_subtypes,
            )

        for name in optional_indexed:
            draws[name] = _normalize_indexed_draws(
                _extract_mcmc(
                    fit,
                    name,
                    required=False,
                ),
                name=name,
                analysis_mode=analysis_mode,
                n_subtypes=n_subtypes,
            )

        if analysis_mode == "subtype_comparison":
            draws["delta_tumor0_log"] = _extract_mcmc(
                fit,
                "delta_tumor0_log",
            )
            draws["delta_scaling"] = _extract_mcmc(
                fit,
                "delta_scaling",
            )
            draws["delta_dev"] = _extract_mcmc(
                fit,
                "delta_dev",
            )

    else:
        draws_df = _vi_draws_dataframe(fit)

        draws["phi"] = _extract_vi_scalar(
            draws_df,
            "phi",
        )
        draws["b_noncancer_log"] = _extract_vi_scalar(
            draws_df,
            "b_noncancer_log",
        )

        for name in required_indexed:
            raw = (
                _extract_vi_scalar(draws_df, name)
                if analysis_mode == "single_group"
                else _extract_vi_vector(
                    draws_df,
                    name,
                    n_subtypes,
                )
            )

            draws[name] = _normalize_indexed_draws(
                raw,
                name=name,
                analysis_mode=analysis_mode,
                n_subtypes=n_subtypes,
            )

        for name in optional_indexed:
            raw = (
                _extract_vi_scalar(
                    draws_df,
                    name,
                    required=False,
                )
                if analysis_mode == "single_group"
                else _extract_vi_vector(
                    draws_df,
                    name,
                    n_subtypes,
                    required=False,
                )
            )

            draws[name] = _normalize_indexed_draws(
                raw,
                name=name,
                analysis_mode=analysis_mode,
                n_subtypes=n_subtypes,
            )

        if analysis_mode == "subtype_comparison":
            draws["delta_tumor0_log"] = _extract_vi_scalar(
                draws_df,
                "delta_tumor0_log",
            )
            draws["delta_scaling"] = _extract_vi_scalar(
                draws_df,
                "delta_scaling",
            )
            draws["delta_dev"] = _extract_vi_scalar(
                draws_df,
                "delta_dev",
            )

    if analysis_mode == "single_group":
        draws["delta_tumor0_log"] = None
        draws["delta_scaling"] = None
        draws["delta_dev"] = None

    return draws


def _summarize_transition(
    output: dict[str, Any],
    log_effect_draws: np.ndarray,
    *,
    transition: str,
    subtype_number: int,
    eps_frac: float,
) -> None:
    log_effect_draws = np.asarray(
        log_effect_draws,
        dtype=float,
    ).reshape(-1)
    log_effect_draws = log_effect_draws[
        np.isfinite(log_effect_draws)
    ]

    fractional_draws = np.expm1(log_effect_draws)

    output.update(
        summarize_draws(
            log_effect_draws,
            f"lp_{transition}_s{subtype_number}",
        )
    )
    output.update(
        summarize_draws(
            fractional_draws,
            f"fracCN_{transition}_s{subtype_number}",
        )
    )

    output[
        f"ppd_fracCN_{transition}_s{subtype_number}"
    ] = probability_of_direction(fractional_draws)

    output[
        f"p_rope_fracCN_{transition}_s{subtype_number}"
    ] = probability_in_rope(
        fractional_draws,
        eps_frac,
    )

    direction = directional_probabilities(
        fractional_draws,
        epsilon=eps_frac,
    )

    output[
        f"p_fracCN_{transition}_pos_s{subtype_number}"
    ] = direction["p_positive"]

    output[
        f"p_fracCN_{transition}_neg_s{subtype_number}"
    ] = direction["p_negative"]


def summarize_posterior(
    fit: Any,
    *,
    engine: InferenceEngine,
    analysis_mode: AnalysisMode,
    subtype_levels: list[str],
    rope_logfc: float = float(np.log(1.2)),
    rope_scaling: float = 0.10,
    rope_deviation: float = 0.10,
    rope_b_deviation: float = 0.10,
    eps_frac: float = 0.10,
    return_all_subtypes: bool = True,
) -> dict[str, Any]:
    if not subtype_levels:
        raise ValueError(
            "subtype_levels must contain at least one level."
        )

    thresholds = {
        "rope_logfc": rope_logfc,
        "rope_scaling": rope_scaling,
        "rope_deviation": rope_deviation,
        "rope_b_deviation": rope_b_deviation,
        "eps_frac": eps_frac,
    }

    for name, value in thresholds.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

    n_subtypes = len(subtype_levels)

    draws = extract_posterior_draws(
        fit,
        engine=engine,
        analysis_mode=analysis_mode,
        n_subtypes=n_subtypes,
    )

    output: dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "subtype_levels": list(subtype_levels),
        "n_subtypes": n_subtypes,
        "engine": engine,
    }

    output.update(
        summarize_draws(draws["phi"], "phi")
    )
    output.update(
        summarize_draws(
            draws["b_noncancer_log"],
            "b_noncancer_log",
        )
    )

    if analysis_mode == "subtype_comparison":
        delta_tumor0_log = np.asarray(
            draws["delta_tumor0_log"],
            dtype=float,
        ).reshape(-1)
        delta_scaling = np.asarray(
            draws["delta_scaling"],
            dtype=float,
        ).reshape(-1)
        delta_deviation = np.asarray(
            draws["delta_dev"],
            dtype=float,
        ).reshape(-1)

        output.update(
            summarize_draws(
                delta_tumor0_log / np.log(2.0),
                "tumor0_lfc",
            )
        )
        output["ppd_tumor"] = probability_of_direction(
            delta_tumor0_log
        )
        output["p_rope_tumor"] = probability_in_rope(
            delta_tumor0_log,
            rope_logfc,
        )

        tumor_direction = directional_probabilities(
            delta_tumor0_log
        )
        output["p_pos_tumor"] = tumor_direction[
            "p_positive"
        ]
        output["p_neg_tumor"] = tumor_direction[
            "p_negative"
        ]

        output.update(
            summarize_draws(
                delta_scaling,
                "delta_scaling",
            )
        )
        output["ppd_scaling"] = probability_of_direction(
            delta_scaling
        )
        output["p_rope_scaling"] = probability_in_rope(
            delta_scaling,
            rope_scaling,
        )

        scaling_direction = directional_probabilities(
            delta_scaling
        )
        output["p_pos_scaling"] = scaling_direction[
            "p_positive"
        ]
        output["p_neg_scaling"] = scaling_direction[
            "p_negative"
        ]

        output.update(
            summarize_draws(
                delta_deviation,
                "delta_dev",
            )
        )
        output["ppd_dev"] = probability_of_direction(
            delta_deviation
        )
        output["p_rope_dev"] = probability_in_rope(
            delta_deviation,
            rope_deviation,
        )

        dev_direction = directional_probabilities(
            delta_deviation
        )
        output["p_pos_dev"] = dev_direction["p_positive"]
        output["p_neg_dev"] = dev_direction["p_negative"]

    subtype_indices = (
        range(n_subtypes)
        if return_all_subtypes
        else range(min(n_subtypes, 2))
    )

    optional_names = [
        "lp_scaling_2to1",
        "lp_dev_2to1",
        "lp_scaling_2to3",
        "lp_dev_2to3",
        "lp_scaling_2to4",
        "lp_dev_2to4",
        "cancel_index_2to1",
        "cancel_index_2to3",
        "cancel_index_2to4",
    ]

    for subtype_index in subtype_indices:
        subtype_number = subtype_index + 1

        output[
            f"subtype_label_s{subtype_number}"
        ] = subtype_levels[subtype_index]

        for parameter in [
            "b0",
            "b_scaling",
            "b_deviation",
        ]:
            output.update(
                summarize_draws(
                    draws[parameter][:, subtype_index],
                    f"{parameter}_s{subtype_number}",
                )
            )

        output[
            f"p_rope_bdev_s{subtype_number}"
        ] = probability_in_rope(
            draws["b_deviation"][:, subtype_index],
            rope_b_deviation,
        )

        for transition in ["2to1", "2to3", "2to4"]:
            _summarize_transition(
                output,
                draws[f"lp_{transition}"][
                    :,
                    subtype_index,
                ],
                transition=transition,
                subtype_number=subtype_number,
                eps_frac=eps_frac,
            )

        for name in optional_names:
            values = draws.get(name)

            if values is not None:
                output.update(
                    summarize_draws(
                        values[:, subtype_index],
                        f"{name}_s{subtype_number}",
                    )
                )

    return output
