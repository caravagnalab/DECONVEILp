from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


"""
Posterior extraction and summary utilities for BDGDM.
"""


# General summary helpers

def posterior_quantiles(
    values: np.ndarray,
    probabilities: tuple[float, float, float] = (0.025, 0.5, 0.975),
) -> tuple[float, float, float]:
    """
    Calculate posterior lower quantile, median, and upper quantile.

    Parameters
    ----------
    values
        One-dimensional array of posterior draws.

    probabilities
        Quantile probabilities. Defaults to the 2.5%, 50%, and 97.5%
        quantiles.

    Returns
    -------
    tuple
        Lower quantile, median, and upper quantile.
    """
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size == 0:
        return np.nan, np.nan, np.nan

    if not np.isfinite(values).any():
        return np.nan, np.nan, np.nan

    quantiles = np.nanquantile(values, probabilities)

    return (
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


def summarize_draws(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """
    Summarize one-dimensional posterior draws.
    """
    q025, median, q975 = posterior_quantiles(values)

    return {
        f"{prefix}_median": median,
        f"{prefix}_q025": q025,
        f"{prefix}_q975": q975,
    }


def probability_of_direction(values: np.ndarray) -> float:
    """
    Compute the posterior probability of direction.

    PPD is the larger of:
      P(theta > 0 | data)
      P(theta < 0 | data)
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    p_positive = np.mean(values > 0)
    p_negative = np.mean(values < 0)

    return float(max(p_positive, p_negative))


def probability_in_rope(
    values: np.ndarray,
    epsilon: float,
) -> float:
    """
    Compute posterior probability that an effect lies inside a ROPE.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.mean(np.abs(values) <= epsilon))


def directional_probabilities(
    values: np.ndarray,
    epsilon: float = 0.0,
) -> dict[str, float]:
    """
    Compute positive and negative posterior probabilities.

    If epsilon > 0:
      positive means theta > epsilon
      negative means theta < -epsilon
    """
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

# Draw extraction

def _is_mcmc_fit(fit: Any) -> bool:
    """
    Check whether the object behaves like a CmdStanMCMC result.
    """
    return callable(getattr(fit, "stan_variable", None))


def _vi_draws_dataframe(fit: Any) -> pd.DataFrame:
    """
    Convert CmdStanVB variational draws to a pandas DataFrame.
    """
    if not hasattr(fit, "variational_sample"):
        raise TypeError(
            "The supplied fit object does not contain variational samples."
        )

    if not hasattr(fit, "column_names"):
        raise TypeError(
            "The supplied variational fit does not contain column names."
        )

    samples = np.asarray(fit.variational_sample)
    column_names = list(fit.column_names)

    if samples.ndim != 2:
        raise ValueError(
            "Expected variational_sample to have shape "
            "(draws, parameters)."
        )

    if samples.shape[1] != len(column_names):
        raise ValueError(
            "Number of variational-sample columns does not match "
            "the number of parameter names."
        )

    return pd.DataFrame(samples, columns=column_names)


def _extract_mcmc_variable(
    fit: Any,
    name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    """
    Extract a variable from a CmdStan MCMC fit.
    """
    try:
        return np.asarray(fit.stan_variable(name))
    except Exception as exc:
        if required:
            raise KeyError(
                f"Required posterior variable '{name}' was not found."
            ) from exc
        return None


def _extract_vi_scalar(
    draws_df: pd.DataFrame,
    name: str,
    *,
    required: bool = True,
) -> np.ndarray | None:
    """
    Extract a scalar parameter from variational draws.
    """
    if name in draws_df.columns:
        return draws_df[name].to_numpy(dtype=float)

    if required:
        raise KeyError(
            f"Required variational parameter '{name}' was not found."
        )

    return None


def _extract_vi_vector(
    draws_df: pd.DataFrame,
    name: str,
    length: int,
    *,
    required: bool = True,
) -> np.ndarray | None:
    """
    Extract a Stan vector from variational draws.

    Returns
    -------
    ndarray
        Shape: (draws, length)
    """
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
            f"Missing columns for '{name}': {missing}"
        )

    return None


def extract_posterior_draws(
    fit: Any,
    *,
    engine: Literal[
        "nuts",
        "vi_meanfield",
        "vi_fullrank",
    ],
    analysis_mode: Literal[
        "single_group",
        "subtype_comparison",
    ],
    n_subtypes: int,
) -> dict[str, np.ndarray | None]:
    """
    Extract posterior arrays from a CmdStan fit.

    Parameters
    ----------
    fit: CmdStanMCMC or CmdStanVB result.

    engine: Inference engine.

    analysis_mode: ``single_group`` or ``subtype_comparison``.

    n_subtypes: Number of subtype levels.

    Returns
    -------
    dict
        Posterior arrays indexed by parameter name.
    """
    if n_subtypes < 1:
        raise ValueError("n_subtypes must be at least 1.")

    engine = engine.lower()

    valid_engines = {
        "nuts",
        "vi_meanfield",
        "vi_fullrank",
    }

    if engine not in valid_engines:
        raise ValueError(
            f"Unknown engine '{engine}'. "
            f"Expected one of {sorted(valid_engines)}."
        )

    if analysis_mode not in {
        "single_group",
        "subtype_comparison",
    }:
        raise ValueError(
            "analysis_mode must be 'single_group' or "
            "'subtype_comparison'."
        )

    vector_parameters = [
        "b0",
        "b_scaling",
        "b_deviation",
        "lp_2to1",
        "lp_2to3",
        "lp_2to4",
    ]

    optional_vector_parameters = [
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
        if not _is_mcmc_fit(fit):
            raise TypeError(
                "engine='nuts' requires a CmdStanMCMC-like object."
            )

        draws["phi"] = _extract_mcmc_variable(fit, "phi")
        draws["b_noncancer_log"] = _extract_mcmc_variable(
            fit,
            "b_noncancer_log",
        )

        for parameter in vector_parameters:
            draws[parameter] = _extract_mcmc_variable(
                fit,
                parameter,
            )

        for parameter in optional_vector_parameters:
            draws[parameter] = _extract_mcmc_variable(
                fit,
                parameter,
                required=False,
            )

        if analysis_mode == "subtype_comparison":
            draws["delta_tumor0_log"] = _extract_mcmc_variable(
                fit,
                "delta_tumor0_log",
            )
            draws["delta_scaling"] = _extract_mcmc_variable(
                fit,
                "delta_scaling",
            )
            draws["delta_dev"] = _extract_mcmc_variable(
                fit,
                "delta_dev",
            )
        else:
            draws["delta_tumor0_log"] = None
            draws["delta_scaling"] = None
            draws["delta_dev"] = None

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

        for parameter in vector_parameters:
            draws[parameter] = _extract_vi_vector(
                draws_df,
                parameter,
                n_subtypes,
            )

        for parameter in optional_vector_parameters:
            draws[parameter] = _extract_vi_vector(
                draws_df,
                parameter,
                n_subtypes,
                required=False,
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
        else:
            draws["delta_tumor0_log"] = None
            draws["delta_scaling"] = None
            draws["delta_dev"] = None

    return draws


# Posterior summary

def _validate_vector_shape(
    values: np.ndarray | None,
    *,
    name: str,
    n_subtypes: int,
) -> None:
    """
    Validate shape of a subtype-indexed posterior array.
    """
    if values is None:
        return

    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(
            f"Posterior variable '{name}' must have shape "
            "(draws, subtypes)."
        )

    if values.shape[1] != n_subtypes:
        raise ValueError(
            f"Posterior variable '{name}' has "
            f"{values.shape[1]} subtype columns, expected "
            f"{n_subtypes}."
        )


def _summarize_subtype_parameter(
    output: dict[str, Any],
    values: np.ndarray | None,
    *,
    name: str,
    subtype_index: int,
    output_prefix: str,
) -> None:
    """
    Add summaries for one subtype-specific parameter.
    """
    if values is None:
        return

    output.update(
        summarize_draws(
            values[:, subtype_index],
            output_prefix,
        )
    )


def _summarize_transition(
    output: dict[str, Any],
    log_effect_draws: np.ndarray,
    *,
    transition: str,
    subtype_number: int,
    eps_frac: float,
) -> None:
    """
    Summarize one CN transition for one subtype.
    """
    log_effect_draws = np.asarray(
        log_effect_draws,
        dtype=float,
    ).reshape(-1)

    fractional_draws = np.expm1(log_effect_draws)

    log_prefix = f"lp_{transition}_s{subtype_number}"
    frac_prefix = f"fracCN_{transition}_s{subtype_number}"

    output.update(
        summarize_draws(
            log_effect_draws,
            log_prefix,
        )
    )

    output.update(
        summarize_draws(
            fractional_draws,
            frac_prefix,
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

    output[
        f"p_fracCN_{transition}_pos_s{subtype_number}"
    ] = float(
        np.mean(fractional_draws > eps_frac)
    )

    output[
        f"p_fracCN_{transition}_neg_s{subtype_number}"
    ] = float(
        np.mean(fractional_draws < -eps_frac)
    )


def summarize_posterior(
    fit: Any,
    *,
    engine: Literal[
        "nuts",
        "vi_meanfield",
        "vi_fullrank",
    ],
    analysis_mode: Literal[
        "single_group",
        "subtype_comparison",
    ],
    subtype_levels: list[str],
    rope_logfc: float = float(np.log(1.2)),
    eps_frac: float = 0.10,
    return_all_subtypes: bool = True,
) -> dict[str, Any]:
    """
    Extract and summarize BDGDM posterior draws.

    Parameters
    ----------
    fit: CmdStanMCMC or CmdStanVB fit.

    engine: Inference engine used to fit the model.

    analysis_mode: ``single_group`` or ``subtype_comparison``.

    subtype_levels: Ordered subtype labels corresponding to Stan subtype indices.

    rope_logfc: ROPE half-width for the diploid baseline contrast. The contrast is
        represented on the natural-log scale in Stan.

    eps_frac
        ROPE half-width for fractional CN transition effects.

    return_all_subtypes
        If False, return summaries only for the first two subtype levels.

    Returns
    -------
    dict
        Flat dictionary of posterior summaries.
    """
    if not subtype_levels:
        raise ValueError(
            "subtype_levels must contain at least one subtype."
        )

    n_subtypes = len(subtype_levels)

    draws = extract_posterior_draws(
        fit,
        engine=engine,
        analysis_mode=analysis_mode,
        n_subtypes=n_subtypes,
    )

    for parameter in [
        "b0",
        "b_scaling",
        "b_deviation",
        "lp_2to1",
        "lp_2to3",
        "lp_2to4",
        "lp_scaling_2to1",
        "lp_dev_2to1",
        "lp_scaling_2to3",
        "lp_dev_2to3",
        "lp_scaling_2to4",
        "lp_dev_2to4",
        "cancel_index_2to1",
        "cancel_index_2to3",
        "cancel_index_2to4",
    ]:
        _validate_vector_shape(
            draws.get(parameter),
            name=parameter,
            n_subtypes=n_subtypes,
        )

    output: dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "subtype_levels": list(subtype_levels),
        "n_subtypes": n_subtypes,
        "engine": engine,
    }

    # Global/non-subtype parameters
    
    output.update(
        summarize_draws(
            np.asarray(draws["phi"]),
            "phi",
        )
    )

    output.update(
        summarize_draws(
            np.asarray(draws["b_noncancer_log"]),
            "b_noncancer_log",
        )
    )

    # Between-subtype contrasts
    
    if analysis_mode == "subtype_comparison":
        delta_tumor0_log = np.asarray(
            draws["delta_tumor0_log"],
            dtype=float,
        ).reshape(-1)

        delta_scaling = np.asarray(
            draws["delta_scaling"],
            dtype=float,
        ).reshape(-1)

        delta_dev = np.asarray(
            draws["delta_dev"],
            dtype=float,
        ).reshape(-1)

        # Stan contrast is on the natural-log scale. Convert to log2 FC.
        tumor0_lfc = delta_tumor0_log / np.log(2.0)

        output.update(
            summarize_draws(
                tumor0_lfc,
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
                delta_dev,
                "delta_dev",
            )
        )
        output["ppd_dev"] = probability_of_direction(
            delta_dev
        )

        dev_direction = directional_probabilities(delta_dev)
        output["p_pos_dev"] = dev_direction["p_positive"]
        output["p_neg_dev"] = dev_direction["p_negative"]

    # Per-subtype parameters and transitions
    
    if return_all_subtypes:
        subtype_indices = range(n_subtypes)
    else:
        subtype_indices = range(min(n_subtypes, 2))

    transition_arrays = {
        "2to1": draws["lp_2to1"],
        "2to3": draws["lp_2to3"],
        "2to4": draws["lp_2to4"],
    }

    optional_arrays = {
        "lp_scaling_2to1": draws.get(
            "lp_scaling_2to1"
        ),
        "lp_dev_2to1": draws.get("lp_dev_2to1"),
        "lp_scaling_2to3": draws.get(
            "lp_scaling_2to3"
        ),
        "lp_dev_2to3": draws.get("lp_dev_2to3"),
        "lp_scaling_2to4": draws.get(
            "lp_scaling_2to4"
        ),
        "lp_dev_2to4": draws.get("lp_dev_2to4"),
        "cancel_index_2to1": draws.get(
            "cancel_index_2to1"
        ),
        "cancel_index_2to3": draws.get(
            "cancel_index_2to3"
        ),
        "cancel_index_2to4": draws.get(
            "cancel_index_2to4"
        ),
    }

    for subtype_index in subtype_indices:
        subtype_number = subtype_index + 1
        subtype_label = subtype_levels[subtype_index]

        output[f"subtype_label_s{subtype_number}"] = (
            subtype_label
        )

        _summarize_subtype_parameter(
            output,
            draws["b0"],
            name="b0",
            subtype_index=subtype_index,
            output_prefix=f"b0_s{subtype_number}",
        )

        _summarize_subtype_parameter(
            output,
            draws["b_scaling"],
            name="b_scaling",
            subtype_index=subtype_index,
            output_prefix=f"b_scaling_s{subtype_number}",
        )

        _summarize_subtype_parameter(
            output,
            draws["b_deviation"],
            name="b_deviation",
            subtype_index=subtype_index,
            output_prefix=f"b_deviation_s{subtype_number}",
        )

        for transition, transition_draws in (
            transition_arrays.items()
        ):
            if transition_draws is None:
                continue

            _summarize_transition(
                output,
                transition_draws[:, subtype_index],
                transition=transition,
                subtype_number=subtype_number,
                eps_frac=eps_frac,
            )

        for parameter_name, parameter_draws in (
            optional_arrays.items()
        ):
            if parameter_draws is None:
                continue

            output.update(
                summarize_draws(
                    parameter_draws[:, subtype_index],
                    f"{parameter_name}_s{subtype_number}",
                )
            )

    return output

