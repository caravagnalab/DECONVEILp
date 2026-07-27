from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


"""
Diagnostics for BDGDM Stan fits.

Primary convergence decisions are based on sampled model parameters only.
Generated quantities such as y_rep, log_lik, mu_rep, transition fold changes,
and cancellation indices are excluded from the primary R-hat/ESS decision.
"""


def _safe_max(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else float("nan")


def _safe_min(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float("nan")


def _find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first available column, ignoring capitalization."""
    direct_columns = {str(column): column for column in dataframe.columns}

    for candidate in candidates:
        if candidate in direct_columns:
            return direct_columns[candidate]

    folded_columns = {
        str(column).casefold(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        match = folded_columns.get(candidate.casefold())
        if match is not None:
            return match

    return None


def _optional_int_attribute(
    obj: Any,
    attribute: str,
) -> int | None:
    """Safely extract an integer-valued object attribute."""
    value = getattr(obj, attribute, None)

    if callable(value):
        try:
            value = value()
        except Exception:
            return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _core_parameter_mask(
    parameter_names: pd.Index,
    *,
    analysis_mode: str,
) -> np.ndarray:
    """
    Select sampled parameters used for the primary convergence decision.

    Notes
    -----
    The subtype model's transformed parameters ``b0[s]``,
    ``b_scaling[s]``, and ``b_deviation[s]`` are deterministic functions of
    the sampled mean and offset parameters. They are intentionally excluded
    from the primary scope, although they can be inspected separately.
    """
    names = parameter_names.astype(str)

    if analysis_mode == "single_group":
        allowed = {
            "b0",
            "b_scaling",
            "b_deviation",
            "b_noncancer_log",
            "phi",
        }
        return names.isin(allowed)

    if analysis_mode == "subtype_comparison":
        pattern = re.compile(
            r"^(?:"
            r"b0_mean|"
            r"b_scaling_mean|"
            r"b_dev_mean|"
            r"b_noncancer_log|"
            r"phi|"
            r"b0_offset\[\d+\]|"
            r"b_scaling_offset\[\d+\]|"
            r"b_dev_offset\[\d+\]"
            r")$"
        )
        return np.asarray(
            [bool(pattern.fullmatch(name)) for name in names],
            dtype=bool,
        )

    raise ValueError(
        "analysis_mode must be 'single_group' or "
        "'subtype_comparison'."
    )


def select_core_parameter_summary(
    summary: pd.DataFrame,
    *,
    analysis_mode: str,
) -> pd.DataFrame:
    """Return the CmdStan summary rows used for convergence assessment."""
    mask = _core_parameter_mask(
        summary.index,
        analysis_mode=analysis_mode,
    )
    core = summary.loc[mask].copy()

    if core.empty:
        raise ValueError(
            "No sampled core parameters were found in fit.summary(). "
            "Check the Stan parameter names and analysis_mode."
        )

    return core


def _extreme_parameter(
    summary: pd.DataFrame,
    column: str | None,
    *,
    mode: str,
) -> str | None:
    """Return the parameter name responsible for an extreme diagnostic."""
    if column is None or column not in summary.columns or summary.empty:
        return None

    values = pd.to_numeric(
        summary[column],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    if not values.notna().any():
        return None

    if mode == "max":
        return str(values.idxmax())

    if mode == "min":
        return str(values.idxmin())

    raise ValueError("mode must be 'max' or 'min'.")


def sampler_diagnostics(
    fit: Any,
    *,
    engine: str,
    analysis_mode: str,
    rhat_threshold: float = 1.05,
    ess_threshold: float = 400.0,
) -> dict[str, Any]:
    """
    Extract diagnostics from a CmdStan fit.

    ``max_rhat`` and ``min_bulk_ess`` retain their original public names but
    now refer to sampled core parameters only. Full-summary extrema are
    reported separately as ``max_rhat_all`` and ``min_bulk_ess_all``.
    """
    engine = str(engine).lower()

    output: dict[str, Any] = {
        "engine": engine,
        "analysis_mode": analysis_mode,
        "diagnostic_scope": "sampled_core_parameters",
        "warnings": [],
    }

    if engine in {"vi_meanfield", "vi_fullrank"}:
        output.update(
            {
                "converged": None,
                "diagnostic_status": "not_applicable",
                "max_rhat": float("nan"),
                "min_rhat": float("nan"),
                "min_bulk_ess": float("nan"),
                "min_tail_ess": float("nan"),
                "max_rhat_all": float("nan"),
                "min_bulk_ess_all": float("nan"),
                "n_divergent": float("nan"),
                "max_treedepth_observed": float("nan"),
                "diagnose": None,
            }
        )
        output["warnings"].append(
            "R-hat, ESS, and divergence diagnostics are not "
            "available for variational inference."
        )
        return output

    if engine != "nuts":
        raise ValueError(
            "engine must be 'nuts', 'vi_meanfield', or 'vi_fullrank'."
        )

    try:
        summary = fit.summary()
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the CmdStan summary table."
        ) from exc

    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame(summary)

    core_summary = select_core_parameter_summary(
        summary,
        analysis_mode=analysis_mode,
    )

    output["summary_columns"] = [
        str(column) for column in summary.columns
    ]
    output["n_summary_rows"] = int(len(summary))
    output["n_core_parameters"] = int(len(core_summary))
    output["core_parameters"] = [
        str(name) for name in core_summary.index
    ]

    rhat_column = _find_column(
        summary,
        ["R_hat", "Rhat", "r_hat"],
    )
    bulk_ess_column = _find_column(
        summary,
        [
            "ESS_bulk",
            "Ess_bulk",
            "ess_bulk",
            "N_Eff",
            "n_eff",
            "Ess",
        ],
    )
    tail_ess_column = _find_column(
        summary,
        ["ESS_tail", "Ess_tail", "ess_tail"],
    )

    output["rhat_column"] = rhat_column
    output["bulk_ess_column"] = bulk_ess_column
    output["tail_ess_column"] = tail_ess_column

    # Full-summary values are informational only.
    if rhat_column is not None:
        output["max_rhat_all"] = _safe_max(
            summary[rhat_column]
        )
        output["max_rhat_all_parameter"] = _extreme_parameter(
            summary,
            rhat_column,
            mode="max",
        )

        output["max_rhat"] = _safe_max(
            core_summary[rhat_column]
        )
        output["min_rhat"] = _safe_min(
            core_summary[rhat_column]
        )
        output["max_rhat_parameter"] = _extreme_parameter(
            core_summary,
            rhat_column,
            mode="max",
        )

        if (
            np.isfinite(output["max_rhat"])
            and output["max_rhat"] > rhat_threshold
        ):
            output["warnings"].append(
                "A sampled core parameter has "
                f"R-hat > {rhat_threshold:.3f}: "
                f"{output['max_rhat_parameter']}."
            )
    else:
        output["max_rhat_all"] = float("nan")
        output["max_rhat"] = float("nan")
        output["min_rhat"] = float("nan")
        output["max_rhat_all_parameter"] = None
        output["max_rhat_parameter"] = None
        output["warnings"].append(
            "No R-hat column was found in fit.summary()."
        )

    if bulk_ess_column is not None:
        output["min_bulk_ess_all"] = _safe_min(
            summary[bulk_ess_column]
        )
        output["min_bulk_ess_all_parameter"] = (
            _extreme_parameter(
                summary,
                bulk_ess_column,
                mode="min",
            )
        )

        output["min_bulk_ess"] = _safe_min(
            core_summary[bulk_ess_column]
        )
        output["min_bulk_ess_parameter"] = (
            _extreme_parameter(
                core_summary,
                bulk_ess_column,
                mode="min",
            )
        )

        if (
            np.isfinite(output["min_bulk_ess"])
            and output["min_bulk_ess"] < ess_threshold
        ):
            output["warnings"].append(
                "A sampled core parameter has bulk ESS below "
                f"{ess_threshold:g}: "
                f"{output['min_bulk_ess_parameter']}."
            )
    else:
        output["min_bulk_ess_all"] = float("nan")
        output["min_bulk_ess"] = float("nan")
        output["min_bulk_ess_all_parameter"] = None
        output["min_bulk_ess_parameter"] = None
        output["warnings"].append(
            "No bulk ESS or legacy N_Eff column was found."
        )

    if tail_ess_column is not None:
        output["min_tail_ess_all"] = _safe_min(
            summary[tail_ess_column]
        )
        output["min_tail_ess"] = _safe_min(
            core_summary[tail_ess_column]
        )
        output["min_tail_ess_parameter"] = (
            _extreme_parameter(
                core_summary,
                tail_ess_column,
                mode="min",
            )
        )

        if (
            np.isfinite(output["min_tail_ess"])
            and output["min_tail_ess"] < ess_threshold
        ):
            output["warnings"].append(
                "A sampled core parameter has tail ESS below "
                f"{ess_threshold:g}: "
                f"{output['min_tail_ess_parameter']}."
            )
    else:
        output["min_tail_ess_all"] = float("nan")
        output["min_tail_ess"] = float("nan")
        output["min_tail_ess_parameter"] = None

    # Make instability outside the core scope visible without failing the fit.
    all_rhat_bad = (
        np.isfinite(output["max_rhat_all"])
        and output["max_rhat_all"] > rhat_threshold
    )
    core_rhat_ok = (
        np.isfinite(output["max_rhat"])
        and output["max_rhat"] <= rhat_threshold
    )
    all_ess_bad = (
        np.isfinite(output["min_bulk_ess_all"])
        and output["min_bulk_ess_all"] < ess_threshold
    )
    core_ess_ok = (
        np.isfinite(output["min_bulk_ess"])
        and output["min_bulk_ess"] >= ess_threshold
    )

    if (all_rhat_bad and core_rhat_ok) or (
        all_ess_bad and core_ess_ok
    ):
        output["warnings"].append(
            "Non-core transformed/generated quantities have unstable "
            "Monte Carlo diagnostics. Inspect "
            f"{output.get('max_rhat_all_parameter')!r} and "
            f"{output.get('min_bulk_ess_all_parameter')!r}; "
            "the primary convergence decision is unchanged."
        )

    output["n_chains"] = _optional_int_attribute(
        fit,
        "chains",
    )
    output["n_draws_per_chain"] = _optional_int_attribute(
        fit,
        "num_draws_sampling",
    )

    if (
        output["n_chains"] is not None
        and output["n_draws_per_chain"] is not None
    ):
        output["n_draws_total"] = (
            output["n_chains"]
            * output["n_draws_per_chain"]
        )
    else:
        output["n_draws_total"] = None

    method_variables: dict[str, Any] = {}

    try:
        method_variables = fit.method_variables()
    except Exception:
        output["warnings"].append(
            "Could not extract CmdStan method variables."
        )

    divergent = method_variables.get("divergent__")

    if divergent is not None:
        output["n_divergent"] = int(
            np.sum(np.asarray(divergent))
        )

        if output["n_divergent"] > 0:
            output["warnings"].append(
                f"{output['n_divergent']} divergent transitions detected."
            )
    else:
        output["n_divergent"] = float("nan")

    treedepth = method_variables.get("treedepth__")

    if treedepth is not None:
        output["max_treedepth_observed"] = int(
            np.max(np.asarray(treedepth))
        )
    else:
        output["max_treedepth_observed"] = float("nan")

    try:
        output["diagnose"] = fit.diagnose()
    except Exception:
        output["diagnose"] = None
        output["warnings"].append(
            "CmdStan diagnose() could not be evaluated."
        )

    checks: list[bool] = []

    if np.isfinite(output["max_rhat"]):
        checks.append(
            output["max_rhat"] <= rhat_threshold
        )

    if np.isfinite(output["min_bulk_ess"]):
        checks.append(
            output["min_bulk_ess"] >= ess_threshold
        )

    if np.isfinite(output["min_tail_ess"]):
        checks.append(
            output["min_tail_ess"] >= ess_threshold
        )

    if np.isfinite(output["n_divergent"]):
        checks.append(output["n_divergent"] == 0)

    output["converged"] = (
        bool(all(checks))
        if checks
        else None
    )

    if output["converged"] is True:
        output["diagnostic_status"] = "ok"
    elif output["converged"] is False:
        output["diagnostic_status"] = "warning"
    else:
        output["diagnostic_status"] = "unknown"

    return output


def get_nuts_diagnostics(
    diagnostics: dict[str, Any],
    *,
    rhat_threshold: float = 1.05,
    minimum_ess: float = 400.0,
) -> pd.DataFrame:
    """Return a compact pass/fail table for primary NUTS diagnostics."""
    max_rhat = diagnostics.get("max_rhat", np.nan)
    min_bulk_ess = diagnostics.get(
        "min_bulk_ess",
        np.nan,
    )
    n_divergent = diagnostics.get(
        "n_divergent",
        np.nan,
    )

    checks = [
        {
            "check": "Overall diagnostic status",
            "value": diagnostics.get(
                "diagnostic_status"
            ),
            "passed": (
                diagnostics.get("diagnostic_status")
                == "ok"
            ),
        },
        {
            "check": "Convergence flag",
            "value": diagnostics.get("converged"),
            "passed": (
                diagnostics.get("converged")
                is True
            ),
        },
        {
            "check": "Maximum core R-hat",
            "value": max_rhat,
            "passed": (
                np.isfinite(max_rhat)
                and max_rhat <= rhat_threshold
            ),
        },
        {
            "check": "Minimum core bulk ESS",
            "value": min_bulk_ess,
            "passed": (
                np.isfinite(min_bulk_ess)
                and min_bulk_ess >= minimum_ess
            ),
        },
        {
            "check": "Divergent transitions",
            "value": n_divergent,
            "passed": n_divergent == 0,
        },
    ]

    return pd.DataFrame(checks)
