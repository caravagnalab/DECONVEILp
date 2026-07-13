from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


"""

Diagnostics for BDGDM Stan fits.

"""

def _safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.max(x))


def _safe_min(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.min(x))

def _find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Find the first available column, ignoring capitalization.

    This supports differences between CmdStan/CmdStanPy versions, such as:

    - ESS_bulk
    - Ess_bulk
    - N_Eff
    """
    direct_columns = set(dataframe.columns)

    for candidate in candidates:
        if candidate in direct_columns:
            return candidate

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


def sampler_diagnostics(
    fit: Any,
    *,
    engine: str,
    analysis_mode: str,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
) -> dict:
    """
    Extract inference diagnostics from a CmdStan fit.

    Parameters
    ----------
    fit
        CmdStanMCMC or CmdStanVB result.

    engine
        Inference engine: ``nuts``, ``vi_meanfield``, or
        ``vi_fullrank``.

    analysis_mode
        ``single_group`` or ``subtype_comparison``.

    rhat_threshold
        Maximum acceptable R-hat.

    ess_threshold
        Minimum acceptable effective sample size.

    Returns
    -------
    dict
        Diagnostic values and warnings.
    """
    engine = engine.lower()

    output: dict = {
        "engine": engine,
        "analysis_mode": analysis_mode,
        "warnings": [],
    }

    # Variational inference
    
    if engine in {"vi_meanfield", "vi_fullrank"}:
        output.update(
            {
                "converged": None,
                "diagnostic_status": "not_applicable",
                "max_rhat": float("nan"),
                "min_bulk_ess": float("nan"),
                "min_tail_ess": float("nan"),
                "n_divergent": float("nan"),
                "max_treedepth_observed": float("nan"),
                "diagnose": None,
            }
        )

        output["warnings"].append(
            "R-hat, effective sample size, and divergence diagnostics "
            "are not available for variational inference."
        )

        return output

    if engine != "nuts":
        raise ValueError(
            "engine must be 'nuts', 'vi_meanfield', or 'vi_fullrank'."
        )

    # CmdStan summary
  
    try:
        summary = fit.summary()
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the CmdStan summary table."
        ) from exc

    output["summary_columns"] = [
        str(column) for column in summary.columns
    ]

    # Support current and legacy naming conventions.
    rhat_column = _find_column(
        summary,
        [
            "R_hat",
            "Rhat",
            "r_hat",
        ],
    )

    bulk_ess_column = _find_column(
        summary,
        [
            "ESS_bulk",
            "Ess_bulk",
            "ess_bulk",
            "N_Eff",
            "n_eff",
        ],
    )

    tail_ess_column = _find_column(
        summary,
        [
            "ESS_tail",
            "Ess_tail",
            "ess_tail",
        ],
    )

    output["rhat_column"] = rhat_column
    output["bulk_ess_column"] = bulk_ess_column
    output["tail_ess_column"] = tail_ess_column

    # R-hat

    if rhat_column is not None:
        rhat_values = summary[rhat_column].to_numpy(dtype=float)

        output["max_rhat"] = _safe_max(rhat_values)
        output["min_rhat"] = _safe_min(rhat_values)

        if (
            np.isfinite(output["max_rhat"])
            and output["max_rhat"] > rhat_threshold
        ):
            output["warnings"].append(
                f"Some parameters have R-hat > {rhat_threshold:.3f}."
            )
    else:
        output["max_rhat"] = float("nan")
        output["min_rhat"] = float("nan")
        output["warnings"].append(
            "No R-hat column was found in fit.summary()."
        )

    # Effective sample size
  
    if bulk_ess_column is not None:
        bulk_ess_values = summary[
            bulk_ess_column
        ].to_numpy(dtype=float)

        output["min_bulk_ess"] = _safe_min(
            bulk_ess_values
        )

        if (
            np.isfinite(output["min_bulk_ess"])
            and output["min_bulk_ess"] < ess_threshold
        ):
            output["warnings"].append(
                f"Minimum bulk ESS is below {ess_threshold:g}."
            )
    else:
        output["min_bulk_ess"] = float("nan")
        output["warnings"].append(
            "No bulk ESS or legacy N_Eff column was found."
        )

    if tail_ess_column is not None:
        tail_ess_values = summary[
            tail_ess_column
        ].to_numpy(dtype=float)

        output["min_tail_ess"] = _safe_min(
            tail_ess_values
        )

        if (
            np.isfinite(output["min_tail_ess"])
            and output["min_tail_ess"] < ess_threshold
        ):
            output["warnings"].append(
                f"Minimum tail ESS is below {ess_threshold:g}."
            )
    else:
        # Some older CmdStan summaries provide N_Eff but not tail ESS.
        output["min_tail_ess"] = float("nan")

    # Sampling dimensions
  
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

    # Sampler method variables
    
    method_variables = {}

    try:
        method_variables = fit.method_variables()
    except Exception:
        output["warnings"].append(
            "Could not extract CmdStan method variables."
        )

    divergent = method_variables.get("divergent__")

    if divergent is not None:
        divergent = np.asarray(divergent)

        output["n_divergent"] = int(
            np.sum(divergent)
        )

        if output["n_divergent"] > 0:
            output["warnings"].append(
                f"{output['n_divergent']} divergent transitions detected."
            )
    else:
        output["n_divergent"] = float("nan")

    treedepth = method_variables.get("treedepth__")

    if treedepth is not None:
        treedepth = np.asarray(treedepth)

        output["max_treedepth_observed"] = int(
            np.max(treedepth)
        )
    else:
        output["max_treedepth_observed"] = float("nan")

    # CmdStan diagnose output
   
    try:
        output["diagnose"] = fit.diagnose()
    except Exception:
        output["diagnose"] = None
        output["warnings"].append(
            "CmdStan diagnose() could not be evaluated."
        )

    # Overall diagnostic status
   
    checks: list[bool] = []

    if np.isfinite(output["max_rhat"]):
        checks.append(
            output["max_rhat"] <= rhat_threshold
        )

    if np.isfinite(output["min_bulk_ess"]):
        checks.append(
            output["min_bulk_ess"] >= ess_threshold
        )

    if np.isfinite(output["n_divergent"]):
        checks.append(
            output["n_divergent"] == 0
        )

    if checks:
        output["converged"] = bool(all(checks))
    else:
        output["converged"] = None

    if output["converged"] is True:
        output["diagnostic_status"] = "ok"
    elif output["converged"] is False:
        output["diagnostic_status"] = "warning"
    else:
        output["diagnostic_status"] = "unknown"

    return output
