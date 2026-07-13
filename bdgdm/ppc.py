from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


"""
Posterior predictive checks (PPC) for BDGDM.

This module provides utilities for extracting posterior predictive
replicates, computing global PPC summaries and saving posterior
predictive draws.
"""

def compute_ppc(
    fit,
    y_obs: np.ndarray,
):
    """
    Compute posterior predictive summaries.

    Parameters
    ----------
    fit: CmdStanMCMC object.

    y_obs: Observed counts.

    Returns
    -------
    dict
    """

    if not hasattr(fit, "stan_variable"):
        raise TypeError(
            "Posterior predictive checks currently require "
            "CmdStanMCMC samples."
        )

    y_rep = fit.stan_variable("y_rep")

    obs_mean = float(np.mean(y_obs))
    obs_var = float(np.var(y_obs, ddof=1))
    obs_zero = float(np.mean(y_obs == 0))

    rep_mean = y_rep.mean(axis=1)
    rep_var = y_rep.var(axis=1, ddof=1)
    rep_zero = (y_rep == 0).mean(axis=1)

    return {

        "obs_mean": obs_mean,
        "obs_var": obs_var,
        "obs_zero_fraction": obs_zero,

        "ppc_mean_median": float(np.median(rep_mean)),
        "ppc_mean_q025": float(np.quantile(rep_mean, 0.025)),
        "ppc_mean_q975": float(np.quantile(rep_mean, 0.975)),

        "ppc_var_median": float(np.median(rep_var)),
        "ppc_var_q025": float(np.quantile(rep_var, 0.025)),
        "ppc_var_q975": float(np.quantile(rep_var, 0.975)),

        "ppc_zero_fraction_median": float(np.median(rep_zero)),
        "ppc_zero_fraction_q025": float(np.quantile(rep_zero, 0.025)),
        "ppc_zero_fraction_q975": float(np.quantile(rep_zero, 0.975)),
    }

# Extract PPC draws

def extract_ppc_draws(
    fit,
):
    """
    Extract posterior predictive draws.

    Returns
    -------
    dict
    """

    if not hasattr(fit, "stan_variable"):
        raise TypeError(
            "PPC extraction requires CmdStanMCMC."
        )

    return {

        "mu_rep":
            fit.stan_variable("mu_rep"),

        "y_rep":
            fit.stan_variable("y_rep"),
    }

# Save PPC draws


def save_ppc_draws(
    fit,
    y_obs,
    output_file,
    thin=10,
):
    """
    Save posterior predictive draws.

    Parameters
    ----------
    fit
    y_obs
    output_file
    thin
    """

    draws = extract_ppc_draws(fit)
    y_rep = draws["y_rep"][::thin]
    mu_rep = draws["mu_rep"][::thin]
    
    np.savez_compressed(
        output_file,
        y_obs=y_obs,
        y_rep=y_rep,
        mu_rep=mu_rep,
    )

    return Path(output_file)