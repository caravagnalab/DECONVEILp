from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

"""

Stan inference utilities for BDGDM.
Supports NUTS and Variational Inference.

"""

def make_initial_values(
    S: int,
    seed: int = 1,
) -> dict:
    """
    Generate random initial values for Stan.

    Parameters
    ----------
    S: Number of subtype groups.

    seed: Random seed.

    Returns
    -------
    dict
    """

    rng = np.random.default_rng(seed)

    phi_init = float(
        np.clip(
            rng.exponential(1.0),
            1e-3,
            100.0,
        )
    )

    return {

        "b0_mean":
            float(rng.normal(0, 0.2)),

        "b_scaling_mean":
            float(rng.normal(0, 0.2)),

        "b_dev_mean":
            float(rng.normal(0, 0.05)),

        "b0_offset":
            rng.normal(0, 0.1, S).tolist(),

        "b_scaling_offset":
            rng.normal(0, 0.1, S).tolist(),

        "b_dev_offset":
            rng.normal(0, 0.05, S).tolist(),

        "b_noncancer_log":
            float(rng.normal(0, 0.2)),

        "phi":
            phi_init,
    }


def run_inference(
    *,
    stan_data: dict,
    analysis_mode: Literal[
        "single_group",
        "subtype_comparison",
    ],
    model_single: CmdStanModel,
    model_subtype: CmdStanModel,
    engine: Literal[
        "nuts",
        "vi_meanfield",
        "vi_fullrank",
    ] = "nuts",
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    adapt_delta: float = 0.90,
    max_treedepth: int = 12,
    seed: int = 1,
    show_progress: bool = False,
    output_dir: str | Path | None = None,
    vi_iter: int = 20000,
    vi_output_samples: int = 2000,
    vi_grad_samples: int = 1,
    vi_elbo_samples: int = 100,
):
    """
    Run Stan inference.

    Returns
    -------
    fit
        CmdStanMCMC or CmdStanVB object.
    """

    if analysis_mode == "single_group":
        model = model_single

    elif analysis_mode == "subtype_comparison":
        model = model_subtype

    else:

        raise ValueError(
            f"Unknown analysis mode: {analysis_mode}"
        )

    init = make_initial_values(
        stan_data["S"],
        seed,
    )

    engine = engine.lower()

    # NUTS
    
    if engine == "nuts":
        fit = model.sample(
            data=stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            inits=init,
            show_progress=show_progress,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
        )

    # Variational inference
    
    elif engine in {"vi_meanfield", "vi_fullrank",
                   }:

        algorithm = (
            "meanfield"
            if engine == "vi_meanfield"
            else "fullrank"
        )

        fit = model.variational(
            data=stan_data,
            seed=seed,
            algorithm=algorithm,
            iter=vi_iter,
            grad_samples=vi_grad_samples,
            elbo_samples=vi_elbo_samples,
            output_samples=vi_output_samples,
            show_console=show_progress,
        )

    else:

        raise ValueError(
            "engine must be "
            "'nuts', "
            "'vi_meanfield', "
            "or "
            "'vi_fullrank'"
        )

    # Save CmdStan csv files (optional)

    if (
        output_dir is not None
        and engine == "nuts"
        and hasattr(fit, "save_csvfiles")
    ):

        csv_dir = Path(output_dir) / "csv"

        csv_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fit.save_csvfiles(
            dir=str(csv_dir)
        )

    return fit