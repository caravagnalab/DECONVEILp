from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from cmdstanpy import CmdStanModel


AnalysisMode = Literal["single_group", "subtype_comparison"]
InferenceEngine = Literal["nuts", "vi_meanfield", "vi_fullrank"]

"""
Stan inference utilities for BDGDM.
Supports NUTS and Variational Inference.
"""

def make_initial_values(
    *,
    analysis_mode: AnalysisMode,
    n_subtypes: int = 1,
    seed: int = 1,
) -> dict[str, Any]:
    """Generate initial values compatible with the selected Stan model."""
    rng = np.random.default_rng(seed)

    common = {
        "b_noncancer_log": float(
            rng.normal(np.log(5.0), 0.15)
        ),
        "phi": float(
            np.clip(
                rng.lognormal(np.log(20.0), 0.15),
                1e-3,
                500.0,
            )
        ),
    }

    if analysis_mode == "single_group":
        return {
            "b0": float(rng.normal(5.0, 0.20)),
            "b_scaling": float(rng.normal(0.4, 0.10)),
            "b_deviation": float(rng.normal(0.0, 0.05)),
            **common,
        }

    if analysis_mode == "subtype_comparison":
        if n_subtypes < 2:
            raise ValueError(
                "subtype_comparison requires at least two subtypes."
            )

        return {
            "b0_mean": float(rng.normal(5.0, 0.20)),
            "b_scaling_mean": float(
                rng.normal(0.4, 0.10)
            ),
            "b_dev_mean": float(rng.normal(0.0, 0.05)),
            "b0_offset": rng.normal(
                0.0, 0.10, size=n_subtypes
            ).tolist(),
            "b_scaling_offset": rng.normal(
                0.0, 0.08, size=n_subtypes
            ).tolist(),
            "b_dev_offset": rng.normal(
                0.0, 0.05, size=n_subtypes
            ).tolist(),
            **common,
        }

    raise ValueError(
        "analysis_mode must be 'single_group' or "
        "'subtype_comparison'."
    )


def _make_chain_initial_values(
    *,
    analysis_mode: AnalysisMode,
    n_subtypes: int,
    chains: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create distinct initial values for each NUTS chain."""
    if chains < 1:
        raise ValueError("chains must be at least 1.")

    return [
        make_initial_values(
            analysis_mode=analysis_mode,
            n_subtypes=n_subtypes,
            seed=seed + 1009 * chain_index,
        )
        for chain_index in range(chains)
    ]


def run_inference(
    *,
    stan_data: dict[str, Any],
    analysis_mode: AnalysisMode,
    model_single: CmdStanModel,
    model_subtype: CmdStanModel,
    engine: InferenceEngine = "nuts",
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
    """Run NUTS or variational inference."""
    engine = engine.lower()

    if engine not in {
        "nuts",
        "vi_meanfield",
        "vi_fullrank",
    }:
        raise ValueError(
            "engine must be 'nuts', 'vi_meanfield', "
            "or 'vi_fullrank'."
        )

    if analysis_mode == "single_group":
        model = model_single
        n_subtypes = 1

        unexpected = {"S", "subtype"} & set(stan_data)
        if unexpected:
            raise ValueError(
                "Single-group data must not contain "
                f"{sorted(unexpected)}."
            )

    elif analysis_mode == "subtype_comparison":
        model = model_subtype

        if "S" not in stan_data or "subtype" not in stan_data:
            raise ValueError(
                "Subtype-comparison data must contain "
                "'S' and 'subtype'."
            )

        n_subtypes = int(stan_data["S"])

        if n_subtypes < 2:
            raise ValueError(
                "Subtype comparison requires S >= 2."
            )

    else:
        raise ValueError(
            f"Unknown analysis mode: {analysis_mode!r}."
        )

    if not 0 < adapt_delta < 1:
        raise ValueError(
            "adapt_delta must lie between 0 and 1."
        )

    if max_treedepth < 1:
        raise ValueError(
            "max_treedepth must be at least 1."
        )

    if engine == "nuts":
        fit = model.sample(
            data=stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            inits=_make_chain_initial_values(
                analysis_mode=analysis_mode,
                n_subtypes=n_subtypes,
                chains=chains,
                seed=seed,
            ),
            show_progress=show_progress,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
        )
    else:
        algorithm = (
            "meanfield"
            if engine == "vi_meanfield"
            else "fullrank"
        )

        fit = model.variational(
            data=stan_data,
            seed=seed,
            inits=make_initial_values(
                analysis_mode=analysis_mode,
                n_subtypes=n_subtypes,
                seed=seed,
            ),
            algorithm=algorithm,
            iter=vi_iter,
            grad_samples=vi_grad_samples,
            elbo_samples=vi_elbo_samples,
            output_samples=vi_output_samples,
            show_console=show_progress,
        )

    if (
        output_dir is not None
        and engine == "nuts"
        and hasattr(fit, "save_csvfiles")
    ):
        csv_dir = Path(output_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        fit.save_csvfiles(dir=str(csv_dir))

    return fit
    