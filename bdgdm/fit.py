from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping
import json

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from .diagnostics import sampler_diagnostics
from .inference import run_inference
from .model import BDGDMFit
from .posterior import extract_posterior_draws, summarize_posterior

from .ppc import compute_ppc, save_ppc_draws
from .preprocessing import prepare_gene_data


""" 
Fitting interface for BDGDM. 
"""

InferenceEngine = Literal[
    "nuts",
    "vi_meanfield",
    "vi_fullrank",
]

# Configuration

@dataclass 
class BDGDMConfig: 
    """ 
    Configuration for fitting a BDGDM model. 
    Parameters 
    ---------- 
    engine
        Inference engine: ``nuts``, ``vi_meanfield``, or ``vi_fullrank``. 
    chains
        Number of MCMC chains. 
    iter_warmup
        Number of warm-up iterations per chain. 
    iter_sampling
        Number of retained sampling iterations per chain. 
    adapt_delta
        Target NUTS acceptance probability. 
    max_treedepth
        Maximum NUTS tree depth. 
    seed
        Random seed. 
    show_progress
        Display CmdStan progress output. 
    vi_iter
        Maximum number of variational optimisation iterations. 
    vi_output_samples
        Number of draws sampled from the variational approximation. 
    vi_grad_samples
        Number of samples used to estimate the ELBO gradient. 
    vi_elbo_samples
        Number of samples used to estimate the ELBO. 
    rope_logfc ROPE
        half-width for baseline subtype contrasts on the natural-log scale. 
    eps_frac
        ROPE half-width for fractional CN effects. 
    return_all_subtypes
        Return posterior summaries for all fitted subtypes. 
    return_ppc
        Compute posterior predictive summaries. PPCs currently require NUTS output containing ``y_rep``. 
    save_ppc_draws
        Save thinned posterior predictive draws. 
    ppc_thin
        Thinning interval for saved PPC draws. 
    save_draws
        Save a compact subset of posterior draws. 
    save_summary
        Save posterior and diagnostic summaries. output_dir Directory in which model outputs are written. 
    """

    engine: InferenceEngine = "nuts"

    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    adapt_delta: float = 0.90
    max_treedepth: int = 12
    seed: int = 1
    show_progress: bool = False

    vi_iter: int = 20_000
    vi_output_samples: int = 2_000
    vi_grad_samples: int = 1
    vi_elbo_samples: int = 100

    # Posterior-summary ROPE thresholds.
    rope_logfc: float = float(np.log(1.2))
    rope_scaling: float = 0.10
    rope_deviation: float = 0.10
    rope_b_deviation: float = 0.10
    eps_frac: float = 0.10
    return_all_subtypes: bool = True

    # Posterior predictive output.
    return_ppc: bool = False
    save_ppc_draws: bool = False
    ppc_thin: int = 10

    # Saved posterior output.
    save_draws: bool = False
    save_summary: bool = True
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.engine = str(self.engine).lower()

        valid_engines = {
            "nuts",
            "vi_meanfield",
            "vi_fullrank",
        }

        if self.engine not in valid_engines:
            raise ValueError(
                f"Unknown inference engine {self.engine!r}. "
                f"Expected one of {sorted(valid_engines)}."
            )

        integer_minimums = {
            "chains": (self.chains, 1),
            "iter_warmup": (self.iter_warmup, 0),
            "iter_sampling": (self.iter_sampling, 1),
            "max_treedepth": (self.max_treedepth, 1),
            "vi_iter": (self.vi_iter, 1),
            "vi_output_samples": (
                self.vi_output_samples,
                1,
            ),
            "vi_grad_samples": (
                self.vi_grad_samples,
                1,
            ),
            "vi_elbo_samples": (
                self.vi_elbo_samples,
                1,
            ),
            "ppc_thin": (self.ppc_thin, 1),
        }

        for name, (value, minimum) in integer_minimums.items():
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")

            if value < minimum:
                raise ValueError(
                    f"{name} must be at least {minimum}."
                )

        if not 0 < self.adapt_delta < 1:
            raise ValueError(
                "adapt_delta must lie between 0 and 1."
            )

        nonnegative_thresholds = {
            "rope_logfc": self.rope_logfc,
            "rope_scaling": self.rope_scaling,
            "rope_deviation": self.rope_deviation,
            "rope_b_deviation": self.rope_b_deviation,
            "eps_frac": self.eps_frac,
        }

        for name, value in nonnegative_thresholds.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )

        if (
            self.return_ppc or self.save_ppc_draws
        ) and self.engine != "nuts":
            raise ValueError(
                "Posterior predictive checks currently require "
                "engine='nuts'."
            )

        if self.save_ppc_draws and self.output_dir is None:
            raise ValueError(
                "save_ppc_draws=True requires output_dir."
            )

        if self.save_draws and self.output_dir is None:
            raise ValueError(
                "save_draws=True requires output_dir."
            )

# Saving utilities

def _json_safe(value: Any) -> Any:
    """Convert common NumPy, pandas, and pathlib values to JSON."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is pd.NA:
        return None

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value

def _write_json(
    values: Mapping[str, Any],
    path: Path,
) -> Path:
    """Write a mapping to a UTF-8 JSON file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(dict(values)),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    return path
            

def _save_compact_draws(
    *,
    fit: Any,
    engine: InferenceEngine,
    analysis_mode: str,
    subtype_levels: list[str],
    output_file: Path,
) -> Path:
    """Save a compact set of extracted posterior arrays."""
    draws = extract_posterior_draws(
        fit,
        engine=engine,
        analysis_mode=analysis_mode,
        n_subtypes=len(subtype_levels),
    )

    save_values = {
        name: np.asarray(values)
        for name, values in draws.items()
        if values is not None
    }

    output_file = Path(output_file)

    if output_file.suffix.lower() != ".npz":
        output_file = Path(f"{output_file}.npz")

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        **save_values,
    )

    return output_file


def _resolve_gene_output_dir(
    base_output_dir: str | Path | None,
    gene: str,
) -> Path | None:
    """Return and create the per-gene output directory."""
    if base_output_dir is None:
        return None

    output_dir = Path(base_output_dir) / gene
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return output_dir

    

# Main fitting function

def fit_one_gene_bdgdm( 
    gene_df: pd.DataFrame,
    *,
    model_single: CmdStanModel,
    model_subtype: CmdStanModel,
    gene: str | None = None,
    subtype_col: str = "subtype",
    subtype_order: list[str] | None = None,
    cna: Literal["all", "amp", "del"] = "all",
    et: float = 0.15,
    min_aneup: int = 5,
    min_unique_counts: int = 5,
    min_cn_abs_sum: float = 1.0,
    config: BDGDMConfig | None = None,
) -> BDGDMFit: 
    """ 
    Fit BDGDM for one gene. 
    The analysis mode is inferred automatically: 
    - One subtype label: ``single_group`` mode. 
    - Two or more subtype labels: ``subtype_comparison`` mode. 
    
    Parameters 
    ----------
    gene_df 
        Long-format gene-level table containing at least: 
          - ``expr``: raw RNA-seq count 
          - ``copies``: gene-level copy number 
          - ``purity``: tumour purity in [0, 1] 
          - ``sf``: positive sample size factor 
          - subtype column specified by ``subtype_col`` 
          A ``gene`` column is recommended when the table contains multiple genes.

    model_single 
        Compiled CmdStan model supporting one subtype group. 
    model_subtype 
        Compiled CmdStan model supporting subtype contrasts. 
    gene 
        Gene to select. Required when ``gene_df`` contains multiple genes. 
    subtype_col 
        Column containing subtype labels. 
    subtype_order 
        Optional explicit subtype ordering. This ordering determines the direction of subtype contrasts in the Stan model. 
    cna 
        CNA subset to analyse: ``all``, ``amp``, or ``del``. 
    et 
        Numerical tolerance used when defining CNA support. 
    min_aneup 
        Minimum number of samples with supported non-diploid CN. 
    min_unique_counts 
        Minimum number of distinct observed count values. 
    min_cn_abs_sum 
        Minimum total absolute CN deviation from diploidy. 
    config 
        BDGDMConfig instance. Defaults are used when omitted.

    Returns 
    ------- 
    BDGDMFit 
        Fitted model container with raw CmdStan output, posterior summaries, diagnostics, PPC summaries, and metadata. 
        """ 
    if config is None:
        config = BDGDMConfig()

    processed_df, stan_data, metadata = prepare_gene_data(
        gene_df=gene_df,
        gene=gene,
        subtype_col=subtype_col,
        subtype_order=subtype_order,
        cna=cna,
        et=et,
        min_aneup=min_aneup,
        min_unique_counts=min_unique_counts,
        min_cn_abs_sum=min_cn_abs_sum,
    )

    gene_name = (
        str(metadata.gene)
        if metadata.gene is not None
        else "unknown_gene"
    )

    output_dir = _resolve_gene_output_dir(
        config.output_dir,
        gene_name,
    )

    stan_fit = run_inference(
        stan_data=stan_data,
        analysis_mode=metadata.analysis_mode,
        model_single=model_single,
        model_subtype=model_subtype,
        engine=config.engine,
        chains=config.chains,
        iter_warmup=config.iter_warmup,
        iter_sampling=config.iter_sampling,
        adapt_delta=config.adapt_delta,
        max_treedepth=config.max_treedepth,
        seed=config.seed,
        show_progress=config.show_progress,
        output_dir=output_dir,
        vi_iter=config.vi_iter,
        vi_output_samples=config.vi_output_samples,
        vi_grad_samples=config.vi_grad_samples,
        vi_elbo_samples=config.vi_elbo_samples,
    )

    posterior = summarize_posterior(
        stan_fit,
        engine=config.engine,
        analysis_mode=metadata.analysis_mode,
        subtype_levels=metadata.subtype_levels,
        rope_logfc=config.rope_logfc,
        rope_scaling=config.rope_scaling,
        rope_deviation=config.rope_deviation,
        rope_b_deviation=config.rope_b_deviation,
        eps_frac=config.eps_frac,
        return_all_subtypes=config.return_all_subtypes,
    )

    diagnostics = sampler_diagnostics(
        stan_fit,
        engine=config.engine,
        analysis_mode=metadata.analysis_mode,
    )

    ppc: dict[str, Any] | None = None
    ppc_path: Path | None = None

    if config.return_ppc:
        ppc = compute_ppc(
            stan_fit,
            stan_data["y"],
        )

    if config.save_ppc_draws:
        if output_dir is None:
            raise RuntimeError(
                "An output directory was expected but was not created."
            )

        ppc_path = save_ppc_draws(
            stan_fit,
            stan_data["y"],
            output_dir / "ppc_draws.npz",
            thin=config.ppc_thin,
        )

    saved_files: dict[str, str] = {}

    if output_dir is not None:
        config_dict = asdict(config)

        if config.save_summary:
            posterior_path = _write_json(
                posterior,
                output_dir / "posterior_summary.json",
            )
            diagnostics_path = _write_json(
                diagnostics,
                output_dir / "diagnostics.json",
            )
            config_path = _write_json(
                config_dict,
                output_dir / "config.json",
            )

            posterior_csv = (
                output_dir / "posterior_summary.csv"
            )
            pd.DataFrame(
                [_json_safe(posterior)]
            ).to_csv(
                posterior_csv,
                index=False,
            )

            saved_files.update(
                {
                    "posterior_json": str(posterior_path),
                    "posterior_csv": str(posterior_csv),
                    "diagnostics_json": str(
                        diagnostics_path
                    ),
                    "config_json": str(config_path),
                }
            )

            if ppc is not None:
                ppc_summary_path = _write_json(
                    ppc,
                    output_dir / "ppc_summary.json",
                )
                saved_files[
                    "ppc_summary_json"
                ] = str(ppc_summary_path)

        if config.save_draws:
            draws_path = _save_compact_draws(
                fit=stan_fit,
                engine=config.engine,
                analysis_mode=metadata.analysis_mode,
                subtype_levels=metadata.subtype_levels,
                output_file=(
                    output_dir / "posterior_draws.npz"
                ),
            )
            saved_files[
                "posterior_draws"
            ] = str(draws_path)

        if ppc_path is not None:
            saved_files["ppc_draws"] = str(ppc_path)

    metadata_output: dict[str, Any] = {
        **asdict(metadata),
        "processed_sample_count": int(
            len(processed_df)
        ),
        "subtype_col": subtype_col,
        "subtype_order": (
            list(subtype_order)
            if subtype_order is not None
            else list(metadata.subtype_levels)
        ),
        "preprocessing": {
            "cna": cna,
            "et": float(et),
            "min_aneup": int(min_aneup),
            "min_unique_counts": int(
                min_unique_counts
            ),
            "min_cn_abs_sum": float(
                min_cn_abs_sum
            ),
        },
        "config": asdict(config),
        "saved_files": saved_files,
    }

    if output_dir is not None and config.save_summary:
        metadata_path = _write_json(
            metadata_output,
            output_dir / "metadata.json",
        )
        saved_files["metadata_json"] = str(metadata_path)
        metadata_output["saved_files"] = saved_files

        # Rewrite once so the metadata file includes its own path.
        _write_json(
            metadata_output,
            metadata_path,
        )

    return BDGDMFit(
        gene=gene_name,
        analysis_mode=metadata.analysis_mode,
        fit=stan_fit,
        posterior=posterior,
        diagnostics=diagnostics,
        ppc=ppc,
        metadata=metadata_output,
    )
