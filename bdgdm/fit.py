from __future__ import annotations

from dataclasses import asdict, dataclass 
from pathlib import Path 
from typing import Literal 
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

    engine: Literal[
        "nuts", 
        "vi_meanfield", 
        "vi_fullrank", 
    ] = "nuts"

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
    
    rope_logfc: float = float(np.log(1.2)) 
    eps_frac: float = 0.10 
    return_all_subtypes: bool = True 
    
    return_ppc: bool = False 
    save_ppc_draws: bool = False 
    ppc_thin: int = 10 
    
    save_draws: bool = False 
    save_summary: bool = True 
    
    output_dir: str | Path | None = None

    def __post_init__(self) -> None: 
        valid_engines = { 
            "nuts", 
            "vi_meanfield", 
            "vi_fullrank", 
        }
        if self.engine not in valid_engines: 
            raise ValueError( 
                f"Unknown inference engine '{self.engine}'. " 
                f"Expected one of {sorted(valid_engines)}." 
            )
        if self.chains < 1: 
            raise ValueError("chains must be at least 1.")

        if self.iter_warmup < 0: 
            raise ValueError("iter_warmup cannot be negative.")

        if self.iter_sampling < 1: 
            raise ValueError("iter_sampling must be at least 1.")

        if not 0 < self.adapt_delta < 1: 
            raise ValueError("adapt_delta must lie between 0 and 1.")

        if self.max_treedepth < 1: 
            raise ValueError("max_treedepth must be at least 1.")

        if self.vi_iter < 1: 
            raise ValueError("vi_iter must be at least 1.")

        if self.vi_output_samples < 1: 
            raise ValueError("vi_output_samples must be at least 1.")

        if self.eps_frac < 0: 
            raise ValueError("eps_frac cannot be negative.")

        if self.rope_logfc < 0: 
            raise ValueError("rope_logfc cannot be negative.")

        if self.ppc_thin < 1: 
            raise ValueError("ppc_thin must be at least 1.")

        if self.save_ppc_draws and not self.return_ppc: 
            raise ValueError( 
                "save_ppc_draws=True requires return_ppc=True." 
            )
# Saving utilities

def _json_safe(value): 
    """ 
    Convert common NumPy and Path objects to JSON-compatible values. 
    """ 
    if isinstance(value, Path): 
        return str(value)  
    if isinstance(value, np.ndarray): 
        return value.tolist() 
    if isinstance(value, np.integer): 
        return int(value) 
    if isinstance(value, np.floating): 
        value = float(value) 
        return value if np.isfinite(value) else None 
    if isinstance(value, float): 
        return value if np.isfinite(value) else None 
    if isinstance(value, dict): 
        return { 
            str(key): _json_safe(item) 
            for key, item in value.items() 
        } 
    if isinstance(value, (list, tuple)): 
        return [_json_safe(item) for item in value] 
        
    return value

def _write_json( 
    values: dict, 
    path: Path, 
) -> Path: 
    """ 
    Write a dictionary to JSON. 
    """ 
    path.parent.mkdir(parents=True, exist_ok=True) 
    with path.open("w", encoding="utf-8") as handle: 
        json.dump( 
            _json_safe(values), 
            handle, 
            indent=2, 
            sort_keys=True, 
        ) 
    return path
            

def _save_compact_draws(
    *, 
    fit, 
    engine: str, 
    analysis_mode: str, 
    subtype_levels: list[str], 
    output_file: Path, 
) -> Path:
    """ 
    Save a compact set of posterior arrays in NPZ format. 
    """ 
    draws = extract_posterior_draws( 
        fit, 
        engine=engine, 
        analysis_mode=analysis_mode, 
        n_subtypes=len(subtype_levels), 
    ) 
    save_values = { 
        name: values for name, 
        values in draws.items() 
        if values is not None 
    } 
    output_file.parent.mkdir( 
        parents=True, 
        exist_ok=True, 
    ) 
    np.savez_compressed( 
        output_file, 
        **save_values, 
    ) 
    return output_file
    

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

    # Output directory
    output_dir: Path | None = None 

    if config.output_dir is not None: 
        output_dir = Path(config.output_dir) 
        if gene is not None: 
            output_dir = output_dir / str(gene) 
            output_dir.mkdir( 
                parents=True, 
                exist_ok=True, 
            )
    # Prepare data

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
    # If the gene was inferred inside preprocessing, update the output path.
    if ( 
        config.output_dir is not None 
        and gene is None 
        and metadata.gene is not None 
    ): 
        output_dir = ( 
            Path(config.output_dir) 
            / str(metadata.gene) 
        ) 
        output_dir.mkdir( 
            parents=True, 
            exist_ok=True, 
        )

    # Run inference
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

    # Posterior summaries
    posterior = summarize_posterior( 
        stan_fit, 
        engine=config.engine, 
        analysis_mode=metadata.analysis_mode, 
        subtype_levels=metadata.subtype_levels, 
        rope_logfc=config.rope_logfc, 
        eps_frac=config.eps_frac, 
        return_all_subtypes=config.return_all_subtypes, 
    )

    # Diagnostics
    diagnostics = sampler_diagnostics( 
        stan_fit, 
        engine=config.engine, 
        analysis_mode=metadata.analysis_mode, 
    )

    # Posterior predictive checks

    ppc = None 
    ppc_path = None

    if config.return_ppc: 
        if config.engine != "nuts": 
            raise ValueError( "Posterior predictive checks currently require " 
                             "engine='nuts'." ) 
        ppc = compute_ppc( 
            stan_fit, 
            stan_data["y"], 
        ) 
        if config.save_ppc_draws: 
            if output_dir is None: 
                raise ValueError( 
                    "save_ppc_draws=True requires config.output_dir." 
                ) 
            ppc_path = save_ppc_draws( 
                stan_fit, 
                stan_data["y"], 
                output_dir / "ppc_draws.npz", 
                thin=config.ppc_thin, 
            )

    # Save summaries and compact draws

    saved_files: dict[str, str] = {}

    if output_dir is not None:
        metadata_dict = asdict(metadata)
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
            metadata_path = _write_json(
                metadata_dict,
                output_dir / "metadata.json",
            )
            config_path = _write_json(
                config_dict,
                output_dir / "config.json",
            )

            posterior_csv = output_dir / "posterior_summary.csv"
            pd.DataFrame([posterior]).to_csv(
                posterior_csv,
                index=False,
            )

            saved_files.update(
                {
                    "posterior_json": str(posterior_path),
                    "posterior_csv": str(posterior_csv),
                    "diagnostics_json": str(diagnostics_path),
                    "metadata_json": str(metadata_path),
                    "config_json": str(config_path),
                }
            )

            if ppc is not None:
                ppc_summary_path = _write_json(
                    ppc,
                    output_dir / "ppc_summary.json",
                )
                saved_files["ppc_summary_json"] = str(ppc_summary_path)

        # Saving posterior draws is independent of save_summary and PPC.
        if config.save_draws:
            draws_path = _save_compact_draws(
                fit=stan_fit,
                engine=config.engine,
                analysis_mode=metadata.analysis_mode,
                subtype_levels=metadata.subtype_levels,
                output_file=output_dir / "posterior_draws.npz",
            )
            saved_files["posterior_draws"] = str(draws_path)

        if ppc_path is not None:
            saved_files["ppc_draws"] = str(ppc_path)

    # Extended metadata
    metadata_output = asdict(metadata)
    metadata_output.update(
        {
            "processed_sample_count": int(len(processed_df)),
            "subtype_col": subtype_col,
            "subtype_order": (
                list(subtype_order)
                if subtype_order is not None
                else list(metadata.subtype_levels)
            ),
            "config": asdict(config),
            "saved_files": saved_files,
        }
    )

    return BDGDMFit(
        gene=str(metadata.gene),
        analysis_mode=metadata.analysis_mode,
        fit=stan_fit,
        posterior=posterior,
        diagnostics=diagnostics,
        ppc=ppc,
        metadata=metadata_output,
    )
