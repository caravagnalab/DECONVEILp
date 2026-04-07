from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as st
import os

from joblib import Parallel, delayed
from tqdm.auto import tqdm
import traceback
import ast

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


def fit_one_gene_de(
    gene_df: pd.DataFrame,
    model: "CmdStanModel",
    gene: str | None = None,
    cna: str = "all",
    et: float = 0.15,
    min_aneup: int = 5,
    min_unique_counts: int = 5,
    min_cn_abs_sum: float = 1.0,
    subtype_col: str = "subtype",
    subtype_order: list[str] | None = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 1,
    show_progress: bool = False,
    adapt_delta: float = 0.9,
    max_treedepth: int = 12,
    rope_logfc: float = float(np.log(1.2)),
    eps_frac: float = 0.10,          # ROPE on fracCN for ~10% change
    return_all_subtypes: bool = True,
    engine: str = "nuts",               # "nuts", "vi_meanfield", "vi_fullrank"
    vi_iter: int = 20000,
    vi_output_samples: int = 2000,
    vi_elbo_samples: int = 100,
    vi_grad_samples: int = 1,
    return_ppc: bool = False,
    save_ppc_draws: bool = False,
    ppc_thin: int = 10,
    save_draws: bool = False,
    output_dir: str | Path | None = None,
):
    """
    Fit the Bayesian differential gene-dosage model for a single gene.

    Required columns in gene_df:
        expr, copies, purity, sf, subtype_col
    Optional:
        gene (if gene_df contains multiple genes)

    Expected generated quantities in Stan model:
        delta_tumor0_log, delta_scaling, delta_dev,
        lp_2to1[s], lp_2to3[s], lp_2to4[s]
        posterior probability of direction (PPD) and ROPE probabilities
    Optional:
        lp_scaling_2to1[s], lp_dev_2to1[s],
        lp_scaling_2to3[s], lp_dev_2to3[s],
        lp_scaling_2to4[s], lp_dev_2to4[s],
        y_rep
        
    """

    # ---------- helpers ----------
    def q(x: np.ndarray) -> list[float]:
        return np.quantile(x, [0.025, 0.5, 0.975]).tolist()

    def summarize_draw_1d(x: np.ndarray, prefix: str) -> dict:
        qi = q(x)
        return {
            f"{prefix}_median": float(qi[1]),
            f"{prefix}_q025": float(qi[0]),
            f"{prefix}_q975": float(qi[2]),
        }

    def summarize_draw_2d(arr_2d: np.ndarray, s_idx0: int, prefix: str) -> dict:
        # arr_2d: (draws, S), s_idx0: 0-based subtype index
        return summarize_draw_1d(arr_2d[:, s_idx0], prefix)

    def ppd_from_draws(x: np.ndarray) -> float:
        return float(max((x > 0).mean(), (x < 0).mean()))

    def maybe_var_mcmc(fit_obj, name: str):
        try:
            return fit_obj.stan_variable(name)
        except Exception:
            return None

    def stack_param_vi(draws_df_local: pd.DataFrame, base: str, S_local: int) -> np.ndarray | None:
        cols = [f"{base}[{s}]" for s in range(1, S_local + 1)]
        if all(c in draws_df_local.columns for c in cols):
            return draws_df_local[cols].to_numpy()
        return None

    # ---------- output directory ----------
    out_dir = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------- subset gene --------------
    df = gene_df.copy()    
    if gene is not None and "gene" in df.columns and df["gene"].nunique() > 1:
        df = df.loc[df["gene"] == gene].copy()
    if gene is None and "gene" in df.columns and df["gene"].nunique() == 1:
        gene = str(df["gene"].iloc[0])

    if df.empty:
        return {"status": "skipped", "gene": gene, "reason": "no_rows_for_gene"}

    # required columns
    required = {"expr", "copies", "purity", "sf", subtype_col}
    missing = required - set(df.columns)
    if missing:
        return {"status": "error", "gene": gene, "reason": f"missing_columns: {sorted(missing)}"}

    # ----------- CNA subset --------------
    if cna == "amp":
        df = df[df["copies"] > (2 - et)]
    elif cna == "del":
        df = df[df["copies"] < (2 + et)]
    elif cna == "all":
        pass
    else:
        raise ValueError("cna must be 'amp', 'del', or 'all'")

    if df.empty:
        return {"status": "skipped", "gene": gene, "reason": "no_samples_after_cna_filter"}

    # -------------- QC -----------------
    df = df.dropna(subset=list(required))
    if df.empty:
        return {"status": "skipped", "gene": gene, "reason": "all_na_after_dropna"}

    if (df["expr"] < 0).any():
        return {"status": "error", "gene": gene, "reason": "negative_counts"}

    if not df["purity"].between(0, 1).all():
        return {"status": "error", "gene": gene, "reason": "purity_out_of_bounds"}

    if not (df["sf"] > 0).all():
        return {"status": "error", "gene": gene, "reason": "nonpositive_sf"}

    if df["expr"].nunique() < min_unique_counts:
        return {"status": "skipped", "gene": gene, "reason": "too_few_unique_counts"}

    n_aneup = int((np.abs(df["copies"].astype(float) - 2.0) > (1.0 - et)).sum())
    if n_aneup < min_aneup or (df["expr"] == 0).all():
        return {
            "status": "skipped",
            "gene": gene,
            "n_aneup": n_aneup,
            "reason": "low_aneup_or_all_zero",
        }

    dev_tmp = (df["copies"].astype(float) - 2.0) / 2.0
    if cna == "all" and float(np.abs(dev_tmp).sum()) < min_cn_abs_sum:
        return {
            "status": "skipped",
            "gene": gene,
            "n_aneup": n_aneup,
            "reason": "too_little_cn_variation",
        }

    # -------------- subtype encoding ---------------
    if subtype_order is not None:
        cat = pd.Categorical(df[subtype_col], categories=subtype_order, ordered=True)
        if cat.isna().any():
            bad = df.loc[cat.isna(), subtype_col].unique().tolist()
            return {
                "status": "error",
                "gene": gene,
                "reason": f"unknown_subtypes: {bad}",
            }
        subtype_codes = (cat.codes + 1).astype(int)
        levels = list(cat.categories)
    else:
        levels = sorted(pd.unique(df[subtype_col]).tolist())
        mapping = {lv: i + 1 for i, lv in enumerate(levels)}
        subtype_codes = df[subtype_col].map(mapping).astype(int).to_numpy()

    S = len(levels)
    if S < 2:
        return {
            "status": "skipped",
            "gene": gene,
            "reason": "need_at_least_2_subtypes_present_for_DE",
        }

    # -------------- CN covariates ---------------
    # effective CN for scaling: treat 0 and 1 as "1 copy"
    CN_eff = df["copies"].astype(float).clip(lower=1.0)
    #df["dose_log"] = np.log(np.maximum(df["copies"].astype(float), 0.1) / 2.0)
    df["dose_log"] = np.log(CN_eff / 2.0)
    df["dev"] = (df["copies"].astype(float) - 2.0) / 2.0

    stan_data = {
        "N": int(len(df)),
        "y": df["expr"].astype(int).to_numpy(),
        "S": int(S),
        "subtype": np.asarray(subtype_codes, dtype=int),
        "sf": df["sf"].to_numpy(dtype=float),
        "purity": df["purity"].to_numpy(dtype=float),
        "dose_log": df["dose_log"].to_numpy(dtype=float),
        "dev": df["dev"].to_numpy(dtype=float),
    }

    # ---------- initial values for NUTS -----------
    rng = np.random.default_rng(seed)
    phi_init = rng.exponential(1.0)
    phi_init = float(np.clip(phi_init, 1e-3, 100.0))
    init_nuts = {
        # global means
        "b0_mean": float(rng.normal(0.0, 0.2)),
        "b_scaling_mean": float(rng.normal(0.0, 0.2)),
        "b_dev_mean": float(rng.normal(0.0, 0.05)),
        
        "b0_offset": rng.normal(0.0, 0.1, size=S).tolist(),
        "b_scaling_offset": rng.normal(0.0, 0.1, size=S).tolist(),
        "b_dev_offset": rng.normal(0.0, 0.05, size=S).tolist(),
        
        # stromal baseline
        "b_noncancer_log": float(rng.normal(0.0, 0.2)),
        
        # dispersion 
        "phi": phi_init
    }

    # ------------- inference -------------
    engine = engine.lower()
    using_mcmc = False
    fit = None
    draws_df = None  # only used for VI; for MCMC we extract arrays directly
    
    if engine == "nuts":
        using_mcmc = True
        fit = model.sample(
            data=stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            inits=init_nuts,
            show_progress=show_progress,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
        )
        # persist csvs in a predictable place
        if out_dir is not None and hasattr(fit, "save_csvfiles"):
            csv_dir = out_dir / "csv"
            csv_dir.mkdir(exist_ok=True)
            fit.save_csvfiles(dir=str(csv_dir))

    elif engine in {"vi_meanfield", "vi_fullrank"}:
        algo = "meanfield" if engine == "vi_meanfield" else "fullrank"

        # NOTE: CmdStan's variational inits arg is a *float range*, not a dict.
        # We'll just let Stan pick its default random init unless you want to
        # pass a scalar, e.g. inits=0.1.
        fit = model.variational(
            data=stan_data,
            seed=seed,
            algorithm=algo,
            iter=vi_iter,
            grad_samples=vi_grad_samples,
            elbo_samples=vi_elbo_samples,
            output_samples=vi_output_samples,
            show_console=show_progress,
        )
        # VI: build a DataFrame manually
        # variational_sample: array of shape (draws, num_params)
        draws_np = fit.variational_sample
        colnames = fit.column_names
        draws_df = pd.DataFrame(draws_np, columns=colnames)

    else:
        raise ValueError("engine must be 'nuts', 'vi_meanfield', or 'vi_fullrank'")

   # ---------- extract draws ----------
    if using_mcmc:
        d_tumor = fit.stan_variable("delta_tumor0_log")
        d_scal = fit.stan_variable("delta_scaling")
        d_dev = fit.stan_variable("delta_dev")
        phi_arr = fit.stan_variable("phi")
        b_nc = fit.stan_variable("b_noncancer_log")

        # subtype-specific vectors (draws, S)
        b0 = fit.stan_variable("b0")
        b_scaling = fit.stan_variable("b_scaling")
        b_deviation = fit.stan_variable("b_deviation")

        # canonical transition summaries (draws, S)
        lp_2to1 = fit.stan_variable("lp_2to1")
        lp_2to3 = fit.stan_variable("lp_2to3")
        lp_2to4 = fit.stan_variable("lp_2to4")

        # optional mechanistic decompositions if present
        lp_scaling_2to1 = maybe_var_mcmc(fit, "lp_scaling_2to1")
        lp_dev_2to1 = maybe_var_mcmc(fit, "lp_dev_2to1")
        lp_scaling_2to3 = maybe_var_mcmc(fit, "lp_scaling_2to3")
        lp_dev_2to3 = maybe_var_mcmc(fit, "lp_dev_2to3")
        lp_scaling_2to4 = maybe_var_mcmc(fit, "lp_scaling_2to4")
        lp_dev_2to4 = maybe_var_mcmc(fit, "lp_dev_2to4")

    else:
        # VI: arrays from draws_df
        req = ["delta_tumor0_log", "delta_scaling", "delta_dev", "phi", "b_noncancer_log"]
        missing_cols = [c for c in req if c not in draws_df.columns]
        if missing_cols:
            return {"status": "error", "gene": gene, "reason": f"missing_draws_columns: {missing_cols}"}

        d_tumor = draws_df["delta_tumor0_log"].to_numpy()
        d_scal = draws_df["delta_scaling"].to_numpy()
        d_dev = draws_df["delta_dev"].to_numpy()
        phi_arr = draws_df["phi"].to_numpy()
        b_nc = draws_df["b_noncancer_log"].to_numpy()

        b0 = stack_param_vi(draws_df, "b0", S)
        b_scaling = stack_param_vi(draws_df, "b_scaling", S)
        b_deviation = stack_param_vi(draws_df, "b_deviation", S)

        lp_2to1 = stack_param_vi(draws_df, "lp_2to1", S)
        lp_2to3 = stack_param_vi(draws_df, "lp_2to3", S)
        lp_2to4 = stack_param_vi(draws_df, "lp_2to4", S)

        lp_scaling_2to1 = stack_param_vi(draws_df, "lp_scaling_2to1", S)
        lp_dev_2to1 = stack_param_vi(draws_df, "lp_dev_2to1", S)
        lp_scaling_2to3 = stack_param_vi(draws_df, "lp_scaling_2to3", S)
        lp_dev_2to3 = stack_param_vi(draws_df, "lp_dev_2to3", S)
        lp_scaling_2to4 = stack_param_vi(draws_df, "lp_scaling_2to4", S)
        lp_dev_2to4 = stack_param_vi(draws_df, "lp_dev_2to4", S)

    # ---------- main summaries -------------------------
    
    ln2 = np.log(2.0)

    # tumor baseline contrast (stored in natural log units in Stan)
    lfc_tumor = d_tumor / ln2
    lfc_ci = q(lfc_tumor)

    p_pos_tumor = float((d_tumor > 0).mean())
    p_neg_tumor = float((d_tumor < 0).mean())
    ppd_tumor = float(max(p_pos_tumor, p_neg_tumor))
    p_rope_tumor = float((np.abs(d_tumor) <= rope_logfc).mean())

    d_scal_q = q(d_scal)
    p_pos_scal = float((d_scal > 0).mean())
    p_neg_scal = float((d_scal < 0).mean())
    ppd_scal = float(max(p_pos_scal, p_neg_scal))

    d_dev_q = q(d_dev)
    p_pos_dev = float((d_dev > 0).mean())
    p_neg_dev = float((d_dev < 0).mean())
    ppd_dev = float(max(p_pos_dev, p_neg_dev))

    phi_q = q(phi_arr)
    b_nc_q = q(b_nc)


    out: dict[str, object] = {
        "status": "ok",
        "gene": gene,
        "N": int(len(df)),
        "n_aneup": n_aneup,
        "cna": cna,
        "subtype_levels": levels,
        "engine": engine,
        "seed": int(seed),
        "output_dir": str(out_dir) if out_dir is not None else None,

        "tumor0_lfc_median": float(lfc_ci[1]),
        "tumor0_lfc_q025": float(lfc_ci[0]),
        "tumor0_lfc_q975": float(lfc_ci[2]),
        "ppd_tumor": ppd_tumor,
        "p_pos_tumor": p_pos_tumor,
        "p_neg_tumor": p_neg_tumor,
        "p_rope_tumor": p_rope_tumor,

        "delta_scaling_median": float(q(d_scal)[1]),
        "delta_scaling_q025": float(q(d_scal)[0]),
        "delta_scaling_q975": float(q(d_scal)[2]),
        "ppd_scaling": ppd_scal,
        "p_pos_scaling": p_pos_scal,
        "p_neg_scaling": p_neg_scal,

        "delta_dev_median": float(q(d_dev)[1]),
        "delta_dev_q025": float(q(d_dev)[0]),
        "delta_dev_q975": float(q(d_dev)[2]),
        "ppd_dev": ppd_dev,
        "p_pos_dev": p_pos_dev, 
        "p_neg_dev": p_neg_dev,
       

        "phi_median": float(q(phi_arr)[1]),
        "phi_q025": float(q(phi_arr)[0]),
        "phi_q975": float(q(phi_arr)[2]),
        "b_noncancer_log_median": float(q(b_nc)[1]),
        "b_noncancer_log_q025": float(q(b_nc)[0]),
        "b_noncancer_log_q975": float(q(b_nc)[2]),
    }

    # ---------- subtype-specific summaries ----------
    s_iter = range(1, S + 1) if return_all_subtypes else range(1, min(S, 2) + 1)

    if b0 is not None:
        for s in s_iter:
            s0 = s - 1
            out.update(summarize_draw_2d(b0, s0, f"b0_s{s}"))
    if b_scaling is not None:
        for s in s_iter:
            s0 = s - 1
            out.update(summarize_draw_2d(b_scaling, s0, f"b_scaling_s{s}"))
    if b_deviation is not None:
        for s in s_iter:
            s0 = s - 1
            out.update(summarize_draw_2d(b_deviation, s0, f"b_deviation_s{s}"))

    # transitions
    for s in s_iter:
        s0 = s - 1
        if lp_2to1 is not None:
            lp21 = lp_2to1[:, s0]
            frac21 = np.expm1(lp21)
            out.update(summarize_draw_1d(lp21, f"lp_2to1_s{s}"))
            out.update(summarize_draw_1d(frac21, f"fracCN_2to1_s{s}"))
            out[f"ppd_fracCN_2to1_s{s}"] = ppd_from_draws(frac21)
            out[f"p_rope_fracCN_2to1_s{s}"] = float((np.abs(frac21) <= eps_frac).mean())
            out[f"p_fracCN_2to1_pos_s{s}"] = float((frac21 > eps_frac).mean())
            out[f"p_fracCN_2to1_neg_s{s}"] = float((frac21 < -eps_frac).mean())

        if lp_2to3 is not None:
            lp23 = lp_2to3[:, s0]
            frac23 = np.expm1(lp23)
            out.update(summarize_draw_1d(lp23, f"lp_2to3_s{s}"))
            out.update(summarize_draw_1d(frac23, f"fracCN_2to3_s{s}"))
            out[f"ppd_fracCN_2to3_s{s}"] = ppd_from_draws(frac23)
            out[f"p_rope_fracCN_2to3_s{s}"] = float((np.abs(frac23) <= eps_frac).mean())
            out[f"p_fracCN_2to3_pos_s{s}"] = float((frac23 > eps_frac).mean())
            out[f"p_fracCN_2to3_neg_s{s}"] = float((frac23 < -eps_frac).mean())

        if lp_2to4 is not None:
            lp24 = lp_2to4[:, s0]
            frac24 = np.expm1(lp24)
            out.update(summarize_draw_1d(lp24, f"lp_2to4_s{s}"))
            out.update(summarize_draw_1d(frac24, f"fracCN_2to4_s{s}"))
            out[f"ppd_fracCN_2to4_s{s}"] = ppd_from_draws(frac24)
            out[f"p_rope_fracCN_2to4_s{s}"] = float((np.abs(frac24) <= eps_frac).mean())
            out[f"p_fracCN_2to4_pos_s{s}"] = float((frac24 > eps_frac).mean())
            out[f"p_fracCN_2to4_neg_s{s}"] = float((frac24 < -eps_frac).mean())

        # optional mechanistic decomposition
        if lp_scaling_2to1 is not None:
            out.update(summarize_draw_1d(lp_scaling_2to1[:, s0], f"lp_scaling_2to1_s{s}"))
        if lp_dev_2to1 is not None:
            out.update(summarize_draw_1d(lp_dev_2to1[:, s0], f"lp_dev_2to1_s{s}"))

        if lp_scaling_2to3 is not None:
            out.update(summarize_draw_1d(lp_scaling_2to3[:, s0], f"lp_scaling_2to3_s{s}"))
        if lp_dev_2to3 is not None:
            out.update(summarize_draw_1d(lp_dev_2to3[:, s0], f"lp_dev_2to3_s{s}"))

        if lp_scaling_2to4 is not None:
            out.update(summarize_draw_1d(lp_scaling_2to4[:, s0], f"lp_scaling_2to4_s{s}"))
        if lp_dev_2to4 is not None:
            out.update(summarize_draw_1d(lp_dev_2to4[:, s0], f"lp_dev_2to4_s{s}"))

    
    # ---------- PPC ----------
    if return_ppc and using_mcmc:
        y_obs = stan_data["y"]
        y_rep = fit.stan_variable("y_rep")  # (draws, N)

        out["ppc_summary"] = {
            "obs_mean": float(y_obs.mean()),
            "obs_var": float(y_obs.var(ddof=1)),
            "obs_zero_frac": float((y_obs == 0).mean()),
            "ppc_mean_median": float(np.median(y_rep.mean(axis=1))),
            "ppc_var_median": float(np.median(y_rep.var(axis=1, ddof=1))),
            "ppc_zero_frac_median": float(np.median((y_rep == 0).mean(axis=1))),
        }

        if save_ppc_draws and out_dir is not None:
            y_rep_thin = y_rep[:: max(1, int(ppc_thin)), :]
            ppc_path = out_dir / "ppc_y_rep_thin.npz"
            np.savez_compressed(ppc_path, y_rep=y_rep_thin, y_obs=y_obs)
            out["ppc_path"] = str(ppc_path)

    # ---------- PPC ----------
    if return_ppc and using_mcmc:
        y_obs = stan_data["y"]
        y_rep = fit.stan_variable("y_rep")  # (draws, N)

        out["ppc_summary"] = {
            "obs_mean": float(y_obs.mean()),
            "obs_var": float(y_obs.var(ddof=1)),
            "obs_zero_frac": float((y_obs == 0).mean()),
            "ppc_mean_median": float(np.median(y_rep.mean(axis=1))),
            "ppc_var_median": float(np.median(y_rep.var(axis=1, ddof=1))),
            "ppc_zero_frac_median": float(np.median((y_rep == 0).mean(axis=1))),
        }

        if save_ppc_draws and out_dir is not None:
            y_rep_thin = y_rep[:: max(1, int(ppc_thin)), :]
            ppc_path = out_dir / "ppc_y_rep_thin.npz"
            np.savez_compressed(ppc_path, y_rep=y_rep_thin, y_obs=y_obs)
            out["ppc_path"] = str(ppc_path)

    # ---------- sampler diagnostics ----------
    if using_mcmc:
        # summary CSV (quick human inspection)
        try:
            summ = fit.summary()
            if out_dir is not None:
                summ_path = out_dir / "summary.csv"
                summ.to_csv(summ_path, index=True)
                out["summary_csv"] = str(summ_path)

            # Rhat / ESS core
            rhat_col = next((c for c in ["R_hat", "Rhat"] if c in summ.columns), None)
            ess_col = next((c for c in ["Ess_bulk", "ESS_bulk", "N_Eff", "Ess"] if c in summ.columns), None)

            core_params = ["delta_tumor0_log", "delta_scaling", "delta_dev", "phi"]
            for s in s_iter:
                core_params += [f"b0[{s}]", f"b_scaling[{s}]", f"b_deviation[{s}]"]
            core_params = [p for p in core_params if p in summ.index]

            if rhat_col and core_params:
                rhat_vals = summ.loc[core_params, rhat_col].to_numpy(dtype=float)
                out["max_Rhat_core"] = float(np.nanmax(rhat_vals))
            else:
                out["max_Rhat_core"] = float("nan")

            if ess_col and core_params:
                ess_vals = summ.loc[core_params, ess_col].to_numpy(dtype=float)
                out["min_ESS_core"] = float(np.nanmin(ess_vals))
            else:
                out["min_ESS_core"] = float("nan")

        except Exception:
            out["summary_csv"] = None
            out["max_Rhat_core"] = float("nan")
            out["min_ESS_core"] = float("nan")

        # diagnose text (contains divergences / treedepth warnings)
        try:
            diag = fit.diagnose()
            out["diagnose"] = diag
            if out_dir is not None:
                (out_dir / "diagnose.txt").write_text(diag)
        except Exception:
            out["diagnose"] = None

        # simple flag
        out["fit_flag"] = "ok"
        if (np.isfinite(out.get("max_Rhat_core", np.nan)) and out["max_Rhat_core"] > 1.05) or (
            np.isfinite(out.get("min_ESS_core", np.nan)) and out["min_ESS_core"] < 100
        ):
            out["fit_flag"] = "warn"

    else:
        out["summary_csv"] = None
        out["max_Rhat_core"] = float("nan")
        out["min_ESS_core"] = float("nan")
        out["diagnose"] = None
        out["fit_flag"] = "ok"

    # ---------- optional: save a compact subset of draws ----------
    if save_draws:
        # Save small NPZ with only core arrays (much smaller than draws_pd)
        if out_dir is not None and using_mcmc:
            keep_path = out_dir / "draws_subset.npz"
            np.savez_compressed(
                keep_path,
                delta_tumor0_log=d_tumor,
                delta_scaling=d_scal,
                delta_dev=d_dev,
                phi=phi_arr,
                b_noncancer_log=b_nc,
                b0=b0 if b0 is not None else np.array([]),
                b_scaling=b_scaling if b_scaling is not None else np.array([]),
                b_deviation=b_deviation if b_deviation is not None else np.array([]),
            )
            out["draws_subset_path"] = str(keep_path)
        else:
            out["draws_subset_path"] = None

    return out