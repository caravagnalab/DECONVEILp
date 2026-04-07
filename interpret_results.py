import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as st
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class InterpretThresholds:
    
    # Bayesian evidence thresholds
    ppd_sig: float = 0.95            # strong directional evidence
    rope_low: float = 0.05           # effect is non-negligible if ROPE prob <= this
    rope_high: float = 0.95          # effect is practically null if ROPE prob >= this

    # Dosage class thresholds
    dose_prob_sens: float = 0.95     # strong evidence of non-negligible directional CN effect
    dose_prob_ins: float = 0.95      # strong evidence effect stays within ROPE
    dev_small_prob: float = 0.75     # relaxed threshold: deviation remains practically small
    dc_prob: float = 0.95            # strong evidence for compensated behavior

    # fallback if explicit p_DC_* not present
    cancel_threshold: float = 0.50   # minimum compensation magnitude for CI-based fallback

    # minimum scaling magnitude for stable CI interpretation
    min_scaling_abs: float = 1e-3


def _get(res: Dict[str, Any], key: str, default=np.nan):
    return res.get(key, default)


def interpret_gene_result(
    res: Dict[str, Any],
    th: InterpretThresholds = InterpretThresholds(),
) -> Dict[str, Any]:
    """
    Interpret one gene's fitted result dict:
      - baseline DE status (tumor baseline contrast)
      - rewiring status (between subtypes)
      - dosage class per subtype (DSG / DIG / DCG / UNC)
      - PPD + ROPE for effect support
      - transition-level fracCN summaries when available

    Returns a flat dict suitable for DataFrame rows.
    """
    out: Dict[str, Any] = {
        "gene": _get(res, "gene"),
        "status": _get(res, "status"),
        "fit_flag": _get(res, "fit_flag", "ok"),
        "N": _get(res, "N"),
        "n_aneup": _get(res, "n_aneup"),
    }

    # Baseline differential expression status
    ppd_tumor = _get(res, "ppd_tumor", np.nan)
    p_rope_tumor = _get(res, "p_rope_tumor", np.nan)

    if np.isfinite(ppd_tumor) and np.isfinite(p_rope_tumor):
        if ppd_tumor >= th.ppd_sig and p_rope_tumor <= th.rope_low:
            de_status = "DE"
        elif p_rope_tumor >= th.rope_high:
            de_status = "DE-null"
        else:
            de_status = "DE-uncertain"
    else:
        de_status = "DE-uncertain"

    out.update({
        "de_status": de_status,
        "ppd_tumor": ppd_tumor,
        "p_rope_tumor": p_rope_tumor,
        "tumor0_lfc_median": _get(res, "tumor0_lfc_median", np.nan),
        "tumor0_lfc_q025": _get(res, "tumor0_lfc_q025", np.nan),
        "tumor0_lfc_q975": _get(res, "tumor0_lfc_q975", np.nan),
    })

    # Rewiring status between subtypes
    ppd_scaling = _get(res, "ppd_scaling", np.nan)
    ppd_dev = _get(res, "ppd_dev", np.nan)

    # Optional ROPEs for rewiring contrasts, if you later add them
    p_rope_scaling = _get(res, "p_rope_scaling", np.nan)
    p_rope_dev = _get(res, "p_rope_dev", np.nan)

    def supported(ppd_val, rope_val):
        if np.isfinite(ppd_val) and np.isfinite(rope_val):
            return (ppd_val >= th.ppd_sig) and (rope_val <= th.rope_low)
        elif np.isfinite(ppd_val):
            # fallback if no explicit ROPE stored
            return ppd_val >= th.ppd_sig
        return False

    scaling_rewired = supported(ppd_scaling, p_rope_scaling)
    dev_rewired = supported(ppd_dev, p_rope_dev)

    if scaling_rewired and dev_rewired:
        rewiring = "rewired:scaling+deviation"
    elif scaling_rewired:
        rewiring = "rewired:scaling"
    elif dev_rewired:
        rewiring = "rewired:deviation"
    else:
        rewiring = "not_rewired"

    out.update({
        "rewiring_status": rewiring,
        "ppd_scaling": ppd_scaling,
        "ppd_dev": ppd_dev,
        "p_rope_scaling": p_rope_scaling,
        "p_rope_dev": p_rope_dev,
        "delta_scaling_median": _get(res, "delta_scaling_median", np.nan),
        "delta_scaling_q025": _get(res, "delta_scaling_q025", np.nan),
        "delta_scaling_q975": _get(res, "delta_scaling_q975", np.nan),
        "delta_dev_median": _get(res, "delta_dev_median", np.nan),
        "delta_dev_q025": _get(res, "delta_dev_q025", np.nan),
        "delta_dev_q975": _get(res, "delta_dev_q975", np.nan),
    })

    # Dosage class per subtype
    subtype_levels = _get(res, "subtype_levels", None)
    if subtype_levels is None:
        subtype_levels = []

    S = len(subtype_levels)
    if S == 0:
        s_candidates = []
        for k in res.keys():
            if k.startswith("ppd_fracCN_2to3_s"):
                try:
                    s_candidates.append(int(k.split("_s")[-1]))
                except Exception:
                    pass
        S = max(s_candidates) if s_candidates else 2

    out["S"] = S
    if subtype_levels:
        out["subtype_levels_str"] = "|".join(map(str, subtype_levels))

    def subtype_name(s: int) -> str:
        if 1 <= s <= len(subtype_levels):
            return str(subtype_levels[s - 1])
        return f"s{s}"

    def dosage_class_for_subtype(s: int) -> Dict[str, Any]:
        name = subtype_name(s)

        # -------- transition summaries --------
        # gains
        p_gain_pos = _get(res, f"p_fracCN_2to3_pos_s{s}", np.nan)
        p_gain_rope = _get(res, f"p_rope_fracCN_2to3_s{s}", np.nan)
        p_amp_pos = _get(res, f"p_fracCN_2to4_pos_s{s}", np.nan)
        p_amp_rope = _get(res, f"p_rope_fracCN_2to4_s{s}", np.nan)

        # losses
        p_loss_neg = _get(res, f"p_fracCN_2to1_neg_s{s}", np.nan)
        p_loss_rope = _get(res, f"p_rope_fracCN_2to1_s{s}", np.nan)

        # optional transition medians
        frac23_median = _get(res, f"fracCN_2to3_s{s}_median", np.nan)
        frac24_median = _get(res, f"fracCN_2to4_s{s}_median", np.nan)
        frac21_median = _get(res, f"fracCN_2to1_s{s}_median", np.nan)

        # -------- subtype-specific coefficients --------
        b_scaling_median = _get(res, f"b_scaling_s{s}_median", np.nan)
        b_dev_median = _get(res, f"b_deviation_s{s}_median", np.nan)

        # posterior probability that deviation is practically small
        # if not explicitly stored, approximate using median + CI if unavailable -> NaN
        # you can improve this later by returning p_rope_bdev_s* from fit fn
        p_dev_small = _get(res, f"p_rope_bdev_s{s}", np.nan)

        # -------- explicit DC probabilities if available --------
        p_dc_gain = _get(res, f"p_DC_gain_s{s}", np.nan)
        p_dc_loss = _get(res, f"p_DC_loss_s{s}", np.nan)

        # -------- fallback using cancellation index --------
        ci_23 = _get(res, f"cancel_index_2to3_s{s}_median", np.nan)
        ci_24 = _get(res, f"cancel_index_2to4_s{s}_median", np.nan)
        ci_21 = _get(res, f"cancel_index_2to1_s{s}_median", np.nan)

        scaling_stable = np.isfinite(b_scaling_median) and (abs(b_scaling_median) > th.min_scaling_abs)

        dc_gain = False
        dc_loss = False

        if np.isfinite(p_dc_gain):
            dc_gain = p_dc_gain >= th.dc_prob
        elif scaling_stable:
            if np.isfinite(ci_23):
                dc_gain = ci_23 < -th.cancel_threshold
            elif np.isfinite(ci_24):
                dc_gain = ci_24 < -th.cancel_threshold

        if np.isfinite(p_dc_loss):
            dc_loss = p_dc_loss >= th.dc_prob
        elif scaling_stable and np.isfinite(ci_21):
            dc_loss = ci_21 > th.cancel_threshold

        any_dc = dc_gain or dc_loss

        # -------- dosage-insensitive --------
        gain_ins = np.isfinite(p_gain_rope) and (p_gain_rope >= th.dose_prob_ins)
        amp_ins = np.isfinite(p_amp_rope) and (p_amp_rope >= th.dose_prob_ins)
        loss_ins = np.isfinite(p_loss_rope) and (p_loss_rope >= th.dose_prob_ins)

        is_dig = gain_ins and loss_ins
        if np.isfinite(p_amp_rope):
            is_dig = is_dig and amp_ins

        # -------- dosage-sensitive --------
        gain_sens = np.isfinite(p_gain_pos) and (p_gain_pos >= th.dose_prob_sens)
        amp_sens = np.isfinite(p_amp_pos) and (p_amp_pos >= th.dose_prob_sens)
        loss_sens = np.isfinite(p_loss_neg) and (p_loss_neg >= th.dose_prob_sens)

        has_supported_cn_effect = gain_sens or loss_sens or amp_sens

        # allow mild deviation if p_dev_small available, otherwise use fallback on |b_dev|
        if np.isfinite(p_dev_small):
            small_dev = p_dev_small >= th.dev_small_prob
        elif np.isfinite(b_dev_median):
            small_dev = abs(b_dev_median) <= th.cancel_threshold
        else:
            small_dev = False

        is_dsg = has_supported_cn_effect and small_dev

        # -------- classify --------
        if any_dc:
            cls = "DCG"
        elif is_dig:
            cls = "DIG"
        elif is_dsg:
            cls = "DSG"
        else:
            cls = "UNC"

        # -------- directional flags --------
        gain_flag = "unknown"
        if gain_sens or amp_sens:
            gain_flag = "sensitive"
        elif gain_ins and (amp_ins if np.isfinite(p_amp_rope) else True):
            gain_flag = "insensitive"

        loss_flag = "unknown"
        if loss_sens:
            loss_flag = "sensitive"
        elif loss_ins:
            loss_flag = "insensitive"

        return {
            f"dosage_class_{name}": cls,
            f"gain_flag_{name}": gain_flag,
            f"loss_flag_{name}": loss_flag,

            f"p_gain_pos_{name}": p_gain_pos,
            f"p_gain_rope_{name}": p_gain_rope,
            f"p_amp_pos_{name}": p_amp_pos,
            f"p_amp_rope_{name}": p_amp_rope,
            f"p_loss_neg_{name}": p_loss_neg,
            f"p_loss_rope_{name}": p_loss_rope,

            f"fracCN_2to3_median_{name}": frac23_median,
            f"fracCN_2to4_median_{name}": frac24_median,
            f"fracCN_2to1_median_{name}": frac21_median,

            f"b_scaling_median_{name}": b_scaling_median,
            f"b_dev_median_{name}": b_dev_median,
            f"p_dev_small_{name}": p_dev_small,

            f"dc_gain_{name}": dc_gain,
            f"dc_loss_{name}": dc_loss,
            f"p_DC_gain_{name}": p_dc_gain,
            f"p_DC_loss_{name}": p_dc_loss,
            f"ci_23_{name}": ci_23,
            f"ci_24_{name}": ci_24,
            f"ci_21_{name}": ci_21,
        }

    for s in range(1, S + 1):
        out.update(dosage_class_for_subtype(s))

    # Combined summary label
    per_sub = []
    for s in range(1, S + 1):
        name = subtype_name(s)
        per_sub.append(f"{name}:{out.get(f'dosage_class_{name}', 'NA')}")

    out["summary_label"] = f"{de_status} | {rewiring} | " + ",".join(per_sub)

    return out