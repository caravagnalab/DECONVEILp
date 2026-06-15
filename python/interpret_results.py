from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class InterpretThresholds:
    # General Bayesian support thresholds
    ppd_sig: float = 0.95
    rope_low: float = 0.05
    rope_high: float = 0.95

    # Transition-level CN effect support
    dose_prob_sens: float = 0.95
    dose_prob_ins: float = 0.95

    # Deviation near-proportional support
    dev_small_prob: float = 0.75

    # Explicit DC posterior support if available
    dc_prob: float = 0.95

    # CI thresholds
    cancel_threshold: float = 0.20
    overcomp_threshold: float = 1.00
    hyper_threshold: float = 0.50

    # Numerical stability
    min_scaling_abs: float = 1e-3

    # Fallback for deviation if posterior ROPE prob is unavailable
    dev_abs_fallback: float = 0.50

    # Minimum number of supported transitions needed before calling Mixed
    min_supported_for_mixed: int = 2

    # DIG-like thresholds
    frac_small_gain: float = 0.15
    frac_small_loss: float = 0.15
    frac_small_amp: float = 0.25

    # Low-CN threshold for UNC-lowCN
    low_cn_aneup_threshold: int = 10


###---------------- Helpers ----------------###

def _get(res: Mapping[str, Any], key: str, default: Any = np.nan) -> Any:
    return res.get(key, default)


def _is_nan_like(x: Any) -> bool:
    return pd.isna(x) and not isinstance(x, (list, tuple, dict))


def _is_finite_number(x: Any) -> bool:
    try:
        return np.isfinite(x)
    except Exception:
        return False


def normalize_subtype_levels(x: Any) -> List[str]:
    if x is None or _is_nan_like(x):
        return []

    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []

        if x.startswith("[") and x.endswith("]"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except Exception:
                pass

        return [x]

    return []


def infer_num_subtypes(res: Mapping[str, Any], default: int = 2) -> int:
    found = set()

    for key in res.keys():
        if "_s" not in key:
            continue
        try:
            suffix = key.split("_s")[-1]
            digits = ""
            for ch in suffix:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                found.add(int(digits))
        except Exception:
            continue

    return max(found) if found else default


def subtype_name(s: int, subtype_levels: List[str]) -> str:
    if 1 <= s <= len(subtype_levels):
        return str(subtype_levels[s - 1])
    return f"s{s}"


def supported_effect(ppd_val: Any, rope_val: Any, th: InterpretThresholds) -> bool:
    if _is_finite_number(ppd_val) and _is_finite_number(rope_val):
        return (ppd_val >= th.ppd_sig) and (rope_val <= th.rope_low)
    if _is_finite_number(ppd_val):
        return ppd_val >= th.ppd_sig
    return False


def ci_from_transition(dev_val: Any, scaling_val: Any, min_scaling_abs: float) -> float:
    if _is_finite_number(dev_val) and _is_finite_number(scaling_val) and abs(scaling_val) > min_scaling_abs:
        return dev_val / abs(scaling_val)
    return np.nan


###---------------- Baseline DE ----------------###

def interpret_baseline_de(
    res: Mapping[str, Any],
    th: InterpretThresholds,
) -> Dict[str, Any]:
    ppd_tumor = _get(res, "ppd_tumor", np.nan)
    p_rope_tumor = _get(res, "p_rope_tumor", np.nan)

    if _is_finite_number(ppd_tumor) and _is_finite_number(p_rope_tumor):
        if ppd_tumor >= th.ppd_sig and p_rope_tumor <= th.rope_low:
            de_status = "DE"
        elif p_rope_tumor >= th.rope_high:
            de_status = "DE-null"
        else:
            de_status = "DE-uncertain"
    else:
        de_status = "DE-uncertain"

    return {
        "de_status": de_status,
        "ppd_tumor": ppd_tumor,
        "p_rope_tumor": p_rope_tumor,
        "tumor0_lfc_median": _get(res, "tumor0_lfc_median", np.nan),
        "tumor0_lfc_q025": _get(res, "tumor0_lfc_q025", np.nan),
        "tumor0_lfc_q975": _get(res, "tumor0_lfc_q975", np.nan),
    }


###---------------- Rewiring ----------------###

def interpret_rewiring(
    res: Mapping[str, Any],
    th: InterpretThresholds,
) -> Dict[str, Any]:
    ppd_scaling = _get(res, "ppd_scaling", np.nan)
    ppd_dev = _get(res, "ppd_dev", np.nan)

    p_rope_scaling = _get(res, "p_rope_scaling", np.nan)
    p_rope_dev = _get(res, "p_rope_dev", np.nan)

    scaling_rewired = supported_effect(ppd_scaling, p_rope_scaling, th)
    dev_rewired = supported_effect(ppd_dev, p_rope_dev, th)

    if scaling_rewired and dev_rewired:
        rewiring_status = "rewired:scaling+deviation"
    elif scaling_rewired:
        rewiring_status = "rewired:scaling"
    elif dev_rewired:
        rewiring_status = "rewired:deviation"
    else:
        rewiring_status = "not_rewired"

    return {
        "rewiring_status": rewiring_status,
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
    }


###--------------Subtype pattern summary---------------###

def summarize_subtype_patterns(
    transition_groups: List[str],
    th: InterpretThresholds,
) -> Dict[str, Any]:
    supported_groups = [g for g in transition_groups if g in {"dsg", "dcg", "hag"}]
    unique_supported = sorted(set(supported_groups))

    if len(supported_groups) >= th.min_supported_for_mixed and len(unique_supported) >= 2:
        return {
            "response_class": "Mixed",
            "response_reason": "conflicting_supported_transition_patterns",
        }

    if len(unique_supported) == 1:
        mapping = {"dsg": "DSG", "dcg": "DCG", "hag": "HAG"}
        return {
            "response_class": mapping[unique_supported[0]],
            "response_reason": f"dominant_{unique_supported[0]}",
        }

    if all(g == "null" for g in transition_groups):
        return {
            "response_class": "Null",
            "response_reason": "all_transitions_null",
        }

    return {
        "response_class": "UNC",
        "response_reason": "insufficient_or_inconsistent_support",
    }


###------------Per-subtype dosage interpretation-----------###

def interpret_subtype_dosage(
    res,
    s,
    subtype_levels,
    th,
):
    name = subtype_name(s, subtype_levels)

    # gain: 2 -> 3
    p_gain_pos = _get(res, f"p_fracCN_2to3_pos_s{s}", np.nan)
    p_gain_rope = _get(res, f"p_rope_fracCN_2to3_s{s}", np.nan)
    frac23 = _get(res, f"fracCN_2to3_s{s}_median", np.nan)

    # amplification: 2 -> 4
    p_amp_pos = _get(res, f"p_fracCN_2to4_pos_s{s}", np.nan)
    p_amp_rope = _get(res, f"p_rope_fracCN_2to4_s{s}", np.nan)
    frac24 = _get(res, f"fracCN_2to4_s{s}_median", np.nan)

    # loss: 2 -> 1
    p_loss_neg = _get(res, f"p_fracCN_2to1_neg_s{s}", np.nan)
    p_loss_rope = _get(res, f"p_rope_fracCN_2to1_s{s}", np.nan)
    frac21 = _get(res, f"fracCN_2to1_s{s}_median", np.nan)

    # subtype-level coefficients
    b_scaling = _get(res, f"b_scaling_s{s}_median", np.nan)
    b_dev = _get(res, f"b_deviation_s{s}_median", np.nan)
    p_dev_small = _get(res, f"p_rope_bdev_s{s}", np.nan)

    # transition-level scaling / deviation summaries
    lp_scaling_21 = _get(res, f"lp_scaling_2to1_s{s}_median", np.nan)
    lp_scaling_23 = _get(res, f"lp_scaling_2to3_s{s}_median", np.nan)
    lp_scaling_24 = _get(res, f"lp_scaling_2to4_s{s}_median", np.nan)

    lp_dev_21 = _get(res, f"lp_dev_2to1_s{s}_median", np.nan)
    lp_dev_23 = _get(res, f"lp_dev_2to3_s{s}_median", np.nan)
    lp_dev_24 = _get(res, f"lp_dev_2to4_s{s}_median", np.nan)

    # support / null
    gain_supported = _is_finite_number(p_gain_pos) and (p_gain_pos >= th.dose_prob_sens)
    amp_supported = _is_finite_number(p_amp_pos) and (p_amp_pos >= th.dose_prob_sens)
    loss_supported = _is_finite_number(p_loss_neg) and (p_loss_neg >= th.dose_prob_sens)

    gain_null = _is_finite_number(p_gain_rope) and (p_gain_rope >= th.dose_prob_ins)
    amp_null = _is_finite_number(p_amp_rope) and (p_amp_rope >= th.dose_prob_ins)
    loss_null = _is_finite_number(p_loss_rope) and (p_loss_rope >= th.dose_prob_ins)

    # cancellation index
    ci23 = _get(res, f"cancel_index_2to3_s{s}_median", np.nan)
    ci24 = _get(res, f"cancel_index_2to4_s{s}_median", np.nan)
    ci21 = _get(res, f"cancel_index_2to1_s{s}_median", np.nan)

    def _ci_fallback(ci_val, lp_dev, lp_scaling):
        if _is_finite_number(ci_val):
            return ci_val
        if (
            _is_finite_number(lp_dev)
            and _is_finite_number(lp_scaling)
            and abs(lp_scaling) > th.min_scaling_abs
        ):
            return lp_dev / abs(lp_scaling)
        return np.nan

    ci23 = _ci_fallback(ci23, lp_dev_23, lp_scaling_23)
    ci24 = _ci_fallback(ci24, lp_dev_24, lp_scaling_24)
    ci21 = _ci_fallback(ci21, lp_dev_21, lp_scaling_21)

    scaling_stable_23 = _is_finite_number(lp_scaling_23) and (abs(lp_scaling_23) > th.min_scaling_abs)
    scaling_stable_24 = _is_finite_number(lp_scaling_24) and (abs(lp_scaling_24) > th.min_scaling_abs)
    scaling_stable_21 = _is_finite_number(lp_scaling_21) and (abs(lp_scaling_21) > th.min_scaling_abs)

    scaling_stable_any = (
        scaling_stable_21
        or scaling_stable_23
        or scaling_stable_24
        or (_is_finite_number(b_scaling) and abs(b_scaling) > th.min_scaling_abs)
    )

    # deviation practically small
    if _is_finite_number(p_dev_small):
        small_dev = p_dev_small >= th.dev_small_prob
        small_dev_method = "posterior_prob"
    elif _is_finite_number(b_dev):
        small_dev = abs(b_dev) <= th.dev_abs_fallback
        small_dev_method = "median_fallback"
    else:
        small_dev = False
        small_dev_method = "unavailable"

    # transition classifier
    def classify_transition(ci, supported, is_null, transition, scaling_stable, small_dev, th, frac_median):
        if is_null:
            return "null"
    
        # Hybrid support (posterior + effect size)
        if not supported:
            if _is_finite_number(frac_median):
                if transition in ("2to3", "2to4"):
                    if abs(frac_median) > th.frac_small_gain:
                        supported = True
                    else:
                        return "weak"
                elif transition == "2to1":
                    if abs(frac_median) > th.frac_small_loss:
                        supported = True
                    else:
                        return "weak"
                else:
                    return "weak"
            else:
                return "weak"

        if not scaling_stable:
            return "weak"

        if not _is_finite_number(ci):
            return "weak"

        # Gains / amplification
        if transition in ("2to3", "2to4"):
            if ci < -th.overcomp_threshold:
                return "overcomp"
            if ci < -th.cancel_threshold:
                return "buffered"
            if ci > th.hyper_threshold:
                return "hyper"
            if small_dev and abs(ci) < th.cancel_threshold:
                return "proportional"
            return "weak"

        # Loss
        if transition == "2to1":
            if ci > th.overcomp_threshold:
                return "overcomp"
            if ci > th.cancel_threshold:
                return "buffered"
            if ci < -th.hyper_threshold:
                return "hyper"
            if small_dev and abs(ci) < th.cancel_threshold:
                return "proportional"
            return "weak"

        return "weak"

    p21 = classify_transition(
        ci=ci21,
        supported=loss_supported,
        is_null=loss_null,
        transition="2to1",
        scaling_stable=scaling_stable_21 or scaling_stable_any,
        small_dev=small_dev,
        frac_median=frac21,
        th=th,
    )
    p23 = classify_transition(
        ci=ci23,
        supported=gain_supported,
        is_null=gain_null,
        transition="2to3",
        scaling_stable=scaling_stable_23 or scaling_stable_any,
        small_dev=small_dev,
        frac_median=frac23,
        th=th,
    )
    p24 = classify_transition(
        ci=ci24,
        supported=amp_supported,
        is_null=amp_null,
        transition="2to4",
        scaling_stable=scaling_stable_24 or scaling_stable_any,
        small_dev=small_dev,
        frac_median=frac24,
        th=th,
    )

    patterns = [p21, p23, p24]

    def group(pattern):
        if pattern == "proportional":
            return "dsg"
        if pattern in ("buffered", "overcomp"):
            return "dcg"
        if pattern == "hyper":
            return "hag"
        if pattern == "null":
            return "null"
        return "other"

    groups = [group(p) for p in patterns]
    summary = summarize_subtype_patterns(groups, th)

    cls = summary["response_class"]
    cls_reason = summary["response_reason"]

    # DIG-like subclass inside UNC
    frac21_small = _is_finite_number(frac21) and (abs(frac21) <= th.frac_small_loss)
    frac23_small = _is_finite_number(frac23) and (abs(frac23) <= th.frac_small_gain)
    frac24_small = _is_finite_number(frac24) and (abs(frac24) <= th.frac_small_amp)

    no_supported = not (gain_supported or loss_supported or amp_supported)
    dig_like = no_supported and frac21_small and frac23_small and frac24_small

    # low-CN variation subclass inside UNC
    n_aneup = _get(res, "n_aneup", np.nan)
    low_cn_variation = (
        _is_finite_number(n_aneup)
        and n_aneup < th.low_cn_aneup_threshold
    )

    if cls == "UNC":
        if dig_like:
            cls = "DIG-like"
            cls_reason = "weak_effects_small_magnitude"
        elif low_cn_variation:
            cls = "UNC-lowCN"
            cls_reason = "insufficient_cn_variation"
        else:
            cls = "UNC-ambiguous"
            cls_reason = "ambiguous_signal"

    # convenience summaries
    dc_gain = p23 in ("buffered", "overcomp") or p24 in ("buffered", "overcomp")
    dc_loss = p21 in ("buffered", "overcomp")

    if dc_gain and dc_loss:
        dc_type = "DC-bi"
    elif dc_gain:
        dc_type = "DC-gain"
    elif dc_loss:
        dc_type = "DC-loss"
    else:
        dc_type = "none"

    return {
        f"response_class_{name}": cls,
        f"response_reason_{name}": cls_reason,

        f"transition_2to1_{name}": p21,
        f"transition_2to3_{name}": p23,
        f"transition_2to4_{name}": p24,

        f"ci_2to1_{name}": ci21,
        f"ci_2to3_{name}": ci23,
        f"ci_2to4_{name}": ci24,

        f"gain_supported_{name}": gain_supported,
        f"loss_supported_{name}": loss_supported,
        f"amp_supported_{name}": amp_supported,

        f"gain_null_{name}": gain_null,
        f"loss_null_{name}": loss_null,
        f"amp_null_{name}": amp_null,

        f"small_dev_{name}": small_dev,
        f"small_dev_method_{name}": small_dev_method,

        f"dc_gain_{name}": dc_gain,
        f"dc_loss_{name}": dc_loss,
        f"dc_type_{name}": dc_type,

        f"dig_like_{name}": dig_like,
        f"low_cn_variation_{name}": low_cn_variation,

        f"fracCN_2to1_median_{name}": frac21,
        f"fracCN_2to3_median_{name}": frac23,
        f"fracCN_2to4_median_{name}": frac24,

        f"b_scaling_median_{name}": b_scaling,
        f"b_dev_median_{name}": b_dev,
        f"p_dev_small_{name}": p_dev_small,
    }


###-------------- Main row interpreter --------------###

def interpret_gene_result(
    res: Mapping[str, Any],
    th: InterpretThresholds = InterpretThresholds(),
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "gene": _get(res, "gene"),
        "status": _get(res, "status"),
        "fit_flag": _get(res, "fit_flag", "ok"),
        "success": _get(res, "success", np.nan),
        "N": _get(res, "N"),
        "n_aneup": _get(res, "n_aneup"),
        "cna": _get(res, "cna", np.nan),
    }

    out.update(interpret_baseline_de(res, th))
    out.update(interpret_rewiring(res, th))

    subtype_levels = normalize_subtype_levels(_get(res, "subtype_levels", None))
    S = len(subtype_levels)
    if S == 0:
        S = infer_num_subtypes(res, default=2)

    out["S"] = S
    out["subtype_levels_str"] = "|".join(subtype_levels) if subtype_levels else ""

    for s in range(1, S + 1):
        out.update(interpret_subtype_dosage(res, s, subtype_levels, th))

    per_sub = []
    for s in range(1, S + 1):
        nm = subtype_name(s, subtype_levels)
        per_sub.append(f"{nm}:{out.get(f'response_class_{nm}', 'NA')}")

    out["summary_label"] = (
        f"{out['de_status']} | {out['rewiring_status']} | " + ",".join(per_sub)
    )

    return out


###---------------- DataFrame wrapper ----------------###

def interpret_results_dataframe(
    df: pd.DataFrame,
    th: InterpretThresholds = InterpretThresholds(),
    keep_original: bool = False,
    drop_duplicate_interpreted_keys: bool = True,
) -> pd.DataFrame:
    records = df.to_dict(orient="records")
    interpreted = pd.DataFrame.from_records(
        [interpret_gene_result(rec, th=th) for rec in records]
    )

    if not keep_original:
        return interpreted

    if drop_duplicate_interpreted_keys:
        overlap = [c for c in interpreted.columns if c in df.columns]
        interpreted = interpreted.drop(columns=overlap, errors="ignore")

    return pd.concat(
        [df.reset_index(drop=True), interpreted.reset_index(drop=True)],
        axis=1,
    )


###---------------- Utilities ----------------###

def summarize_response_classes(interpreted_df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = [c for c in interpreted_df.columns if c.startswith("response_class_")]
    return {col: interpreted_df[col].value_counts(dropna=False) for col in cols}


def summarize_transition_patterns(interpreted_df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = [c for c in interpreted_df.columns if c.startswith("transition_")]
    return {col: interpreted_df[col].value_counts(dropna=False) for col in cols}


def genes_with_response_class(
    interpreted_df: pd.DataFrame,
    target_class: str,
    mode: str = "any",
) -> pd.DataFrame:
    cols = [c for c in interpreted_df.columns if c.startswith("response_class_")]
    if not cols:
        return interpreted_df.iloc[0:0].copy()

    mask = interpreted_df[cols].eq(target_class)
    keep = mask.all(axis=1) if mode == "all" else mask.any(axis=1)
    return interpreted_df.loc[keep].copy()