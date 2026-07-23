from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


"""
Posterior classification utilities for BDGDM.

This module converts posterior summaries produced by ``summarize_posterior``into interpretable gene-dosage classes:

- DSG: proportional dosage-sensitive response
- DCG: buffered or compensated response
- HYPER: hyperactive response
- Mixed: conflicting supported transition patterns
- DIG: dosage-insensitive response
- UNC: uncertain response

The module supports both a fitted ``BDGDMFit`` object and flat summary
dictionaries/data frames.
"""

__all__ = [
    "ClassificationThresholds",
    "InterpretThresholds",
    "classify_fit",
    "classify_gene",
    "classify_gene_result",
    "classify_fits",
    "classify_results_dataframe",
    "genes_with_response_class",
    "summarize_response_classes",
    "summarize_transition_patterns",
]

@dataclass(frozen=True)
class ClassificationThresholds:
    """Thresholds controlling BDGDM posterior classification."""

    # General Bayesian evidence thresholds.
    ppd_sig: float = 0.95
    rope_low: float = 0.05
    rope_high: float = 0.95

    # Directional support for CN transitions.
    dose_prob_sensitive: float = 0.95
    dose_prob_insensitive: float = 0.95

    # Evidence that the deviation coefficient is practically small.
    dev_small_prob: float = 0.75

    # Cancellation-index thresholds.
    cancel_threshold: float = 0.20
    overcomp_threshold: float = 1.00
    hyper_threshold: float = 0.50

    # Numerical stability.
    min_scaling_abs: float = 1e-3

    # Used only when p_rope_bdev is unavailable.
    dev_abs_fallback: float = 0.50

    # At least this many supported transitions are required for Mixed.
    min_supported_for_mixed: int = 2

    # Small fractional-effect thresholds used for DIG assessment.
    frac_small_loss: float = 0.15
    frac_small_gain: float = 0.15
    frac_small_amp: float = 0.25

    # Low-CN support flag.
    low_cn_aneup_threshold: int = 10

    # By default, a large median cannot replace weak posterior support.
    allow_median_support_fallback: bool = False

    def __post_init__(self) -> None:
        probability_fields = {
            "ppd_sig": self.ppd_sig,
            "rope_low": self.rope_low,
            "rope_high": self.rope_high,
            "dose_prob_sensitive": self.dose_prob_sensitive,
            "dose_prob_insensitive": self.dose_prob_insensitive,
            "dev_small_prob": self.dev_small_prob,
        }

        for name, value in probability_fields.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie between 0 and 1.")

        if self.rope_low > self.rope_high:
            raise ValueError("rope_low cannot exceed rope_high.")

        positive_fields = {
            "cancel_threshold": self.cancel_threshold,
            "overcomp_threshold": self.overcomp_threshold,
            "hyper_threshold": self.hyper_threshold,
            "min_scaling_abs": self.min_scaling_abs,
            "dev_abs_fallback": self.dev_abs_fallback,
            "frac_small_loss": self.frac_small_loss,
            "frac_small_gain": self.frac_small_gain,
            "frac_small_amp": self.frac_small_amp,
        }

        for name, value in positive_fields.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")

        if self.min_supported_for_mixed < 2:
            raise ValueError("min_supported_for_mixed must be at least 2.")

        if self.low_cn_aneup_threshold < 0:
            raise ValueError("low_cn_aneup_threshold cannot be negative.")


# Backward-compatible name used by the original script.
InterpretThresholds = ClassificationThresholds


def _get(
    result: Mapping[str, Any],
    key: str,
    default: Any = np.nan,
) -> Any:
    return result.get(key, default)


def _is_nan_like(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_finite_number(value: Any) -> bool:
    try:
        array = np.asarray(value)

        if array.ndim != 0:
            return False

        return bool(np.isfinite(float(array)))
    except (TypeError, ValueError):
        return False


def _safe_label(value: str) -> str:
    """Convert a subtype name to a stable output-column suffix."""
    label = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return label or "unknown"

def normalize_subtype_levels(value: Any) -> list[str]:
    """Normalize subtype levels stored as a list or serialized string."""
    if value is None or _is_nan_like(value):
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)

                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                pass

        if "|" in stripped:
            return [
                part.strip()
                for part in stripped.split("|")
                if part.strip()
            ]

        return [stripped]

    return []


def infer_num_subtypes(
    result: Mapping[str, Any],
    *,
    default: int = 1,
) -> int:
    """Infer S from summary keys such as b_scaling_s2_median."""
    found: set[int] = set()

    for key in result:
        for match in re.finditer(r"_s(\d+)(?:_|$)", str(key)):
            found.add(int(match.group(1)))

    return max(found) if found else default


def subtype_name(
    subtype_index: int,
    subtype_levels: Sequence[str],
) -> str:
    """Return the human-readable subtype name for a one-based index."""
    if 1 <= subtype_index <= len(subtype_levels):
        return str(subtype_levels[subtype_index - 1])

    return f"s{subtype_index}"


def _effect_supported(
    ppd_value: Any,
    rope_value: Any,
    thresholds: ClassificationThresholds,
) -> tuple[bool, str]:
    """Evaluate contrast support and record the evidence source."""
    ppd_available = _is_finite_number(ppd_value)
    rope_available = _is_finite_number(rope_value)

    if ppd_available and rope_available:
        supported = (
            float(ppd_value) >= thresholds.ppd_sig
            and float(rope_value) <= thresholds.rope_low
        )
        return supported, "ppd_and_rope"

    if ppd_available:
        supported = float(ppd_value) >= thresholds.ppd_sig
        return supported, "ppd_only"

    return False, "unavailable"


def _effect_null(
    rope_value: Any,
    thresholds: ClassificationThresholds,
) -> bool:
    return (
        _is_finite_number(rope_value)
        and float(rope_value) >= thresholds.rope_high
    )


def _transition_ci(
    result: Mapping[str, Any],
    subtype_index: int,
    transition: str,
    thresholds: ClassificationThresholds,
) -> float:
    """Read a cancellation index or derive it from transition components."""
    stored = _get(
        result,
        f"cancel_index_{transition}_s{subtype_index}_median",
        np.nan,
    )

    if _is_finite_number(stored):
        return float(stored)

    lp_deviation = _get(
        result,
        f"lp_dev_{transition}_s{subtype_index}_median",
        np.nan,
    )
    lp_scaling = _get(
        result,
        f"lp_scaling_{transition}_s{subtype_index}_median",
        np.nan,
    )

    if (
        _is_finite_number(lp_deviation)
        and _is_finite_number(lp_scaling)
        and abs(float(lp_scaling)) > thresholds.min_scaling_abs
    ):
        return float(lp_deviation) / abs(float(lp_scaling))

    return float("nan")


def _transition_support(
    *,
    transition: str,
    direction_probability: Any,
    rope_probability: Any,
    fractional_median: Any,
    thresholds: ClassificationThresholds,
) -> tuple[bool, bool, str]:
    """Return supported, null, and evidence method for one transition."""
    is_null = (
        _is_finite_number(rope_probability)
        and float(rope_probability)
        >= thresholds.dose_prob_insensitive
    )

    if (
        _is_finite_number(direction_probability)
        and float(direction_probability)
        >= thresholds.dose_prob_sensitive
    ):
        return True, is_null, "posterior_direction"

    if (
        thresholds.allow_median_support_fallback
        and _is_finite_number(fractional_median)
    ):
        median = float(fractional_median)

        if transition == "2to1":
            supported = median < -thresholds.frac_small_loss
        elif transition == "2to3":
            supported = median > thresholds.frac_small_gain
        elif transition == "2to4":
            supported = median > thresholds.frac_small_amp
        else:
            supported = False

        if supported:
            return True, is_null, "median_fallback"

    return False, is_null, "insufficient_support"


def _classify_transition(
    *,
    transition: str,
    cancellation_index: float,
    supported: bool,
    is_null: bool,
    scaling_stable: bool,
    small_deviation: bool,
    thresholds: ClassificationThresholds,
) -> str:
    """Classify one CN transition."""
    if is_null:
        return "null"

    if not supported:
        return "weak"

    if not scaling_stable or not _is_finite_number(cancellation_index):
        return "weak"

    ci = float(cancellation_index)

    if transition in {"2to3", "2to4"}:
        if ci < -thresholds.overcomp_threshold:
            return "overcompensated"
        if ci < -thresholds.cancel_threshold:
            return "buffered"
        if ci > thresholds.hyper_threshold:
            return "hyperactive"
        if abs(ci) <= thresholds.cancel_threshold:
            return "proportional"
        return "weak"

    if transition == "2to1":
        if ci > thresholds.overcomp_threshold:
            return "overcompensated"
        if ci > thresholds.cancel_threshold:
            return "buffered"
        if ci < -thresholds.hyper_threshold:
            return "hyperactive"
        if abs(ci) <= thresholds.cancel_threshold:
            return "proportional"
        return "weak"

    return "weak"


def interpret_baseline_de(
    result: Mapping[str, Any],
    thresholds: ClassificationThresholds,
    *,
    analysis_mode: str,
) -> dict[str, Any]:
    """Interpret the baseline subtype-expression contrast."""
    ppd_tumor = _get(result, "ppd_tumor", np.nan)
    p_rope_tumor = _get(result, "p_rope_tumor", np.nan)

    if analysis_mode != "subtype_comparison":
        status = "not_applicable"
    elif (
        _is_finite_number(ppd_tumor)
        and _is_finite_number(p_rope_tumor)
        and float(ppd_tumor) >= thresholds.ppd_sig
        and float(p_rope_tumor) <= thresholds.rope_low
    ):
        status = "DE"
    elif (
        _is_finite_number(p_rope_tumor)
        and float(p_rope_tumor) >= thresholds.rope_high
    ):
        status = "DE-null"
    else:
        status = "DE-uncertain"

    return {
        "de_status": status,
        "ppd_tumor": ppd_tumor,
        "p_rope_tumor": p_rope_tumor,
        "tumor0_lfc_median": _get(
            result,
            "tumor0_lfc_median",
            np.nan,
        ),
        "tumor0_lfc_q025": _get(
            result,
            "tumor0_lfc_q025",
            np.nan,
        ),
        "tumor0_lfc_q975": _get(
            result,
            "tumor0_lfc_q975",
            np.nan,
        ),
    }


def interpret_rewiring(
    result: Mapping[str, Any],
    thresholds: ClassificationThresholds,
    *,
    analysis_mode: str,
) -> dict[str, Any]:
    """Interpret subtype differences in dosage scaling and deviation."""
    ppd_scaling = _get(result, "ppd_scaling", np.nan)
    ppd_deviation = _get(result, "ppd_dev", np.nan)

    p_rope_scaling = _get(result, "p_rope_scaling", np.nan)
    p_rope_deviation = _get(result, "p_rope_dev", np.nan)

    if analysis_mode != "subtype_comparison":
        rewiring_status = "not_applicable"
        scaling_rewired = False
        deviation_rewired = False
        scaling_method = "not_applicable"
        deviation_method = "not_applicable"
    else:
        scaling_rewired, scaling_method = _effect_supported(
            ppd_scaling,
            p_rope_scaling,
            thresholds,
        )
        deviation_rewired, deviation_method = _effect_supported(
            ppd_deviation,
            p_rope_deviation,
            thresholds,
        )

        if scaling_rewired and deviation_rewired:
            rewiring_status = "rewired:scaling+deviation"
        elif scaling_rewired:
            rewiring_status = "rewired:scaling"
        elif deviation_rewired:
            rewiring_status = "rewired:deviation"
        else:
            scaling_null = _effect_null(
                p_rope_scaling,
                thresholds,
            )
            deviation_null = _effect_null(
                p_rope_deviation,
                thresholds,
            )

            if scaling_null and deviation_null:
                rewiring_status = "not_rewired"
            else:
                rewiring_status = "rewiring_uncertain"

    return {
        "rewiring_status": rewiring_status,
        "scaling_rewired": scaling_rewired,
        "deviation_rewired": deviation_rewired,
        "scaling_rewiring_evidence": scaling_method,
        "deviation_rewiring_evidence": deviation_method,
        "ppd_scaling": ppd_scaling,
        "ppd_dev": ppd_deviation,
        "p_rope_scaling": p_rope_scaling,
        "p_rope_dev": p_rope_deviation,
        "delta_scaling_median": _get(
            result,
            "delta_scaling_median",
            np.nan,
        ),
        "delta_scaling_q025": _get(
            result,
            "delta_scaling_q025",
            np.nan,
        ),
        "delta_scaling_q975": _get(
            result,
            "delta_scaling_q975",
            np.nan,
        ),
        "delta_dev_median": _get(
            result,
            "delta_dev_median",
            np.nan,
        ),
        "delta_dev_q025": _get(
            result,
            "delta_dev_q025",
            np.nan,
        ),
        "delta_dev_q975": _get(
            result,
            "delta_dev_q975",
            np.nan,
        ),
    }


def _summarize_transition_patterns(
    patterns: Sequence[str],
    *,
    fractional_medians: Sequence[Any],
    n_aneup: Any,
    thresholds: ClassificationThresholds,
) -> tuple[str, str, str | None]:
    """Convert transition-level patterns into one gene-level class."""
    group_map = {
        "proportional": "DSG",
        "buffered": "DCG",
        "overcompensated": "DCG",
        "hyperactive": "HYPER",
    }

    supported_groups = [
        group_map[pattern]
        for pattern in patterns
        if pattern in group_map
    ]
    unique_supported = sorted(set(supported_groups))

    if (
        len(supported_groups) >= thresholds.min_supported_for_mixed
        and len(unique_supported) >= 2
    ):
        return (
            "Mixed",
            "conflicting_supported_transition_patterns",
            None,
        )

    if len(unique_supported) == 1:
        return (
            unique_supported[0],
            f"supported_{unique_supported[0].lower()}_pattern",
            None,
        )

    if patterns and all(pattern == "null" for pattern in patterns):
        return "DIG", "all_transitions_practically_null", None

    small_thresholds = [
        thresholds.frac_small_loss,
        thresholds.frac_small_gain,
        thresholds.frac_small_amp,
    ]

    small_effects = all(
        _is_finite_number(value)
        and abs(float(value)) <= limit
        for value, limit in zip(
            fractional_medians,
            small_thresholds,
        )
    )

    if not supported_groups and small_effects:
        return "DIG", "all_transition_effects_small", "median_based"

    low_cn = (
        _is_finite_number(n_aneup)
        and int(float(n_aneup))
        < thresholds.low_cn_aneup_threshold
    )

    if low_cn:
        return "UNC", "insufficient_cn_variation", "low_CN"

    return "UNC", "insufficient_or_inconsistent_support", "ambiguous"


def interpret_subtype_dosage(
    result: Mapping[str, Any],
    subtype_index: int,
    subtype_levels: Sequence[str],
    thresholds: ClassificationThresholds,
) -> dict[str, Any]:
    """Interpret CN-response patterns for one subtype."""
    human_name = subtype_name(
        subtype_index,
        subtype_levels,
    )
    safe_name = _safe_label(human_name)
    canonical_suffix = f"s{subtype_index}"

    transition_inputs = {
        "2to1": {
            "direction_probability": _get(
                result,
                f"p_fracCN_2to1_neg_s{subtype_index}",
                np.nan,
            ),
            "rope_probability": _get(
                result,
                f"p_rope_fracCN_2to1_s{subtype_index}",
                np.nan,
            ),
            "fractional_median": _get(
                result,
                f"fracCN_2to1_s{subtype_index}_median",
                np.nan,
            ),
        },
        "2to3": {
            "direction_probability": _get(
                result,
                f"p_fracCN_2to3_pos_s{subtype_index}",
                np.nan,
            ),
            "rope_probability": _get(
                result,
                f"p_rope_fracCN_2to3_s{subtype_index}",
                np.nan,
            ),
            "fractional_median": _get(
                result,
                f"fracCN_2to3_s{subtype_index}_median",
                np.nan,
            ),
        },
        "2to4": {
            "direction_probability": _get(
                result,
                f"p_fracCN_2to4_pos_s{subtype_index}",
                np.nan,
            ),
            "rope_probability": _get(
                result,
                f"p_rope_fracCN_2to4_s{subtype_index}",
                np.nan,
            ),
            "fractional_median": _get(
                result,
                f"fracCN_2to4_s{subtype_index}_median",
                np.nan,
            ),
        },
    }

    b_scaling = _get(
        result,
        f"b_scaling_s{subtype_index}_median",
        np.nan,
    )
    b_deviation = _get(
        result,
        f"b_deviation_s{subtype_index}_median",
        np.nan,
    )
    p_deviation_small = _get(
        result,
        f"p_rope_bdev_s{subtype_index}",
        np.nan,
    )

    if _is_finite_number(p_deviation_small):
        small_deviation = (
            float(p_deviation_small)
            >= thresholds.dev_small_prob
        )
        small_deviation_method = "posterior_probability"
    elif _is_finite_number(b_deviation):
        small_deviation = (
            abs(float(b_deviation))
            <= thresholds.dev_abs_fallback
        )
        small_deviation_method = "median_fallback"
    else:
        small_deviation = False
        small_deviation_method = "unavailable"

    transition_results: dict[str, dict[str, Any]] = {}

    for transition, inputs in transition_inputs.items():
        supported, is_null, support_method = _transition_support(
            transition=transition,
            direction_probability=inputs[
                "direction_probability"
            ],
            rope_probability=inputs["rope_probability"],
            fractional_median=inputs["fractional_median"],
            thresholds=thresholds,
        )

        lp_scaling = _get(
            result,
            (
                f"lp_scaling_{transition}_"
                f"s{subtype_index}_median"
            ),
            np.nan,
        )

        scaling_stable = (
            (
                _is_finite_number(lp_scaling)
                and abs(float(lp_scaling))
                > thresholds.min_scaling_abs
            )
            or (
                not _is_finite_number(lp_scaling)
                and _is_finite_number(b_scaling)
                and abs(float(b_scaling))
                > thresholds.min_scaling_abs
            )
        )

        cancellation_index = _transition_ci(
            result,
            subtype_index,
            transition,
            thresholds,
        )

        pattern = _classify_transition(
            transition=transition,
            cancellation_index=cancellation_index,
            supported=supported,
            is_null=is_null,
            scaling_stable=scaling_stable,
            small_deviation=small_deviation,
            thresholds=thresholds,
        )

        transition_results[transition] = {
            "pattern": pattern,
            "supported": supported,
            "null": is_null,
            "support_method": support_method,
            "cancellation_index": cancellation_index,
            "fractional_median": inputs[
                "fractional_median"
            ],
            "direction_probability": inputs[
                "direction_probability"
            ],
            "rope_probability": inputs[
                "rope_probability"
            ],
            "scaling_stable": scaling_stable,
        }

    ordered_transitions = ["2to1", "2to3", "2to4"]
    patterns = [
        transition_results[transition]["pattern"]
        for transition in ordered_transitions
    ]
    fractional_medians = [
        transition_results[transition][
            "fractional_median"
        ]
        for transition in ordered_transitions
    ]

    response_class, response_reason, response_subclass = (
        _summarize_transition_patterns(
            patterns,
            fractional_medians=fractional_medians,
            n_aneup=_get(result, "n_aneup", np.nan),
            thresholds=thresholds,
        )
    )

    dc_gain = any(
        transition_results[transition]["pattern"]
        in {"buffered", "overcompensated"}
        for transition in ["2to3", "2to4"]
    )
    dc_loss = (
        transition_results["2to1"]["pattern"]
        in {"buffered", "overcompensated"}
    )

    if dc_gain and dc_loss:
        dc_type = "DC-bi"
    elif dc_gain:
        dc_type = "DC-gain"
    elif dc_loss:
        dc_type = "DC-loss"
    else:
        dc_type = "none"

    output: dict[str, Any] = {
        f"subtype_{canonical_suffix}": human_name,
        f"response_class_{canonical_suffix}": response_class,
        f"response_reason_{canonical_suffix}": response_reason,
        f"response_subclass_{canonical_suffix}": response_subclass,
        f"small_deviation_{canonical_suffix}": small_deviation,
        (
            f"small_deviation_method_{canonical_suffix}"
        ): small_deviation_method,
        f"b_scaling_median_{canonical_suffix}": b_scaling,
        f"b_deviation_median_{canonical_suffix}": b_deviation,
        f"p_deviation_small_{canonical_suffix}": p_deviation_small,
        f"dc_gain_{canonical_suffix}": dc_gain,
        f"dc_loss_{canonical_suffix}": dc_loss,
        f"dc_type_{canonical_suffix}": dc_type,
    }

    for transition in ordered_transitions:
        values = transition_results[transition]

        output.update(
            {
                (
                    f"transition_{transition}_"
                    f"{canonical_suffix}"
                ): values["pattern"],
                (
                    f"transition_supported_{transition}_"
                    f"{canonical_suffix}"
                ): values["supported"],
                (
                    f"transition_null_{transition}_"
                    f"{canonical_suffix}"
                ): values["null"],
                (
                    f"transition_support_method_{transition}_"
                    f"{canonical_suffix}"
                ): values["support_method"],
                (
                    f"cancel_index_{transition}_"
                    f"{canonical_suffix}"
                ): values["cancellation_index"],
                (
                    f"fracCN_{transition}_median_"
                    f"{canonical_suffix}"
                ): values["fractional_median"],
                (
                    f"direction_probability_{transition}_"
                    f"{canonical_suffix}"
                ): values["direction_probability"],
                (
                    f"rope_probability_{transition}_"
                    f"{canonical_suffix}"
                ): values["rope_probability"],
            }
        )

    # Human-readable aliases retained for compatibility with the old script.
    if safe_name != canonical_suffix:
        output.update(
            {
                f"response_class_{safe_name}": response_class,
                f"response_reason_{safe_name}": response_reason,
                f"response_subclass_{safe_name}": response_subclass,
                f"transition_2to1_{safe_name}": (
                    transition_results["2to1"]["pattern"]
                ),
                f"transition_2to3_{safe_name}": (
                    transition_results["2to3"]["pattern"]
                ),
                f"transition_2to4_{safe_name}": (
                    transition_results["2to4"]["pattern"]
                ),
            }
        )

    return output


def _fit_to_result(fit: Any) -> dict[str, Any]:
    """Merge BDGDMFit posterior, metadata, and diagnostics."""
    posterior = getattr(fit, "posterior", None)

    if not isinstance(posterior, Mapping):
        raise TypeError(
            "fit must expose a mapping-valued 'posterior' attribute."
        )

    result = dict(posterior)

    metadata = getattr(fit, "metadata", {})

    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            result.setdefault(key, value)

    diagnostics = getattr(fit, "diagnostics", {})

    if isinstance(diagnostics, Mapping):
        result.setdefault(
            "fit_flag",
            (
                "warn"
                if diagnostics.get("converged") is False
                else "ok"
            ),
        )
        result.setdefault(
            "diagnostic_status",
            diagnostics.get("diagnostic_status"),
        )

    result.setdefault("gene", getattr(fit, "gene", None))
    result.setdefault(
        "analysis_mode",
        getattr(fit, "analysis_mode", None),
    )

    if "subtype_levels" not in result:
        result["subtype_levels"] = result.get(
            "subtype_order",
            [],
        )

    return result


def classify_gene_result(
    result: Mapping[str, Any],
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """Classify one flat BDGDM posterior-summary mapping."""
    if thresholds is None:
        thresholds = ClassificationThresholds()

    analysis_mode = str(
        _get(result, "analysis_mode", "")
    ).strip()

    subtype_levels = normalize_subtype_levels(
        _get(
            result,
            "subtype_levels",
            _get(result, "subtype_order", None),
        )
    )

    if not analysis_mode:
        inferred_s = infer_num_subtypes(
            result,
            default=max(1, len(subtype_levels)),
        )
        analysis_mode = (
            "single_group"
            if inferred_s == 1
            else "subtype_comparison"
        )

    default_s = 1 if analysis_mode == "single_group" else 2

    number_of_subtypes = len(subtype_levels)

    if number_of_subtypes == 0:
        number_of_subtypes = infer_num_subtypes(
            result,
            default=default_s,
        )

    output: dict[str, Any] = {
        "gene": _get(result, "gene", None),
        "status": _get(result, "status", "ok"),
        "fit_flag": _get(result, "fit_flag", "ok"),
        "analysis_mode": analysis_mode,
        "N": _get(result, "N", np.nan),
        "n_aneup": _get(result, "n_aneup", np.nan),
        "cna": _get(result, "cna", np.nan),
        "S": number_of_subtypes,
        "subtype_levels": subtype_levels,
        "subtype_levels_str": "|".join(subtype_levels),
    }

    output.update(
        interpret_baseline_de(
            result,
            thresholds,
            analysis_mode=analysis_mode,
        )
    )
    output.update(
        interpret_rewiring(
            result,
            thresholds,
            analysis_mode=analysis_mode,
        )
    )

    subtype_labels: list[str] = []

    for subtype_index in range(1, number_of_subtypes + 1):
        subtype_output = interpret_subtype_dosage(
            result,
            subtype_index,
            subtype_levels,
            thresholds,
        )
        output.update(subtype_output)

        name = subtype_name(
            subtype_index,
            subtype_levels,
        )
        subtype_class = subtype_output.get(
            f"response_class_s{subtype_index}",
            "NA",
        )
        subtype_labels.append(
            f"{name}:{subtype_class}"
        )

    output["summary_label"] = (
        f"{output['de_status']} | "
        f"{output['rewiring_status']} | "
        + ",".join(subtype_labels)
    )

    return output


def classify_fit(
    fit: Any,
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """Classify one BDGDMFit object."""
    return classify_gene_result(
        _fit_to_result(fit),
        thresholds=thresholds,
    )


def classify_gene(
    fit_or_result: Any,
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """Classify either a BDGDMFit object or a summary mapping."""
    if isinstance(fit_or_result, Mapping):
        return classify_gene_result(
            fit_or_result,
            thresholds=thresholds,
        )

    return classify_fit(
        fit_or_result,
        thresholds=thresholds,
    )


def classify_fits(
    fits: Mapping[str, Any],
    thresholds: ClassificationThresholds | None = None,
) -> pd.DataFrame:
    """Classify a dictionary such as ``fits_rep``."""
    records: list[dict[str, Any]] = []

    for gene, fit in fits.items():
        classified = classify_fit(
            fit,
            thresholds=thresholds,
        )

        if not classified.get("gene"):
            classified["gene"] = str(gene)

        records.append(classified)

    return pd.DataFrame.from_records(records)


def classify_results_dataframe(
    dataframe: pd.DataFrame,
    thresholds: ClassificationThresholds | None = None,
    *,
    keep_original: bool = False,
    drop_duplicate_classified_keys: bool = True,
) -> pd.DataFrame:
    """Classify every row of a posterior-summary data frame."""
    records = dataframe.to_dict(orient="records")

    classified = pd.DataFrame.from_records(
        [
            classify_gene_result(
                record,
                thresholds=thresholds,
            )
            for record in records
        ]
    )

    if not keep_original:
        return classified

    original = dataframe.reset_index(drop=True)

    if drop_duplicate_classified_keys:
        overlap = [
            column
            for column in classified.columns
            if column in original.columns
        ]
        classified = classified.drop(
            columns=overlap,
            errors="ignore",
        )

    return pd.concat(
        [
            original,
            classified.reset_index(drop=True),
        ],
        axis=1,
    )


def summarize_response_classes(
    classified_dataframe: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Count response classes for canonical subtype columns."""
    columns = [
        column
        for column in classified_dataframe.columns
        if re.fullmatch(r"response_class_s\d+", column)
    ]

    return {
        column: classified_dataframe[column].value_counts(
            dropna=False
        )
        for column in columns
    }


def summarize_transition_patterns(
    classified_dataframe: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Count transition patterns for canonical subtype columns."""
    columns = [
        column
        for column in classified_dataframe.columns
        if re.fullmatch(
            r"transition_2to[134]_s\d+",
            column,
        )
    ]

    return {
        column: classified_dataframe[column].value_counts(
            dropna=False
        )
        for column in columns
    }


def genes_with_response_class(
    classified_dataframe: pd.DataFrame,
    target_class: str,
    *,
    mode: str = "any",
) -> pd.DataFrame:
    """Return genes assigned to a response class in any or all subtypes."""
    if mode not in {"any", "all"}:
        raise ValueError("mode must be 'any' or 'all'.")

    columns = [
        column
        for column in classified_dataframe.columns
        if re.fullmatch(r"response_class_s\d+", column)
    ]

    if not columns:
        return classified_dataframe.iloc[0:0].copy()

    matches = classified_dataframe[columns].eq(
        target_class
    )

    keep = (
        matches.all(axis=1)
        if mode == "all"
        else matches.any(axis=1)
    )

    return classified_dataframe.loc[keep].copy()




