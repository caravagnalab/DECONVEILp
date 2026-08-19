from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


"""
Posterior classification for BDGDM.

This module converts posterior summaries produced by ``summarize_posterior``into interpretable gene-dosage classes:

- DSG: proportional dosage-sensitive response
- DCG: buffered or compensated response
- HYPER: hyperactive response
- Mixed: conflicting supported transition patterns
- DIG: dosage-insensitive response
- UNC: uncertain response

The module supports both a fitted ``BDGDMFit`` object and flat summary dictionaries/data frames.

- Failed or non-converged fits are not assigned DSG/DCG/DIG/HYPER/Mixed/UNC.
  They remain in tabular outputs with ``classification_eligible=False`` and a missing response-class value, so they can be audited without entering
  biological class proportions.
- Transition directional support requires an available PPD together with the expected/reverse directional probability. When a transition ROPE
  probability is available, it must also pass the practical-effect threshold.
- Baseline differential-expression and subtype-rewiring calls require both PPD and ROPE evidence.
- Transition-level biological calls can be gated by empirical CN-state support.
  In strict mode, both CN=2 and the target CN state must meet their sample-count
  thresholds; unsupported transitions remain available as model predictions but
  cannot create DSG/DCG/HYPER/Mixed/DIG calls.
"""

__all__ = [
    "ClassificationThresholds",
    "InterpretThresholds",
    "classify_fit",
    "classify_gene",
    "classify_gene_result",
    "classify_fits",
    "classify_results_dataframe",
    "get_response_class",
    "get_subtype_classification",
    "get_transition_classification",
    "get_rewiring_summary",
    "classification_to_subtype_df",
    "classification_to_transition_df",
    "classification_to_rewiring_df",
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

    # Transition-response thresholds relative to the canonical proportional log-response.
    # response_ratio = observed_log_effect / canonical_log_effect
    # cancel_threshold=0.50: response_ratio < 0.50 is buffered.
    # overcomp_threshold=1.00: response_ratio < 0 indicates sign reversal/overcompensation.
    # hyper_threshold=0.30: response_ratio > 1.30 is hyper-responsive.
    #
    # The stored cancellation index is retained as a descriptive output, but it no longer defines HYPER.
    cancel_threshold: float = 0.50
    overcomp_threshold: float = 1.00
    hyper_threshold: float = 0.30

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

    # Overall low-CN support flag retained for UNC-lowCN.
    low_cn_aneup_threshold: int = 10

    # Transition-specific empirical CN support.  A CN transition is allowed
    # to determine DSG/DCG/HYPER/Mixed/DIG only when both the diploid
    # reference state (CN=2) and the target state are sufficiently represented.
    # Expected result keys are, for example, n_cn2_s1 and n_cn3_s1 (with
    # global n_cn2/n_cn3 accepted as a single-group fallback).
    min_reference_cn2_support: int = 10
    min_transition_cn_support: int = 10
    require_empirical_transition_support: bool = True

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

        if self.min_reference_cn2_support < 0:
            raise ValueError("min_reference_cn2_support cannot be negative.")

        if self.min_transition_cn_support < 0:
            raise ValueError("min_transition_cn_support cannot be negative.")


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
    """Evaluate subtype-contrast support using both PPD and ROPE."""
    ppd_available = _is_finite_number(ppd_value)
    rope_available = _is_finite_number(rope_value)

    if not ppd_available or not rope_available:
        return False, "incomplete_ppd_rope"

    supported = (
        float(ppd_value) >= thresholds.ppd_sig
        and float(rope_value) <= thresholds.rope_low
    )
    return supported, "ppd_and_rope"


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



def _canonical_log_effect(transition: str) -> float:
    """Return the canonical proportional log-expression change."""
    canonical = {
        "2to1": float(np.log(1.0 / 2.0)),
        "2to3": float(np.log(3.0 / 2.0)),
        "2to4": float(np.log(4.0 / 2.0)),
        "2to5": float(np.log(5.0 / 2.0)),
    }

    if transition not in canonical:
        raise ValueError(
            "transition must be one of '2to1', '2to3', '2to4' or '2to5'."
        )

    return canonical[transition]


def _transition_log_effect(
    *,
    log_effect_median: Any,
    fractional_median: Any,
) -> float:
    """Read the log effect or derive it from the fractional change."""
    if _is_finite_number(log_effect_median):
        return float(log_effect_median)

    if _is_finite_number(fractional_median):
        fractional = float(fractional_median)

        if fractional > -1.0:
            return float(np.log1p(fractional))

    return float("nan")


def _transition_response_ratio(
    *,
    transition: str,
    log_effect_median: Any,
    fractional_median: Any,
) -> float:
    """
    Compare the fitted transition effect with proportional dosage.

    A value of 1 is proportional, values between 0 and 1 are attenuated,
    values above 1 are stronger than proportional, and values below zero
    reverse the canonical direction.
    """
    observed = _transition_log_effect(
        log_effect_median=log_effect_median,
        fractional_median=fractional_median,
    )
    canonical = _canonical_log_effect(transition)

    if not _is_finite_number(observed):
        return float("nan")

    return float(observed / canonical)


def _hyper_supported(
    *,
    transition: str,
    log_effect_q025: Any,
    log_effect_q975: Any,
    response_ratio: Any,
    thresholds: ClassificationThresholds,
) -> tuple[bool, str]:
    """
    Test whether the complete credible interval is stronger than proportional.

    ``hyper_threshold=0.30`` means that the log-response must be more than
    1.3 times the canonical proportional log-response.
    """
    canonical = _canonical_log_effect(transition)
    boundary = canonical * (1.0 + thresholds.hyper_threshold)

    if (
        _is_finite_number(log_effect_q025)
        and _is_finite_number(log_effect_q975)
    ):
        if canonical > 0:
            supported = float(log_effect_q025) > boundary
        else:
            supported = float(log_effect_q975) < boundary

        return supported, "credible_interval"

    if (
        thresholds.allow_median_support_fallback
        and _is_finite_number(response_ratio)
    ):
        return (
            float(response_ratio)
            > 1.0 + thresholds.hyper_threshold,
            "median_fallback",
        )

    return False, "unavailable"


def _transition_support(
    *,
    transition: str,
    ppd_value: Any,
    expected_direction_probability: Any,
    opposite_direction_probability: Any,
    rope_probability: Any,
    fractional_median: Any,
    thresholds: ClassificationThresholds,
) -> tuple[bool, bool, bool, str]:
    """
    Return expected-direction support, reverse-direction support, and null.

    When PPD and ROPE probabilities are available, they are included in the
    evidence rule rather than relying on a directional probability alone.
    """
    is_null = (
        _is_finite_number(rope_probability)
        and float(rope_probability)
        >= thresholds.dose_prob_insensitive
    )

    if is_null:
        return False, False, True, "posterior_rope_null"

    def direction_supported(probability: Any) -> bool:
        if (
            not _is_finite_number(probability)
            or float(probability)
            < thresholds.dose_prob_sensitive
        ):
            return False

        # PPD is mandatory for a supported biological direction.
        if (
            not _is_finite_number(ppd_value)
            or float(ppd_value) < thresholds.ppd_sig
        ):
            return False

        # When ROPE is available it must satisfy the practical-effect rule.
        # If ROPE is unavailable, retain the documented PPD + directional
        # probability fallback for legacy/saved posterior summaries.
        if (
            _is_finite_number(rope_probability)
            and float(rope_probability) > thresholds.rope_low
        ):
            return False

        return True

    expected_supported = direction_supported(
        expected_direction_probability
    )
    reverse_supported = direction_supported(
        opposite_direction_probability
    )

    if expected_supported:
        return True, False, False, "posterior_expected_direction"

    if reverse_supported:
        return False, True, False, "posterior_reverse_direction"

    if (
        thresholds.allow_median_support_fallback
        and _is_finite_number(fractional_median)
    ):
        median = float(fractional_median)

        if transition == "2to1":
            expected_supported = median < -thresholds.frac_small_loss
            reverse_supported = median > thresholds.frac_small_loss
        elif transition == "2to3":
            expected_supported = median > thresholds.frac_small_gain
            reverse_supported = median < -thresholds.frac_small_gain
        elif transition == "2to4":
            expected_supported = median > thresholds.frac_small_amp
            reverse_supported = median < -thresholds.frac_small_amp
        elif transition == "2to5":
            expected_supported = median > thresholds.frac_small_amp
            reverse_supported = median < -thresholds.frac_small_amp
        else:
            expected_supported = False
            reverse_supported = False

        if expected_supported:
            return True, False, False, "median_expected_fallback"

        if reverse_supported:
            return False, True, False, "median_reverse_fallback"

    return False, False, False, "insufficient_support"


def _classify_transition(
    *,
    response_ratio: float,
    expected_supported: bool,
    reverse_supported: bool,
    is_null: bool,
    hyper_supported: bool,
    thresholds: ClassificationThresholds,
) -> str:
    """
    Classify one CN transition relative to proportional dosage.

    The classification is based on the total fitted transition effect, not on
    the deviation-to-scaling cancellation index alone.
    """
    if is_null:
        return "null"

    if not _is_finite_number(response_ratio):
        return "weak"

    ratio = float(response_ratio)

    if reverse_supported:
        if ratio < 1.0 - thresholds.overcomp_threshold:
            return "overcompensated"

        return "weak"

    if not expected_supported:
        return "weak"

    if hyper_supported:
        return "hyperactive"

    if ratio < 1.0 - thresholds.overcomp_threshold:
        return "overcompensated"

    if ratio < 1.0 - thresholds.cancel_threshold:
        return "buffered"

    # Mild enhancement that does not pass the HYPER credible-interval rule
    # remains dosage-sensitive/proportional rather than becoming UNC.
    return "proportional"

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


_TRANSITION_TARGET_CN: dict[str, int] = {
    "2to1": 1,
    "2to3": 3,
    "2to4": 4,
    "2to5": 5,
}


def _empirical_cn_count(
    result: Mapping[str, Any],
    subtype_index: int,
    cn_state: int,
) -> float:
    """Read an empirical sample count for one absolute-CN state.

    The preferred keys are ``n_cn{state}_s{subtype}`` for subtype analyses and
    ``n_cn{state}`` for a single group.  A few harmless naming variants and
    nested count mappings are accepted to make saved results easier to reuse.

    For continuous CN estimates, these counts should be created upstream from
    prespecified CN-state bins; the classifier deliberately does not infer bins
    from posterior summaries.
    """
    state = int(cn_state)
    suffix = f"s{subtype_index}"

    candidate_keys = [
        f"n_cn{state}_{suffix}",
        f"n_cn_{state}_{suffix}",
        f"n_CN{state}_{suffix}",
        f"n_cn{state}",
        f"n_cn_{state}",
        f"n_CN{state}",
    ]

    if state == 5:
        candidate_keys.extend(
            [
                f"n_cn5plus_{suffix}",
                f"n_cn5_plus_{suffix}",
                "n_cn5plus",
                "n_cn5_plus",
            ]
        )

    for key in candidate_keys:
        value = _get(result, key, np.nan)
        if _is_finite_number(value):
            return float(value)

    mapping_keys = [
        f"cn_state_counts_{suffix}",
        f"cn_counts_{suffix}",
        "cn_state_counts",
        "cn_counts",
    ]

    for mapping_key in mapping_keys:
        counts = _get(result, mapping_key, None)
        if not isinstance(counts, Mapping):
            continue

        for state_key in (
            state,
            str(state),
            f"CN{state}",
            f"cn{state}",
        ):
            value = counts.get(state_key, np.nan)
            if _is_finite_number(value):
                return float(value)

    return float("nan")


def _transition_empirical_support(
    result: Mapping[str, Any],
    subtype_index: int,
    transition: str,
    thresholds: ClassificationThresholds,
) -> dict[str, Any]:
    """Evaluate whether a transition is empirically represented in the data.

    A class-defining transition requires sufficient samples at both CN=2 and
    the transition target state.  Posterior predictions are still retained for
    unsupported states, but they are treated as model extrapolations and cannot
    create DSG/DCG/HYPER/Mixed/DIG calls in strict empirical-support mode.
    """
    if transition not in _TRANSITION_TARGET_CN:
        raise ValueError(f"Unknown CN transition {transition!r}.")

    target_cn = _TRANSITION_TARGET_CN[transition]
    reference_count = _empirical_cn_count(result, subtype_index, 2)
    target_count = _empirical_cn_count(result, subtype_index, target_cn)

    reference_available = _is_finite_number(reference_count)
    target_available = _is_finite_number(target_count)
    support_available = reference_available and target_available

    empirical_supported: bool | None
    if support_available:
        empirical_supported = (
            float(reference_count) >= thresholds.min_reference_cn2_support
            and float(target_count) >= thresholds.min_transition_cn_support
        )
    else:
        empirical_supported = None

    if thresholds.require_empirical_transition_support:
        classifiable = empirical_supported is True
    else:
        classifiable = True

    if not thresholds.require_empirical_transition_support:
        method = "empirical_support_not_required"
    elif not support_available:
        method = "empirical_cn_counts_unavailable"
    elif empirical_supported:
        method = "reference_and_target_cn_supported"
    else:
        ref_ok = (
            reference_available
            and float(reference_count) >= thresholds.min_reference_cn2_support
        )
        target_ok = (
            target_available
            and float(target_count) >= thresholds.min_transition_cn_support
        )
        if not ref_ok and not target_ok:
            method = "insufficient_reference_and_target_cn_support"
        elif not ref_ok:
            method = "insufficient_reference_cn2_support"
        else:
            method = "insufficient_target_cn_support"

    return {
        "target_cn": target_cn,
        "reference_count": reference_count,
        "target_count": target_count,
        "support_available": support_available,
        "empirical_supported": empirical_supported,
        "classifiable": classifiable,
        "method": method,
    }


def _transition_evidence_scope(
    classifiable_transitions: Sequence[str],
) -> str:
    """Describe which observed CN states support the final gene-level call."""
    transitions = set(classifiable_transitions)

    if not transitions:
        return "none"

    has_loss = "2to1" in transitions
    has_gain = "2to3" in transitions
    has_amp = bool(transitions.intersection({"2to4", "2to5"}))

    if has_loss and has_gain and has_amp:
        return "broad"
    if has_loss and has_gain:
        return "loss+gain"
    if has_loss and has_amp:
        return "loss+amplification"
    if has_gain and has_amp:
        return "gain+amplification"
    if has_loss:
        return "loss_only"
    if has_gain:
        return "gain_only"
    return "amplification_only"


def _summarize_transition_patterns(
    patterns: Sequence[str],
    *,
    transitions: Sequence[str],
    fractional_medians: Sequence[Any],
    classifiable: Sequence[bool],
    n_aneup: Any,
    thresholds: ClassificationThresholds,
) -> tuple[str, str, str | None, str, str | None]:
    """Convert empirically supported transition patterns into one gene class.

    Returns
    -------
    response_class, reason, subclass, evidence_scope, classification_basis

    ``patterns`` may contain model predictions for empirically unsupported CN
    states, but only entries marked ``classifiable=True`` participate in the
    biological class.  This prevents extrapolated CN1/CN4/CN5 predictions from
    creating Mixed or otherwise changing the gene class.
    """
    group_map = {
        "proportional": "DSG",
        "buffered": "DCG",
        "overcompensated": "DCG",
        "hyperactive": "HYPER",
    }

    eligible_records = [
        (transition, pattern, fractional)
        for transition, pattern, fractional, use_transition in zip(
            transitions,
            patterns,
            fractional_medians,
            classifiable,
        )
        if bool(use_transition)
    ]

    classifiable_transitions = [
        transition for transition, _, _ in eligible_records
    ]
    evidence_scope = _transition_evidence_scope(classifiable_transitions)
    classification_basis = (
        "|".join(classifiable_transitions)
        if classifiable_transitions
        else None
    )

    # No locally supported target transition means that a global biological
    # response class cannot be inferred, even if the parametric model can
    # extrapolate effects to those states.
    if not eligible_records:
        low_cn = (
            _is_finite_number(n_aneup)
            and int(float(n_aneup)) < thresholds.low_cn_aneup_threshold
        )

        if low_cn:
            return (
                "UNC",
                "insufficient_cn_variation",
                "low_CN",
                evidence_scope,
                classification_basis,
            )

        return (
            "UNC",
            "insufficient_transition_specific_cn_support",
            "limited_transition_support",
            evidence_scope,
            classification_basis,
        )

    supported_groups = [
        group_map[pattern]
        for _, pattern, _ in eligible_records
        if pattern in group_map
    ]
    unique_supported = sorted(set(supported_groups))

    if (
        len(supported_groups) >= thresholds.min_supported_for_mixed
        and len(unique_supported) >= 2
    ):
        return (
            "Mixed",
            "conflicting_empirically_supported_transition_patterns",
            None,
            evidence_scope,
            classification_basis,
        )

    if len(unique_supported) == 1:
        response_class = unique_supported[0]
        return (
            response_class,
            f"supported_{response_class.lower()}_pattern",
            None,
            evidence_scope,
            classification_basis,
        )

    eligible_patterns = [pattern for _, pattern, _ in eligible_records]
    if eligible_patterns and all(pattern == "null" for pattern in eligible_patterns):
        return (
            "DIG",
            "all_empirically_supported_transitions_practically_null",
            None,
            evidence_scope,
            classification_basis,
        )

    # Secondary DIG remains anchored to the original 2->1/2->3/2->4
    # practical-effect thresholds.  CN5 is not assigned an arbitrary median
    # threshold; a classifiable non-null CN5 transition therefore blocks this
    # fallback and leaves the gene unresolved.
    small_threshold_map = {
        "2to1": thresholds.frac_small_loss,
        "2to3": thresholds.frac_small_gain,
        "2to4": thresholds.frac_small_amp,
    }

    dig_reference_records = [
        (transition, pattern, fractional)
        for transition, pattern, fractional in eligible_records
        if transition in small_threshold_map
    ]

    small_effects = bool(dig_reference_records) and all(
        _is_finite_number(fractional)
        and abs(float(fractional)) <= small_threshold_map[transition]
        for transition, _, fractional in dig_reference_records
    )

    cn5_blocks_secondary_dig = any(
        transition == "2to5" and pattern != "null"
        for transition, pattern, _ in eligible_records
    )

    if (
        not supported_groups
        and small_effects
        and not cn5_blocks_secondary_dig
    ):
        return (
            "DIG",
            "empirically_supported_transition_effects_small",
            "median_based",
            evidence_scope,
            classification_basis,
        )

    low_cn = (
        _is_finite_number(n_aneup)
        and int(float(n_aneup)) < thresholds.low_cn_aneup_threshold
    )

    if low_cn:
        return (
            "UNC",
            "insufficient_cn_variation",
            "low_CN",
            evidence_scope,
            classification_basis,
        )

    if len(eligible_records) < thresholds.min_supported_for_mixed:
        return (
            "UNC",
            "limited_or_ambiguous_transition_support",
            "limited_transition_support",
            evidence_scope,
            classification_basis,
        )

    return (
        "UNC",
        "insufficient_or_inconsistent_support",
        "ambiguous",
        evidence_scope,
        classification_basis,
    )


def interpret_subtype_dosage(
    result: Mapping[str, Any],
    subtype_index: int,
    subtype_levels: Sequence[str],
    thresholds: ClassificationThresholds,
) -> dict[str, Any]:
    """Interpret CN-response patterns for one subtype.

    Posterior effects are evaluated for CN 2->1/3/4/5, but only transitions
    with sufficient empirical support at both CN=2 and the target CN state are
    allowed to determine the final biological class when
    ``require_empirical_transition_support=True``.
    """
    human_name = subtype_name(subtype_index, subtype_levels)
    safe_name = _safe_label(human_name)
    canonical_suffix = f"s{subtype_index}"
    ordered_transitions = ["2to1", "2to3", "2to4", "2to5"]

    transition_inputs: dict[str, dict[str, Any]] = {}
    for transition in ordered_transitions:
        expected_direction = "neg" if transition == "2to1" else "pos"
        opposite_direction = "pos" if transition == "2to1" else "neg"

        transition_inputs[transition] = {
            "ppd": _get(
                result,
                f"ppd_fracCN_{transition}_s{subtype_index}",
                np.nan,
            ),
            "expected_direction_probability": _get(
                result,
                f"p_fracCN_{transition}_{expected_direction}_s{subtype_index}",
                np.nan,
            ),
            "opposite_direction_probability": _get(
                result,
                f"p_fracCN_{transition}_{opposite_direction}_s{subtype_index}",
                np.nan,
            ),
            "rope_probability": _get(
                result,
                f"p_rope_fracCN_{transition}_s{subtype_index}",
                np.nan,
            ),
            "fractional_median": _get(
                result,
                f"fracCN_{transition}_s{subtype_index}_median",
                np.nan,
            ),
            "lp_median": _get(
                result,
                f"lp_{transition}_s{subtype_index}_median",
                np.nan,
            ),
            "lp_q025": _get(
                result,
                f"lp_{transition}_s{subtype_index}_q025",
                np.nan,
            ),
            "lp_q975": _get(
                result,
                f"lp_{transition}_s{subtype_index}_q975",
                np.nan,
            ),
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
            float(p_deviation_small) >= thresholds.dev_small_prob
        )
        small_deviation_method = "posterior_probability"
    elif _is_finite_number(b_deviation):
        small_deviation = (
            abs(float(b_deviation)) <= thresholds.dev_abs_fallback
        )
        small_deviation_method = "median_fallback"
    else:
        small_deviation = False
        small_deviation_method = "unavailable"

    transition_results: dict[str, dict[str, Any]] = {}

    for transition, inputs in transition_inputs.items():
        (
            expected_supported,
            reverse_supported,
            is_null,
            posterior_support_method,
        ) = _transition_support(
            transition=transition,
            ppd_value=inputs["ppd"],
            expected_direction_probability=inputs[
                "expected_direction_probability"
            ],
            opposite_direction_probability=inputs[
                "opposite_direction_probability"
            ],
            rope_probability=inputs["rope_probability"],
            fractional_median=inputs["fractional_median"],
            thresholds=thresholds,
        )

        lp_scaling = _get(
            result,
            f"lp_scaling_{transition}_s{subtype_index}_median",
            np.nan,
        )
        scaling_stable = (
            (
                _is_finite_number(lp_scaling)
                and abs(float(lp_scaling)) > thresholds.min_scaling_abs
            )
            or (
                not _is_finite_number(lp_scaling)
                and _is_finite_number(b_scaling)
                and abs(float(b_scaling)) > thresholds.min_scaling_abs
            )
        )

        cancellation_index = _transition_ci(
            result,
            subtype_index,
            transition,
            thresholds,
        )
        response_ratio = _transition_response_ratio(
            transition=transition,
            log_effect_median=inputs["lp_median"],
            fractional_median=inputs["fractional_median"],
        )
        (
            posterior_hyper_supported,
            hyper_evidence_method,
        ) = _hyper_supported(
            transition=transition,
            log_effect_q025=inputs["lp_q025"],
            log_effect_q975=inputs["lp_q975"],
            response_ratio=response_ratio,
            thresholds=thresholds,
        )

        model_pattern = _classify_transition(
            response_ratio=response_ratio,
            expected_supported=expected_supported,
            reverse_supported=reverse_supported,
            is_null=is_null,
            hyper_supported=posterior_hyper_supported,
            thresholds=thresholds,
        )

        empirical = _transition_empirical_support(
            result,
            subtype_index,
            transition,
            thresholds,
        )
        classifiable = bool(empirical["classifiable"])

        # Keep the model-predicted transition pattern for descriptive use, but
        # do not let an unsupported target state alter the gene-level class.
        effective_pattern = (
            model_pattern if classifiable else "unsupported_cn_state"
        )
        posterior_supported = expected_supported or reverse_supported
        effective_supported = posterior_supported and classifiable
        effective_null = is_null and classifiable
        effective_hyper_supported = (
            posterior_hyper_supported and classifiable
        )
        effective_reverse_supported = reverse_supported and classifiable

        transition_results[transition] = {
            "pattern": effective_pattern,
            "model_pattern": model_pattern,
            "supported": effective_supported,
            "posterior_supported": posterior_supported,
            "expected_supported": expected_supported and classifiable,
            "posterior_expected_supported": expected_supported,
            "reverse_supported": effective_reverse_supported,
            "posterior_reverse_supported": reverse_supported,
            "null": effective_null,
            "posterior_null": is_null,
            "support_method": posterior_support_method,
            "cancellation_index": cancellation_index,
            "response_ratio": response_ratio,
            "hyper_supported": effective_hyper_supported,
            "posterior_hyper_supported": posterior_hyper_supported,
            "hyper_evidence_method": hyper_evidence_method,
            "fractional_median": inputs["fractional_median"],
            "direction_probability": inputs[
                "expected_direction_probability"
            ],
            "opposite_direction_probability": inputs[
                "opposite_direction_probability"
            ],
            "rope_probability": inputs["rope_probability"],
            "ppd": inputs["ppd"],
            "scaling_stable": scaling_stable,
            "classifiable": classifiable,
            "empirical_supported": empirical["empirical_supported"],
            "empirical_support_available": empirical["support_available"],
            "empirical_support_method": empirical["method"],
            "reference_cn_count": empirical["reference_count"],
            "target_cn_count": empirical["target_count"],
            "target_cn": empirical["target_cn"],
        }

    patterns = [
        transition_results[transition]["pattern"]
        for transition in ordered_transitions
    ]
    fractional_medians = [
        transition_results[transition]["fractional_median"]
        for transition in ordered_transitions
    ]
    classifiable = [
        bool(transition_results[transition]["classifiable"])
        for transition in ordered_transitions
    ]

    (
        response_class,
        response_reason,
        response_subclass,
        evidence_scope,
        classification_basis,
    ) = _summarize_transition_patterns(
        patterns,
        transitions=ordered_transitions,
        fractional_medians=fractional_medians,
        classifiable=classifiable,
        n_aneup=_get(result, "n_aneup", np.nan),
        thresholds=thresholds,
    )

    classifiable_transitions = [
        transition
        for transition in ordered_transitions
        if transition_results[transition]["classifiable"]
    ]

    dc_gain = any(
        transition_results[transition]["pattern"]
        in {"buffered", "overcompensated"}
        for transition in ["2to3", "2to4", "2to5"]
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
        f"evidence_scope_{canonical_suffix}": evidence_scope,
        f"classification_basis_{canonical_suffix}": classification_basis,
        f"n_classifiable_transitions_{canonical_suffix}": len(
            classifiable_transitions
        ),
        f"n_empirically_supported_transitions_{canonical_suffix}": sum(
            transition_results[transition]["empirical_supported"] is True
            for transition in ordered_transitions
        ),
        f"small_deviation_{canonical_suffix}": small_deviation,
        f"small_deviation_method_{canonical_suffix}": small_deviation_method,
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
                f"transition_{transition}_{canonical_suffix}": values[
                    "pattern"
                ],
                f"transition_model_pattern_{transition}_{canonical_suffix}": values[
                    "model_pattern"
                ],
                f"transition_classifiable_{transition}_{canonical_suffix}": values[
                    "classifiable"
                ],
                f"transition_empirical_support_{transition}_{canonical_suffix}": values[
                    "empirical_supported"
                ],
                f"transition_empirical_support_available_{transition}_{canonical_suffix}": values[
                    "empirical_support_available"
                ],
                f"transition_empirical_support_method_{transition}_{canonical_suffix}": values[
                    "empirical_support_method"
                ],
                f"transition_reference_cn_count_{transition}_{canonical_suffix}": values[
                    "reference_cn_count"
                ],
                f"transition_target_cn_count_{transition}_{canonical_suffix}": values[
                    "target_cn_count"
                ],
                f"transition_target_cn_{transition}_{canonical_suffix}": values[
                    "target_cn"
                ],
                f"transition_supported_{transition}_{canonical_suffix}": values[
                    "supported"
                ],
                f"transition_posterior_supported_{transition}_{canonical_suffix}": values[
                    "posterior_supported"
                ],
                f"transition_null_{transition}_{canonical_suffix}": values[
                    "null"
                ],
                f"transition_posterior_null_{transition}_{canonical_suffix}": values[
                    "posterior_null"
                ],
                f"transition_support_method_{transition}_{canonical_suffix}": values[
                    "support_method"
                ],
                f"cancel_index_{transition}_{canonical_suffix}": values[
                    "cancellation_index"
                ],
                f"fracCN_{transition}_median_{canonical_suffix}": values[
                    "fractional_median"
                ],
                f"direction_probability_{transition}_{canonical_suffix}": values[
                    "direction_probability"
                ],
                f"rope_probability_{transition}_{canonical_suffix}": values[
                    "rope_probability"
                ],
                f"ppd_{transition}_{canonical_suffix}": values["ppd"],
                f"response_ratio_{transition}_{canonical_suffix}": values[
                    "response_ratio"
                ],
                f"hyper_supported_{transition}_{canonical_suffix}": values[
                    "hyper_supported"
                ],
                f"posterior_hyper_supported_{transition}_{canonical_suffix}": values[
                    "posterior_hyper_supported"
                ],
                f"hyper_evidence_method_{transition}_{canonical_suffix}": values[
                    "hyper_evidence_method"
                ],
                f"reverse_supported_{transition}_{canonical_suffix}": values[
                    "reverse_supported"
                ],
                f"posterior_reverse_supported_{transition}_{canonical_suffix}": values[
                    "posterior_reverse_supported"
                ],
            }
        )

    # Human-readable aliases retained for backward compatibility.
    if safe_name != canonical_suffix:
        output.update(
            {
                f"response_class_{safe_name}": response_class,
                f"response_reason_{safe_name}": response_reason,
                f"response_subclass_{safe_name}": response_subclass,
                f"evidence_scope_{safe_name}": evidence_scope,
                f"classification_basis_{safe_name}": classification_basis,
                f"transition_2to1_{safe_name}": transition_results[
                    "2to1"
                ]["pattern"],
                f"transition_2to3_{safe_name}": transition_results[
                    "2to3"
                ]["pattern"],
                f"transition_2to4_{safe_name}": transition_results[
                    "2to4"
                ]["pattern"],
                f"transition_2to5_{safe_name}": transition_results[
                    "2to5"
                ]["pattern"],
            }
        )

    return output


def _optional_bool(value: Any) -> bool | None:
    """Interpret common boolean representations; return None if unknown."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None or _is_nan_like(value):
        return None

    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False

    return None


def _classification_eligibility(
    result: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Determine whether a fit may receive a biological response class.

    Failed sampling and non-converged fits are excluded, as specified in the
    manuscript. Legacy summaries without diagnostic metadata remain eligible
    because convergence cannot be reconstructed from them.
    """
    success = _optional_bool(
        _get(
            result,
            "batch_success",
            _get(result, "success", np.nan),
        )
    )
    if success is False:
        return False, "fit_failed"

    converged = _optional_bool(_get(result, "converged", np.nan))
    if converged is False:
        return False, "nonconverged"

    fit_flag = str(_get(result, "fit_flag", "")).strip().casefold()

    if fit_flag in {
        "warn",
        "warning",
        "nonconverged",
        "non-converged",
    }:
        return False, "nonconverged"

    if fit_flag in {
        "fail",
        "failed",
        "failure",
        "error",
    }:
        return False, "fit_failed"

    status = str(_get(result, "status", "")).strip().casefold()
    if status in {
        "fail",
        "failed",
        "failure",
        "error",
        "skipped",
    }:
        return False, f"status_{status}"

    return True, None


def _excluded_subtype_output(
    result: Mapping[str, Any],
    subtype_index: int,
    subtype_levels: Sequence[str],
    reason: str,
    thresholds: ClassificationThresholds,
) -> dict[str, Any]:
    """Return schema-compatible fields for a computationally excluded fit."""
    human_name = subtype_name(subtype_index, subtype_levels)
    safe_name = _safe_label(human_name)
    suffix = f"s{subtype_index}"

    output: dict[str, Any] = {
        f"subtype_{suffix}": human_name,
        f"response_class_{suffix}": None,
        f"response_reason_{suffix}": f"excluded:{reason}",
        f"response_subclass_{suffix}": reason,
        f"evidence_scope_{suffix}": "not_evaluated",
        f"classification_basis_{suffix}": None,
        f"n_classifiable_transitions_{suffix}": 0,
        f"n_empirically_supported_transitions_{suffix}": 0,
        f"small_deviation_{suffix}": None,
        f"small_deviation_method_{suffix}": "not_evaluated",
        f"b_scaling_median_{suffix}": _get(
            result,
            f"b_scaling_s{subtype_index}_median",
            np.nan,
        ),
        f"b_deviation_median_{suffix}": _get(
            result,
            f"b_deviation_s{subtype_index}_median",
            np.nan,
        ),
        f"p_deviation_small_{suffix}": _get(
            result,
            f"p_rope_bdev_s{subtype_index}",
            np.nan,
        ),
        f"dc_gain_{suffix}": False,
        f"dc_loss_{suffix}": False,
        f"dc_type_{suffix}": "not_evaluated",
    }

    for transition in ("2to1", "2to3", "2to4", "2to5"):
        fractional = _get(
            result,
            f"fracCN_{transition}_s{subtype_index}_median",
            np.nan,
        )
        log_effect = _get(
            result,
            f"lp_{transition}_s{subtype_index}_median",
            np.nan,
        )
        target_cn = _TRANSITION_TARGET_CN[transition]
        reference_count = _empirical_cn_count(result, subtype_index, 2)
        target_count = _empirical_cn_count(result, subtype_index, target_cn)

        output.update(
            {
                f"transition_{transition}_{suffix}": None,
                f"transition_model_pattern_{transition}_{suffix}": None,
                f"transition_classifiable_{transition}_{suffix}": False,
                f"transition_empirical_support_{transition}_{suffix}": None,
                f"transition_empirical_support_available_{transition}_{suffix}": False,
                f"transition_empirical_support_method_{transition}_{suffix}": "not_evaluated",
                f"transition_reference_cn_count_{transition}_{suffix}": reference_count,
                f"transition_target_cn_count_{transition}_{suffix}": target_count,
                f"transition_target_cn_{transition}_{suffix}": target_cn,
                f"transition_supported_{transition}_{suffix}": False,
                f"transition_posterior_supported_{transition}_{suffix}": False,
                f"transition_null_{transition}_{suffix}": False,
                f"transition_posterior_null_{transition}_{suffix}": False,
                f"transition_support_method_{transition}_{suffix}": "not_evaluated",
                f"cancel_index_{transition}_{suffix}": _transition_ci(
                    result,
                    subtype_index,
                    transition,
                    thresholds,
                ),
                f"fracCN_{transition}_median_{suffix}": fractional,
                f"direction_probability_{transition}_{suffix}": np.nan,
                f"rope_probability_{transition}_{suffix}": _get(
                    result,
                    f"p_rope_fracCN_{transition}_s{subtype_index}",
                    np.nan,
                ),
                f"ppd_{transition}_{suffix}": _get(
                    result,
                    f"ppd_fracCN_{transition}_s{subtype_index}",
                    np.nan,
                ),
                f"response_ratio_{transition}_{suffix}": (
                    _transition_response_ratio(
                        transition=transition,
                        log_effect_median=log_effect,
                        fractional_median=fractional,
                    )
                ),
                f"hyper_supported_{transition}_{suffix}": False,
                f"posterior_hyper_supported_{transition}_{suffix}": False,
                f"hyper_evidence_method_{transition}_{suffix}": "not_evaluated",
                f"reverse_supported_{transition}_{suffix}": False,
                f"posterior_reverse_supported_{transition}_{suffix}": False,
            }
        )

    if safe_name != suffix:
        output.update(
            {
                f"response_class_{safe_name}": None,
                f"response_reason_{safe_name}": f"excluded:{reason}",
                f"response_subclass_{safe_name}": reason,
                f"evidence_scope_{safe_name}": "not_evaluated",
                f"classification_basis_{safe_name}": None,
                f"transition_2to1_{safe_name}": None,
                f"transition_2to3_{safe_name}": None,
                f"transition_2to4_{safe_name}": None,
                f"transition_2to5_{safe_name}": None,
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
            "converged",
            diagnostics.get("converged"),
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

    eligible, exclusion_reason = _classification_eligibility(result)

    output: dict[str, Any] = {
        "gene": _get(result, "gene", None),
        "status": _get(result, "status", "ok"),
        "fit_flag": _get(result, "fit_flag", "ok"),
        "converged": _get(result, "converged", np.nan),
        "analysis_mode": analysis_mode,
        "N": _get(result, "N", np.nan),
        "n_aneup": _get(result, "n_aneup", np.nan),
        "cna": _get(result, "cna", np.nan),
        "S": number_of_subtypes,
        "subtype_levels": subtype_levels,
        "subtype_levels_str": "|".join(subtype_levels),
        "classification_eligible": eligible,
        "classification_exclusion_reason": exclusion_reason,
    }

    # A computational failure is not UNC. UNC is reserved for an ambiguous but valid posterior fit.
    if not eligible:
        output.update(
            {
                "de_status": "not_evaluated",
                "ppd_tumor": _get(result, "ppd_tumor", np.nan),
                "p_rope_tumor": _get(result, "p_rope_tumor", np.nan),
                "tumor0_lfc_median": _get(result, "tumor0_lfc_median", np.nan),
                "tumor0_lfc_q025": _get(result, "tumor0_lfc_q025", np.nan),
                "tumor0_lfc_q975": _get(result, "tumor0_lfc_q975", np.nan),
                "rewiring_status": "not_evaluated",
                "scaling_rewired": False,
                "deviation_rewired": False,
                "scaling_rewiring_evidence": "not_evaluated",
                "deviation_rewiring_evidence": "not_evaluated",
                "ppd_scaling": _get(result, "ppd_scaling", np.nan),
                "ppd_dev": _get(result, "ppd_dev", np.nan),
                "p_rope_scaling": _get(result, "p_rope_scaling", np.nan),
                "p_rope_dev": _get(result, "p_rope_dev", np.nan),
                "delta_scaling_median": _get(
                    result, "delta_scaling_median", np.nan
                ),
                "delta_scaling_q025": _get(
                    result, "delta_scaling_q025", np.nan
                ),
                "delta_scaling_q975": _get(
                    result, "delta_scaling_q975", np.nan
                ),
                "delta_dev_median": _get(result, "delta_dev_median", np.nan),
                "delta_dev_q025": _get(result, "delta_dev_q025", np.nan),
                "delta_dev_q975": _get(result, "delta_dev_q975", np.nan),
            }
        )

        for subtype_index in range(1, number_of_subtypes + 1):
            output.update(
                _excluded_subtype_output(
                    result,
                    subtype_index,
                    subtype_levels,
                    exclusion_reason or "ineligible_fit",
                    thresholds,
                )
            )

        output["summary_label"] = (
            f"excluded:{exclusion_reason or 'ineligible_fit'}"
        )
        return output

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

def _ast_dotted_name(node: ast.AST) -> str | None:
    """Return a dotted name such as ``np.float64`` for a simple AST node."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = _ast_dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"

    return None

class _SerializedResultNormalizer(ast.NodeTransformer):
    """Normalize safe NumPy/pandas tokens found in dict strings from CSV files.

    ``ast.literal_eval`` intentionally rejects names such as ``nan`` and calls
    such as ``np.float64(0.5)``.  These representations commonly appear after
    a Python dictionary containing NumPy scalars or missing values has been
    converted to text and round-tripped through CSV.

    Only a small whitelist of numeric/missing-value constructs is accepted.
    Arbitrary names, attributes, and function calls remain rejected.
    """

    _NAN_NAMES = {
        "nan",
        "NaN",
        "NAN",
    }

    _INF_NAMES = {
        "inf",
        "Inf",
        "INF",
        "Infinity",
        "infinity",
    }

    _NAN_ATTRIBUTES = {
        "np.nan",
        "numpy.nan",
        "pd.NA",
        "pandas.NA",
    }

    _INF_ATTRIBUTES = {
        "np.inf",
        "numpy.inf",
    }

    _NUMERIC_WRAPPERS = {
        "float",
        "int",
        "np.float16",
        "np.float32",
        "np.float64",
        "numpy.float16",
        "numpy.float32",
        "numpy.float64",
        "np.int8",
        "np.int16",
        "np.int32",
        "np.int64",
        "numpy.int8",
        "numpy.int16",
        "numpy.int32",
        "numpy.int64",
        "np.uint8",
        "np.uint16",
        "np.uint32",
        "np.uint64",
        "numpy.uint8",
        "numpy.uint16",
        "numpy.uint32",
        "numpy.uint64",
        "np.bool_",
        "numpy.bool_",
        "bool",
    }

    _ARRAY_WRAPPERS = {
        "array",
        "np.array",
        "numpy.array",
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._NAN_NAMES:
            return ast.copy_location(
                ast.Constant(value=float("nan")),
                node,
            )

        if node.id in self._INF_NAMES:
            return ast.copy_location(
                ast.Constant(value=float("inf")),
                node,
            )

        # Do not silently evaluate any other bare name.
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        dotted = _ast_dotted_name(node)

        if dotted in self._NAN_ATTRIBUTES:
            return ast.copy_location(
                ast.Constant(value=float("nan")),
                node,
            )

        if dotted in self._INF_ATTRIBUTES:
            return ast.copy_location(
                ast.Constant(value=float("inf")),
                node,
            )

        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        dotted = _ast_dotted_name(node.func)

        if dotted in self._NUMERIC_WRAPPERS:
            if len(node.args) != 1 or node.keywords:
                return node

            argument = self.visit(node.args[0])

            # Convert simple constants to their ordinary Python scalar form.
            if isinstance(argument, ast.Constant):
                value = argument.value

                try:
                    if dotted.endswith(("bool", "bool_")):
                        converted = bool(value)
                    elif ".int" in dotted or ".uint" in dotted or dotted == "int":
                        converted = int(value)
                    else:
                        converted = float(value)
                except (TypeError, ValueError, OverflowError):
                    return node

                return ast.copy_location(
                    ast.Constant(value=converted),
                    node,
                )

            # A signed numeric expression such as np.float64(-1.2) is already
            # safe for literal_eval once the wrapper is removed.
            if isinstance(argument, ast.UnaryOp):
                return ast.copy_location(argument, node)

            return node

        if dotted in self._ARRAY_WRAPPERS:
            if len(node.args) == 1 and not node.keywords:
                argument = self.visit(node.args[0])

                # Preserve the serialized information as an ordinary list or
                # tuple; classification code does not require ndarray methods.
                if isinstance(
                    argument,
                    (ast.List, ast.Tuple, ast.Set, ast.Constant),
                ):
                    return ast.copy_location(argument, node)

            return node

        return self.generic_visit(node)

def _safe_serialized_literal_eval(value: str) -> Any:
    """Safely parse a serialized Python literal with common NumPy NA tokens."""
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            "Serialized result is not valid Python-literal syntax."
        ) from exc

    tree = _SerializedResultNormalizer().visit(tree)
    ast.fix_missing_locations(tree)

    try:
        return ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError) as exc:
        # Report unresolved names/calls to make problematic saved rows easier
        # to diagnose without executing arbitrary text.
        unresolved_names = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
            }
        )
        unresolved_calls = sorted(
            {
                name
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                for name in [_ast_dotted_name(node.func)]
                if name is not None
            }
        )

        details: list[str] = []

        if unresolved_names:
            details.append(
                "unrecognized names="
                + repr(unresolved_names[:10])
            )

        if unresolved_calls:
            details.append(
                "unrecognized calls="
                + repr(unresolved_calls[:10])
            )

        suffix = (
            " (" + "; ".join(details) + ")"
            if details
            else ""
        )

        raise ValueError(
            "Could not safely parse a serialized result mapping"
            f"{suffix}."
        ) from exc
    

def _coerce_saved_result(
    value: Any,
) -> dict[str, Any]:
    """Convert a saved ``result`` cell into a flat result mapping.

    The preferred representation is a real mapping.  CSV round-trips may turn
    dictionaries into strings containing ordinary Python literals and, in some
    datasets, NumPy/pandas representations such as ``nan``, ``np.nan``,
    ``np.float64(...)`` or ``array([...])``.  These common safe constructs are
    normalized before ``ast.literal_eval``; arbitrary code is never executed.
    """
    if isinstance(value, Mapping):
        return dict(value)

    if value is None or _is_nan_like(value):
        return {}

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return {}

        parsed = _safe_serialized_literal_eval(stripped)

        if not isinstance(parsed, Mapping):
            raise TypeError(
                "Serialized result must evaluate to a mapping; "
                f"received {type(parsed).__name__}."
            )

        return dict(parsed)

    raise TypeError(
        "result must be a mapping, a serialized mapping string, "
        f"or a missing value; received {type(value).__name__}."
    )



def _result_record_from_dataframe_row(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one row from either a flat or nested results data frame."""
    # Existing/legacy format: every posterior-summary field is already a
    # top-level data-frame column.
    if "result" not in record:
        return dict(record)

    # Saved batch format: posterior summary is nested in the ``result`` cell,
    # while columns such as gene/success/error remain outside it.
    result = _coerce_saved_result(record.get("result"))

    outer_gene = record.get("gene")
    if not result.get("gene") and not _is_nan_like(outer_gene):
        result["gene"] = outer_gene

    # Preserve batch-execution metadata without overwriting posterior fields.
    if "success" in record:
        result.setdefault("batch_success", record.get("success"))

    if "converged" in record:
        result.setdefault("converged", record.get("converged"))

    if "fit_flag" in record:
        result.setdefault("fit_flag", record.get("fit_flag"))

    if "diagnostic_status" in record:
        result.setdefault(
            "diagnostic_status",
            record.get("diagnostic_status"),
        )

    if "error" in record and not _is_nan_like(record.get("error")):
        result.setdefault("batch_error", record.get("error"))

    return result


def classify_fits(
    fits: Mapping[str, Any] | pd.DataFrame,
    thresholds: ClassificationThresholds | None = None,
) -> pd.DataFrame:
    """Classify fitted objects or a saved results data frame.

    ``Mapping[str, BDGDMFit]`` inputs retain the original behaviour.  A pandas
    DataFrame is delegated to :func:`classify_results_dataframe`, including
    the saved batch format with a nested/serialized ``result`` column.
    """
    if isinstance(fits, pd.DataFrame):
        return classify_results_dataframe(
            fits,
            thresholds=thresholds,
        )

    if not isinstance(fits, Mapping):
        raise TypeError(
            "fits must be a mapping of gene -> fit or a pandas DataFrame."
        )

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
    """Classify every row of a posterior-summary data frame.

    Two input layouts are supported:

    1. a flat table in which posterior-summary fields are ordinary columns;
    2. a saved batch table with columns such as ``gene``, ``success``,
       ``result`` and ``error``, where ``result`` is either a mapping or a
       string representation of one.
    """
    source_records = dataframe.to_dict(orient="records")
    result_records: list[dict[str, Any]] = []

    for row_number, source_record in enumerate(source_records):
        try:
            result_record = _result_record_from_dataframe_row(source_record)
        except (TypeError, ValueError) as exc:
            gene = source_record.get("gene", f"row_{row_number}")
            raise ValueError(
                f"Could not normalize result for gene {gene!r} "
                f"at row {row_number}."
            ) from exc

        result_records.append(result_record)

    classified = pd.DataFrame.from_records(
        [
            classify_gene_result(
                record,
                thresholds=thresholds,
            )
            for record in result_records
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


def _is_classified_result(result: Mapping[str, Any]) -> bool:
    """Return True when a mapping already contains subtype class outputs."""
    return any(
        re.fullmatch(r"response_class_s\d+", str(key))
        for key in result
    )

def _coerce_classification(
    fit_or_result: Any,
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """
    Return a classified result from a fit, posterior summary, or class mapping.
    """
    if isinstance(fit_or_result, Mapping):
        if _is_classified_result(fit_or_result):
            return dict(fit_or_result)

        return classify_gene_result(
            fit_or_result,
            thresholds=thresholds,
        )

    return classify_fit(
        fit_or_result,
        thresholds=thresholds,
    )


def _resolve_subtype_index(
    classification: Mapping[str, Any],
    subtype: int | str | None,
) -> int:
    """
    Resolve a one-based subtype index from an index or human-readable label.
    """
    subtype_levels = normalize_subtype_levels(
        classification.get(
            "subtype_levels",
            classification.get(
                "subtype_order",
                None,
            ),
        )
    )

    number_of_subtypes = classification.get(
        "S",
        classification.get(
            "n_subtypes",
            None,
        ),
    )

    if not _is_finite_number(number_of_subtypes):
        number_of_subtypes = (
            len(subtype_levels)
            or infer_num_subtypes(
                classification,
                default=1,
            )
        )

    number_of_subtypes = int(number_of_subtypes)

    if subtype is None:
        if number_of_subtypes == 1:
            return 1

        raise ValueError(
            "subtype must be specified when more than one subtype "
            "is present."
        )

    if isinstance(subtype, (int, np.integer)):
        subtype_index = int(subtype)

    elif isinstance(subtype, str):
        requested = subtype.strip()

        if not requested:
            raise ValueError("subtype label must be non-empty.")

        match = re.fullmatch(r"[sS]?(\d+)", requested)

        if match:
            subtype_index = int(match.group(1))
        else:
            labels = [
                str(
                    classification.get(
                        f"subtype_s{index}",
                        subtype_name(
                            index,
                            subtype_levels,
                        ),
                    )
                )
                for index in range(
                    1,
                    number_of_subtypes + 1,
                )
            ]

            exact_matches = [
                index
                for index, label in enumerate(
                    labels,
                    start=1,
                )
                if label == requested
            ]

            if not exact_matches:
                folded = requested.casefold()
                exact_matches = [
                    index
                    for index, label in enumerate(
                        labels,
                        start=1,
                    )
                    if label.casefold() == folded
                ]

            if len(exact_matches) != 1:
                raise KeyError(
                    f"Unknown subtype {subtype!r}. "
                    f"Available subtypes: {labels}"
                )

            subtype_index = exact_matches[0]

    else:
        raise TypeError(
            "subtype must be None, a one-based integer, or a label."
        )

    if not 1 <= subtype_index <= number_of_subtypes:
        raise IndexError(
            f"Subtype index must lie between 1 and "
            f"{number_of_subtypes}; received {subtype_index}."
        )

    return subtype_index


def get_subtype_classification(
    fit_or_result: Any,
    subtype: int | str | None = None,
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """
    Return a compact classification summary for one subtype.

    Parameters
    ----------
    fit_or_result
        A ``BDGDMFit`` object, a flat posterior-summary mapping, or an
        already classified mapping.

    subtype
        One-based subtype index or human-readable subtype label. It may be
        omitted when the result contains only one subtype.

    thresholds
        Classification thresholds used only when classification has not
        already been performed.

    Returns
    -------
    dict
        Compact subtype-level classification information.
    """
    classification = _coerce_classification(
        fit_or_result,
        thresholds=thresholds,
    )
    subtype_index = _resolve_subtype_index(
        classification,
        subtype,
    )
    suffix = f"s{subtype_index}"
    subtype_label = classification.get(
        f"subtype_{suffix}",
        subtype_name(
            subtype_index,
            normalize_subtype_levels(
                classification.get(
                    "subtype_levels",
                    None,
                )
            ),
        ),
    )

    return {
        "gene": classification.get("gene"),
        "status": classification.get("status"),
        "fit_flag": classification.get("fit_flag"),
        "classification_eligible": classification.get(
            "classification_eligible"
        ),
        "classification_exclusion_reason": classification.get(
            "classification_exclusion_reason"
        ),
        "analysis_mode": classification.get("analysis_mode"),
        "N": classification.get("N"),
        "n_aneup": classification.get("n_aneup"),
        "cna": classification.get("cna"),
        "subtype_index": subtype_index,
        "subtype": subtype_label,
        "response_class": classification.get(
            f"response_class_{suffix}"
        ),
        "response_reason": classification.get(
            f"response_reason_{suffix}"
        ),
        "response_subclass": classification.get(
            f"response_subclass_{suffix}"
        ),
        "evidence_scope": classification.get(
            f"evidence_scope_{suffix}"
        ),
        "classification_basis": classification.get(
            f"classification_basis_{suffix}"
        ),
        "n_classifiable_transitions": classification.get(
            f"n_classifiable_transitions_{suffix}"
        ),
        "n_empirically_supported_transitions": classification.get(
            f"n_empirically_supported_transitions_{suffix}"
        ),
        "b_scaling_median": classification.get(
            f"b_scaling_median_{suffix}"
        ),
        "b_deviation_median": classification.get(
            f"b_deviation_median_{suffix}"
        ),
        "p_deviation_small": classification.get(
            f"p_deviation_small_{suffix}"
        ),
        "small_deviation": classification.get(
            f"small_deviation_{suffix}"
        ),
        "small_deviation_method": classification.get(
            f"small_deviation_method_{suffix}"
        ),
        "dc_gain": classification.get(
            f"dc_gain_{suffix}"
        ),
        "dc_loss": classification.get(
            f"dc_loss_{suffix}"
        ),
        "dc_type": classification.get(
            f"dc_type_{suffix}"
        ),
    }


def get_response_class(
    fit_or_result: Any,
    subtype: int | str | None = None,
    thresholds: ClassificationThresholds | None = None,
) -> str | None:
    """
    Return only the final response class for one subtype.
    """
    return get_subtype_classification(
        fit_or_result,
        subtype=subtype,
        thresholds=thresholds,
    )["response_class"]


def get_transition_classification(
    fit_or_result: Any,
    transition: str,
    subtype: int | str | None = None,
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """Return posterior and empirical-support evidence for one CN transition."""
    normalized_transition = (
        str(transition)
        .strip()
        .replace("→", "to")
        .replace("->", "to")
        .replace(" ", "")
    )

    if normalized_transition not in {
        "2to1",
        "2to3",
        "2to4",
        "2to5",
    }:
        raise ValueError(
            "transition must be one of '2to1', '2to3', '2to4', or '2to5'."
        )

    classification = _coerce_classification(
        fit_or_result,
        thresholds=thresholds,
    )
    subtype_index = _resolve_subtype_index(classification, subtype)
    suffix = f"s{subtype_index}"
    subtype_label = classification.get(
        f"subtype_{suffix}",
        subtype_name(
            subtype_index,
            normalize_subtype_levels(
                classification.get("subtype_levels", None)
            ),
        ),
    )

    return {
        "gene": classification.get("gene"),
        "classification_eligible": classification.get(
            "classification_eligible"
        ),
        "classification_exclusion_reason": classification.get(
            "classification_exclusion_reason"
        ),
        "subtype_index": subtype_index,
        "subtype": subtype_label,
        "transition": normalized_transition.replace("to", "→"),
        "transition_key": normalized_transition,
        "pattern": classification.get(
            f"transition_{normalized_transition}_{suffix}"
        ),
        "model_pattern": classification.get(
            f"transition_model_pattern_{normalized_transition}_{suffix}"
        ),
        "classifiable": classification.get(
            f"transition_classifiable_{normalized_transition}_{suffix}"
        ),
        "empirical_support": classification.get(
            f"transition_empirical_support_{normalized_transition}_{suffix}"
        ),
        "empirical_support_available": classification.get(
            f"transition_empirical_support_available_{normalized_transition}_{suffix}"
        ),
        "empirical_support_method": classification.get(
            f"transition_empirical_support_method_{normalized_transition}_{suffix}"
        ),
        "reference_cn_count": classification.get(
            f"transition_reference_cn_count_{normalized_transition}_{suffix}"
        ),
        "target_cn_count": classification.get(
            f"transition_target_cn_count_{normalized_transition}_{suffix}"
        ),
        "target_cn": classification.get(
            f"transition_target_cn_{normalized_transition}_{suffix}"
        ),
        "supported": classification.get(
            f"transition_supported_{normalized_transition}_{suffix}"
        ),
        "posterior_supported": classification.get(
            f"transition_posterior_supported_{normalized_transition}_{suffix}"
        ),
        "null": classification.get(
            f"transition_null_{normalized_transition}_{suffix}"
        ),
        "posterior_null": classification.get(
            f"transition_posterior_null_{normalized_transition}_{suffix}"
        ),
        "support_method": classification.get(
            f"transition_support_method_{normalized_transition}_{suffix}"
        ),
        "fractional_change_median": classification.get(
            f"fracCN_{normalized_transition}_median_{suffix}"
        ),
        "direction_probability": classification.get(
            f"direction_probability_{normalized_transition}_{suffix}"
        ),
        "rope_probability": classification.get(
            f"rope_probability_{normalized_transition}_{suffix}"
        ),
        "ppd": classification.get(
            f"ppd_{normalized_transition}_{suffix}"
        ),
        "response_ratio": classification.get(
            f"response_ratio_{normalized_transition}_{suffix}"
        ),
        "hyper_supported": classification.get(
            f"hyper_supported_{normalized_transition}_{suffix}"
        ),
        "posterior_hyper_supported": classification.get(
            f"posterior_hyper_supported_{normalized_transition}_{suffix}"
        ),
        "hyper_evidence_method": classification.get(
            f"hyper_evidence_method_{normalized_transition}_{suffix}"
        ),
        "reverse_supported": classification.get(
            f"reverse_supported_{normalized_transition}_{suffix}"
        ),
        "posterior_reverse_supported": classification.get(
            f"posterior_reverse_supported_{normalized_transition}_{suffix}"
        ),
        "cancellation_index": classification.get(
            f"cancel_index_{normalized_transition}_{suffix}"
        ),
    }


def classification_to_subtype_df(
    fit_or_result: Any,
    thresholds: ClassificationThresholds | None = None,
) -> pd.DataFrame:
    """
    Convert one classified result into one tidy row per subtype.
    """
    classification = _coerce_classification(
        fit_or_result,
        thresholds=thresholds,
    )

    number_of_subtypes = classification.get(
        "S",
        classification.get(
            "n_subtypes",
            infer_num_subtypes(
                classification,
                default=1,
            ),
        ),
    )
    number_of_subtypes = int(number_of_subtypes)

    return pd.DataFrame.from_records(
        [
            get_subtype_classification(
                classification,
                subtype=index,
            )
            for index in range(
                1,
                number_of_subtypes + 1,
            )
        ]
    )


def get_transition_df(
    fit_or_result: Any,
    subtype: int | str | None = None,
    thresholds: ClassificationThresholds | None = None,
) -> pd.DataFrame:
    """
    Convert transition evidence into a tidy DataFrame.

    When ``subtype`` is omitted for a multi-subtype result, transitions for
    all subtypes are returned.
    """
    classification = _coerce_classification(
        fit_or_result,
        thresholds=thresholds,
    )

    if subtype is None:
        number_of_subtypes = classification.get(
            "S",
            classification.get(
                "n_subtypes",
                infer_num_subtypes(
                    classification,
                    default=1,
                ),
            ),
        )
        subtype_indices = range(
            1,
            int(number_of_subtypes) + 1,
        )
    else:
        subtype_indices = [
            _resolve_subtype_index(
                classification,
                subtype,
            )
        ]

    records = []

    for subtype_index in subtype_indices:
        for transition in (
            "2to1",
            "2to3",
            "2to4",
            "2to5",
        ):
            records.append(
                get_transition_classification(
                    classification,
                    transition=transition,
                    subtype=subtype_index,
                )
            )

    return pd.DataFrame.from_records(records)


def classification_to_transition_df(
    fit_or_result: Any,
    subtype: int | str | None = None,
    thresholds: ClassificationThresholds | None = None,
) -> pd.DataFrame:
    """Backward-compatible tidy transition DataFrame wrapper."""
    return get_transition_df(
        fit_or_result,
        subtype=subtype,
        thresholds=thresholds,
    )


def summarize_response_classes(
    classified_dataframe: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Count response classes among classification-eligible fits."""
    dataframe = classified_dataframe

    if "classification_eligible" in dataframe.columns:
        eligible = dataframe["classification_eligible"].map(_optional_bool)
        dataframe = dataframe.loc[eligible.eq(True)].copy()

    columns = [
        column
        for column in dataframe.columns
        if re.fullmatch(r"response_class_s\d+", column)
    ]

    return {
        column: dataframe[column].value_counts(
            dropna=True
        )
        for column in columns
    }


def summarize_transition_patterns(
    classified_dataframe: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Count transition patterns among classification-eligible fits."""
    dataframe = classified_dataframe

    if "classification_eligible" in dataframe.columns:
        eligible = dataframe["classification_eligible"].map(_optional_bool)
        dataframe = dataframe.loc[eligible.eq(True)].copy()

    columns = [
        column
        for column in dataframe.columns
        if re.fullmatch(
            r"transition_2to[1345]_s\d+",
            column,
        )
    ]

    return {
        column: dataframe[column].value_counts(
            dropna=True
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


def get_rewiring_summary(
    fit_or_result: Any,
    thresholds: ClassificationThresholds | None = None,
) -> dict[str, Any]:
    """
    Return a compact subtype-rewiring summary.

    Parameters
    ----------
    fit_or_result
        A ``BDGDMFit`` object, a flat posterior-summary mapping,
        or an already classified mapping.

    thresholds
        Classification thresholds used only when classification
        has not already been performed.

    Returns
    -------
    dict
        Rewiring status and posterior contrast evidence.
    """
    classification = _coerce_classification(
        fit_or_result,
        thresholds=thresholds,
    )

    subtype_levels = normalize_subtype_levels(
        classification.get(
            "subtype_levels",
            classification.get(
                "subtype_order",
                None,
            ),
        )
    )

    return {
        "gene": classification.get("gene"),
        "status": classification.get("status"),
        "fit_flag": classification.get("fit_flag"),
        "classification_eligible": classification.get(
            "classification_eligible"
        ),
        "classification_exclusion_reason": classification.get(
            "classification_exclusion_reason"
        ),
        "analysis_mode": classification.get(
            "analysis_mode"
        ),
        "n_subtypes": classification.get(
            "S",
            classification.get(
                "n_subtypes",
                len(subtype_levels),
            ),
        ),
        "subtype_levels": subtype_levels,
        "subtype_levels_str": "|".join(
            subtype_levels
        ),

        # Overall rewiring interpretation
        "rewiring_status": classification.get(
            "rewiring_status"
        ),
        "scaling_rewired": classification.get(
            "scaling_rewired"
        ),
        "deviation_rewired": classification.get(
            "deviation_rewired"
        ),

        # Evidence source
        "scaling_rewiring_evidence": (
            classification.get(
                "scaling_rewiring_evidence"
            )
        ),
        "deviation_rewiring_evidence": (
            classification.get(
                "deviation_rewiring_evidence"
            )
        ),

        # Scaling contrast
        "delta_scaling_median": classification.get(
            "delta_scaling_median"
        ),
        "delta_scaling_q025": classification.get(
            "delta_scaling_q025"
        ),
        "delta_scaling_q975": classification.get(
            "delta_scaling_q975"
        ),
        "ppd_scaling": classification.get(
            "ppd_scaling"
        ),
        "p_rope_scaling": classification.get(
            "p_rope_scaling"
        ),

        # Deviation contrast
        "delta_dev_median": classification.get(
            "delta_dev_median"
        ),
        "delta_dev_q025": classification.get(
            "delta_dev_q025"
        ),
        "delta_dev_q975": classification.get(
            "delta_dev_q975"
        ),
        "ppd_dev": classification.get(
            "ppd_dev"
        ),
        "p_rope_dev": classification.get(
            "p_rope_dev"
        ),

        # Convenient complete label
        "summary_label": classification.get(
            "summary_label"
        ),
    }

def classification_to_rewiring_df(
    fit_or_result: Any,
    thresholds: ClassificationThresholds | None = None,
) -> pd.DataFrame:
    """
    Convert a rewiring summary into a one-row tidy DataFrame.

    Parameters
    ----------
    fit_or_result
        A ``BDGDMFit`` object, a posterior-summary mapping,
        or an already classified mapping.

    thresholds
        Classification thresholds used only when classification
        has not already been performed.

    Returns
    -------
    pandas.DataFrame
        One-row rewiring summary.
    """
    rewiring = get_rewiring_summary(
        fit_or_result,
        thresholds=thresholds,
    )

    return pd.DataFrame(
        [rewiring]
    )
