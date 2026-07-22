#!/usr/bin/env python3
"""
End-to-end simulation and benchmarking workflow for BDGDM and DeConveil.

The script provides:

1. A shared normal-plus-tumour negative-binomial RNA-seq simulator.
2. Known latent mechanisms: NEUTRAL, DSG, DIG, DCG, HYPER, and MIXED.
3. Method-specific oracle labels:
   - DeConveil: non-DEG, DSG, DIG, DCG;
   - BDGDM: DIG/no-dosage, DSG, DCG, HYPER, MIXED.
4. A collapsed common taxonomy for a fair head-to-head comparison.
5. Optional execution of:
   - DeConveil through a built-in independent-BH adapter;
   - DeConveil through a user-supplied adapter, e.g. the official stageR pipeline;
   - BDGDM through a user-supplied per-gene adapter.
6. Classification, parameter-recovery, and uncertainty metrics.
7. CSV outputs and manuscript-oriented Matplotlib figures.

DeConveil reference
-------------------
Davydzenka K, Caravagna G, Sanguinetti G.
Extending differential gene expression testing to handle genome aneuploidy
in cancer. PLoS Computational Biology. 2026;22(3):e1014134.
https://doi.org/10.1371/journal.pcbi.1014134

Important scientific distinction
--------------------------------
DeConveil and BDGDM do not estimate identical quantities.

DeConveil uses normal and tumour samples to estimate tumour-normal
differential expression after accounting for copy number. BDGDM uses tumour
samples only to estimate the within-tumour copy-number/expression response.

For this reason, the simulator stores:
- one latent generative class;
- a DeConveil-specific oracle label;
- a BDGDM-specific oracle label;
- a collapsed common label.

UNC is never simulated as biological truth. It is treated as an abstention
or uncertainty outcome.

External adapter contracts
--------------------------

DeConveil adapter
~~~~~~~~~~~~~~~~~
Pass ``--deconveil-adapter package.module:function``.

The callable must have the signature:

    adapter(dataset, context) -> pandas.DataFrame

The returned DataFrame must contain:
- ``gene`` or a gene index;
- ``deconveil_class``.

Recommended optional columns:
- ``naive_log2fc``;
- ``aware_log2fc``;
- ``naive_padj``;
- ``aware_padj``;
- ``confidence``.

This hook is the preferred way to connect the official omnibus/stageR
classification workflow.

BDGDM adapter
~~~~~~~~~~~~~
Pass ``--bdgdm-adapter package.module:function``.

The callable is invoked once per gene:

    adapter(gene_df, context) -> Mapping | pandas.Series

Required output:
- ``bdgdm_class`` or ``predicted_class``.

Recommended optional output:
- ``b_scaling_median``;
- ``b_scaling_q025``;
- ``b_scaling_q975``;
- ``b_deviation_median``;
- ``b_deviation_q025``;
- ``b_deviation_q975``;
- ``confidence``;
- ``status``.

The ``gene_df`` argument is tumour-only and contains:
- sample;
- gene;
- count and expression;
- copies;
- size_factor and sf;
- purity;
- subtype;
- truth columns.

The context mapping includes the scenario, replicate, seed, output directory,
and requested BDGDM engine.

Examples
--------

Pipeline smoke test without external methods:

    python benchmark_bdgdm_deconveil.py \
        --preset smoke \
        --demo-predictions \
        --output-dir benchmark_smoke \
        --overwrite

Development simulation plus built-in DeConveil:

    python benchmark_bdgdm_deconveil.py \
        --preset development \
        --run-deconveil \
        --output-dir benchmark_development

Run both methods with a custom BDGDM adapter:

    python benchmark_bdgdm_deconveil.py \
        --preset development \
        --run-deconveil \
        --bdgdm-adapter my_project.benchmark_adapter:run_one_gene \
        --output-dir benchmark_development

Manuscript-scale configurations can be very expensive. Test adapters with the
smoke preset before launching a large grid.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("bdgdm_benchmark")


LATENT_CLASS_ORDER: tuple[str, ...] = (
    "NEUTRAL",
    "DSG",
    "DIG",
    "DCG",
    "HYPER",
    "MIXED",
)

DECONVEIL_CLASS_ORDER: tuple[str, ...] = (
    "non-DEG",
    "DSG",
    "DIG",
    "DCG",
    "OTHER",
)

BDGDM_CLASS_ORDER: tuple[str, ...] = (
    "DIG",
    "DSG",
    "DCG",
    "HYPER",
    "MIXED",
    "UNC",
)

COMMON_CLASS_ORDER: tuple[str, ...] = (
    "NO_DOSAGE",
    "DOSAGE_SENSITIVE",
    "COMPENSATED",
    "COMPLEX",
    "UNC",
)

CLASS_PALETTE: dict[str, str] = {
    "NEUTRAL": "#7A7A7A",
    "non-DEG": "#7A7A7A",
    "NO_DOSAGE": "#7A7A7A",
    "DIG": "#E69F00",
    "DSG": "#009E73",
    "DOSAGE_SENSITIVE": "#009E73",
    "DCG": "#0072B2",
    "COMPENSATED": "#0072B2",
    "HYPER": "#D55E00",
    "MIXED": "#CC79A7",
    "COMPLEX": "#CC79A7",
    "UNC": "#BDBDBD",
    "OTHER": "#999999",
}


@dataclass(frozen=True)
class Scenario:
    """One simulation condition."""

    scenario_id: str
    n_genes: int
    n_normal: int
    n_tumor: int
    class_probabilities: Mapping[str, float]
    cn_signal: str = "moderate"
    purity_low: float = 1.0
    purity_high: float = 1.0
    cn_noise_rate: float = 0.0
    dispersion_scale: float = 1.0
    baseline_log_mean: float = float(np.log(100.0))
    baseline_log_sd: float = 1.0
    phi_log_mean: float = float(np.log(20.0))
    phi_log_sd: float = 0.50
    size_factor_log_sd: float = 0.25
    gain_probability: float = 0.75
    subtype_label: str = "simulated"

    def validate(self) -> None:
        if self.n_genes <= 0:
            raise ValueError("n_genes must be positive.")

        if self.n_normal <= 1 or self.n_tumor <= 1:
            raise ValueError(
                "n_normal and n_tumor must each be greater than one."
            )

        probabilities = {
            canonical_latent_class(key): float(value)
            for key, value in self.class_probabilities.items()
        }

        unknown = sorted(set(probabilities) - set(LATENT_CLASS_ORDER))
        if unknown:
            raise ValueError(
                "Unknown latent classes: " + ", ".join(unknown)
            )

        if any(value < 0 for value in probabilities.values()):
            raise ValueError("Class probabilities must be nonnegative.")

        if not np.isclose(sum(probabilities.values()), 1.0):
            raise ValueError("Class probabilities must sum to one.")

        if self.cn_signal not in {"weak", "moderate", "strong"}:
            raise ValueError(
                "cn_signal must be 'weak', 'moderate', or 'strong'."
            )

        if not 0 < self.purity_low <= self.purity_high <= 1:
            raise ValueError(
                "Purity bounds must satisfy 0 < low <= high <= 1."
            )

        if not 0 <= self.cn_noise_rate <= 1:
            raise ValueError("cn_noise_rate must lie in [0, 1].")

        if self.dispersion_scale <= 0:
            raise ValueError("dispersion_scale must be positive.")

        if not 0 <= self.gain_probability <= 1:
            raise ValueError("gain_probability must lie in [0, 1].")


@dataclass
class SimulatedDataset:
    """Shared simulation output for one scenario and replicate."""

    counts: pd.DataFrame
    copy_number_true: pd.DataFrame
    copy_number_observed: pd.DataFrame
    metadata: pd.DataFrame
    size_factors: pd.Series
    purity: pd.Series
    expected_counts: pd.DataFrame
    truth: pd.DataFrame
    bdgdm_long: pd.DataFrame
    scenario: Scenario
    replicate: int
    seed: int

    @property
    def normal_samples(self) -> pd.Index:
        return self.metadata.index[
            self.metadata["condition"].eq("normal")
        ]

    @property
    def tumor_samples(self) -> pd.Index:
        return self.metadata.index[
            self.metadata["condition"].eq("tumor")
        ]

    def validate(self) -> None:
        matrices = (
            self.copy_number_true,
            self.copy_number_observed,
            self.expected_counts,
        )

        for matrix in matrices:
            if not self.counts.index.equals(matrix.index):
                raise ValueError("Simulation sample indices are not aligned.")

            if not self.counts.columns.equals(matrix.columns):
                raise ValueError("Simulation gene columns are not aligned.")

        for vector in (
            self.size_factors,
            self.purity,
        ):
            if not self.counts.index.equals(vector.index):
                raise ValueError("Simulation sample vectors are not aligned.")

        if not self.counts.index.equals(self.metadata.index):
            raise ValueError("Metadata is not aligned with samples.")

        if not self.counts.columns.equals(self.truth.index):
            raise ValueError("Truth is not aligned with genes.")

    def deconveil_inputs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self.counts.copy(),
            self.copy_number_observed.copy(),
            self.metadata.copy(),
        )

    def bdgdm_gene_data(
        self,
        gene: str,
    ) -> pd.DataFrame:
        gene_data = self.bdgdm_long.loc[
            self.bdgdm_long["gene"].eq(gene)
        ].copy()

        if gene_data.empty:
            raise KeyError(f"Unknown simulated gene: {gene!r}")

        return gene_data.reset_index(drop=True)


@dataclass
class BenchmarkOutputs:
    """Tables produced by the benchmark runner."""

    metrics: list[dict[str, Any]] = field(default_factory=list)
    per_class_metrics: list[dict[str, Any]] = field(default_factory=list)
    parameter_metrics: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[pd.DataFrame] = field(default_factory=list)
    risk_coverage: list[pd.DataFrame] = field(default_factory=list)


def canonical_latent_class(value: Any) -> str:
    text = str(value).strip().upper()

    aliases = {
        "MIXED": "MIXED",
        "MIX": "MIXED",
        "NONE": "NEUTRAL",
        "NULL": "NEUTRAL",
        "NON-DEG": "NEUTRAL",
    }

    return aliases.get(text, text)


def canonical_deconveil_class(value: Any) -> str:
    if value is None or pd.isna(value):
        return "OTHER"

    text = str(value).strip().casefold()

    aliases = {
        "dsg": "DSG",
        "dsgs": "DSG",
        "dosage-sensitive": "DSG",
        "dig": "DIG",
        "digs": "DIG",
        "dosage-insensitive": "DIG",
        "dcg": "DCG",
        "dcgs": "DCG",
        "dosage-compensated": "DCG",
        "non-deg": "non-DEG",
        "non-degs": "non-DEG",
        "non_deg": "non-DEG",
        "nonde": "non-DEG",
        "n.s.": "non-DEG",
        "ns": "non-DEG",
        "neutral": "non-DEG",
        "other": "OTHER",
        "unc": "OTHER",
        "unknown": "OTHER",
    }

    return aliases.get(text, str(value).strip())


def canonical_bdgdm_class(value: Any) -> str:
    if value is None or pd.isna(value):
        return "UNC"

    text = str(value).strip().casefold()

    aliases = {
        "dsg": "DSG",
        "dcg": "DCG",
        "hyper": "HYPER",
        "hyper-responsive": "HYPER",
        "mixed": "MIXED",
        "mix": "MIXED",
        "dig": "DIG",
        "no_dosage": "DIG",
        "no dosage": "DIG",
        "neutral": "DIG",
        "unc": "UNC",
        "uncertain": "UNC",
        "unknown": "UNC",
        "unclassified": "UNC",
    }

    return aliases.get(text, str(value).strip().upper())


def latent_to_deconveil_truth(value: Any) -> str:
    latent = canonical_latent_class(value)

    mapping = {
        "NEUTRAL": "non-DEG",
        "DSG": "DSG",
        "DIG": "DIG",
        "DCG": "DCG",
        "HYPER": "DSG",
        "MIXED": "OTHER",
    }

    return mapping[latent]


def latent_to_bdgdm_truth(value: Any) -> str:
    latent = canonical_latent_class(value)

    mapping = {
        "NEUTRAL": "DIG",
        "DSG": "DSG",
        "DIG": "DIG",
        "DCG": "DCG",
        "HYPER": "HYPER",
        "MIXED": "MIXED",
    }

    return mapping[latent]


def latent_to_common_truth(value: Any) -> str:
    latent = canonical_latent_class(value)

    mapping = {
        "NEUTRAL": "NO_DOSAGE",
        "DIG": "NO_DOSAGE",
        "DSG": "DOSAGE_SENSITIVE",
        "HYPER": "DOSAGE_SENSITIVE",
        "DCG": "COMPENSATED",
        "MIXED": "COMPLEX",
    }

    return mapping[latent]


def deconveil_to_common(value: Any) -> str:
    predicted = canonical_deconveil_class(value)

    mapping = {
        "non-DEG": "NO_DOSAGE",
        "DIG": "NO_DOSAGE",
        "DSG": "DOSAGE_SENSITIVE",
        "DCG": "COMPENSATED",
        "OTHER": "UNC",
    }

    return mapping.get(predicted, "UNC")


def bdgdm_to_common(value: Any) -> str:
    predicted = canonical_bdgdm_class(value)

    mapping = {
        "DIG": "NO_DOSAGE",
        "DSG": "DOSAGE_SENSITIVE",
        "HYPER": "DOSAGE_SENSITIVE",
        "DCG": "COMPENSATED",
        "MIXED": "COMPLEX",
        "UNC": "UNC",
    }

    return mapping.get(predicted, "UNC")


def parse_csv_values(
    text: str,
    cast: Callable[[str], Any],
) -> list[Any]:
    values = [
        item.strip()
        for item in str(text).split(",")
        if item.strip()
    ]

    if not values:
        raise ValueError("Expected at least one comma-separated value.")

    return [cast(value) for value in values]


def parse_class_probabilities(
    text: str | None,
    *,
    extended: bool,
) -> dict[str, float]:
    if text is None:
        if extended:
            return {
                "NEUTRAL": 0.15,
                "DSG": 0.20,
                "DIG": 0.15,
                "DCG": 0.20,
                "HYPER": 0.15,
                "MIXED": 0.15,
            }

        return {
            "NEUTRAL": 0.25,
            "DSG": 0.25,
            "DIG": 0.25,
            "DCG": 0.25,
            "HYPER": 0.0,
            "MIXED": 0.0,
        }

    result: dict[str, float] = {}

    for item in text.split(","):
        if "=" not in item:
            raise ValueError(
                "Class probabilities must use CLASS=value entries."
            )

        key, value = item.split("=", 1)
        result[canonical_latent_class(key)] = float(value)

    for class_label in LATENT_CLASS_ORDER:
        result.setdefault(class_label, 0.0)

    if not np.isclose(sum(result.values()), 1.0):
        raise ValueError("Class probabilities must sum to one.")

    return result


def safe_scenario_id(text: str) -> str:
    return (
        str(text)
        .replace(".", "p")
        .replace(" ", "_")
    )


def geometric_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    if np.any(values <= 0):
        raise ValueError("Geometric mean requires positive values.")

    return float(np.exp(np.mean(np.log(values))))


def cn_state_distribution(
    signal: str,
    *,
    gain: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if gain:
        if signal == "weak":
            return (
                np.asarray([2, 3, 4]),
                np.asarray([0.70, 0.25, 0.05]),
            )

        if signal == "moderate":
            return (
                np.asarray([2, 3, 4]),
                np.asarray([0.40, 0.45, 0.15]),
            )

        return (
            np.asarray([2, 3, 4, 5]),
            np.asarray([0.20, 0.35, 0.30, 0.15]),
        )

    if signal == "weak":
        return (
            np.asarray([1, 2]),
            np.asarray([0.20, 0.80]),
        )

    if signal == "moderate":
        return (
            np.asarray([1, 2]),
            np.asarray([0.45, 0.55]),
        )

    return (
        np.asarray([1, 2]),
        np.asarray([0.70, 0.30]),
    )


def guarantee_cn_variation(
    values: np.ndarray,
    *,
    gain: bool,
) -> np.ndarray:
    result = np.asarray(values, dtype=int).copy()

    if np.unique(result).size >= 2:
        return result

    result[0] = 3 if gain else 1
    result[1] = 2
    return result


def simulate_true_cn(
    rng: np.random.Generator,
    latent_classes: np.ndarray,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    cn = np.full(
        (scenario.n_tumor, scenario.n_genes),
        2,
        dtype=int,
    )
    direction = np.full(
        scenario.n_genes,
        "diploid",
        dtype=object,
    )

    for gene_index, latent_class in enumerate(latent_classes):
        if latent_class in {"NEUTRAL", "DIG"}:
            continue

        if latent_class == "MIXED":
            if scenario.cn_signal == "weak":
                states = np.asarray([1, 2, 3])
                probabilities = np.asarray([0.15, 0.60, 0.25])
            elif scenario.cn_signal == "moderate":
                states = np.asarray([1, 2, 3, 4])
                probabilities = np.asarray([0.20, 0.30, 0.35, 0.15])
            else:
                states = np.asarray([1, 2, 3, 4, 5])
                probabilities = np.asarray([0.20, 0.15, 0.25, 0.25, 0.15])

            values = rng.choice(
                states,
                size=scenario.n_tumor,
                replace=True,
                p=probabilities,
            )

            if not np.any(values < 2):
                values[0] = 1

            if not np.any(values > 2):
                values[1] = 3

            cn[:, gene_index] = values
            direction[gene_index] = "mixed"
            continue

        gain = bool(rng.random() < scenario.gain_probability)
        states, probabilities = cn_state_distribution(
            scenario.cn_signal,
            gain=gain,
        )
        values = rng.choice(
            states,
            size=scenario.n_tumor,
            replace=True,
            p=probabilities,
        )
        cn[:, gene_index] = guarantee_cn_variation(
            values,
            gain=gain,
        )
        direction[gene_index] = "gain" if gain else "loss"

    return cn, direction


def perturb_copy_number(
    rng: np.random.Generator,
    true_cn: np.ndarray,
    *,
    noise_rate: float,
    minimum: int = 1,
    maximum: int = 6,
) -> np.ndarray:
    observed = np.asarray(true_cn, dtype=int).copy()

    if noise_rate <= 0:
        return observed

    mask = rng.random(observed.shape) < noise_rate
    possible_noise = np.asarray([-2, -1, 1, 2])
    changes = rng.choice(
        possible_noise,
        size=observed.shape,
        replace=True,
    )

    observed[mask] = np.clip(
        observed[mask] + changes[mask],
        minimum,
        maximum,
    )
    return observed


def nb2_sample(
    rng: np.random.Generator,
    mean: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if not np.isfinite(mean).all() or np.any(mean < 0):
        raise ValueError("NB means must be finite and nonnegative.")

    if not np.isfinite(phi).all() or np.any(phi <= 0):
        raise ValueError("NB precision must be finite and positive.")

    probability = phi / (phi + mean)

    return rng.negative_binomial(
        phi,
        probability,
    )


def sample_gene_parameters(
    rng: np.random.Generator,
    latent_classes: np.ndarray,
    tumor_cn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_genes = len(latent_classes)
    b_scaling = np.zeros(n_genes, dtype=float)
    b_deviation = np.zeros(n_genes, dtype=float)
    delta_log2 = np.zeros(n_genes, dtype=float)

    for gene_index, latent_class in enumerate(latent_classes):
        cn_values = tumor_cn[:, gene_index].astype(float)
        dose = np.log2(
            np.maximum(cn_values, 1.0) / 2.0
        )
        deviation = np.abs(cn_values - 2.0)

        if latent_class == "NEUTRAL":
            continue

        if latent_class == "DIG":
            sign = rng.choice(np.asarray([-1.0, 1.0]))
            delta_log2[gene_index] = (
                sign * rng.uniform(1.20, 2.00)
            )
            continue

        if latent_class == "DSG":
            b_scaling[gene_index] = rng.uniform(0.90, 1.10)
            b_deviation[gene_index] = rng.normal(0.0, 0.03)
            continue

        if latent_class == "HYPER":
            b_scaling[gene_index] = rng.uniform(1.30, 1.80)
            b_deviation[gene_index] = rng.normal(0.0, 0.03)
            continue

        if latent_class == "DCG":
            b_scaling[gene_index] = rng.uniform(0.15, 0.55)
            b_deviation[gene_index] = rng.normal(0.0, 0.03)

            response_log2 = (
                b_scaling[gene_index] * dose
                + b_deviation[gene_index] * deviation
            )
            delta_log2[gene_index] = -np.log2(
                np.mean(
                    np.exp2(response_log2)
                )
            )
            continue

        if latent_class == "MIXED":
            b_scaling[gene_index] = rng.uniform(0.65, 1.15)
            magnitude = rng.uniform(0.25, 0.55)
            b_deviation[gene_index] = (
                magnitude
                if rng.random() < 0.5
                else -magnitude
            )

            # Keep the average tumour-normal shift modest so that the mixed
            # transition pattern, rather than a large global shift, dominates.
            response_log2 = (
                b_scaling[gene_index] * dose
                + b_deviation[gene_index] * deviation
            )
            delta_log2[gene_index] = (
                -0.35
                * np.log2(
                    np.mean(
                        np.exp2(response_log2)
                    )
                )
            )
            continue

        raise ValueError(f"Unknown latent class: {latent_class}")

    return b_scaling, b_deviation, delta_log2


def transition_effect_log2(
    copy_number: float,
    *,
    b_scaling: np.ndarray,
    b_deviation: np.ndarray,
) -> np.ndarray:
    dose = np.log2(
        np.maximum(float(copy_number), 1.0) / 2.0
    )
    deviation = abs(float(copy_number) - 2.0)

    return (
        b_scaling * dose
        + b_deviation * deviation
    )


def simulate_dataset(
    scenario: Scenario,
    *,
    replicate: int,
    seed: int,
) -> SimulatedDataset:
    scenario.validate()
    rng = np.random.default_rng(seed)

    genes = pd.Index(
        [
            f"SIM_GENE_{index:05d}"
            for index in range(scenario.n_genes)
        ],
        name="gene",
    )
    normal_samples = pd.Index(
        [
            f"NORMAL_{index:03d}"
            for index in range(scenario.n_normal)
        ],
        name="sample",
    )
    tumor_samples = pd.Index(
        [
            f"TUMOR_{index:03d}"
            for index in range(scenario.n_tumor)
        ],
        name="sample",
    )
    samples = normal_samples.append(tumor_samples)

    probability_vector = np.asarray(
        [
            float(
                scenario.class_probabilities.get(
                    class_label,
                    0.0,
                )
            )
            for class_label in LATENT_CLASS_ORDER
        ]
    )
    latent_classes = rng.choice(
        np.asarray(LATENT_CLASS_ORDER, dtype=object),
        size=scenario.n_genes,
        replace=True,
        p=probability_vector,
    )

    tumor_cn_true, cn_direction = simulate_true_cn(
        rng,
        latent_classes,
        scenario,
    )
    tumor_cn_observed = perturb_copy_number(
        rng,
        tumor_cn_true,
        noise_rate=scenario.cn_noise_rate,
    )

    normal_cn = np.full(
        (scenario.n_normal, scenario.n_genes),
        2,
        dtype=int,
    )
    cn_true = np.vstack(
        [
            normal_cn,
            tumor_cn_true,
        ]
    )
    cn_observed = np.vstack(
        [
            normal_cn,
            tumor_cn_observed,
        ]
    )

    baseline = rng.lognormal(
        mean=scenario.baseline_log_mean,
        sigma=scenario.baseline_log_sd,
        size=scenario.n_genes,
    )
    phi = (
        rng.lognormal(
            mean=scenario.phi_log_mean,
            sigma=scenario.phi_log_sd,
            size=scenario.n_genes,
        )
        / scenario.dispersion_scale
    )
    phi = np.clip(phi, 0.25, None)

    b_scaling, b_deviation, delta_log2 = sample_gene_parameters(
        rng,
        latent_classes,
        tumor_cn_true,
    )

    raw_size_factors = rng.lognormal(
        mean=0.0,
        sigma=scenario.size_factor_log_sd,
        size=len(samples),
    )
    raw_size_factors /= geometric_mean(raw_size_factors)
    size_factors = pd.Series(
        raw_size_factors,
        index=samples,
        name="size_factor",
    )

    normal_purity = np.zeros(
        scenario.n_normal,
        dtype=float,
    )
    tumor_purity = rng.uniform(
        scenario.purity_low,
        scenario.purity_high,
        size=scenario.n_tumor,
    )
    purity = pd.Series(
        np.concatenate(
            [
                normal_purity,
                tumor_purity,
            ]
        ),
        index=samples,
        name="purity",
    )

    normal_component = np.broadcast_to(
        baseline[None, :],
        (scenario.n_normal, scenario.n_genes),
    )

    tumor_cn_float = tumor_cn_true.astype(float)
    tumor_dose = np.log2(
        np.maximum(tumor_cn_float, 1.0) / 2.0
    )
    tumor_deviation = np.abs(
        tumor_cn_float - 2.0
    )

    tumor_response_log2 = (
        delta_log2[None, :]
        + b_scaling[None, :] * tumor_dose
        + b_deviation[None, :] * tumor_deviation
    )
    tumor_cell_component = (
        baseline[None, :]
        * np.exp2(tumor_response_log2)
    )
    tumor_bulk_component = (
        tumor_purity[:, None] * tumor_cell_component
        + (1.0 - tumor_purity[:, None])
        * baseline[None, :]
    )

    expression_component = np.vstack(
        [
            normal_component,
            tumor_bulk_component,
        ]
    )
    expected = (
        size_factors.to_numpy()[:, None]
        * expression_component
    )

    phi_matrix = np.broadcast_to(
        phi[None, :],
        expected.shape,
    )
    counts_array = nb2_sample(
        rng,
        expected,
        phi_matrix,
    )

    counts = pd.DataFrame(
        counts_array,
        index=samples,
        columns=genes,
        dtype=int,
    )
    copy_number_true = pd.DataFrame(
        cn_true,
        index=samples,
        columns=genes,
        dtype=int,
    )
    copy_number_observed = pd.DataFrame(
        cn_observed,
        index=samples,
        columns=genes,
        dtype=int,
    )
    expected_counts = pd.DataFrame(
        expected,
        index=samples,
        columns=genes,
    )
    metadata = pd.DataFrame(
        {
            "condition": (
                ["normal"] * scenario.n_normal
                + ["tumor"] * scenario.n_tumor
            ),
        },
        index=samples,
    )

    pure_tumor_factor = np.exp2(
        tumor_response_log2
    )
    true_naive_log2fc = np.log2(
        np.mean(
            pure_tumor_factor,
            axis=0,
        )
    )
    true_regulatory_log2fc = delta_log2.copy()
    true_cn_contribution_log2 = (
        true_naive_log2fc
        - true_regulatory_log2fc
    )

    transition_2to1 = transition_effect_log2(
        1,
        b_scaling=b_scaling,
        b_deviation=b_deviation,
    )
    transition_2to3 = transition_effect_log2(
        3,
        b_scaling=b_scaling,
        b_deviation=b_deviation,
    )
    transition_2to4 = transition_effect_log2(
        4,
        b_scaling=b_scaling,
        b_deviation=b_deviation,
    )

    truth = pd.DataFrame(
        {
            "truth_class_latent": latent_classes,
            "truth_class_deconveil": [
                latent_to_deconveil_truth(value)
                for value in latent_classes
            ],
            "truth_class_bdgdm": [
                latent_to_bdgdm_truth(value)
                for value in latent_classes
            ],
            "truth_class_common": [
                latent_to_common_truth(value)
                for value in latent_classes
            ],
            "cn_direction": cn_direction,
            "true_baseline_mean": baseline,
            "true_phi": phi,
            "true_b_scaling": b_scaling,
            "true_b_deviation": b_deviation,
            "true_delta_log2": delta_log2,
            "true_naive_log2fc": true_naive_log2fc,
            "true_regulatory_log2fc": true_regulatory_log2fc,
            "true_cn_contribution_log2": true_cn_contribution_log2,
            "true_transition_2to1": transition_2to1,
            "true_transition_2to3": transition_2to3,
            "true_transition_2to4": transition_2to4,
            "mean_tumor_cn_true": tumor_cn_true.mean(axis=0),
            "sd_tumor_cn_true": tumor_cn_true.std(axis=0),
            "n_unique_tumor_cn_true": np.apply_along_axis(
                lambda values: np.unique(values).size,
                axis=0,
                arr=tumor_cn_true,
            ),
            "mean_tumor_cn_observed": tumor_cn_observed.mean(axis=0),
            "sd_tumor_cn_observed": tumor_cn_observed.std(axis=0),
        },
        index=genes,
    )

    tumor_counts_long = (
        counts.loc[tumor_samples]
        .rename_axis(index="sample", columns="gene")
        .stack()
        .rename("count")
        .reset_index()
    )
    tumor_cn_long = (
        copy_number_observed.loc[tumor_samples]
        .rename_axis(index="sample", columns="gene")
        .stack()
        .rename("copies")
        .reset_index()
    )

    bdgdm_long = tumor_counts_long.merge(
        tumor_cn_long,
        on=[
            "sample",
            "gene",
        ],
        how="inner",
        validate="one_to_one",
    )
    bdgdm_long["expression"] = bdgdm_long["count"]
    bdgdm_long["size_factor"] = bdgdm_long[
        "sample"
    ].map(size_factors)
    bdgdm_long["sf"] = bdgdm_long["size_factor"]
    bdgdm_long["purity"] = bdgdm_long[
        "sample"
    ].map(purity)
    bdgdm_long["subtype"] = scenario.subtype_label
    bdgdm_long["condition"] = "tumor"
    bdgdm_long = bdgdm_long.merge(
        truth[
            [
                "truth_class_latent",
                "truth_class_bdgdm",
                "true_b_scaling",
                "true_b_deviation",
                "true_delta_log2",
                "true_transition_2to1",
                "true_transition_2to3",
                "true_transition_2to4",
            ]
        ].reset_index(),
        on="gene",
        how="left",
        validate="many_to_one",
    )

    dataset = SimulatedDataset(
        counts=counts,
        copy_number_true=copy_number_true,
        copy_number_observed=copy_number_observed,
        metadata=metadata,
        size_factors=size_factors,
        purity=purity,
        expected_counts=expected_counts,
        truth=truth,
        bdgdm_long=bdgdm_long,
        scenario=scenario,
        replicate=replicate,
        seed=seed,
    )
    dataset.validate()
    return dataset


def write_dataset(
    dataset: SimulatedDataset,
    directory: Path,
) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.counts.to_csv(
        directory / "counts.csv.gz"
    )
    dataset.copy_number_true.to_csv(
        directory / "copy_number_true.csv.gz"
    )
    dataset.copy_number_observed.to_csv(
        directory / "copy_number_observed.csv.gz"
    )
    dataset.metadata.to_csv(
        directory / "metadata.csv"
    )
    dataset.size_factors.to_csv(
        directory / "size_factors.csv"
    )
    dataset.purity.to_csv(
        directory / "purity.csv"
    )
    dataset.expected_counts.to_csv(
        directory / "expected_counts.csv.gz"
    )
    dataset.truth.to_csv(
        directory / "truth.csv"
    )
    dataset.bdgdm_long.to_csv(
        directory / "bdgdm_tumor_long.csv.gz",
        index=False,
    )

    metadata = {
        "scenario": asdict(dataset.scenario),
        "replicate": dataset.replicate,
        "seed": dataset.seed,
    }

    with (
        directory / "simulation_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            default=float,
        )


def load_callable(
    specification: str,
) -> Callable[..., Any]:
    if ":" not in specification:
        raise ValueError(
            "Adapter must use 'package.module:function' syntax."
        )

    module_name, function_name = specification.split(
        ":",
        1,
    )
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    if not callable(function):
        raise TypeError(
            f"Adapter {specification!r} is not callable."
        )

    return function


def normalize_prediction_frame(
    value: Any,
    *,
    required_class_column: str,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, pd.Series):
        frame = value.to_frame().T
    elif isinstance(value, Mapping):
        frame = pd.DataFrame([dict(value)])
    else:
        raise TypeError(
            "Method output must be a DataFrame, Series, or mapping."
        )

    if "gene" not in frame.columns:
        if frame.index.name == "gene":
            frame = frame.reset_index()
        else:
            frame = frame.reset_index().rename(
                columns={
                    frame.index.name or "index": "gene",
                }
            )

    if required_class_column not in frame.columns:
        alternative = (
            "predicted_class"
            if "predicted_class" in frame.columns
            else None
        )

        if alternative is None:
            raise KeyError(
                f"Method output must contain {required_class_column!r} "
                "or 'predicted_class'."
            )

        frame = frame.rename(
            columns={
                alternative: required_class_column,
            }
        )

    frame["gene"] = frame["gene"].astype(str)
    return frame


def run_deconveil_builtin(
    dataset: SimulatedDataset,
    context: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Run CN-aware and constant-diploid CN-naive DeConveil fits.

    This built-in classification uses independent adjusted p-values and
    the same logical class definitions as ``define_gene_groups``. It is
    appropriate for development and smoke testing.

    For final manuscript analysis, connect the official omnibus/stageR
    workflow through ``--deconveil-adapter``.
    """
    try:
        from deconveil import deconveil_fit, deconveil_stats
    except ImportError as exc:
        raise ImportError(
            "DeConveil is not installed. Install deconveil and its "
            "required R/stageR dependencies."
        ) from exc

    counts, cnv, metadata = dataset.deconveil_inputs()
    alpha = float(context["deconveil_alpha"])
    lfc_threshold = float(context["deconveil_lfc_threshold"])
    n_cpus = context.get("n_cpus")
    quiet = bool(context.get("quiet", True))

    def fit_one(
        cnv_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        dds = deconveil_fit(
            counts=counts,
            cnv=cnv_matrix,
            metadata=metadata,
            design="~condition",
            n_cpus=n_cpus,
            quiet=quiet,
        )
        dds.deseq2()

        stats = deconveil_stats(
            dds,
            contrast=[
                "condition",
                "tumor",
                "normal",
            ],
            alpha=alpha,
            quiet=quiet,
        )
        stats.summary()

        result = stats.results_df.copy()
        result.index.name = "gene"
        return result

    aware = fit_one(cnv)
    diploid = pd.DataFrame(
        2,
        index=cnv.index,
        columns=cnv.columns,
        dtype=int,
    )
    naive = fit_one(diploid)

    common_genes = naive.index.intersection(
        aware.index
    )
    naive = naive.loc[common_genes]
    aware = aware.loc[common_genes]

    naive_de = (
        naive["padj"].fillna(1.0).lt(alpha)
        & naive["log2FoldChange"]
        .abs()
        .ge(lfc_threshold)
    )
    aware_de = (
        aware["padj"].fillna(1.0).lt(alpha)
        & aware["log2FoldChange"]
        .abs()
        .ge(lfc_threshold)
    )
    same_direction = (
        np.sign(naive["log2FoldChange"])
        == np.sign(aware["log2FoldChange"])
    )

    predicted = np.select(
        [
            naive_de & ~aware_de,
            naive_de & aware_de & same_direction,
            ~naive_de & aware_de,
            ~naive_de & ~aware_de,
        ],
        [
            "DSG",
            "DIG",
            "DCG",
            "non-DEG",
        ],
        default="OTHER",
    )

    result = pd.DataFrame(
        {
            "gene": common_genes.astype(str),
            "deconveil_class": predicted,
            "naive_log2fc": naive[
                "log2FoldChange"
            ].to_numpy(),
            "aware_log2fc": aware[
                "log2FoldChange"
            ].to_numpy(),
            "naive_padj": naive[
                "padj"
            ].to_numpy(),
            "aware_padj": aware[
                "padj"
            ].to_numpy(),
            "deconveil_pipeline": (
                "builtin_independent_bh"
            ),
        }
    )
    return result


def call_deconveil_adapter(
    adapter: Callable[..., Any],
    dataset: SimulatedDataset,
    context: Mapping[str, Any],
) -> pd.DataFrame:
    result = adapter(
        dataset,
        context,
    )
    return normalize_prediction_frame(
        result,
        required_class_column="deconveil_class",
    )


def call_bdgdm_adapter(
    adapter: Callable[..., Any],
    dataset: SimulatedDataset,
    context: Mapping[str, Any],
    *,
    max_genes: int | None,
) -> pd.DataFrame:
    genes = dataset.truth.index.astype(str)

    if max_genes is not None:
        genes = genes[:max_genes]

    rows: list[dict[str, Any]] = []

    for gene_number, gene in enumerate(genes, start=1):
        gene_data = dataset.bdgdm_gene_data(gene)
        gene_context = dict(context)
        gene_context.update(
            {
                "gene": gene,
                "gene_number": gene_number,
                "n_genes_requested": len(genes),
                "truth": dataset.truth.loc[gene].to_dict(),
            }
        )

        start = time.perf_counter()

        try:
            raw_result = adapter(
                gene_data,
                gene_context,
            )

            if isinstance(raw_result, pd.Series):
                row = raw_result.to_dict()
            elif isinstance(raw_result, Mapping):
                row = dict(raw_result)
            else:
                raise TypeError(
                    "BDGDM adapter must return a mapping or Series."
                )

            row.setdefault("status", "ok")

        except Exception as exc:
            LOGGER.exception(
                "BDGDM failed for %s",
                gene,
            )
            row = {
                "status": "failed",
                "error": str(exc),
                "bdgdm_class": "UNC",
            }

        elapsed = time.perf_counter() - start
        row["gene"] = gene
        row["runtime_seconds"] = elapsed

        if "bdgdm_class" not in row:
            if "predicted_class" in row:
                row["bdgdm_class"] = row.pop(
                    "predicted_class"
                )
            else:
                row["bdgdm_class"] = "UNC"

        rows.append(row)

    return pd.DataFrame(rows)


def demo_method_predictions(
    dataset: SimulatedDataset,
    *,
    method: str,
    seed: int,
) -> pd.DataFrame:
    """
    Generate noisy oracle-like predictions for pipeline testing only.

    These outputs are labelled ``demo_only=True`` and must never be used
    in scientific results.
    """
    rng = np.random.default_rng(seed)
    truth = dataset.truth.copy()

    information = (
        math.log2(dataset.scenario.n_tumor + 1)
        / math.log2(61)
    )
    purity_factor = (
        dataset.scenario.purity_low
        + dataset.scenario.purity_high
    ) / 2
    noise_penalty = dataset.scenario.cn_noise_rate

    base_accuracy = np.clip(
        0.55
        + 0.35 * information
        + 0.10 * purity_factor
        - 0.55 * noise_penalty,
        0.40,
        0.96,
    )

    if method == "deconveil":
        correct = truth[
            "truth_class_deconveil"
        ].astype(str).to_numpy()
        choices = np.asarray(
            [
                "non-DEG",
                "DSG",
                "DIG",
                "DCG",
                "OTHER",
            ],
            dtype=object,
        )
        predicted = correct.copy()

        for index in range(len(predicted)):
            if rng.random() > base_accuracy:
                alternatives = choices[
                    choices != correct[index]
                ]
                predicted[index] = rng.choice(
                    alternatives
                )

        return pd.DataFrame(
            {
                "gene": truth.index.astype(str),
                "deconveil_class": predicted,
                "aware_log2fc": (
                    truth[
                        "true_regulatory_log2fc"
                    ].to_numpy()
                    + rng.normal(
                        0,
                        0.25,
                        size=len(truth),
                    )
                ),
                "naive_log2fc": (
                    truth[
                        "true_naive_log2fc"
                    ].to_numpy()
                    + rng.normal(
                        0,
                        0.30,
                        size=len(truth),
                    )
                ),
                "confidence": rng.uniform(
                    0.55,
                    0.99,
                    size=len(truth),
                ),
                "demo_only": True,
            }
        )

    if method == "bdgdm":
        correct = truth[
            "truth_class_bdgdm"
        ].astype(str).to_numpy()
        choices = np.asarray(
            [
                "DIG",
                "DSG",
                "DCG",
                "HYPER",
                "MIXED",
            ],
            dtype=object,
        )
        predicted = correct.copy()
        confidence = rng.uniform(
            0.55,
            0.99,
            size=len(truth),
        )

        difficult = (
            truth["sd_tumor_cn_observed"].to_numpy()
            < 0.25
        )
        abstention_probability = np.clip(
            0.04
            + 0.35 * difficult.astype(float)
            + 0.45 * noise_penalty
            + 0.20 * (1.0 - purity_factor),
            0.02,
            0.65,
        )

        for index in range(len(predicted)):
            if rng.random() < abstention_probability[index]:
                predicted[index] = "UNC"
                confidence[index] = rng.uniform(
                    0.05,
                    0.45,
                )
            elif rng.random() > base_accuracy:
                alternatives = choices[
                    choices != correct[index]
                ]
                predicted[index] = rng.choice(
                    alternatives
                )
                confidence[index] = rng.uniform(
                    0.35,
                    0.75,
                )

        return pd.DataFrame(
            {
                "gene": truth.index.astype(str),
                "bdgdm_class": predicted,
                "b_scaling_median": (
                    truth["true_b_scaling"].to_numpy()
                    + rng.normal(
                        0,
                        0.12,
                        size=len(truth),
                    )
                ),
                "b_scaling_q025": (
                    truth["true_b_scaling"].to_numpy()
                    - 0.25
                ),
                "b_scaling_q975": (
                    truth["true_b_scaling"].to_numpy()
                    + 0.25
                ),
                "b_deviation_median": (
                    truth[
                        "true_b_deviation"
                    ].to_numpy()
                    + rng.normal(
                        0,
                        0.10,
                        size=len(truth),
                    )
                ),
                "b_deviation_q025": (
                    truth[
                        "true_b_deviation"
                    ].to_numpy()
                    - 0.22
                ),
                "b_deviation_q975": (
                    truth[
                        "true_b_deviation"
                    ].to_numpy()
                    + 0.22
                ),
                "confidence": confidence,
                "demo_only": True,
            }
        )

    raise ValueError(f"Unknown demo method: {method}")


def aligned_labels(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    exclude_truth: Iterable[str] = (),
) -> pd.DataFrame:
    frame = pd.concat(
        [
            truth.rename("truth"),
            prediction.rename("prediction"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    excluded = set(exclude_truth)

    if excluded:
        frame = frame.loc[
            ~frame["truth"].isin(excluded)
        ].copy()

    return frame


def confusion_matrix_frame(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    labels: Sequence[str] | None = None,
    exclude_truth: Iterable[str] = (),
) -> pd.DataFrame:
    frame = aligned_labels(
        truth,
        prediction,
        exclude_truth=exclude_truth,
    )

    matrix = pd.crosstab(
        frame["truth"],
        frame["prediction"],
    )

    if labels is not None:
        ordered = list(labels)
        extra_rows = [
            value
            for value in matrix.index
            if value not in ordered
        ]
        extra_columns = [
            value
            for value in matrix.columns
            if value not in ordered
        ]
        matrix = matrix.reindex(
            index=ordered + extra_rows,
            columns=ordered + extra_columns,
            fill_value=0,
        )

    return matrix


def multiclass_mcc(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    sample_count = matrix.sum()

    if sample_count <= 0:
        return float("nan")

    correct = np.trace(matrix)
    predicted_totals = matrix.sum(axis=0)
    truth_totals = matrix.sum(axis=1)

    numerator = (
        correct * sample_count
        - np.dot(
            predicted_totals,
            truth_totals,
        )
    )
    denominator = math.sqrt(
        (
            sample_count**2
            - np.dot(
                predicted_totals,
                predicted_totals,
            )
        )
        * (
            sample_count**2
            - np.dot(
                truth_totals,
                truth_totals,
            )
        )
    )

    if denominator <= 0:
        return float("nan")

    return float(numerator / denominator)


def classification_metrics(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    class_order: Sequence[str],
    uncertainty_label: str | None = None,
    exclude_truth: Iterable[str] = (),
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    frame = aligned_labels(
        truth,
        prediction,
        exclude_truth=exclude_truth,
    )

    if frame.empty:
        empty = {
            "n": 0.0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "mcc": float("nan"),
            "coverage": float("nan"),
            "selective_accuracy": float("nan"),
            "wrong_confident_rate": float("nan"),
        }
        return (
            empty,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    observed_labels = list(
        dict.fromkeys(
            list(class_order)
            + frame["truth"].astype(str).tolist()
            + frame["prediction"].astype(str).tolist()
        )
    )

    matrix = confusion_matrix_frame(
        frame["truth"],
        frame["prediction"],
        labels=observed_labels,
    )
    matrix_values = matrix.to_numpy(dtype=float)
    total = matrix_values.sum()
    accuracy = (
        float(np.trace(matrix_values) / total)
        if total > 0
        else float("nan")
    )

    per_class_rows: list[dict[str, Any]] = []

    for label in matrix.index:
        if uncertainty_label is not None and label == uncertainty_label:
            continue

        true_positive = float(
            matrix.loc[label, label]
            if label in matrix.columns
            else 0.0
        )
        false_negative = float(
            matrix.loc[label].sum() - true_positive
        )
        false_positive = float(
            matrix[label].sum() - true_positive
            if label in matrix.columns
            else 0.0
        )

        precision_denominator = (
            true_positive + false_positive
        )
        recall_denominator = (
            true_positive + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else float("nan")
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else float("nan")
        )
        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (
                np.isfinite(precision)
                and np.isfinite(recall)
                and precision + recall > 0
            )
            else float("nan")
        )

        per_class_rows.append(
            {
                "class": label,
                "support": recall_denominator,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    per_class = pd.DataFrame(
        per_class_rows
    )
    balanced_accuracy = (
        float(
            per_class["recall"].dropna().mean()
        )
        if not per_class.empty
        else float("nan")
    )
    macro_f1 = (
        float(
            per_class["f1"].dropna().mean()
        )
        if not per_class.empty
        else float("nan")
    )

    if uncertainty_label is None:
        coverage = 1.0
        selective_accuracy = accuracy
        wrong_confident_rate = (
            float(1.0 - accuracy)
            if np.isfinite(accuracy)
            else float("nan")
        )
    else:
        classified = frame[
            "prediction"
        ].ne(uncertainty_label)
        coverage = float(classified.mean())

        if classified.any():
            selective_accuracy = float(
                (
                    frame.loc[classified, "truth"]
                    == frame.loc[
                        classified,
                        "prediction",
                    ]
                ).mean()
            )
        else:
            selective_accuracy = float("nan")

        wrong_confident_rate = float(
            (
                classified
                & frame["truth"].ne(
                    frame["prediction"]
                )
            ).mean()
        )

    overall = {
        "n": float(len(frame)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mcc": multiclass_mcc(
            matrix_values
        ),
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "wrong_confident_rate": wrong_confident_rate,
    }

    return overall, per_class, matrix


def risk_coverage_curve(
    truth: pd.Series,
    prediction: pd.Series,
    confidence: pd.Series,
    *,
    uncertainty_label: str = "UNC",
) -> pd.DataFrame:
    frame = pd.concat(
        [
            truth.rename("truth"),
            prediction.rename("prediction"),
            pd.to_numeric(
                confidence,
                errors="coerce",
            ).rename("confidence"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    frame = frame.loc[
        frame["prediction"].ne(
            uncertainty_label
        )
    ].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "coverage",
                "risk",
                "n_selected",
            ]
        )

    frame = frame.sort_values(
        "confidence",
        ascending=False,
    )
    errors = frame["truth"].ne(
        frame["prediction"]
    ).to_numpy(dtype=float)
    cumulative_errors = np.cumsum(errors)
    selected = np.arange(
        1,
        len(frame) + 1,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "coverage": selected / len(truth),
            "risk": cumulative_errors / selected,
            "n_selected": selected.astype(int),
        }
    )


def parameter_recovery_metrics(
    truth: pd.Series,
    estimate: pd.Series,
    *,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
) -> dict[str, float]:
    components = [
        pd.to_numeric(
            truth,
            errors="coerce",
        ).rename("truth"),
        pd.to_numeric(
            estimate,
            errors="coerce",
        ).rename("estimate"),
    ]

    if lower is not None:
        components.append(
            pd.to_numeric(
                lower,
                errors="coerce",
            ).rename("lower")
        )

    if upper is not None:
        components.append(
            pd.to_numeric(
                upper,
                errors="coerce",
            ).rename("upper")
        )

    frame = pd.concat(
        components,
        axis=1,
        join="inner",
    ).dropna()

    if frame.empty:
        return {
            "n": 0.0,
            "bias": float("nan"),
            "rmse": float("nan"),
            "correlation": float("nan"),
            "coverage_95": float("nan"),
            "median_interval_width": float("nan"),
        }

    error = (
        frame["estimate"] - frame["truth"]
    )
    correlation = (
        float(
            frame[
                [
                    "truth",
                    "estimate",
                ]
            ].corr().iloc[0, 1]
        )
        if (
            frame["truth"].nunique() > 1
            and frame["estimate"].nunique() > 1
        )
        else float("nan")
    )

    if {"lower", "upper"}.issubset(frame.columns):
        coverage = float(
            (
                frame["lower"].le(
                    frame["truth"]
                )
                & frame["upper"].ge(
                    frame["truth"]
                )
            ).mean()
        )
        median_width = float(
            (
                frame["upper"] - frame["lower"]
            ).median()
        )
    else:
        coverage = float("nan")
        median_width = float("nan")

    return {
        "n": float(len(frame)),
        "bias": float(error.mean()),
        "rmse": float(
            np.sqrt(
                np.mean(error**2)
            )
        ),
        "correlation": correlation,
        "coverage_95": coverage,
        "median_interval_width": median_width,
    }


def scenario_metadata(
    dataset: SimulatedDataset,
) -> dict[str, Any]:
    scenario = dataset.scenario

    return {
        "scenario_id": scenario.scenario_id,
        "replicate": dataset.replicate,
        "seed": dataset.seed,
        "n_genes": scenario.n_genes,
        "n_normal": scenario.n_normal,
        "n_tumor": scenario.n_tumor,
        "cn_signal": scenario.cn_signal,
        "cn_noise_rate": scenario.cn_noise_rate,
        "purity_low": scenario.purity_low,
        "purity_high": scenario.purity_high,
        "dispersion_scale": scenario.dispersion_scale,
    }


def evaluate_method(
    dataset: SimulatedDataset,
    predictions: pd.DataFrame,
    *,
    method: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame,
]:
    truth = dataset.truth.copy()
    prediction_frame = predictions.copy()

    if "gene" not in prediction_frame.columns:
        raise KeyError("Prediction table must contain 'gene'.")

    prediction_frame = prediction_frame.drop_duplicates(
        subset=["gene"],
        keep="last",
    ).set_index("gene")

    combined = truth.join(
        prediction_frame,
        how="left",
    )
    metadata = scenario_metadata(dataset)

    metrics_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    confusion_tables: dict[str, pd.DataFrame] = {}
    risk_tables: list[pd.DataFrame] = []

    if method == "deconveil":
        combined["predicted_fine"] = combined[
            "deconveil_class"
        ].map(canonical_deconveil_class)
        fine_truth = combined[
            "truth_class_deconveil"
        ]
        fine_prediction = combined[
            "predicted_fine"
        ]

        fine_overall, fine_per_class, fine_confusion = (
            classification_metrics(
                fine_truth,
                fine_prediction,
                class_order=DECONVEIL_CLASS_ORDER,
                exclude_truth={"OTHER"},
            )
        )
        confusion_tables[
            "deconveil_fine"
        ] = fine_confusion

        for metric_name, metric_value in fine_overall.items():
            metrics_rows.append(
                {
                    **metadata,
                    "method": method,
                    "taxonomy": "deconveil_fine",
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

        if not fine_per_class.empty:
            fine_per_class = fine_per_class.assign(
                **metadata,
                method=method,
                taxonomy="deconveil_fine",
            )
            per_class_rows.extend(
                fine_per_class.to_dict(
                    orient="records"
                )
            )

        combined["truth_common_primary"] = combined[
            "truth_class_common"
        ]
        combined["predicted_common"] = combined[
            "predicted_fine"
        ].map(deconveil_to_common)

        common_overall, common_per_class, common_confusion = (
            classification_metrics(
                combined[
                    "truth_common_primary"
                ],
                combined["predicted_common"],
                class_order=COMMON_CLASS_ORDER,
                uncertainty_label="UNC",
                exclude_truth={"COMPLEX"},
            )
        )
        confusion_tables[
            "common_primary"
        ] = common_confusion

        for metric_name, metric_value in common_overall.items():
            metrics_rows.append(
                {
                    **metadata,
                    "method": method,
                    "taxonomy": "common_primary",
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

        if not common_per_class.empty:
            common_per_class = common_per_class.assign(
                **metadata,
                method=method,
                taxonomy="common_primary",
            )
            per_class_rows.extend(
                common_per_class.to_dict(
                    orient="records"
                )
            )

        if "aware_log2fc" in combined.columns:
            recovery = parameter_recovery_metrics(
                combined[
                    "true_regulatory_log2fc"
                ],
                combined["aware_log2fc"],
            )

            for metric_name, metric_value in recovery.items():
                parameter_rows.append(
                    {
                        **metadata,
                        "method": method,
                        "parameter": (
                            "regulatory_log2fc"
                        ),
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )

        if {
            "naive_log2fc",
            "aware_log2fc",
        }.issubset(combined.columns):
            estimated_cn_contribution = (
                combined["naive_log2fc"]
                - combined["aware_log2fc"]
            )
            recovery = parameter_recovery_metrics(
                combined[
                    "true_cn_contribution_log2"
                ],
                estimated_cn_contribution,
            )

            for metric_name, metric_value in recovery.items():
                parameter_rows.append(
                    {
                        **metadata,
                        "method": method,
                        "parameter": (
                            "cn_contribution_log2"
                        ),
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )

        combined["method"] = method
        return (
            metrics_rows,
            per_class_rows,
            parameter_rows,
            confusion_tables,
            risk_tables,
            combined.reset_index(),
        )

    if method == "bdgdm":
        combined["predicted_fine"] = combined[
            "bdgdm_class"
        ].map(canonical_bdgdm_class)
        fine_truth = combined[
            "truth_class_bdgdm"
        ]
        fine_prediction = combined[
            "predicted_fine"
        ]

        fine_overall, fine_per_class, fine_confusion = (
            classification_metrics(
                fine_truth,
                fine_prediction,
                class_order=BDGDM_CLASS_ORDER,
                uncertainty_label="UNC",
            )
        )
        confusion_tables[
            "bdgdm_fine"
        ] = fine_confusion

        for metric_name, metric_value in fine_overall.items():
            metrics_rows.append(
                {
                    **metadata,
                    "method": method,
                    "taxonomy": "bdgdm_fine",
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

        if not fine_per_class.empty:
            fine_per_class = fine_per_class.assign(
                **metadata,
                method=method,
                taxonomy="bdgdm_fine",
            )
            per_class_rows.extend(
                fine_per_class.to_dict(
                    orient="records"
                )
            )

        combined["predicted_common"] = combined[
            "predicted_fine"
        ].map(bdgdm_to_common)

        common_overall, common_per_class, common_confusion = (
            classification_metrics(
                combined["truth_class_common"],
                combined["predicted_common"],
                class_order=COMMON_CLASS_ORDER,
                uncertainty_label="UNC",
                exclude_truth={"COMPLEX"},
            )
        )
        confusion_tables[
            "common_primary"
        ] = common_confusion

        for metric_name, metric_value in common_overall.items():
            metrics_rows.append(
                {
                    **metadata,
                    "method": method,
                    "taxonomy": "common_primary",
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

        if not common_per_class.empty:
            common_per_class = common_per_class.assign(
                **metadata,
                method=method,
                taxonomy="common_primary",
            )
            per_class_rows.extend(
                common_per_class.to_dict(
                    orient="records"
                )
            )

        if "confidence" in combined.columns:
            risk = risk_coverage_curve(
                combined[
                    "truth_class_bdgdm"
                ],
                combined[
                    "predicted_fine"
                ],
                combined["confidence"],
                uncertainty_label="UNC",
            )

            if not risk.empty:
                for key, value in metadata.items():
                    risk[key] = value

                risk["method"] = method
                risk["taxonomy"] = "bdgdm_fine"
                risk_tables.append(risk)

        parameter_specs = (
            (
                "b_scaling",
                "true_b_scaling",
                "b_scaling_median",
                "b_scaling_q025",
                "b_scaling_q975",
            ),
            (
                "b_deviation",
                "true_b_deviation",
                "b_deviation_median",
                "b_deviation_q025",
                "b_deviation_q975",
            ),
        )

        for (
            parameter_name,
            truth_column,
            estimate_column,
            lower_column,
            upper_column,
        ) in parameter_specs:
            if estimate_column not in combined.columns:
                continue

            lower = (
                combined[lower_column]
                if lower_column in combined.columns
                else None
            )
            upper = (
                combined[upper_column]
                if upper_column in combined.columns
                else None
            )

            recovery = parameter_recovery_metrics(
                combined[truth_column],
                combined[estimate_column],
                lower=lower,
                upper=upper,
            )

            for metric_name, metric_value in recovery.items():
                parameter_rows.append(
                    {
                        **metadata,
                        "method": method,
                        "parameter": parameter_name,
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )

        combined["method"] = method
        return (
            metrics_rows,
            per_class_rows,
            parameter_rows,
            confusion_tables,
            risk_tables,
            combined.reset_index(),
        )

    raise ValueError(f"Unknown method: {method}")


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    normalize_rows: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if matrix.empty:
        return

    values = matrix.astype(float)

    if normalize_rows:
        values = values.div(
            values.sum(axis=1).replace(
                0,
                np.nan,
            ),
            axis=0,
        ).fillna(0.0)

    figure_width = max(
        6.0,
        0.75 * len(values.columns) + 2.5,
    )
    figure_height = max(
        5.0,
        0.65 * len(values.index) + 2.2,
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )
    image = ax.imshow(
        values.to_numpy(),
        vmin=0.0,
        vmax=1.0 if normalize_rows else None,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(values.columns)),
        labels=values.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(values.index)),
        labels=values.index,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    threshold = (
        0.5
        if normalize_rows
        else (
            float(values.to_numpy().max()) / 2
            if values.to_numpy().size
            else 0.0
        )
    )

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values.iat[
                row_index,
                column_index,
            ]

            text = (
                f"{value:.2f}"
                if normalize_rows
                else f"{int(value)}"
            )
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )
    colorbar.set_label(
        "Row proportion"
        if normalize_rows
        else "Count"
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_class_balance(
    truth: pd.DataFrame,
    path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    counts = (
        truth["truth_class_latent"]
        .value_counts()
        .reindex(
            LATENT_CLASS_ORDER,
            fill_value=0,
        )
    )
    colors = [
        CLASS_PALETTE.get(
            class_label,
            "#999999",
        )
        for class_label in counts.index
    ]

    fig, ax = plt.subplots(
        figsize=(8.0, 4.8)
    )
    bars = ax.bar(
        counts.index,
        counts.to_numpy(),
        color=colors,
    )

    for bar, value in zip(
        bars,
        counts.to_numpy(),
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Latent class")
    ax.set_ylabel("Number of genes")
    ax.set_title(title)
    ax.grid(
        axis="y",
        alpha=0.20,
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_parameter_recovery(
    table: pd.DataFrame,
    path: Path,
    *,
    truth_column: str,
    estimate_column: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    import matplotlib.pyplot as plt

    if (
        truth_column not in table.columns
        or estimate_column not in table.columns
    ):
        return

    frame = table[
        [
            truth_column,
            estimate_column,
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if frame.empty:
        return

    lower = float(
        min(
            frame[truth_column].min(),
            frame[estimate_column].min(),
        )
    )
    upper = float(
        max(
            frame[truth_column].max(),
            frame[estimate_column].max(),
        )
    )
    padding = (
        0.05 * (upper - lower)
        if upper > lower
        else 0.1
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 5.5)
    )
    ax.scatter(
        frame[truth_column],
        frame[estimate_column],
        alpha=0.55,
        s=24,
    )
    ax.plot(
        [
            lower - padding,
            upper + padding,
        ],
        [
            lower - padding,
            upper + padding,
        ],
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlim(
        lower - padding,
        upper + padding,
    )
    ax.set_ylim(
        lower - padding,
        upper + padding,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_risk_coverage(
    table: pd.DataFrame,
    path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if table.empty:
        return

    fig, ax = plt.subplots(
        figsize=(6.4, 5.0)
    )

    grouping_columns = [
        column
        for column in (
            "scenario_id",
            "replicate",
        )
        if column in table.columns
    ]

    if grouping_columns:
        for _, group in table.groupby(
            grouping_columns,
            dropna=False,
        ):
            ax.plot(
                group["coverage"],
                group["risk"],
                alpha=0.30,
                linewidth=1.0,
            )

        mean_curve = (
            table.assign(
                coverage_bin=pd.cut(
                    table["coverage"],
                    bins=np.linspace(
                        0,
                        1,
                        21,
                    ),
                    include_lowest=True,
                )
            )
            .groupby(
                "coverage_bin",
                observed=False,
            )[
                [
                    "coverage",
                    "risk",
                ]
            ]
            .mean()
            .dropna()
        )

        if not mean_curve.empty:
            ax.plot(
                mean_curve["coverage"],
                mean_curve["risk"],
                linewidth=2.4,
                label="Mean",
            )
            ax.legend(
                frameon=False
            )
    else:
        ax.plot(
            table["coverage"],
            table["risk"],
            linewidth=2.0,
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(
        bottom=0.0
    )
    ax.set_title(title)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_by_sample_size(
    metrics: pd.DataFrame,
    path: Path,
    *,
    metric: str = "macro_f1",
    taxonomy: str = "common_primary",
) -> None:
    import matplotlib.pyplot as plt

    frame = metrics.loc[
        metrics["metric"].eq(metric)
        & metrics["taxonomy"].eq(taxonomy)
    ].copy()

    if frame.empty:
        return

    summary = (
        frame.groupby(
            [
                "method",
                "n_tumor",
            ],
            dropna=False,
        )["value"]
        .agg(
            mean="mean",
            std="std",
            n="size",
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0)
    )

    for method, group in summary.groupby(
        "method"
    ):
        group = group.sort_values(
            "n_tumor"
        )
        standard_deviation = group[
            "std"
        ].fillna(0.0)

        ax.plot(
            group["n_tumor"],
            group["mean"],
            marker="o",
            label=method,
        )
        ax.fill_between(
            group["n_tumor"],
            group["mean"] - standard_deviation,
            group["mean"] + standard_deviation,
            alpha=0.18,
        )

    ax.set_xlabel("Tumour samples")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"{metric.replace('_', ' ').title()} — {taxonomy}"
    )
    ax.grid(alpha=0.20)
    ax.legend(
        frameon=False
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def aggregate_confusions(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()

    all_rows: list[str] = []
    all_columns: list[str] = []

    for table in tables:
        for value in table.index:
            if value not in all_rows:
                all_rows.append(value)

        for value in table.columns:
            if value not in all_columns:
                all_columns.append(value)

    total = pd.DataFrame(
        0,
        index=all_rows,
        columns=all_columns,
        dtype=int,
    )

    for table in tables:
        total = total.add(
            table.reindex(
                index=all_rows,
                columns=all_columns,
                fill_value=0,
            ),
            fill_value=0,
        )

    return total.astype(int)


def build_scenarios(
    args: argparse.Namespace,
) -> list[Scenario]:
    extended = args.classes == "extended"
    class_probabilities = parse_class_probabilities(
        args.class_probabilities,
        extended=extended,
    )
    sample_sizes = parse_csv_values(
        args.sample_sizes,
        int,
    )
    cn_signals = parse_csv_values(
        args.cn_signals,
        str,
    )
    cn_noise_rates = parse_csv_values(
        args.cn_noise_rates,
        float,
    )

    scenarios: list[Scenario] = []

    for sample_size in sample_sizes:
        for cn_signal in cn_signals:
            for cn_noise_rate in cn_noise_rates:
                scenario_id = safe_scenario_id(
                    (
                        f"n{sample_size}"
                        f"_cn-{cn_signal}"
                        f"_noise-{cn_noise_rate:.2f}"
                        f"_purity-{args.purity_low:.2f}"
                        f"-{args.purity_high:.2f}"
                        f"_disp-{args.dispersion_scale:.2f}"
                    )
                )

                scenarios.append(
                    Scenario(
                        scenario_id=scenario_id,
                        n_genes=args.n_genes,
                        n_normal=sample_size,
                        n_tumor=sample_size,
                        class_probabilities=class_probabilities,
                        cn_signal=cn_signal,
                        purity_low=args.purity_low,
                        purity_high=args.purity_high,
                        cn_noise_rate=cn_noise_rate,
                        dispersion_scale=(
                            args.dispersion_scale
                        ),
                    )
                )

    return scenarios


def apply_preset(
    args: argparse.Namespace,
) -> None:
    if args.preset == "smoke":
        if args.n_genes is None:
            args.n_genes = 120

        if args.sample_sizes is None:
            args.sample_sizes = "10"

        if args.replicates is None:
            args.replicates = 1

    elif args.preset == "development":
        if args.n_genes is None:
            args.n_genes = 1000

        if args.sample_sizes is None:
            args.sample_sizes = "20,40"

        if args.replicates is None:
            args.replicates = 5

    elif args.preset == "manuscript":
        if args.n_genes is None:
            args.n_genes = 5000

        if args.sample_sizes is None:
            args.sample_sizes = "10,20,40,60"

        if args.replicates is None:
            args.replicates = 20

    else:
        raise ValueError(
            f"Unknown preset: {args.preset}"
        )


def benchmark_context(
    dataset: SimulatedDataset,
    replicate_directory: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        **scenario_metadata(dataset),
        "output_dir": str(replicate_directory),
        "bdgdm_engine": args.bdgdm_engine,
        "deconveil_alpha": args.deconveil_alpha,
        "deconveil_lfc_threshold": (
            args.deconveil_lfc_threshold
        ),
        "n_cpus": args.n_cpus,
        "quiet": not args.verbose_methods,
    }


def run_benchmark(
    args: argparse.Namespace,
) -> BenchmarkOutputs:
    output_root = Path(
        args.output_dir
    ).expanduser().resolve()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_directory = (
        output_root / "figures"
    )
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = build_scenarios(args)
    deconveil_adapter = (
        load_callable(
            args.deconveil_adapter
        )
        if args.deconveil_adapter
        else None
    )
    bdgdm_adapter = (
        load_callable(
            args.bdgdm_adapter
        )
        if args.bdgdm_adapter
        else None
    )

    outputs = BenchmarkOutputs()
    confusion_store: dict[
        tuple[str, str],
        list[pd.DataFrame],
    ] = {}

    configuration = {
        "arguments": vars(args),
        "scenarios": [
            asdict(scenario)
            for scenario in scenarios
        ],
    }

    with (
        output_root / "benchmark_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            configuration,
            handle,
            indent=2,
            default=float,
        )

    for scenario_number, scenario in enumerate(
        scenarios
    ):
        LOGGER.info(
            "Scenario %s (%d/%d)",
            scenario.scenario_id,
            scenario_number + 1,
            len(scenarios),
        )

        for replicate in range(args.replicates):
            replicate_seed = (
                args.seed
                + scenario_number * 100_000
                + replicate
            )
            replicate_directory = (
                output_root
                / scenario.scenario_id
                / f"replicate_{replicate:03d}"
            )
            replicate_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            truth_path = (
                replicate_directory
                / "truth.csv"
            )

            if (
                truth_path.exists()
                and not args.overwrite
            ):
                LOGGER.warning(
                    "Existing simulation found at %s; "
                    "the replicate will be regenerated in memory "
                    "for method execution and evaluation.",
                    replicate_directory,
                )

            dataset = simulate_dataset(
                scenario,
                replicate=replicate,
                seed=replicate_seed,
            )
            write_dataset(
                dataset,
                replicate_directory,
            )
            plot_class_balance(
                dataset.truth,
                replicate_directory
                / "class_balance.png",
                title=(
                    f"Latent class balance — "
                    f"{scenario.scenario_id}, "
                    f"replicate {replicate}"
                ),
            )

            context = benchmark_context(
                dataset,
                replicate_directory,
                args,
            )

            method_tables: list[
                tuple[str, pd.DataFrame]
            ] = []

            if args.demo_predictions:
                LOGGER.warning(
                    "Generating DEMO predictions. "
                    "These are not scientific method results."
                )
                method_tables.append(
                    (
                        "deconveil",
                        demo_method_predictions(
                            dataset,
                            method="deconveil",
                            seed=replicate_seed + 11,
                        ),
                    )
                )
                method_tables.append(
                    (
                        "bdgdm",
                        demo_method_predictions(
                            dataset,
                            method="bdgdm",
                            seed=replicate_seed + 29,
                        ),
                    )
                )

            else:
                if args.run_deconveil:
                    deconveil_path = (
                        replicate_directory
                        / "deconveil_predictions.csv"
                    )

                    if (
                        deconveil_path.exists()
                        and not args.overwrite
                    ):
                        deconveil_predictions = pd.read_csv(
                            deconveil_path
                        )
                    elif deconveil_adapter is not None:
                        deconveil_predictions = (
                            call_deconveil_adapter(
                                deconveil_adapter,
                                dataset,
                                context,
                            )
                        )
                        deconveil_predictions.to_csv(
                            deconveil_path,
                            index=False,
                        )
                    else:
                        deconveil_predictions = (
                            run_deconveil_builtin(
                                dataset,
                                context,
                            )
                        )
                        deconveil_predictions.to_csv(
                            deconveil_path,
                            index=False,
                        )

                    method_tables.append(
                        (
                            "deconveil",
                            deconveil_predictions,
                        )
                    )

                bdgdm_path = (
                    replicate_directory
                    / "bdgdm_predictions.csv"
                )

                if (
                    bdgdm_path.exists()
                    and not args.overwrite
                ):
                    LOGGER.info(
                        "Loading existing BDGDM predictions: %s",
                        bdgdm_path,
                    )
                    bdgdm_predictions = pd.read_csv(
                        bdgdm_path
                    )
                    method_tables.append(
                        (
                            "bdgdm",
                            bdgdm_predictions,
                        )
                    )

                elif bdgdm_adapter is not None:
                    bdgdm_predictions = (
                        call_bdgdm_adapter(
                            bdgdm_adapter,
                            dataset,
                            context,
                            max_genes=(
                                args.max_bdgdm_genes
                            ),
                        )
                    )
                    bdgdm_predictions.to_csv(
                        bdgdm_path,
                        index=False,
                    )
                    method_tables.append(
                        (
                            "bdgdm",
                            bdgdm_predictions,
                        )
                    )

            for method, method_predictions in method_tables:
                predictions_path = (
                    replicate_directory
                    / f"{method}_predictions.csv"
                )
                method_predictions.to_csv(
                    predictions_path,
                    index=False,
                )

                (
                    metric_rows,
                    class_rows,
                    parameter_rows,
                    confusion_tables,
                    risk_tables,
                    combined,
                ) = evaluate_method(
                    dataset,
                    method_predictions,
                    method=method,
                )

                outputs.metrics.extend(
                    metric_rows
                )
                outputs.per_class_metrics.extend(
                    class_rows
                )
                outputs.parameter_metrics.extend(
                    parameter_rows
                )
                outputs.predictions.append(
                    combined
                )
                outputs.risk_coverage.extend(
                    risk_tables
                )

                combined.to_csv(
                    replicate_directory
                    / f"{method}_evaluation.csv",
                    index=False,
                )

                for taxonomy, matrix in confusion_tables.items():
                    matrix.to_csv(
                        replicate_directory
                        / (
                            f"{method}_{taxonomy}"
                            "_confusion.csv"
                        )
                    )
                    plot_confusion_matrix(
                        matrix,
                        replicate_directory
                        / (
                            f"{method}_{taxonomy}"
                            "_confusion.png"
                        ),
                        title=(
                            f"{method.title()} — "
                            f"{taxonomy} — "
                            f"{scenario.scenario_id}"
                        ),
                    )
                    confusion_store.setdefault(
                        (
                            method,
                            taxonomy,
                        ),
                        [],
                    ).append(matrix)

                if method == "bdgdm":
                    plot_parameter_recovery(
                        combined,
                        replicate_directory
                        / (
                            "bdgdm_b_scaling"
                            "_recovery.png"
                        ),
                        truth_column=(
                            "true_b_scaling"
                        ),
                        estimate_column=(
                            "b_scaling_median"
                        ),
                        title=(
                            "BDGDM scaling recovery"
                        ),
                        x_label=(
                            "True scaling coefficient"
                        ),
                        y_label=(
                            "Estimated posterior median"
                        ),
                    )
                    plot_parameter_recovery(
                        combined,
                        replicate_directory
                        / (
                            "bdgdm_b_deviation"
                            "_recovery.png"
                        ),
                        truth_column=(
                            "true_b_deviation"
                        ),
                        estimate_column=(
                            "b_deviation_median"
                        ),
                        title=(
                            "BDGDM deviation recovery"
                        ),
                        x_label=(
                            "True deviation coefficient"
                        ),
                        y_label=(
                            "Estimated posterior median"
                        ),
                    )

                if method == "deconveil":
                    plot_parameter_recovery(
                        combined,
                        replicate_directory
                        / (
                            "deconveil_regulatory"
                            "_lfc_recovery.png"
                        ),
                        truth_column=(
                            "true_regulatory_log2fc"
                        ),
                        estimate_column=(
                            "aware_log2fc"
                        ),
                        title=(
                            "DeConveil regulatory "
                            "effect recovery"
                        ),
                        x_label=(
                            "True regulatory log2FC"
                        ),
                        y_label=(
                            "Estimated CN-aware log2FC"
                        ),
                    )

    metrics = pd.DataFrame(
        outputs.metrics
    )
    per_class = pd.DataFrame(
        outputs.per_class_metrics
    )
    parameter_metrics = pd.DataFrame(
        outputs.parameter_metrics
    )
    predictions = (
        pd.concat(
            outputs.predictions,
            ignore_index=True,
            sort=False,
        )
        if outputs.predictions
        else pd.DataFrame()
    )
    risk_coverage = (
        pd.concat(
            outputs.risk_coverage,
            ignore_index=True,
            sort=False,
        )
        if outputs.risk_coverage
        else pd.DataFrame()
    )

    metrics.to_csv(
        output_root / "classification_metrics.csv",
        index=False,
    )
    per_class.to_csv(
        output_root / "per_class_metrics.csv",
        index=False,
    )
    parameter_metrics.to_csv(
        output_root / "parameter_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        output_root / "all_predictions.csv.gz",
        index=False,
    )
    risk_coverage.to_csv(
        output_root / "risk_coverage.csv",
        index=False,
    )

    if not metrics.empty:
        aggregate_metrics = (
            metrics.groupby(
                [
                    "scenario_id",
                    "n_normal",
                    "n_tumor",
                    "cn_signal",
                    "cn_noise_rate",
                    "purity_low",
                    "purity_high",
                    "method",
                    "taxonomy",
                    "metric",
                ],
                dropna=False,
            )["value"]
            .agg(
                mean="mean",
                std="std",
                median="median",
                minimum="min",
                maximum="max",
                n="size",
            )
            .reset_index()
        )
        aggregate_metrics.to_csv(
            output_root
            / "classification_metrics_aggregated.csv",
            index=False,
        )
        plot_metric_by_sample_size(
            metrics,
            figures_directory
            / "macro_f1_by_sample_size.png",
            metric="macro_f1",
            taxonomy="common_primary",
        )
        plot_metric_by_sample_size(
            metrics,
            figures_directory
            / "balanced_accuracy_by_sample_size.png",
            metric="balanced_accuracy",
            taxonomy="common_primary",
        )

    if not parameter_metrics.empty:
        aggregate_parameters = (
            parameter_metrics.groupby(
                [
                    "scenario_id",
                    "n_normal",
                    "n_tumor",
                    "method",
                    "parameter",
                    "metric",
                ],
                dropna=False,
            )["value"]
            .agg(
                mean="mean",
                std="std",
                median="median",
                n="size",
            )
            .reset_index()
        )
        aggregate_parameters.to_csv(
            output_root
            / "parameter_metrics_aggregated.csv",
            index=False,
        )

    if not risk_coverage.empty:
        plot_risk_coverage(
            risk_coverage,
            figures_directory
            / "bdgdm_risk_coverage.png",
            title="BDGDM risk–coverage",
        )

    for (
        method,
        taxonomy,
    ), tables in confusion_store.items():
        aggregate = aggregate_confusions(
            tables
        )
        aggregate.to_csv(
            output_root
            / (
                f"{method}_{taxonomy}"
                "_confusion_aggregated.csv"
            )
        )
        plot_confusion_matrix(
            aggregate,
            figures_directory
            / (
                f"{method}_{taxonomy}"
                "_confusion_aggregated.png"
            ),
            title=(
                f"{method.title()} — "
                f"{taxonomy} — aggregated"
            ),
        )

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate and benchmark BDGDM and DeConveil."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
    )
    parser.add_argument(
        "--preset",
        choices=[
            "smoke",
            "development",
            "manuscript",
        ],
        default="smoke",
    )
    parser.add_argument(
        "--classes",
        choices=[
            "shared",
            "extended",
        ],
        default="extended",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sample-sizes",
        default=None,
        help=(
            "Comma-separated sample sizes used for "
            "both normal and tumour groups."
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--class-probabilities",
        default=None,
        help=(
            "Comma-separated CLASS=value entries. "
            "Example: NEUTRAL=.2,DSG=.2,DIG=.2,"
            "DCG=.2,HYPER=.1,MIXED=.1"
        ),
    )
    parser.add_argument(
        "--cn-signals",
        default="moderate",
        help=(
            "Comma-separated weak, moderate, or strong."
        ),
    )
    parser.add_argument(
        "--cn-noise-rates",
        default="0.0",
        help=(
            "Comma-separated proportions of tumour "
            "gene-sample CN entries to perturb."
        ),
    )
    parser.add_argument(
        "--purity-low",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--purity-high",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dispersion-scale",
        type=float,
        default=1.0,
        help=(
            "Values above one increase overdispersion "
            "by reducing NB precision."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--run-deconveil",
        action="store_true",
    )
    parser.add_argument(
        "--deconveil-adapter",
        default=None,
        help=(
            "Optional package.module:function implementing "
            "the official DeConveil/stageR pipeline."
        ),
    )
    parser.add_argument(
        "--deconveil-alpha",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--deconveil-lfc-threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--bdgdm-adapter",
        default=None,
        help=(
            "package.module:function called once per gene."
        ),
    )
    parser.add_argument(
        "--bdgdm-engine",
        default="vi_meanfield",
    )
    parser.add_argument(
        "--max-bdgdm-genes",
        type=int,
        default=None,
        help=(
            "Optional development limit. Do not use "
            "for final class-frequency comparisons."
        ),
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--demo-predictions",
        action="store_true",
        help=(
            "Generate noisy oracle predictions to test "
            "the pipeline. Never use these as results."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--verbose-methods",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
        default="INFO",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_preset(args)

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
    )

    if (
        not args.demo_predictions
        and not args.run_deconveil
        and args.bdgdm_adapter is None
    ):
        LOGGER.warning(
            "No method runner was requested. The script will "
            "generate simulation inputs and truth only."
        )

    if args.preset == "manuscript":
        LOGGER.warning(
            "The manuscript preset is computationally expensive: "
            "%s genes, sample sizes %s, %s replicates.",
            args.n_genes,
            args.sample_sizes,
            args.replicates,
        )

    try:
        run_benchmark(args)
    except Exception:
        LOGGER.exception(
            "Benchmark failed."
        )
        return 1

    LOGGER.info(
        "Benchmark completed: %s",
        Path(
            args.output_dir
        ).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
