"""
BDGDM
-----

Bayesian Differential Gene Dosage Model.

A Bayesian framework for modelling copy number–dependent transcriptional
responses from bulk RNA-seq data.

Main functionality
------------------
- Fit BDGDM models
- Posterior summaries
- Gene dosage classification
- Posterior predictive checks
- Example datasets
"""

__version__ = "0.1.0"


# Model fitting interface
from .fit import fit_one_gene_bdgdm, BDGDMConfig

# Example datasets
from .data import load_example_data

# Posterior summaries
from .posterior import summarize_posterior, extract_posterior_draws

# Diagnostics
from .diagnostics import sampler_diagnostics

# Gene-dosage classification
from .classify import (
    ClassificationThresholds,
    classify_fit,
    classify_fits,
    classify_gene,
    classify_gene_result,
    classify_results_dataframe,
    genes_with_response_class,
    summarize_response_classes,
    summarize_transition_patterns,
)

# Main exported objects
__all__ = [

    # fitting
    "fit_one_gene_bdgdm",
    "BDGDMConfig",
    "BDGDMFit",

    # example data
    "load_example_data",

    # posterior
    "summarize_posterior",
    "extract_posterior_draws",

    # diagnostics
    "sampler_diagnostics",

    # gene-dosage classification
    "ClassificationThresholds",
    "classify_fit",
    "classify_fits",
    "classify_gene",
    "classify_gene_result",
    "classify_results_dataframe",
    "genes_with_response_class",
    "summarize_response_classes",
    "summarize_transition_patterns",

]
