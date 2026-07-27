# BDGDM

**BDGDM — Bayesian Differential Gene Dosage Model** is a Bayesian framework for quantifying copy-number-dependent transcriptional responses from bulk
RNA-seq data using tumour purity and gene-level absolute copy-number estimates.


The package supports:

- tumour-only transcriptomic datasets without requiring matched normal samples; 
- single-group and subtype-comparison dosage-response analysis;
- posterior classification of dosage responses as dosage-sensitive (`DSG`), dosage-compensated (`DCG`), hyper-responsive (`HYPER`), mixed (`Mixed`),
  dosage-insensitive (`DIG`), or uncertain (`UNC`);
- direct Bayesian assessment of subtype-specific dosage-response rewiring.

## Installation

BDGDM requires Python 3.10 or later and uses [CmdStanPy](https://mc-stan.org/cmdstanpy/) for Bayesian inference.
Before fitting BDGDM models, install CmdStan and a compatible C++ toolchain by following the official
[CmdStanPy installation instructions](https://mc-stan.org/cmdstanpy/installation.html).

### Install from source

BDGDM is currently installed directly from its source repository:

```bash
git clone https://github.com/caravagnalab/DECONVEILp.git
cd DECONVEILp
python -m pip install -e .

```

## Workflow overview

### Single-group analysis

The [single-group vignette](https://github.com/caravagnalab/DECONVEILp/blob/main/docs/single_group.ipynb) demonstrates how to fit and interpret one gene in a single tumour type/subtype. It covers posterior
parameters, CN-transition effects, diagnostics, posterior predictive checks, and dosage-response classification.

### Subtype comparison

The [subtype-comparison vignette](https://github.com/caravagnalab/DECONVEILp/blob/main/docs/subtype_comparison.ipynb) demonstrates how to jointly estimate gene-dosage responses across tumour subtypes. It covers subtype-specific parameters, direct scaling and deviation contrasts, classification, and dosage-response rewiring.

### Multigene analysis

The [multigene-analysis vignette](https://github.com/caravagnalab/DECONVEILp/blob/main/docs/multigene_analysis.ipynb) extends the workflow to a set of informative genes. It covers gene selection, batch fitting, diagnostic filtering, response-class summaries, transition patterns and dosage-response rewiring.


### Copyright and contacts

Katsiaryna Davydzenka, Cancer Data Science (CDS) Laboratory.

[![](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab)
[![](https://img.shields.io/badge/CDS%20Lab%20webpage-https://www.caravagnalab.org/-red.svg)](https://www.caravagnalab.org/)