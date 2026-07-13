# DECONVEILp
BDGDM is a Bayesian framework for modelling copy-number-dependent transcriptional responses from bulk RNA-seq data.

### Features

- Bayesian inference of copy number–expression relationships.
- Tumour-only and subtype-specific dosage-response analysis.
- Dosage-response classification (DSG, DCG, HYPER, Mixed, DIG, UNC).
- Explicit tumour purity correction.
- Posterior predictive checks for model validation.


### Key advantages

- Supports **tumour-only** transcriptomic datasets without requiring matched normal samples.
- Quantifies **uncertainty** through posterior distributions and credible intervals.
- Detects **subtype-specific dosage rewiring** by comparing CN–expression relationships across tumour subtypes.
- Scales from **single-group analyses** to **multi-subtype comparisons** within a unified Bayesian framework.

### Installation

BDGDM requires Python 3.10 or later and uses CmdStanPy as its interface to Stan.

Install the current development version:

- clone or download the repository, then open a terminal in the project root—the directory containing `pyproject.toml`:

`cd DECONVEILp`

- create and activate a dedicated Conda environment:
  
`conda create -n bdgdm python=3.11`
`conda activate bdgdm`

- install the package:

`python -m pip install .`

**Install CmdStan**

`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`

### Copyright and contacts

Katsiaryna Davydzenka, Cancer Data Science (CDS) Laboratory.

[![](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab)
[![](https://img.shields.io/badge/CDS%20Lab%20webpage-https://www.caravagnalab.org/-red.svg)](https://www.caravagnalab.org/)