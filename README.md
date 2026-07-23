# DECONVEILp
BDGDM is a Bayesian framework that quantify copy-number-dependent transcriptional signal from bulk RNA-seq and WGS.

### Features

- Bayesian inference of copy-number–expression relationships.
- Single-tumour and subtype-specific dosage-response analysis.
- Dosage-response classification (dosage-sensitive, dosage-compensated, HYPER, Mixed, dosage-insensitive, Uncertain).
- Explicit tumour purity correction.


### Key advantages

- Supports **tumour-only** transcriptomic datasets without requiring matched normal samples.
- Quantifies **uncertainty** through posterior distributions and credible intervals.
- Detects **subtype-specific dosage rewiring** by comparing CN-expression relationships across tumour subtypes.

  
### Installation

BDGDM requires Python 3.10 or later and uses CmdStanPy as its interface to Stan.
Install the current development version:
- clone or download the repository, then open a terminal in the project root—the directory containing `pyproject.toml`:
  `cd DECONVEILp`
- install the package:
  `python -m pip install .`

**Install CmdStan**
`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`


### Copyright and contacts

Katsiaryna Davydzenka, Cancer Data Science (CDS) Laboratory.

[![](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab)
[![](https://img.shields.io/badge/CDS%20Lab%20webpage-https://www.caravagnalab.org/-red.svg)](https://www.caravagnalab.org/)