from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BDGDMFit:
    """
    Object for a fitted BDGDM model.
    """

    gene: str
    
    analysis_mode: str

    fit: Any

    posterior: dict

    diagnostics: dict

    ppc: dict | None

    metadata: dict