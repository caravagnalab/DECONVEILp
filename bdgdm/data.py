from __future__ import annotations
from importlib.resources import files
import pandas as pd


"""
Example datasets to test BDGDM.
"""

def load_example_data() -> pd.DataFrame:
    """
    Load the bundled CRC example dataset.

    Returns
    -------
    pandas.DataFrame
        Long-format CRC example dataset.
    """

    path = files("bdgdm").joinpath(
        "data",
        "crc_joint_long_drivers.csv",
    )

    return pd.read_csv(path)