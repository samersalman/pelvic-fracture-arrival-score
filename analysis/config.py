"""Shared path configuration for the analysis scripts.

Layout (flat, relative to the repository root):

    analysis/   this directory (code)
    inputs/     small reference inputs shipped with the repository
    derived/    intermediate and aggregate outputs of the pipeline
    tables/     table outputs
    figures/    figure outputs
    logs/       run logs, manifests and QC outputs

Every directory except NTDB_DATA is derived from the location of this file, so
the scripts run from any working directory.

NTDB_DATA is the only external dependency. It must point to the folder that
holds the National Trauma Data Bank files, with this layout:

    $NTDB_DATA/compiled_NTDB/compiled_NTDB_2019_2024/
        compiled_PUF_AY_2019_2024_with_VL.csv
        ntdb_2019_2024_icd_diagnoses.csv
        ntdb_2019_2024_ais_diagnoses.csv
        ntdb_2019_2024_preexisting.csv
    $NTDB_DATA/PUF/PUF AY <year>/CSV/PUF_TRAUMA.csv     (year = 2019 .. 2024)

The variable is read lazily: importing this module never fails. Only code that
actually needs the raw data (01_build_cohort.py) calls ntdb_data() and fails
with a clear message when the variable is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

ANALYSIS_DIR = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(__file__).resolve().parent.parent)

INPUTS = os.path.join(REPO_ROOT, "inputs")
DERIVED = os.path.join(REPO_ROOT, "derived")
TABLES = os.path.join(REPO_ROOT, "tables")
FIGURES = os.path.join(REPO_ROOT, "figures")
LOGS = os.path.join(REPO_ROOT, "logs")

_OUTPUT_DIRS = (DERIVED, TABLES, FIGURES, LOGS)

# Relative sub-paths below NTDB_DATA.
NTDB_COMPILED_SUBDIR = os.path.join("compiled_NTDB", "compiled_NTDB_2019_2024")
NTDB_PUF_SUBDIR = "PUF"


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for d in _OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)


def ntdb_data() -> str:
    """Return the NTDB data root from the NTDB_DATA environment variable."""
    value = os.environ.get("NTDB_DATA", "").strip()
    if not value:
        raise RuntimeError(
            "The environment variable NTDB_DATA is not set. Set it to the "
            "folder that holds the National Trauma Data Bank files (see the "
            "docstring of analysis/config.py for the expected layout), e.g. "
            "export NTDB_DATA=/path/to/NTDB"
        )
    if not os.path.isdir(value):
        raise RuntimeError(f"NTDB_DATA points to a folder that does not exist: {value}")
    return value


def ntdb_compiled_dir() -> str:
    return os.path.join(ntdb_data(), NTDB_COMPILED_SUBDIR)


def ntdb_puf_dir() -> str:
    return os.path.join(ntdb_data(), NTDB_PUF_SUBDIR)


def __getattr__(name: str):
    # Lazy attribute so that `config.NTDB_DATA` works without failing at import.
    if name == "NTDB_DATA":
        return ntdb_data()
    raise AttributeError(f"module 'config' has no attribute {name!r}")
