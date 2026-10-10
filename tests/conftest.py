"""Make the calculator modules importable when pytest is run from the repository root.

The analysis package also has a module named config, so the calculator directory is put
first on sys.path and any other already-imported "config" module is dropped.
"""
import sys
from pathlib import Path

CALC_DIR = Path(__file__).resolve().parent.parent / "calculator"

if str(CALC_DIR) in sys.path:
    sys.path.remove(str(CALC_DIR))
sys.path.insert(0, str(CALC_DIR))

for _name in ("config", "prediction", "app"):
    _mod = sys.modules.get(_name)
    if _mod is not None and Path(getattr(_mod, "__file__", "") or "").resolve().parent != CALC_DIR:
        del sys.modules[_name]
