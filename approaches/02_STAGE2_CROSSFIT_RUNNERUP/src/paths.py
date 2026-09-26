"""
Path handling for this approach.

SHARED INPUTS (read-only, produced by 00_BASELINE):   ER_CACHE_DIR  (default <repo>/cache)
                                                       ER_DATA_DIR   (default <repo>/dataset)
APPROACH-SPECIFIC ARTIFACTS (written):                <approach>/cache, <approach>/results, <approach>/output
                                                       (ER_OUTPUT_DIR may override output/)

The baseline modules (business_entity_resolution/src) are imported read-only for shared logic
(labels, folds, metric, feature list, TSV writer). Their config.py creates OUTPUT_DIR on import, so
ER_OUTPUT_DIR is pointed at this approach *before* they are imported.
"""
import json
import os
import sys
from pathlib import Path

APPROACH_DIR = Path(__file__).resolve().parents[1]
REPO = APPROACH_DIR.parents[1]
SHARED_CACHE = Path(os.environ.get("ER_CACHE_DIR", REPO / "cache"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", APPROACH_DIR / "output"))
RESULTS_DIR = APPROACH_DIR / "results"
APPROACH_CACHE = APPROACH_DIR / "cache"
BASELINE_SRC = REPO / "business_entity_resolution" / "src"

# never write into the baseline / production output folder
_FORBIDDEN = {(REPO / "output").resolve(), SHARED_CACHE.resolve(), BASELINE_SRC.resolve()}
if OUTPUT_DIR.resolve() in _FORBIDDEN:
    raise SystemExit(f"refusing to write outputs to {OUTPUT_DIR}: that is a shared/baseline location")

os.environ["ER_CACHE_DIR"] = str(SHARED_CACHE)
os.environ["ER_OUTPUT_DIR"] = str(OUTPUT_DIR)
for d in (OUTPUT_DIR, RESULTS_DIR, APPROACH_CACHE):
    d.mkdir(parents=True, exist_ok=True)
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

CONFIG = json.load(open(APPROACH_DIR / "config" / "config.json", encoding="utf-8"))


def shared(name: str) -> Path:
    p = SHARED_CACHE / name
    if not p.exists():
        raise SystemExit(f"missing shared artifact {p} (produced by 00_BASELINE); check ER_CACHE_DIR")
    return p


def mine(name: str) -> Path:
    return APPROACH_CACHE / name
