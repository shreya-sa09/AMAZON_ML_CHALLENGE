"""Option B full-corpus run: executes the FROZEN approaches/FINAL_BEST_APPROACH/src/final_run.py code unchanged, with its
output/cache/results paths redirected into approaches/FINAL_CANDIDATE_VERIFY/. FINAL_BEST_APPROACH is not written.
Run with PYTHONDONTWRITEBYTECODE=1 (so no __pycache__ appears inside FINAL_BEST_APPROACH/src)."""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
sys.path.insert(0, str(ROOT / "approaches/FINAL_BEST_APPROACH/src"))
import final_run  # noqa: E402

assert not final_run.SMOKE, "full run only"
final_run.CACHE = APP / "cache"
final_run.RES = APP / "results"
final_run.OUT = APP / "output"
for d in (final_run.CACHE, final_run.RES, final_run.OUT):
    d.mkdir(parents=True, exist_ok=True)
final_run.log(f"redirected outputs -> {final_run.OUT}; cache -> {final_run.CACHE}")
final_run.main()
