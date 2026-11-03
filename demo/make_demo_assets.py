"""Generate demo images: a clean sheet, a sheet with planted errors, the legend, and a noisy scan.

  python demo/make_demo_assets.py        ->  demo/*.png
"""
from pathlib import Path

from pid.synth.generator import make_sheet, render_legend

out = Path(__file__).parent
make_sheet(seed=11, n_rows=2).image.save(out / "clean_sheet.png")
planted = make_sheet(seed=11, n_rows=2, errors=["missing_check_valve", "missing_relief"])
planted.image.save(out / "planted_errors.png")
make_sheet(seed=12, n_rows=3, errors=["missing_signal"], scan_noise=True).image.save(out / "noisy_scan.png")
render_legend().save(out / "legend.png")
print("planted:", planted.planted)
