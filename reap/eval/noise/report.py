"""The table of the appendix "Execution Under Perturbed Observations" (Completed and Recovered, REAP-GBFS / REAP-greedy,
per perturbation and level) from the sweep cells. The shared unperturbed cells (level 0) are not part of the table."""
from __future__ import annotations

from pathlib import Path
from typing import List

from reap.eval.noise import metrics

PERTURBATIONS = [("n1", "Placement drift, sigma", lambda v: "%g mm" % (1000 * v)),
                 ("n2", "Occupancy errors, p", lambda v: "%.2f" % v),
                 ("n3", "Grasp failures, p", lambda v: "%.2f" % v)]
MODES = ("gbfs", "greedy")


def _pct(x):
    return f"{100.0 * x:.1f}" if x is not None else "n/a"


def render_table(cells: List[dict]) -> str:
    lines = ["| Perturbation | Level | Completed (%) | Recovered (%) |", "|---|---|---|---|"]
    for noise, label, level in PERTURBATIONS:
        rows = {(r["value"], r["mode"]): r for r in metrics.per_noise_model_table(cells, noise)}
        for v in sorted({v for v, _ in rows if v}):
            by = [rows.get((v, m)) for m in MODES]
            lines.append("| %s | %s | %s | %s |" % (
                label, level(v),
                " / ".join(_pct(r["fraction_completed_without_replanning"]) if r else "n/a" for r in by),
                " / ".join(_pct(r["replan_recovery_rate"]) if r else "n/a" for r in by)))
    return "\n".join(lines)


def build_report(sweep_dir: Path, baseline_json: Path = None) -> str:
    cells = metrics.load_all_sweep_cells(sweep_dir)
    return "\n".join([
        "# Execution under perturbation, REAP-GBFS / REAP-greedy\n",
        "Completed: episodes finished without replanning; recovered: episodes that succeed after replanning, "
        "among those that replan.\n",
        render_table(cells), "",
    ])


def main(argv=None):
    import argparse
    from reap.paths import out_root
    ap = argparse.ArgumentParser(description="The table of the appendix \"Execution Under Perturbed Observations\" "
                                             "from the sweep cells (reap.eval.noise.sweep).")
    ap.add_argument("sweep_dir", nargs="?", default=None, help="directory of the sweep cell JSONs "
                    "(default: $REAP_OUT/noise/sweep)")
    ap.add_argument("--out", default=None, help="markdown file to write (default: print)")
    a = ap.parse_args(argv)
    text = build_report(Path(a.sweep_dir) if a.sweep_dir else out_root() / "noise" / "sweep")
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
