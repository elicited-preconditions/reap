"""eval_protocol.py for policies trained on the reap-unified domain (joint multi-environment training).

Registers the per-environment evaluation "domains" u-confined / u-shelf / u-tabletop (ID + OOD cells = unified copies
of each environment's usual sets, dataset $U_EVAL_DATA, default joint), then calls eval_protocol.main(): same GBFS,
same rows.
One difference: eval_protocol reads MP calls from the checker registered under the problem's DOMAIN name; for reap-unified
that is not a checker (sim keys checkers by the '; reap-env:' tag), so mp_calls would read 0. Here the lookup for
'reap-unified' returns the run's single environment checker (one u-* domain per invocation).

Registration and the patch run at import time, so eval_protocol's spawned --workers processes (which re-import this
file as __mp_main__) get them too.

    CONFINED_CHECKER=honest SHELF_ACCESS_CHECKER=v2 \\
    python -m reap.eval.joint_protocol --domain u-confined --set OOD --subdir eval_confined_nonmonotone_16 \\
        --model_path <ckpt dir> --snapshot live --label <label> --output_dir <dir> --time_limit_s 100 --workers 10
"""
import argparse
import os
import sys
from pathlib import Path

from reap.eval.eval_config import EVAL_DOMAINS, Cell, Dom  # noqa: E402

_TT = ("obj10", "obj12", "obj14", "obj16", "obj18", "obj20", "obj22", "obj24", "obj10_plan3", "obj12_plan3",
       "obj14_plan5", "obj16_plan5", "obj10_plan5_biased2", "obj12_plan7_biased3", "obj14_plan9_biased3")
# per environment of the joint dataset (data/joint): the ID set and the OOD sets
SETS = {"confined": ("eval_confined", tuple("eval_confined_nonmonotone_%d" % k for k in (10, 12, 14, 16))),
        "shelf": ("eval_multilevel_shelf", ("eval_multilevel_shelf_ood",)),
        "tabletop": ("eval_tabletop/obj10", tuple("eval_tabletop/" + s for s in _TT))}


def register(D):
    """Register the per-environment evaluation domains u-confined / u-shelf / u-tabletop of the dataset ``D``."""
    for env, (id_set, cells) in SETS.items():
        EVAL_DOMAINS["u-" + env] = Dom(
            data=D, id_set=id_set, ood_set=cells[0], train_set="train", wandb_project="joint", obj_prefixes=(),
            domain_file="domain.pddl",
            ood_cells=tuple(Cell(c, D, c, domain_file="domain.pddl") for c in cells))


register(os.environ.get("U_EVAL_DATA") or "joint")

from reap.eval import eval_protocol as EP  # noqa: E402
from reap.rl.av_star import sim  # noqa: E402

_get = sim.get_checker_for


def _checker_for(domain_name):
    ck = _get(domain_name)
    if ck is None and domain_name == sim.UNIFIED_DOMAIN and len(sim._CHECKERS) == 1:
        ck = next(iter(sim._CHECKERS.values()))
    return ck


sim.get_checker_for = _checker_for

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=sorted(k for k in EVAL_DOMAINS if k.startswith("u-")))
    ap.add_argument("--set", dest="set_kind", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--time_limit_s", type=int, default=300)
    ap.add_argument("--max_budget", type=int, default=16384)
    ap.add_argument("--snapshot", default="live", choices=("best", "live"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--subdir", default=None)
    a = ap.parse_args(argv)
    EP.main(a.domain, a.set_kind, a.model_path, a.label, output_dir=a.output_dir, time_limit_s=a.time_limit_s,
            max_budget=a.max_budget, snapshot=a.snapshot, workers=a.workers, subdir=a.subdir)


if __name__ == "__main__":
    main()
