"""Domain table for the uniform GBFS eval protocol.

ONE place to extend: add a `Dom(...)` entry keyed by the registered PDDL
domain name (must already be registered in reap.rl.av_star.sim).
Everything else (eval_protocol.py, reproduce_eval.py) is domain-agnostic and reads from here.

Per `Dom` entry:
  data          InputData domain dir name (under $REAP_DATA, or under `pddl_domains_dir` when set)
  domain_file   domain file to load ("domain.pddl")
  id_set        subdir for in-distribution eval
  ood_set       the first OOD subdir (== ood_cells[0].subdir)
  train_set     subdir for the training distribution (metadata only)
  wandb_project wandb project name of the environment (metadata only)
  obj_prefixes  object-name prefixes counted as "blocks/movables" for the
                dataset metadata (object-count distribution)
  pddl_domains_dir  root dir holding `data` (None => $REAP_DATA)
  ood_cells     the full (possibly multi-bucket) OOD set as `Cell`s

Per `Cell` entry (one OOD bucket):
  label             short bucket label, e.g. "eval_ood", "obj14_plan9_biased3"
  data              InputData data dir name
  subdir            validation subdir within `data`
  domain_file       domain file to load
  pddl_domains_dir  root dir holding `data` (None => default)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    label: str           # short bucket label, e.g. "eval_ood", "obj14_plan9_biased3"
    data: str            # InputData data dir name
    subdir: str          # the set directory within data (e.g. "eval", "eval/obj10")
    domain_file: str = "domain.pddl"
    pddl_domains_dir: str | None = None   # None => $REAP_DATA


@dataclass(frozen=True)
class Dom:
    data: str
    id_set: str                 # ID validation subdir
    ood_set: str                # the first OOD subdir (= ood_cells[0].subdir)
    train_set: str
    wandb_project: str
    obj_prefixes: tuple[str, ...]
    domain_file: str = "domain.pddl"
    pddl_domains_dir: str | None = None
    ood_cells: tuple[Cell, ...] = ()    # the full OOD set (multi-cell)


def _tabletop_ood_cells() -> tuple[Cell, ...]:
    """The 15 tabletop OOD buckets.

    Each lives under data/tabletop/eval/<label> and uses data/tabletop/domain.pddl.
    """
    labels = (
        "obj10", "obj12", "obj14", "obj16", "obj18", "obj20", "obj22", "obj24",
        "obj10_plan3", "obj12_plan3", "obj14_plan5", "obj16_plan5",
        "obj10_plan5_biased2", "obj12_plan7_biased3", "obj14_plan9_biased3",
    )
    return tuple(
        Cell(
            label=lbl,
            data="tabletop",
            subdir=f"eval/{lbl}",
        )
        for lbl in labels
    )


# Keyed by the registered domain name (== state.problem.domain.name).
EVAL_DOMAINS: dict[str, Dom] = {
    "confined-shelf": Dom(
        data="confined", id_set="eval", ood_set="eval_nonmonotone_10",
        train_set="train", wandb_project="confined", obj_prefixes=("cyl_",),
        ood_cells=(
            Cell("eval_nonmonotone_10", "confined", "eval_nonmonotone_10"),
            Cell("eval_nonmonotone_12", "confined", "eval_nonmonotone_12"),
            Cell("eval_nonmonotone_14", "confined", "eval_nonmonotone_14"),
            Cell("eval_nonmonotone_16", "confined", "eval_nonmonotone_16"),
        ),
    ),
    "shelf-access": Dom(
        data="multilevel_shelf", id_set="eval", ood_set="eval_ood",
        train_set="train", wandb_project="multilevel_shelf",
        obj_prefixes=("blocker_", "ooi"),
        ood_cells=(
            Cell("eval_ood", "multilevel_shelf", "eval_ood"),
        ),
    ),
    "multilevel-blocks": Dom(
        data="stacking", id_set="eval", ood_set="eval_ood",
        train_set="train", wandb_project="stacking",
        obj_prefixes=("cube_", "oblong_", "long_"),
        ood_cells=(
            Cell("eval_ood", "stacking", "eval_ood"),
        ),
    ),
    "tabletop": Dom(
        data="tabletop", id_set="validation", ood_set="eval/obj14_plan9_biased3", train_set="train",
        wandb_project="tabletop", obj_prefixes=("cylinder_",),
        ood_cells=_tabletop_ood_cells(),
    ),
}


def set_subdir(domain: str, set_kind: str) -> str:
    """Map ('ID'|'OOD') -> the subdir name for `domain`."""
    d = EVAL_DOMAINS[domain]
    return {"ID": d.id_set, "OOD": d.ood_set}[set_kind]
