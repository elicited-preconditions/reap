"""The SAHS abstract predicate set of one xmimir XState in REAP's open-tabletop domain, computed with the
InstrumentedFeasibilityChecker.

IsObject/IsRegion/IsGoal/InRegion are trivial/symbolic (no motion-planner
calls). PreFree/ManipFree cost one MP call each.

OccludesPre/OccludesManip have two implementations, selected by whether an
`occlusion_checker` is passed:
- `occlusion_checker=None` (default): counterfactual-removal approximation.
  PreFree/ManipFree's own result is reused as the "baseline" half, so each
  occlusion query costs one *additional* MP call per candidate blocker --
  O(N) planner calls per target.
- `occlusion_checker=<OcclusionChecker>`: the SAHS computation (occlusion.py):
  plan once ignoring collisions, check what the resulting path hits. O(1) planner calls per target regardless of N.
  Requires the PreFree/ManipFree failure `reason` (captured into
  `pre_free_reason`/`manip_free_reason`) to decide whether attempting the
  ignore-collision fallback even makes sense -- see occlusion.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import reap.xmimir as xmi

from sahs import state_adapter
from sahs.checker import InstrumentedFeasibilityChecker
from sahs.occlusion import OcclusionChecker


@dataclass
class PredicateSet:
    is_object: list[str]
    is_region: list[str]
    is_goal: dict[str, bool]
    in_region: dict[str, str | None]
    pre_free: dict[str, bool]
    manip_free: dict[tuple[str, str], bool]
    occludes_pre: dict[tuple[str, str], bool]
    occludes_manip: dict[tuple[str, str, str], bool]
    pre_free_reason: dict[str, str] = field(default_factory=dict)
    manip_free_reason: dict[tuple[str, str], str] = field(default_factory=dict)


def compute_predicate_set(
    state: xmi.XState,
    target: str,
    cylinders: list[str],
    all_cells: list[str],
    checker: InstrumentedFeasibilityChecker,
    regions: list[str],
    occlusion_checker: OcclusionChecker | None = None,
) -> PredicateSet:
    base_rs = state_adapter.state_to_robot_state(state)

    # PreFree(o): one MP call per object.
    pre_free: dict[str, bool] = {}
    pre_free_reason: dict[str, str] = {}
    for o in cylinders:
        feasible, info = checker.check(base_rs, "pick", o, predicate_label="PreFree")
        pre_free[o] = feasible
        pre_free_reason[o] = info.get("reason", "")

    # ManipFree(o,r): one MP call per (object, region); needs o "as-if held".
    held_rs = {o: state_adapter.with_synthetic_holding(base_rs, o) for o in cylinders}
    manip_free: dict[tuple[str, str], bool] = {}
    manip_free_reason: dict[tuple[str, str], str] = {}
    for o in cylinders:
        for r in regions:
            feasible, info = checker.check(held_rs[o], "put", o, r, predicate_label="ManipFree")
            manip_free[(o, r)] = feasible
            manip_free_reason[(o, r)] = info.get("reason", "")

    occludes_pre: dict[tuple[str, str], bool] = {}
    occludes_manip: dict[tuple[str, str, str], bool] = {}

    if occlusion_checker is not None:
        for o2 in cylinders:
            candidates = [o1 for o1 in cylinders if o1 != o2]
            row = occlusion_checker.occludes_pre_row(base_rs, o2, candidates, pre_free_reason[o2])
            for o1, v in row.items():
                occludes_pre[(o1, o2)] = v

        for o2 in cylinders:
            for r in regions:
                candidates = [o1 for o1 in cylinders if o1 != o2]
                row = occlusion_checker.occludes_manip_row(
                    held_rs[o2], o2, r, candidates, manip_free_reason[(o2, r)],
                )
                for o1, v in row.items():
                    occludes_manip[(o1, o2, r)] = v
    else:
        # OccludesPre(o1,o2): true iff pick(o2) is infeasible with o1 present
        # (reuses pre_free[o2], no extra call) AND feasible with o1 removed
        # (one new MP call).
        for o2 in cylinders:
            if pre_free[o2]:
                # already feasible with everyone present -> nothing occludes it.
                for o1 in cylinders:
                    if o1 != o2:
                        occludes_pre[(o1, o2)] = False
                continue
            for o1 in cylinders:
                if o1 == o2:
                    continue
                rs_without_o1 = state_adapter.without_object(base_rs, o1)
                feasible_without, _ = checker.check(
                    rs_without_o1, "pick", o2, predicate_label="OccludesPre",
                )
                occludes_pre[(o1, o2)] = feasible_without

        # OccludesManip(o1,o2,r): same pattern, but against manip_free[(o2,r)]
        # and with o2 already synthetically held.
        for o2 in cylinders:
            for r in regions:
                if manip_free[(o2, r)]:
                    for o1 in cylinders:
                        if o1 != o2:
                            occludes_manip[(o1, o2, r)] = False
                    continue
                for o1 in cylinders:
                    if o1 == o2:
                        continue
                    rs_without_o1 = state_adapter.without_object(held_rs[o2], o1)
                    feasible_without, _ = checker.check(
                        rs_without_o1, "put", o2, r, predicate_label="OccludesManip",
                    )
                    occludes_manip[(o1, o2, r)] = feasible_without

    return PredicateSet(
        is_object=list(cylinders),
        is_region=list(all_cells),
        is_goal={o: (o == target) for o in cylinders},
        in_region={o: base_rs["cylinders"].get(o, [None])[0] for o in cylinders},
        pre_free=pre_free,
        manip_free=manip_free,
        occludes_pre=occludes_pre,
        occludes_manip=occludes_manip,
        pre_free_reason=pre_free_reason,
        manip_free_reason=manip_free_reason,
    )


def compute_predicate_set_incremental(
    state: xmi.XState,
    target: str,
    cylinders: list[str],
    all_cells: list[str],
    pool,
    regions: list[str],
    parent_ps: PredicateSet,
    parent_regions: list[str],
    moved_object: str,
    deadline: float | None = None,
) -> PredicateSet:
    """Incremental version of `compute_predicate_set_parallel`: the predicates of a child state are taken over from
    its parent except where the last action can have changed them. This is the reuse rule of the SAHS authors' code
    for the one-arm domain (`OneArmPaPState.initialize_pap_pick_place_params`, github.com/beomjoonkim/guiding_gtamp,
    trajectory_representation/one_arm_pap_state.py), which keeps the cached pick-and-place parameters of every
    (object, region) pair except those of the object the last action moved, and of the paper's supplement ("many
    predicates will not change... so we reuse the predicate value without recomputing it"). Their code also
    re-validates a reused witness pose with a collision check; tampanda's ActionFeasibilityChecker.check() returns no
    path, so that step is left out. A carried-over value can be out of date: e.g. occludes_pre(o1, o2) with o1, o2 !=
    moved_object is kept although moved_object may now block or unblock that pair.

    A predicate is refreshed this round iff its OWN gate could have changed:
    - pre_free/manip_free: fresh only for `moved_object` (all regions), plus
      any OTHER object at a region that's newly available this round (a
      region with no parent-state entry to reuse at all -- only possible
      after a pick frees up the moved object's old cell).
    - occludes_pre/occludes_manip: fresh only for rows whose gate
      (pre_free_reason/manip_free_reason) was itself refreshed above --
      i.e. target=moved_object, or (for manip) a newly-available region.
      Everything else (including moved_object's status as a BLOCKER of
      other, unrefreshed targets) is carried forward from `parent_ps`.
    """
    base_rs = state_adapter.state_to_robot_state(state)
    held_rs = {o: state_adapter.with_synthetic_holding(base_rs, o) for o in cylinders}

    region_set = set(regions)
    parent_region_set = set(parent_regions)
    new_regions = [r for r in regions if r not in parent_region_set]

    # ---- Round 1: pre_free / manip_free ----
    round1_keys: list[tuple] = [("pre_free", moved_object)]
    round1_requests: list[tuple[str, tuple]] = [("pre_free", (base_rs, moved_object))]

    for r in regions:
        round1_keys.append(("manip_free", moved_object, r))
        round1_requests.append(("manip_free", (held_rs[moved_object], moved_object, r)))
    for o in cylinders:
        if o == moved_object:
            continue
        for r in new_regions:
            round1_keys.append(("manip_free", o, r))
            round1_requests.append(("manip_free", (held_rs[o], o, r)))

    round1_results = pool.dispatch_batch(round1_requests, deadline=deadline)

    pre_free = dict(parent_ps.pre_free)
    pre_free_reason = dict(parent_ps.pre_free_reason)
    manip_free = {(o, r): v for (o, r), v in parent_ps.manip_free.items() if r in region_set}
    manip_free_reason = {
        (o, r): v for (o, r), v in parent_ps.manip_free_reason.items() if r in region_set
    }

    for key, (feasible, reason) in zip(round1_keys, round1_results):
        if key[0] == "pre_free":
            _, o = key
            pre_free[o] = feasible
            pre_free_reason[o] = reason
        else:
            _, o, r = key
            manip_free[(o, r)] = feasible
            manip_free_reason[(o, r)] = reason

    # ---- Round 2: occludes_pre / occludes_manip ----
    round2_keys: list[tuple] = []
    round2_requests: list[tuple[str, tuple]] = []

    moved_candidates = [o1 for o1 in cylinders if o1 != moved_object]
    round2_keys.append(("occludes_pre", moved_object))
    round2_requests.append(
        ("occludes_pre_row", (base_rs, moved_object, moved_candidates, pre_free_reason[moved_object]))
    )
    for r in regions:
        round2_keys.append(("occludes_manip", moved_object, r))
        round2_requests.append(
            (
                "occludes_manip_row",
                (held_rs[moved_object], moved_object, r, moved_candidates, manip_free_reason[(moved_object, r)]),
            )
        )
    for o in cylinders:
        if o == moved_object:
            continue
        for r in new_regions:
            candidates_o = [o1 for o1 in cylinders if o1 != o]
            round2_keys.append(("occludes_manip", o, r))
            round2_requests.append(
                (
                    "occludes_manip_row",
                    (held_rs[o], o, r, candidates_o, manip_free_reason[(o, r)]),
                )
            )

    round2_results = pool.dispatch_batch(round2_requests, deadline=deadline)

    occludes_pre = dict(parent_ps.occludes_pre)
    occludes_manip = {k: v for k, v in parent_ps.occludes_manip.items() if k[2] in region_set}

    for key, row in zip(round2_keys, round2_results):
        if key[0] == "occludes_pre":
            _, o2 = key
            for o1, v in row.items():
                occludes_pre[(o1, o2)] = v
        else:
            _, o2, r = key
            for o1, v in row.items():
                occludes_manip[(o1, o2, r)] = v

    return PredicateSet(
        is_object=list(cylinders),
        is_region=list(all_cells),
        is_goal={o: (o == target) for o in cylinders},
        in_region={o: base_rs["cylinders"].get(o, [None])[0] for o in cylinders},
        pre_free=pre_free,
        manip_free=manip_free,
        occludes_pre=occludes_pre,
        occludes_manip=occludes_manip,
        pre_free_reason=pre_free_reason,
        manip_free_reason=manip_free_reason,
    )


def compute_predicate_set_parallel(
    state: xmi.XState,
    target: str,
    cylinders: list[str],
    all_cells: list[str],
    pool,
    regions: list[str],
    deadline: float | None = None,
) -> PredicateSet:
    """`compute_predicate_set(..., occlusion_checker=<OcclusionChecker>)` on the parallel pool (the occlusion
    computation of occlusion.py only; the removal test needs one planner call per candidate blocker).

    `pool` is a `parallel_pool.ParallelFeasibilityPool` (untyped here to
    avoid a hard import of `multiprocessing`-touching code into this
    otherwise-pure module). Two dispatch rounds, matching the real
    dependency structure: PreFree/ManipFree (round 1) are independent of
    each other and of everything else, but OccludesPre/OccludesManip
    (round 2) each need their own PreFree/ManipFree `reason` to decide
    whether attempting the ignore-collision fallback makes sense at all
    (see occlusion.py) -- so round 2 cannot be dispatched until round 1's
    results are back.
    """
    base_rs = state_adapter.state_to_robot_state(state)
    held_rs = {o: state_adapter.with_synthetic_holding(base_rs, o) for o in cylinders}

    round1_keys: list[tuple] = []
    round1_requests: list[tuple[str, tuple]] = []
    for o in cylinders:
        round1_keys.append(("pre_free", o))
        round1_requests.append(("pre_free", (base_rs, o)))
    for o in cylinders:
        for r in regions:
            round1_keys.append(("manip_free", o, r))
            round1_requests.append(("manip_free", (held_rs[o], o, r)))

    round1_results = pool.dispatch_batch(round1_requests, deadline=deadline)

    pre_free: dict[str, bool] = {}
    pre_free_reason: dict[str, str] = {}
    manip_free: dict[tuple[str, str], bool] = {}
    manip_free_reason: dict[tuple[str, str], str] = {}
    for key, (feasible, reason) in zip(round1_keys, round1_results):
        if key[0] == "pre_free":
            _, o = key
            pre_free[o] = feasible
            pre_free_reason[o] = reason
        else:
            _, o, r = key
            manip_free[(o, r)] = feasible
            manip_free_reason[(o, r)] = reason

    round2_keys: list[tuple] = []
    round2_requests: list[tuple[str, tuple]] = []
    for o2 in cylinders:
        candidates = [o1 for o1 in cylinders if o1 != o2]
        round2_keys.append(("occludes_pre", o2))
        round2_requests.append(
            ("occludes_pre_row", (base_rs, o2, candidates, pre_free_reason[o2]))
        )
    for o2 in cylinders:
        for r in regions:
            candidates = [o1 for o1 in cylinders if o1 != o2]
            round2_keys.append(("occludes_manip", o2, r))
            round2_requests.append(
                (
                    "occludes_manip_row",
                    (held_rs[o2], o2, r, candidates, manip_free_reason[(o2, r)]),
                )
            )

    round2_results = pool.dispatch_batch(round2_requests, deadline=deadline)

    occludes_pre: dict[tuple[str, str], bool] = {}
    occludes_manip: dict[tuple[str, str, str], bool] = {}
    for key, row in zip(round2_keys, round2_results):
        if key[0] == "occludes_pre":
            _, o2 = key
            for o1, v in row.items():
                occludes_pre[(o1, o2)] = v
        else:
            _, o2, r = key
            for o1, v in row.items():
                occludes_manip[(o1, o2, r)] = v

    return PredicateSet(
        is_object=list(cylinders),
        is_region=list(all_cells),
        is_goal={o: (o == target) for o in cylinders},
        in_region={o: base_rs["cylinders"].get(o, [None])[0] for o in cylinders},
        pre_free=pre_free,
        manip_free=manip_free,
        occludes_pre=occludes_pre,
        occludes_manip=occludes_manip,
        pre_free_reason=pre_free_reason,
        manip_free_reason=manip_free_reason,
    )
