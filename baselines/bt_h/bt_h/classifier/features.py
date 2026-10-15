"""Token featurization for the feasibility classifier (Sec 4.2/Appendix C
of Cieslar et al., scaled down).

One token per (state, ground-lifted-action) pair in the FULL original task
plan pi_full (length n): for already-refined actions (0-indexed j < i,
where i = len(sigma)-1 = number of actions applied so far) the token uses
the real resulting state sigma[j+1]; for not-yet-refined actions (j >= i)
every token shares the latest known state sigma[i] == sigma[-1] (their true
future state is unknown).

"Relevant objects" per token = the acting cylinder plus up to R_MAXOBJ-1
spatially-nearest other cylinders in that state: unlike the paper's own domains, where every relevant object
apparently eventually becomes some action's own parameter, bystander
cylinders here can occlude without ever being an action's argument, so they
have to be pulled in by spatial neighborhood instead of by PDDL parameter
list.
"""
from __future__ import annotations

import random

from bt_h.grid import cell_coords

R_MAXOBJ = 6                # acting cylinder + up to 5 spatial neighbors
IDENTITY_ONEHOT_SIZE = 8    # randomized per training example (Sec 4.2 of the paper)
ROLES = ("on-table", "held", "parked")
ACTION_KINDS = ("pick", "relocate")

# per-object feature layout: [x, y, size_onehot(3), role_onehot(3), identity_onehot(8)]
OBJECT_FEATURE_SIZE = 2 + 3 + len(ROLES) + IDENTITY_ONEHOT_SIZE


def cylinder_role_at(cyl: str, m: int, pi_full: list) -> str:
    """m = number of actions applied to reach the state snapshot in
    question. Role is derived purely from which of cyl's own pick/relocate
    steps have index < m in pi_full -- the raw robot_state dict alone
    cannot distinguish 'parked' from 'on-table' (both just have a cell), so
    this walks the plan instead of reading the state dict for this part."""
    last_kind = None
    for a in pi_full[:m]:
        if a.cyl == cyl:
            last_kind = a.kind
    if last_kind is None:
        return "on-table"
    return {"pick": "held", "relocate": "parked"}[last_kind]


def cylinder_size_class(cyl: str) -> int:
    """0=thin, 1=medium, 2=thick, per tampanda's
    StateManager.CYLINDER_SPECS index pools (see bt_h.config)."""
    idx = int(cyl.split("_")[1])
    if idx < 15:
        return 0
    if idx < 25:
        return 1
    return 2


def _onehot(i: int, n: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def _object_feature(cyl: str, state: dict, m: int, pi_full: list, identity_map: dict) -> list[float]:
    role = cylinder_role_at(cyl, m, pi_full)
    cells = state["cylinders"].get(cyl)
    if cells:
        ix, iy = cell_coords(cells[0])
        xy = [ix / 9.0, iy / 6.0]  # normalize by grid extent (GRID_NX-1, GRID_NY-1)
    else:
        xy = [0.0, 0.0]  # held -- no cell, position not meaningful

    return (
        xy
        + _onehot(cylinder_size_class(cyl), 3)
        + _onehot(ROLES.index(role), len(ROLES))
        + _onehot(identity_map[cyl], IDENTITY_ONEHOT_SIZE)
    )


def _nearest_neighbors(cyl: str, state: dict, k: int) -> list[str]:
    others = [c for c in state["cylinders"] if c != cyl]
    if cyl not in state["cylinders"]:
        # acting cylinder currently held -- no anchor cell to rank by
        # distance from; fall back to a stable arbitrary order.
        return sorted(others, key=lambda c: int(c.split("_")[1]))[:k]
    cx, cy = cell_coords(state["cylinders"][cyl][0])

    def dist2(c: str) -> int:
        ox, oy = cell_coords(state["cylinders"][c][0])
        return (ox - cx) ** 2 + (oy - cy) ** 2

    return sorted(others, key=dist2)[:k]


def build_random_identity_map(cylinders: list[str], rng: random.Random) -> dict[str, int]:
    """A fresh, randomized cylinder-name -> one-hot-slot assignment per
    training example, so the classifier can't overfit to specific object
    instance identities (Sec 4.2 of the paper)."""
    slots = list(range(IDENTITY_ONEHOT_SIZE))
    rng.shuffle(slots)
    return {cyl: slots[j % IDENTITY_ONEHOT_SIZE] for j, cyl in enumerate(cylinders)}


def build_token_object_features(sigma: list[dict], pi_full: list, identity_map: dict) -> list[list[float]]:
    """One flattened R_MAXOBJ*OBJECT_FEATURE_SIZE vector per token
    (len(pi_full) tokens total, in plan order)."""
    i = len(sigma) - 1
    token_vectors = []
    for j, action in enumerate(pi_full):
        refined = j < i
        m = (j + 1) if refined else i
        state = sigma[m]
        neighbors = _nearest_neighbors(action.cyl, state, R_MAXOBJ - 1)
        objs = [action.cyl] + neighbors
        objs += [None] * (R_MAXOBJ - len(objs))  # zero-pad absent slots

        vec: list[float] = []
        for obj in objs:
            if obj is None:
                vec.extend([0.0] * OBJECT_FEATURE_SIZE)
            else:
                vec.extend(_object_feature(obj, state, m, pi_full, identity_map))
        token_vectors.append(vec)
    return token_vectors


def build_action_type_and_position_flags(sigma: list[dict], pi_full: list) -> tuple[list[list[float]], list[float]]:
    i = len(sigma) - 1
    latest_refined_j = i - 1  # 0-indexed action index of the most recently applied action, or -1 if none
    type_flags, latest_flags = [], []
    for j, action in enumerate(pi_full):
        type_flags.append(_onehot(ACTION_KINDS.index(action.kind), len(ACTION_KINDS)))
        latest_flags.append(1.0 if j == latest_refined_j else 0.0)
    return type_flags, latest_flags


def build_refined_flags(sigma: list[dict], pi_full: list) -> list[float]:
    i = len(sigma) - 1
    return [1.0 if j < i else 0.0 for j in range(len(pi_full))]


def featurize_example(sigma: list[dict], pi_full: list, rng: random.Random) -> dict:
    # Must cover every cylinder that could ever appear in a token's object
    # features, not just the ones acted on by pi_full: _nearest_neighbors
    # draws from state["cylinders"], which on instances with untouched
    # decoy cylinders (never picked/relocated; REAP's problems have up to
    # 20+ of these) includes objects that never appear as any action's own cylinder.
    # sigma[0]["cylinders"] is the authoritative full set: cylinders are
    # only ever removed from this dict (once held), never added.
    cylinders = sorted({a.cyl for a in pi_full} | set(sigma[0]["cylinders"]))
    identity_map = build_random_identity_map(cylinders, rng)
    type_flags, latest_flags = build_action_type_and_position_flags(sigma, pi_full)
    return {
        "object_features": build_token_object_features(sigma, pi_full, identity_map),
        "type_flags": type_flags,
        "latest_flags": latest_flags,
        "refined_flags": build_refined_flags(sigma, pi_full),
        "n_tokens": len(pi_full),
    }
