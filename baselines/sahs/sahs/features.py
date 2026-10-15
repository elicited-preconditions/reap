"""Feature tensors of the SAHS networks (TripleRelationalQNet, gnn.py) from the predicate set of a state.

The encoding is the one of the IJRR 2022 version (Kim, Shimanuki, Kaelbling and Lozano-Perez, "Representation,
learning, and planning algorithms for geometric task and motion planning", Section 5.2 and Appendix A.3), the edge
feature set "ijrr22". Only objects (the cylinders) are graph nodes; a region (a cell) enters through the edge vector,
which gnn.py's theta3 embeds:
  Node(oi)         = x_oi = [IsObject, IsRegion, IsGoal, PreFree]
  Edge(oi, oj, rk) = [x_oi, x_oj, x_oi_oj, x_oj_oi, x_oi_rk, x_oj_rk, x_oi_oj_rk, x_oj_oi_rk]   (width 22)
The binary slot is [InRegion, OccludesPre, ManipFree]: x_oi_oj holds OccludesPre(oi, oj), x_oi_rk holds InRegion and
ManipFree, and the ternary x_oi_oj_rk is OccludesManip(oi, oj, rk). The paper prints OCCLUDESMANIP(ei,ej) in x_eiej;
OccludesManip is ternary and has its own slot, so the binary slot holds OccludesPre. Both directions are read from the
same PredicateSet, which holds the occlusion predicates of every ordered object pair, so the reverse-direction terms
need no extra checker calls. A checkpoint's edge feature set follows from its edge width (gnn.load_qnet).
"""
from __future__ import annotations

import torch

NULL_REGION = "__null_region__"

UNARY_DIM = 4  # IsObject, IsRegion, IsGoal, PreFree -- Node(oi), objects only
BINARY_DIM = 3  # InRegion, OccludesPre, ManipFree -- shared vector shape for both
# x_oi_oj (object-object: OccludesPre populated, InRegion/ManipFree always 0)
# and x_oi_rk (object-region: InRegion/ManipFree populated, OccludesPre always 0)
TERNARY_DIM = 1  # OccludesManip -- x_oi_oj_rk

EDGE_DIMS = {
    "ijrr22": 2 * UNARY_DIM + 2 * BINARY_DIM + 2 * BINARY_DIM + 2 * TERNARY_DIM,  # 22
}
DEFAULT_EDGE_FEATURES = "ijrr22"
EDGE_DIM = EDGE_DIMS[DEFAULT_EDGE_FEATURES]


def edge_dim(feature_set: str) -> int:
    if feature_set not in EDGE_DIMS:
        raise ValueError(f"unknown edge-feature set {feature_set!r} (known: {sorted(EDGE_DIMS)})")
    return EDGE_DIMS[feature_set]


def feature_set_for_edge_dim(dim: int) -> str:
    for name, d in EDGE_DIMS.items():
        if d == dim:
            return name
    raise ValueError(f"no edge-feature set has width {dim} (known: {EDGE_DIMS})")


def entities_for_pick(cylinders: list[str]) -> tuple[list[str], list[str]]:
    """Q_pick has no real region choice -- collapse the region set to one
    synthetic all-zero-feature 'null region' (one architecture class, two
    instantiations, no separate degenerate code path)."""
    return list(cylinders), [NULL_REGION]


def entities_for_put(cylinders: list[str], regions: list[str]) -> tuple[list[str], list[str]]:
    return list(cylinders), list(regions)


def _unary(entity: str, predicate_set, target: str) -> list[float]:
    """Node(oi) -- objects only, matching the paper (regions never get a
    unary Node vector of their own in this design)."""
    return [
        1.0,
        0.0,
        1.0 if entity == target else 0.0,
        1.0 if predicate_set.pre_free.get(entity, False) else 0.0,
    ]


def _binary_object_object(oi: str, oj: str, predicate_set) -> list[float]:
    occludes_pre = 1.0 if predicate_set.occludes_pre.get((oi, oj), False) else 0.0
    return [0.0, occludes_pre, 0.0]


def _binary_object_region(oi: str, rk: str, predicate_set) -> list[float]:
    in_region = 1.0 if predicate_set.in_region.get(oi) == rk else 0.0
    manip_free = 1.0 if predicate_set.manip_free.get((oi, rk), False) else 0.0
    return [in_region, 0.0, manip_free]


def build_edge_tensors(
    object_entities: list[str],
    region_entities: list[str],
    predicate_set,
    target: str,
    feature_set: str = DEFAULT_EDGE_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (x_obj, edge): x_obj (N_obj, UNARY_DIM) is Node(oi) for each
    object; edge (N_obj, N_obj, R, edge_dim(feature_set)) is Edge(oi,oj,rk)
    for every ordered object pair (oi != oj) and region."""
    n_obj = len(object_entities)
    r = len(region_entities)

    x_obj = torch.zeros(n_obj, UNARY_DIM)
    unary_cache = {}
    for i, o in enumerate(object_entities):
        v = _unary(o, predicate_set, target)
        unary_cache[o] = torch.tensor(v)
        x_obj[i] = unary_cache[o]

    edge = torch.zeros(n_obj, n_obj, r, edge_dim(feature_set))
    for i, oi in enumerate(object_entities):
        for j, oj in enumerate(object_entities):
            if i == j:
                continue
            x_oi = unary_cache[oi]
            x_oj = unary_cache[oj]
            x_oi_oj = torch.tensor(_binary_object_object(oi, oj, predicate_set))
            x_oj_oi = torch.tensor(_binary_object_object(oj, oi, predicate_set))
            for k, rk in enumerate(region_entities):
                x_oi_rk = torch.tensor(_binary_object_region(oi, rk, predicate_set))
                x_oj_rk = torch.tensor(_binary_object_region(oj, rk, predicate_set))
                x_oi_oj_rk = 1.0 if predicate_set.occludes_manip.get((oi, oj, rk), False) else 0.0
                x_oj_oi_rk = 1.0 if predicate_set.occludes_manip.get((oj, oi, rk), False) else 0.0
                edge[i, j, k] = torch.cat([
                    x_oi, x_oj, x_oi_oj, x_oj_oi, x_oi_rk, x_oj_rk,
                    torch.tensor([x_oi_oj_rk, x_oj_oi_rk]),
                ])

    return x_obj, edge


def delta_index(object_entities: list[str], region_entities: list[str], delta: dict) -> tuple[int, int]:
    """Row/col index of a {"object":.., "region":..} delta into the Q-table
    returned by TripleRelationalQNet.forward (delta["region"]=None maps to
    index 0, matching the null-region convention for Q_pick)."""
    row = object_entities.index(delta["object"])
    region = delta.get("region")
    col = 0 if region is None else region_entities.index(region)
    return row, col
