# The graph encoding is adapted from github.com/tomsilver/ploi (guidance/gnn_guidance.py, _state_to_graph).
# MIT License, Copyright (c) 2020 Tom Silver and Rohan Chitnis; the notice is in LICENSES/ploi-MIT.txt.
"""Node/edge featurization for PLOI's object-importance GNN (Silver et al.,
AAAI 2021, Section 5).

Per-predicate dimension counts, derived from the paper's
text: the initial state I is a *total* assignment (no
presence bit needed -- every unary/binary property is either true or false
for every relevant object tuple), so I contributes 1 dim per unary predicate
and 2 dims per binary predicate (one per ordering). The goal G is *sparse*
(most properties aren't mentioned at all), so it contributes 2 dims per
unary predicate (presence + value) and 4 dims per binary predicate (2
orderings x 2), collapsing under the paper's own stated no-negation
simplification -- true here, since every goal in this dataset is a single
positive `(holding ?cyl)` literal -- to 1 dim per unary and 2 dims per
binary (presence == value when there is no negation).

    | contribution         | from I | from G (simplified) | total |
    |-----------------------|--------|----------------------|-------|
    | object type            | 1      | -- (never in :goal)  | 1     |
    | unary predicate         | 1      | 1                    | 2     |
    | binary predicate         | 2      | 2                    | 4     |

Domain predicates used, after two drops:
`gripper-empty` (arity-0, constant across the whole dataset -- true in
every problem's initial state by construction, never in any goal -- zero
training signal under any encoding) and `discarded` (never asserted true
anywhere in REAP's dataset; `allow_drop: False` on every generated
problem).

`adjacent` is arity-3 (`?dir ?cel1 ?cel2`), outside PLOI's arity<=2 scheme.
The paper's own text prescribes exactly the fix: "higher-order [properties]
can often be converted to an equivalent set of binary properties." Desugared
here into `adjacent-north`/`adjacent-east` (the domain's only two
`direction` constants) -- this is the only structural predicate that lets
K=3-hop message passing reason about cell-to-cell distance, so it is kept,
not dropped.

POSITIONAL NODE FEATURES (`pose-x`/`pose-y`, feature set "positional", off by
default; the paper's appendix: this variant does not perform better). Without them, a cylinder's location reaches the network only
through the `occupied(cell,cyl)` edge into the 70-cell lattice, so a
cylinder two cells away is 4 message-passing hops distant, while REAP's
occlusion span (frontal/south-approach blocking) extends up to 6 cells
within a column, outside K=3's receptive field. The paper's Section 3
defines exactly this case: "A property is a real-valued function on a tuple
of objects. For example, in the expression `pose(cup3) = 5.7`, the property
is `pose`... Predicates... are a special case of properties where the
output is binary." Appendix B does this for the paper's geometric domain
(PyBullet robotic simulation): "we treat the continuous object poses as
node features." So `pose-x`/`pose-y` are two UNARY real-valued properties,
with the same architecture as `UNARY_PREDICATES` below; each gets 1
dimension, I-only, per Section 5's formula (I is total, no presence bit
needed; the G-derived component is always zero in this dataset since
`pose` never appears in any goal here, so it is dropped, as for
`gripper-empty`/`discarded`).

- The reference GNN's graph is SPARSE, as here: `tomsilver/ploi`'s
  `gnn_guidance.py::_state_to_graph` does
  `adjacency_mat = np.any(all_edge_features, axis=2)` then
  `np.argwhere(adjacency_mat)`, so edges are kept only where at least one
  predicate holds.
- The exact PyBullet-domain encoding (raw vs. normalized, 2D vs. 3D,
  orientation) is not open source (neither `tomsilver/ploi` nor
  `tomsilver/pddlgym`, ploi's environment dependency, contains it). So the
  MECHANISM above (real-valued property -> 1 dim from I) follows the paper;
  the specific parameterization below (2D, no orientation, [-1,1]
  normalization) is an extension to REAP's flat 2D grid, not a reproduction
  of unpublished code.

Coordinate source: cell names encode grid indices, `cell_{ix}_{iy}`
(`adjacent east cell_0_0 cell_1_0` -> x increases via east;
`adjacent north cell_0_0 cell_0_1` -> y increases via north; tampanda's
`GridDomain._generate_cells` uses `cell_id = f"cell_{ix}_{iy}"`). A cell's
ABSOLUTE physical center depends on the MuJoCo table geom's live position
(`GridDomain._parse_table_geometry`'s `geom_xpos`) and is not computable
without a running MuJoCo scene, but a NORMALIZED [-1,1] position needs only
`(ix, iy)` and the `config.GRID_CELL_SIZE`/`GRID_WORKING_AREA` constants
(`cells_x = working_area[0]/cell_size = 10`,
`cells_y = working_area[1]/cell_size = 7`, the 70-cell grid), with the
absolute table offset canceling out. So featurization needs no
MuJoCo/tampanda call: pure arithmetic on parsed cell names, like every
other feature in this file. The paper does not specify a normalization
(Section 3's `pose(cup3)=5.7` example is a raw value); [-1,1] is a choice
made here.

Applied to CYLINDER nodes only: cells are a fixed lattice with static
positions, and the paper's PyBullet domain has no discrete-cell
abstraction to follow.
"""
from __future__ import annotations

import os

import torch

from ploi import config

# The positional node features are off by default (feature set "default", 6 node inputs: the released weights of the
# Table 1 rows); the pose columns are the last two.
# set_feature_set("positional") switches them on (run_evaluation.py / train_model.py --features positional), as does
# PLOI_POSITIONAL_FEATURES=1 at import; run_evaluation.py takes the feature set from the weights it loads.
POSITIONAL = os.environ.get("PLOI_POSITIONAL_FEATURES", "0") == "1"

OBJECT_TYPES = ("cell", "cylinder")
UNARY_PREDICATES = ("empty", "holding")
# (predicate_name, direction_constant_or_None) -- direction_constant is only
# set for the desugared `adjacent` cases.
BINARY_SPECS = (("occupied", None), ("adjacent", "north"), ("adjacent", "east"))

FEATURE_SETS = {"default": False, "positional": True}  # feature set -> positional node features on


def _node_dim(positional: bool) -> int:
    return len(OBJECT_TYPES) + 2 * len(UNARY_PREDICATES) + (2 if positional else 0)  # 2 + 2*2 (+ 2) = 6 or 8


NODE_DIM = _node_dim(POSITIONAL)
EDGE_DIM = 4 * len(BINARY_SPECS)                             # 4*3 = 12


def feature_set() -> str:
    """The active feature set, "default" or "positional"."""
    return "positional" if POSITIONAL else "default"


def set_feature_set(name: str) -> None:
    """Selects the node features: "default" (6 inputs, the reported weights) or "positional" (8, + cylinder pose).
    Call it before building a model or featurizing a problem."""
    global POSITIONAL, NODE_DIM
    POSITIONAL = FEATURE_SETS[name]
    NODE_DIM = _node_dim(POSITIONAL)


def feature_set_of(state_dict) -> str:
    """The feature set an ObjectImportanceGNN state dict was trained with, from its node encoder's input width."""
    width = state_dict["node_encoder.net.0.weight"].shape[1]
    for name, positional in FEATURE_SETS.items():
        if _node_dim(positional) == width:
            return name
    raise ValueError(f"node encoder input width {width} matches no feature set "
                     f"({', '.join(f'{n}: {_node_dim(p)}' for n, p in FEATURE_SETS.items())})")

# int()-truncated to match GridDomain.__init__'s own
# `self.cells_x = int(working_area[0] / cell_size)` exactly -- a raw float
# division (0.3/0.04 = 7.5) overstates the grid extent relative to the real
# system's actual cell count (7), understating how close corner cells sit to
# the normalized +-1 boundary.
_CELLS_X = int(config.GRID_WORKING_AREA[0] / config.GRID_CELL_SIZE)  # 10
_CELLS_Y = int(config.GRID_WORKING_AREA[1] / config.GRID_CELL_SIZE)  # 7


def _cell_grid_indices(cell_name: str) -> tuple[int, int]:
    """Parses `cell_{ix}_{iy}` (tampanda's naming scheme, see module
    docstring)."""
    _, ix, iy = cell_name.split("_")
    return int(ix), int(iy)


def _normalized_cell_position(cell_name: str) -> tuple[float, float]:
    """Cell center, normalized to roughly [-1, 1] over the grid extent --
    see module docstring for why the absolute table offset can be ignored."""
    ix, iy = _cell_grid_indices(cell_name)
    norm_x = 2 * (ix + 0.5) / _CELLS_X - 1
    norm_y = 2 * (iy + 0.5) / _CELLS_Y - 1
    return norm_x, norm_y


def _object_list(problem):
    """All (cell + cylinder) objects declared in the problem -- excludes the
    domain's `north`/`east` direction constants, which xmimir's
    XProblem.objects already does not include."""
    return list(problem.objects)


def build_node_features(problem) -> tuple[list, dict, torch.Tensor]:
    """Returns (objects, name_to_index, x) where x has shape (n_objects, NODE_DIM)."""
    objects = _object_list(problem)
    idx = {o.name: i for i, o in enumerate(objects)}
    x = torch.zeros(len(objects), NODE_DIM)

    for i, o in enumerate(objects):
        type_col = OBJECT_TYPES.index(o.type_.name)
        x[i, type_col] = 1.0

    base = len(OBJECT_TYPES)
    for k, pred in enumerate(UNARY_PREDICATES):
        i_col, g_col = base + 2 * k, base + 2 * k + 1
        for atom in problem.static_atoms:
            if atom.predicate.name == pred and atom.predicate.arity == 1:
                x[idx[atom.objects[0].name], i_col] = 1.0
        for lit in problem.initial_fluents:
            atom = lit.atom
            if lit.is_negated:
                continue
            if atom.predicate.name == pred and atom.predicate.arity == 1:
                x[idx[atom.objects[0].name], i_col] = 1.0
        for lit in problem.literal_goal():
            if lit.is_negated:
                continue  # no-negation simplification's precondition
            atom = lit.atom
            if atom.predicate.name == pred and atom.predicate.arity == 1:
                x[idx[atom.objects[0].name], g_col] = 1.0

    pose_x_col, pose_y_col = base + 2 * len(UNARY_PREDICATES), base + 2 * len(UNARY_PREDICATES) + 1
    for lit in problem.initial_fluents if POSITIONAL else ():
        if lit.is_negated:
            continue
        atom = lit.atom
        if atom.predicate.name == "occupied" and atom.predicate.arity == 2:
            cell_name, cyl_name = atom.objects[0].name, atom.objects[1].name
            norm_x, norm_y = _normalized_cell_position(cell_name)
            x[idx[cyl_name], pose_x_col] = norm_x
            x[idx[cyl_name], pose_y_col] = norm_y

    return objects, idx, x


def _desugared_args(atom, dir_const) -> tuple[str, str] | None:
    """For a plain binary predicate (dir_const is None), returns the atom's
    two object-name arguments. For `adjacent`, returns the two cell
    arguments only if the atom's direction argument matches dir_const,
    otherwise None (this atom belongs to the OTHER desugared predicate)."""
    names = [o.name for o in atom.objects]
    if dir_const is not None:
        if not names or names[0] != dir_const:
            return None
        names = names[1:]
    if len(names) != 2:
        return None
    return names[0], names[1]


def build_edge_features(problem, idx: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (edge_index, edge_attr). Per the paper's own stated
    optimization, edges with all-zero features are sparsified away entirely
    -- in this domain that prunes every cylinder-cylinder pair (no binary
    predicate here ever relates two cylinders) and every non-adjacent
    cell-cell pair."""
    dense: dict[tuple[int, int], torch.Tensor] = {}

    def cell(u: int, v: int) -> torch.Tensor:
        key = (u, v)
        if key not in dense:
            dense[key] = torch.zeros(EDGE_DIM)
        return dense[key]

    def mark(k: int, col_offset: int, a0: str, a1: str) -> None:
        if a0 not in idx or a1 not in idx:
            return
        u, v = idx[a0], idx[a1]
        # atom's own (a0, a1) order is the "forward" direction of the u->v
        # edge; the same atom also marks the reverse edge's "backward" dim,
        # so message passing in either direction sees the full relationship.
        cell(u, v)[4 * k + col_offset] = 1.0
        cell(v, u)[4 * k + col_offset + 1] = 1.0

    def scan(atoms_or_literals, is_literal: bool, col_offset: int) -> None:
        for item in atoms_or_literals:
            if is_literal:
                if item.is_negated:
                    continue
                atom = item.atom
            else:
                atom = item
            for k, (pred, dir_const) in enumerate(BINARY_SPECS):
                if atom.predicate.name != pred:
                    continue
                args = _desugared_args(atom, dir_const)
                if args is not None:
                    mark(k, col_offset, args[0], args[1])

    scan(problem.static_atoms, is_literal=False, col_offset=0)      # I, e.g. adjacent-*
    scan(problem.initial_fluents, is_literal=True, col_offset=0)    # I, e.g. occupied
    scan(problem.literal_goal(), is_literal=True, col_offset=2)     # G (simplified: presence==value)

    if not dense:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, EDGE_DIM)

    edge_index = torch.tensor(list(dense.keys()), dtype=torch.long).T
    edge_attr = torch.stack(list(dense.values()))
    return edge_index, edge_attr


def featurize(problem):
    """Full featurization of a single xmimir XProblem: returns
    (objects, name_to_index, node_features, edge_index, edge_attr)."""
    objects, idx, x = build_node_features(problem)
    edge_index, edge_attr = build_edge_features(problem, idx)
    return objects, idx, x, edge_index, edge_attr
