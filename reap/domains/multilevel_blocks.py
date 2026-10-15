"""Multi-level-blocks domain (3D stacking): feasibility checker and eval suites.

Wraps tampanda's grip-aware check
(:mod:`tampanda.symbolic.domains.multilevel_blocks.honest_grip`, ``check_action_grip``)
for use in the bwqs / GBFS search loops.

The PDDL domain name is ``multilevel-blocks`` (hyphen), per the
``(define (domain multilevel-blocks))`` line in
``tampanda/symbolic/domains/multilevel_blocks/pddl/domain.pddl``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import reap.xmimir as xmi

from tampanda.symbolic.domains.multilevel_blocks import MultilevelBlocksConfig

from reap.domains import checker_info
from reap.domains.registry import DomainSpec, EvalSuite, register_domain


class MultilevelBlocksFeasibilityChecker:
    """Feasibility check via the tampanda multilevel_blocks grip-aware executor.

    Each per-worker instance constructs its own MuJoCo env + workspace +
    config + executor on the first :meth:`check` call.  Results
    are cached on a key derived from the per-block cell occupancy and
    the action string.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._scratch: tempfile.TemporaryDirectory | None = None
        self._env = None
        self._ws = None
        self._cfg: MultilevelBlocksConfig | None = None
        self._executor = None
        self._inferred_signature: tuple | None = None  # (n_cubes, n_oblong, n_long)
        self._signature_cache: dict = {}  # problem -> signature, memoization
        self._cache: dict = {}
        # MLB_CHECKER must be set explicitly to "honest_grip" (checker_info.require): the grip-aware check (tampanda
        # multilevel_blocks/honest_grip.py), which needs the grip-aware domain (fluents grip-centre / grip-end); exact
        # 2 mm clearance, block-centric puts, the held block checked in every transit, release <= 2 mm above the
        # support where possible
        checker_info.require(checker_info.MLB_VAR)

    @classmethod
    def _infer_config(cls, problem: xmi.XProblem) -> MultilevelBlocksConfig:
        """Build a :class:`MultilevelBlocksConfig` matching the dataset.

        Block counts are read from the problem's object list (names follow
        ``{cube,oblong,long}_<idx>`` per tampanda's env_builder).  Grid
        sizes / table positions / cube extent use config defaults
        (15×15 parts, 10×10×5 stack, 30 mm cubes); problems with other
        grid dimensions are not supported.
        """
        n_cubes, n_oblong, n_long = cls._infer_signature(problem)
        return MultilevelBlocksConfig(
            n_cubes=n_cubes, n_oblong=n_oblong, n_long=n_long,
        )

    def init(self, state: xmi.XState | None = None) -> None:
        if self._initialized:
            return
        if state is None:
            raise RuntimeError(
                "MultilevelBlocksFeasibilityChecker.init() called without a "
                "state: config must be inferred from the first problem.  "
                "Call check(state, action) instead of init() directly."
            )
        self._initialized = True
        self._cfg = self._infer_config(state.problem)
        self._inferred_signature = (
            self._cfg.n_cubes, self._cfg.n_oblong, self._cfg.n_long,
        )
        self._scratch = tempfile.TemporaryDirectory(prefix="reap_mlblocks_")
        from tampanda.symbolic.domains.multilevel_blocks.honest_grip import build_grip_env, make_grip_executor
        self._env, self._ws, self._cfg = build_grip_env(Path(self._scratch.name), config=self._cfg)
        self._executor = make_grip_executor(self._env, self._ws, self._cfg)

    def _assert_signature_matches(self, state: xmi.XState) -> None:
        """Sanity-check that the current problem matches the checker's
        cached config.  Memoized per problem so objects are not recounted
        on every check call."""
        problem = state.problem
        if problem in self._signature_cache:
            sig = self._signature_cache[problem]
        else:
            sig = self._infer_signature(problem)
            self._signature_cache[problem] = sig
        if sig != self._inferred_signature:
            raise RuntimeError(
                f"MultilevelBlocksFeasibilityChecker initialized with "
                f"signature {self._inferred_signature} but received state from "
                f"a problem with signature {sig}.  The current implementation "
                f"assumes a single block-count regime per worker process; if "
                f"the curriculum mixes them, rebuild the env per signature."
            )

    @staticmethod
    def _infer_signature(problem: xmi.XProblem) -> tuple:
        return (
            sum(1 for o in problem.objects if o.name.startswith("cube_")),
            sum(1 for o in problem.objects if o.name.startswith("oblong_")),
            sum(1 for o in problem.objects if o.name.startswith("long_")),
        )

    @staticmethod
    def _state_to_pddl_dict(state: xmi.XState) -> dict:
        """Translate xmimir XState into the dict format
        ``check_action`` expects.

        ``restore_state`` inside the executor only reads the
        ``(in, block, cell)`` and ``held-*`` fluents; every other
        predicate (static adjacency, shape markers, ``empty``) is
        either derived from the layout or irrelevant for restoration.
        So only those two families are emitted.

        Cell names from PDDL come through lowercased (pymimir folds case
        on parse).  tampanda's Workspace.region_of does case-insensitive
        lookup, so no per-cell normalization is needed.
        """
        d: dict[tuple, bool] = {}
        for atom in state.fluent_atoms:
            pname = atom.predicate.name
            objs = tuple(o.name for o in atom.objects)
            if pname == "in" and len(objs) == 2:
                d[("in",) + objs] = True
            elif pname in ("held-cube", "held-flat-x", "held-flat-y", "held-upright"):
                d[(pname,) + objs] = True
            elif pname in ("grip-centre", "grip-end"):
                # grip-aware domain: only MLB_CHECKER=honest_grip models the grip
                raise RuntimeError("state has %s atoms (grip-aware domain): use MLB_CHECKER=honest_grip" % pname)
        return d

    @staticmethod
    def _state_to_pddl_dict_grip(state: xmi.XState) -> dict:
        """_state_to_pddl_dict plus the grip fluents of the grip-aware domain (MLB_CHECKER=honest_grip)."""
        d: dict[tuple, bool] = {}
        for atom in state.fluent_atoms:
            pname = atom.predicate.name
            objs = tuple(o.name for o in atom.objects)
            if pname == "in" and len(objs) == 2:
                d[("in",) + objs] = True
            elif pname in ("held-cube", "held-flat-x", "held-flat-y", "held-upright", "grip-centre", "grip-end"):
                d[(pname,) + objs] = True
        return d

    @staticmethod
    def _action_to_tuple(action: xmi.XAction) -> tuple:
        """``(action_name, *object_names)``, matching tampanda's
        ``dispatch_action`` signature.

        ``action.name`` is the PDDL action name (with hyphens preserved);
        dispatch_action handles the hyphen→underscore conversion to find
        the executor method.
        """
        return (action.name, *(o.name for o in action.objects))

    @staticmethod
    def _cache_key(pddl_state: dict, action_tuple: tuple) -> tuple:
        # The pddl_state dict is small (only in/held entries); freeze it for
        # use as a dict key.  Action tuple is already hashable.
        return (frozenset(pddl_state.keys()), action_tuple)

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        self.init(state=state)
        # Quick consistency check for mixed-problem regimes; cheap.
        self._assert_signature_matches(state)
        pddl_state = self._state_to_pddl_dict_grip(state)
        action_tuple = self._action_to_tuple(action)
        key = self._cache_key(pddl_state, action_tuple)
        if key in self._cache:
            return self._cache[key]

        from tampanda.symbolic.domains.multilevel_blocks.honest_grip import check_action_grip
        result_dict = check_action_grip(
            self._env, self._ws, self._cfg, pddl_state, action_tuple, executor=self._executor,
        )
        # check_action_grip reports success (True = feasible); check returns
        # True for INFEASIBLE (filter_action contract).
        infeasible = not result_dict.get("success", False)
        self._cache[key] = infeasible
        return infeasible


# Eval suites.  Dataset layout: data/stacking/{domain.pddl, train/, validation/, eval/, eval_ood/}
# eval/ is the in-distribution set (levels L0-L4), eval_ood/ the L5 set.
MULTILEVEL_BLOCKS_EVAL_SUITES: tuple[EvalSuite, ...] = (
    EvalSuite(label="eval", path="data/stacking/eval"),
    EvalSuite(label="eval_ood", path="data/stacking/eval_ood"),
)


register_domain(DomainSpec(
    name="multilevel-blocks",  # matches (define (domain multilevel-blocks))
    feasibility_checker=MultilevelBlocksFeasibilityChecker,
    reference_problem_dir="stacking",
    eval_suites=MULTILEVEL_BLOCKS_EVAL_SUITES,
    domain_files=("domain.pddl",),
))
