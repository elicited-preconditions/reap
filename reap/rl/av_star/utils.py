# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

import random
from itertools import chain

import numpy as np
import torch
from multimethod import multimethod
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import softmax

import reap.xmimir as xmimir
import reap.xmimir as xmi
import reap.xmimir.iw as iw
from reap.encoding import HorizonHGraphEncoder


def squeeze_last_dim_if_one(tensor):
    if tensor.ndim > 1 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


# State Manager (cachable state manager that provides a state embedding and a cached output of values)
# Handles the input of [states] -> [embedding]
# [states] -> [[values]]
# [embedding] -> [[values]]
# if [states] is to big it also handles batching
# does not need to be async / parallel


class StateEmbedder:
    batch_size = 512
    max_size = 10_000  # prevent memory overflow / to many indexed states

    def __init__(self, model, low_memory=False):
        """
        low_memory does not store the state encodings, only all the computed values per state,
        this reduces the memory footprint in evaluation, where the state encodings are not needed afterwards.
        """
        self.model = model
        self.successor_generators = {}
        self.states_encoding = {}
        self.states_policy = {}
        self.states_q = {}
        self.device = model.q_gnn.device
        self.low_memory = low_memory

    def get_batch(self, states, device=None, goals=None):
        if states and isinstance(states[0], tuple):
            st, gt = zip(*states)
            enc = self.get_encoding(list(st), list(gt))
        else:
            enc = self.get_encoding(states, goals)

        if device is None:
            device = self.device

        # The encoder sets the graph attribute `action_args` only when every applicable action has arity 2
        # (hetero_encoder); a domain with mixed arities (multilevel-blocks: 1-4 arguments) yields graphs with and
        # without it, which Batch.from_data_list cannot collate (KeyError: 'action_args'). It is read only under
        # pooling mode "args" (_action_args_tensor), so it is left out exactly when the batch is mixed and no pooling
        # uses it; a batch where every graph has it, or none does, is collated unchanged.
        has = [getattr(e, "action_args", None) is not None for e in enc]
        if any(has) and not all(has):
            if "args" in (getattr(self.model.q_pooling, "mode", None), getattr(self.model.policy_pooling, "mode", None)):
                raise RuntimeError("pooling_mode='args' but only some graphs carry action_args (mixed action arities)")
            batch = Batch.from_data_list(enc, exclude_keys=["action_args"]).to(device)
        else:
            batch = Batch.from_data_list(enc).to(device)
        return batch

    def get_encoding(self, states, goals=None):
        # cached states need no recomputation
        if goals is None:
            goals = [None for _ in states]

        output = [None for _ in states]

        to_encode = []
        to_encode_i = []

        for i, (s, g) in enumerate(zip(states, goals)):
            if (s, g) in self.states_encoding:
                output[i] = self.states_encoding[(s, g)]
            else:
                to_encode.append((s, g))
                to_encode_i.append(i)

        if len(to_encode) > 0:
            for i, (s, g) in zip(to_encode_i, to_encode):
                encoder = self.model.encoder

                if s.problem not in self.successor_generators:
                    self.successor_generators[s.problem] = xmi.XSuccessorGenerator(
                        s.problem
                    )
                succ = self.successor_generators[s.problem]

                if isinstance(encoder, HorizonHGraphEncoder):
                    transitions = list(succ.successors(s))
                    transitions.sort(
                        key=lambda t: str(t.source.fluent_atoms)
                        + str(t.target.fluent_atoms)
                    )

                    encoding_graph = encoder.encode(s, transitions=transitions)
                    pyg_data = encoder.to_pyg_data(encoding_graph)

                else:
                    actions = list(succ.action_generator.generate_actions(s))
                    actions.sort(key=lambda key: str(key))

                    pyg_data = encoder.encode(s, actions=actions)

                if not self.low_memory:
                    self.states_encoding[(s, g)] = pyg_data
                output[i] = pyg_data

        return output

    def _run_policy(self, data_batch):
        embedding, batch = self.model.q_gnn.forward(
            data_batch.x_dict,
            data_batch.edge_index_dict,
            data_batch.batch_dict,
        )
        action_tensor = torch.tensor(
            list(chain.from_iterable(data_batch.action)),
            device=self.device,
            dtype=torch.bool,
        )
        paired_embeddings, action_batch = self.model.policy_pooling.pool(
            embedding, batch, action_tensor,
            action_args=self._action_args_tensor(data_batch, self.model.policy_pooling),
        )
        scalar = squeeze_last_dim_if_one(self.model.policy_readout(paired_embeddings))
        return softmax(scalar, action_batch), action_batch

    def _action_args_tensor(self, data_batch, pooling):
        """Flatten the encoder's per-graph action_args lists for "args" mode.

        None for every other mode (pool() ignores the kwarg there), so the
        other modes are unaffected.  A batch missing the attribute under
        "args" mode raises here rather than deep in pool(); that means the
        encoder withheld it (non-arity-2 domain) or stale cached encodings.
        """
        if getattr(pooling, "mode", "mask") != "args":
            return None
        aa = getattr(data_batch, "action_args", None)
        if aa is None:
            raise RuntimeError(
                "pooling_mode='args' but the batch has no action_args attribute"
            )
        # Batch.from_data_list collates graph attrs per graph (same layout the
        # `action` mask flattening above relies on), so flatten unconditionally.
        flat = list(chain.from_iterable(aa))
        if not flat:
            return torch.zeros((0, 2), dtype=torch.long, device=self.device)
        return torch.tensor(flat, dtype=torch.long, device=self.device)

    def _run_q(self, data_batch):
        embedding, batch = self.model.q_gnn.forward(
            data_batch.x_dict, data_batch.edge_index_dict, data_batch.batch_dict
        )
        if isinstance(self.model.encoder, HorizonHGraphEncoder):
            action_tensor = data_batch.target_mask
        else:
            action_tensor = torch.tensor(
                list(chain.from_iterable(data_batch.action)),
                device=self.device,
                dtype=torch.bool,
            )
        paired_embeddings, action_batch = self.model.q_pooling.pool(
            embedding, batch, action_tensor,
            action_args=self._action_args_tensor(data_batch, self.model.q_pooling),
        )
        scalar = squeeze_last_dim_if_one(self.model.q_readout(paired_embeddings))
        return scalar, action_batch

    def _run_model(self, batch, q_only=False):
        policy_values = None
        if not q_only:
            policy_values, action_batch = self._run_policy(batch)
        q_values, action_batch = self._run_q(batch)

        return policy_values, q_values, action_batch

    @multimethod
    def run_model_no_cache(self, states: list, q_only=False):
        if states and isinstance(states[0], tuple):
            st, gt = zip(*states)
            batch = self.get_batch(states=list(st), goals=list(gt))
        else:
            batch = self.get_batch(states)

        return self._run_model(batch, q_only)

    @multimethod
    def run_model_no_cache(self, batch: HeteroData, q_only=False):
        return self._run_model(batch, q_only)

    def _run_model_on(self, states, q_only=False, goal=None):
        if len(states) == 0:
            return

        with torch.no_grad():
            if states and isinstance(states[0], tuple):
                st, gt = zip(*states)
                batch = self.get_batch(states=list(st), goals=list(gt))
            else:
                batch = self.get_batch(states, goals=[goal for _ in states])
            policy_values, q_values, action_batch = self._run_model(batch, q_only)
            if q_only:
                for i, s in enumerate(states):
                    self.states_q[s] = q_values[action_batch == i].to("cpu")
            else:
                for i, s in enumerate(states):
                    self.states_q[s] = q_values[action_batch == i].to("cpu")
                    self.states_policy[s] = policy_values[action_batch == i].to("cpu")

    def get_q_values(self, states, goal=None):
        output = []
        to_encode = []

        for i, s in enumerate(states):
            if s not in self.states_q:
                to_encode.append(s)

        batches = (len(to_encode) + self.batch_size) // self.batch_size

        for b in range(batches):
            n = to_encode[b * self.batch_size : (b + 1) * self.batch_size]
            self._run_model_on(n, q_only=True, goal=goal)

        for i, s in enumerate(states):
            if s not in self.states_q:
                assert "something went wrong in the model run"

            output.append(self.states_q[s])

        # Concatenate all tensors into one
        big_tensor = torch.cat(output)

        # Create the index tensor
        index_tensor = torch.cat(
            [
                torch.full((t.size(0),), i, dtype=torch.long)
                for i, t in enumerate(output)
            ]
        )

        return big_tensor, index_tensor

    def get_policy_values(self, states):
        output = []

        to_encode = []

        for i, s in enumerate(states):
            if s not in self.states_policy:
                to_encode.append(s)

        batches = (len(to_encode) + self.batch_size) // self.batch_size

        for b in range(batches):
            n = to_encode[b * self.batch_size : (b + 1) * self.batch_size]
            self._run_model_on(n)

        for i, s in enumerate(states):
            if s not in self.states_policy:
                assert "something went wrong in the model run"

            output.append(self.states_policy[s])

        # Concatenate all tensors into one
        big_tensor = torch.cat(output)

        # Create the index tensor
        index_tensor = torch.cat(
            [
                torch.full((t.size(0),), i, dtype=torch.long)
                for i, t in enumerate(output)
            ]
        )

        return big_tensor, index_tensor

    def clean(self):
        if len(self.states_encoding) > self.max_size:
            self.states_encoding = {}
            self.states_policy = {}
            self.states_q = {}
            self.successor_generators = {}


import random

import numpy as np


class ProblemSampler:
    """
    Sampler with per-problem status (solved / medium / unsolved) and adaptive
    budget growth.  Sampling weights:

      - medium    : base 1.0, decays by DECAY_RATE per step past MEDIUM_PEAK_ITERS
                    (down to MEDIUM_FLOOR).  Fresh medium problems get top
                    priority, but the curriculum stops fixating on
                    long-stuck-in-medium problems.
      - unsolved  : base UNSOLVED_BASE (default 0.1), boosted by fail_streak
                    (per-failure increment FAIL_STREAK_BOOST_PER, capped at
                    FAIL_STREAK_BOOST_CAP).  Stuck-unsolved problems get
                    progressively more attention so they can graduate.
      - solved    : flat SOLVED_WEIGHT (default 0.05), de-emphasised but
                    not invisible, so the model can occasionally re-confirm.

    Top/bottom ratio is ~20×, so unsolved problems get meaningful
    share even when many problems are in 'medium' status.

    Tracked state per problem:
      iteration[i]            : cumulative sample count (just increments)
      iter_in_status[i]       : sample count since last status change (resets
                                on status change), drives the medium decay
      fail_streak[i]          : consecutive failures (resets on solved/medium),
                                drives the unsolved boost
    """

    # --- weighting knobs ---
    # A never-solved problem tops out at
    # UNSOLVED_BASE * FAIL_STREAK_BOOST_CAP = 0.3, against a fresh medium
    # problem's 1.0, i.e. unsolved problems are DE-emphasised, not boosted.
    MEDIUM_BASE          = 1.0
    MEDIUM_PEAK_ITERS    = 20
    DECAY_RATE           = 0.9
    MEDIUM_FLOOR         = 0.1
    UNSOLVED_BASE        = 0.1
    FAIL_STREAK_BOOST_PER = 0.1
    FAIL_STREAK_BOOST_CAP = 3.0
    SOLVED_WEIGHT        = 0.05

    def __init__(self, problems, search_budget=2048, search_batching=64,
                 budget_growth_threshold=3, budget_max_multiplier=8,
                 budget_growth_increment=None,
                 budget_growth_mode="additive",
                 easy_first=False, easy_first_alpha=0.15):
        self.k = len(problems)
        self.problems = [
            {
                "problem": p,
                "batch": search_batching,
                "init_budget": search_budget,
                "budget": search_budget,
                "status": "unsolved",  # "medium", "solved", "unsolved"
            }
            for p in problems
        ]
        self.iteration = np.zeros(self.k, dtype=int)
        # iter_in_status[i] is the # of sample() calls since this problem's
        # status last changed.  Resets to 0 in update() whenever status
        # transitions.  Drives the medium-weight decay (so the sampler does
        # not keep focusing on a long-stuck medium problem).
        self.iter_in_status = np.zeros(self.k, dtype=int)
        self.fail_streak = np.zeros(self.k, dtype=int)  # consecutive failures
        self.budget_growth_threshold = budget_growth_threshold
        self.budget_max_multiplier = budget_max_multiplier
        self.budget_growth_increment = (
            budget_growth_increment if budget_growth_increment is not None
            else search_budget
        )
        if budget_growth_mode not in ("additive", "exponential"):
            raise ValueError(
                f"ProblemSampler: unknown budget_growth_mode {budget_growth_mode!r}"
            )
        self.budget_growth_mode = budget_growth_mode
        # easy-first curriculum (opt-in, default off): down-weight harder
        # (larger) problems so abundant small instances get harvested before
        # the sampler fixates on chronically-failing hard ones.
        self.easy_first = easy_first
        self.easy_first_alpha = easy_first_alpha
        if easy_first:
            self.num_obj = np.array(
                [ProblemSampler._objcount(p["problem"]) for p in self.problems],
                dtype=float)
            self._min_num = float(self.num_obj.min())
        else:
            self.num_obj = None
            self._min_num = 0.0

    @staticmethod
    def _objcount(p):
        try:
            return float(len(list(p.objects)))
        except Exception:
            return 1.0

    def _easy_factor(self, i):
        if not self.easy_first or self.num_obj is None:
            return 1.0
        return float(np.exp(-self.easy_first_alpha * (self.num_obj[i] - self._min_num)))

    def _weight_for(self, i):
        return self._base_weight_for(i) * self._easy_factor(i)

    def _base_weight_for(self, i):
        """Compute the sampling weight for problem i based on its current
        status, time-in-status, and fail_streak."""
        status = self.problems[i]["status"]
        if status == "medium":
            t = self.iter_in_status[i]
            if t <= self.MEDIUM_PEAK_ITERS:
                return self.MEDIUM_BASE
            decayed = self.MEDIUM_BASE * (self.DECAY_RATE ** (t - self.MEDIUM_PEAK_ITERS))
            return max(decayed, self.MEDIUM_FLOOR)
        if status == "unsolved":
            boost = min(
                1.0 + self.FAIL_STREAK_BOOST_PER * self.fail_streak[i],
                self.FAIL_STREAK_BOOST_CAP,
            )
            return self.UNSOLVED_BASE * boost
        # status == "solved"
        return self.SOLVED_WEIGHT

    def sample(self, include_solved=True, include_unsolved=True):
        """
        Sample a problem with weights based on (status, iter_in_status, fail_streak).

        Parameters:
        - include_solved: include already-solved problems in sampling
        - include_unsolved: include very-hard never-solved problems in sampling
        """
        candidates = []
        weights = []

        for i, p in enumerate(self.problems):
            if p["status"] == "solved" and not include_solved:
                continue
            if p["status"] == "unsolved" and not include_unsolved:
                continue
            candidates.append(i)
            weights.append(self._weight_for(i))

        # Fallback: if the filters leave no candidates, sample uniformly.
        if not candidates:
            candidates = list(range(self.k))
            weights = [1.0] * self.k

        idx = random.choices(candidates, weights=weights, k=1)[0]
        # Plain per-problem increments, no cross-bucket coupling.
        self.iteration[idx] += 1
        self.iter_in_status[idx] += 1
        return self.problems[idx], idx

    def _set_status(self, idx, new_status):
        """Set a problem's status and reset its iter_in_status counter."""
        if self.problems[idx]["status"] != new_status:
            self.iter_in_status[idx] = 0
        self.problems[idx]["status"] = new_status

    def update(self, idx, mark_unsolved=False, mark_solved=False):
        """
        Update problem status, fail streak, and budget.

        Parameters:
        - mark_solved: goal found, dif_ratio ≤ 1.0 (trivially solved)
        - mark_unsolved: no goal found at all
        - else (both False): goal found with dif_ratio > 1.0 (medium difficulty,
          model directed search through multi-step backtracking)
        """
        problem = self.problems[idx]

        if mark_solved:
            self._set_status(idx, "solved")
            self.fail_streak[idx] = 0
            problem["budget"] = max(
                problem["init_budget"],
                problem["budget"] // 2,
            )
        elif mark_unsolved:
            self._set_status(idx, "unsolved")
            self.fail_streak[idx] += 1
            if self.fail_streak[idx] >= self.budget_growth_threshold:
                cap = problem["init_budget"] * self.budget_max_multiplier
                if self.budget_growth_mode == "exponential":
                    new_budget = problem["budget"] * 2
                else:
                    new_budget = problem["budget"] + self.budget_growth_increment
                problem["budget"] = min(cap, new_budget)
        else:
            # mark_solved=False, mark_unsolved=False → "medium":
            # goal found but with dif_ratio > 1.0 (model directed multi-step search)
            self._set_status(idx, "medium")
            self.fail_streak[idx] = 0
            problem["budget"] = problem["init_budget"]

    def stats(self):
        status_counts = {"solved": 0, "unsolved": 0, "medium": 0}
        for p in self.problems:
            status_counts[p["status"]] += 1

        budgets = [p["budget"] for p in self.problems]
        return {
            "solved_problems": status_counts["solved"],
            "unsolved_problems": status_counts["unsolved"],
            "medium_problems": status_counts["medium"],
            "average_fail_streak": float(np.mean(self.fail_streak)),
            "median_iter_in_status_medium": (
                float(np.median(self.iter_in_status[
                    [i for i, p in enumerate(self.problems) if p["status"] == "medium"]
                ]))
                if any(p["status"] == "medium" for p in self.problems) else 0.0
            ),
            "budget_mean": float(np.mean(budgets)),
            "budget_max": int(np.max(budgets)),
            "budget_min": int(np.min(budgets)),
        }
