"""A copy of REAP's greedy rollout control flow, plus action capture.

Copy of ``reap.rl.av_star.search.rollout()``. The real function does not
record which action was chosen at each step (only Q-value bookkeeping via
``history``); N1's headline metric needs the actual plan of the greedy arm,
so this copy adds that recording. The SEARCH LOGIC itself (candidate
selection, retry/masking, termination conditions) is the same.

Simplification versus the real function: the ``compute_loss``/``log_level``
return-shape variants are dropped, since the greedy evaluation
(``search.rollouts()`` with ``compute_rmse=False``) never uses them.
"""
from __future__ import annotations

import time

import torch
import reap.xmimir as xmi
import reap.xmimir.iw as iw

from reap.encoding import HorizonHGraphEncoder
from reap.encoding.horizon_hetero_encoder import TransitionDAG
from reap.rl.av_star.search import next_real


def rollout_with_capture(
    generator,
    initial_state,
    state_embedder,
    checker,
    max_expansion: int = 1000,
    closed: bool = True,
    stochastic: bool = False,
    time_limit_s: float = 90,
    max_infeasible_retries: int = 2,
):
    """Same control flow as ``search.rollout()``, plus action-sequence
    capture. Defaults match the greedy evaluation of this protocol
    (``max_expansion=1000``, ``closed=True``, ``stochastic=False``,
    ``max_infeasible_retries=2``, ``time_limit_s=90``) -- see config.py.

    One deliberate difference from the real function: takes an explicit
    ``checker`` object (``checker.check(state, action) -> bool``, True iff
    infeasible, same contract as ``reap.rl.av_star.sim.filter_action``)
    instead of calling that module-level function directly. ``filter_action``
    reads a process-global singleton checker
    (``sim._CHECKERS``); this evaluation needs a
    PER-ROLLOUT checker parametrized by a true-pose table / occupancy-flip
    RNG (:class:`reap.eval.noise.noisy_checker.NoisyTabletopFeasibilityChecker`).
    Passing a plain, unparametrized instance of that same class (or the real
    global ``filter_action``) reproduces the real function's behavior exactly.

    Returns ``(counter, solved, chosen_actions)``: ``chosen_actions`` is
    the addition; ``counter``/``solved`` match the real function's base
    (non-``compute_loss``, non-``log_level``) return shape.

    Must be kept in sync with search.rollout().
    """
    epsilon = 0.5
    history = []
    chosen_actions = []

    current = initial_state
    counter = 0
    visited = set()
    start_time = time.time()
    actions = None
    target = None

    while (
        not current.is_goal()
        and counter < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        visited.add(current)
        counter += 1

        # As in search.rollout(), including its reliance on `actions`
        # persisting across loop iterations if the HorizonHGraphEncoder
        # branch is ever taken (that branch does not itself (re)set
        # `actions`); copied as is, for fidelity.
        if isinstance(state_embedder.model.encoder, HorizonHGraphEncoder):
            transitions = list(generator.successors(current))
            transitions.sort(
                key=lambda t: str(t.source.fluent_atoms) + str(t.target.fluent_atoms)
            )
            dag = TransitionDAG(current, transitions, ignore_actions=True, actions_only=False)
            target = [t for t, idx in dag if idx != 0]
        elif isinstance(generator, xmi.XSuccessorGenerator) and not isinstance(
            generator, iw.IWSuccessorGenerator
        ):
            actions = list(generator.action_generator.generate_actions(current))
            actions.sort(key=lambda key: str(key))
            target = [next_real(t.apply(current), generator) for t in actions]
            # occluded_objects/not_allowed_mask are computed in
            # search.rollout() but never used; omitted here.

        output = state_embedder.get_q_values([current])
        og_output = output[0].clone()

        if len(output[0]) == 0:
            return counter, current.is_goal(), chosen_actions

        if closed:
            visited_maks = torch.tensor([t not in visited for t in target])
            output[0][~visited_maks] = -torch.inf

        q_vals = output[0].clone()
        infeasible_count = 0
        i = None
        while True:
            if stochastic:
                best_q = torch.max(q_vals)
                near_optimal_idxs = torch.where((best_q - q_vals) <= epsilon)[0]
                candidate = near_optimal_idxs[torch.randint(len(near_optimal_idxs), (1,))]
            else:
                candidate = torch.argmax(q_vals)

            if torch.isinf(q_vals[candidate]) and q_vals[candidate] < 0:
                break

            if checker.check(current, actions[candidate]):
                infeasible_count += 1
                q_vals[candidate] = -torch.inf
                if infeasible_count > max_infeasible_retries:
                    break
            else:
                i = candidate
                break

        if i is None:
            return counter, current.is_goal(), chosen_actions

        history.append((len(output[0]), og_output[i]))
        chosen_actions.append(actions[i])
        current = target[i]

    return counter, current.is_goal(), chosen_actions
