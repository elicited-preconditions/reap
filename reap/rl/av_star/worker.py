# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

import copy
import gc
import io
import itertools
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

import reap.xmimir as xmimir
from reap.rl.av_star.agent import Model
from reap.rl.av_star.search import (
    bwqs,
    rollouts,
)
from reap.rl.av_star.utils import (
    ProblemSampler,
    StateEmbedder,
)
from reap.rl.data_layout import InputData


def _extend_batch(batch_states, batch_targets, batch_target_map, batch_bound,
                  states, targets, target_map, bound):
    """Append one bwqs attempt's (states, targets, target_map, bound) onto the
    running per-chunk accumulators, remapping target_map ids (0=nan-sentinel,
    1=dead-end-sentinel, >=2=index into targets) so they land after whatever
    is already in batch_targets.  Mutates the list accumulators in place and
    returns the updated batch_target_map tensor (torch.cat can't mutate
    in place)."""
    target_map = torch.tensor(target_map, dtype=torch.long)
    target_map[target_map > 1] += len(batch_targets)
    batch_states += states
    batch_target_map = torch.cat([batch_target_map, target_map])
    batch_targets += targets
    batch_bound += bound
    return batch_target_map


@dataclass(order=True)
class Chunk:
    """
    batched target states may hold similar states to batched states

    """

    batched_states: HeteroData  # SS -> AS when run
    batched_targets: HeteroData  # AS -> AS when run and max pooled
    target_map: torch.tensor  # AS [maps for the batched states to the target space]
    bounds: torch.tensor  # AS

    @staticmethod
    def copy(c: "Chunk") -> "Chunk":
        # Use .clone() to force a new memory allocation; for HeteroData the
        # underlying storage must be cloned as well.

        # 1. Clone the HeteroData (Graph)
        # Note: PyG's .clone() is usually shallow-ish depending on version,
        # but .to('cpu', non_blocking=False) isn't enough.
        # The safest way for PyG data in this context is explicit cloning or detaching.
        batched_states = c.batched_states.clone()
        batched_targets = c.batched_targets.clone()

        # 2. Clone the Tensors
        target_map = c.target_map.clone()
        bounds = c.bounds.clone()

        return Chunk(batched_states, batched_targets, target_map, bounds)


def worker_process(rank, save_path, data_path, wandb_queue, chunk_queue,
                   shared_param_tensors=None,
                   shared_buffer_tensors=None,
                   search_budget=4096, search_batching=1, action_cost=0.5,
                   budget_growth_threshold=3, budget_max_multiplier=8,
                   budget_growth_increment=None,
                   budget_growth_mode="additive",
                   max_infeasible_retries=2, domain_file="domain.pddl", train_dir="train",
                   seed=None,
                   # Per-bwqs-attempt wall budget in seconds.  0 = no time
                   # limit (the search is purely expansion-budget-bound).  A
                   # finite limit also caps the adaptive budget growth.
                   worker_time_limit_s=60,
                   # ε-exploration probability passed to bwqs (per-child).
                   # 0.0 = Q-greedy; >0 adds exploration during the random-Q
                   # bootstrap.
                   random_exploration=0.0,
                   easy_first=False):
    # Spawn-started child process: RNG state isn't inherited, so re-seed.
    # Each worker uses seed + rank so problem sampling order differs across
    # workers but is deterministic given (seed, num_workers).
    if seed is not None:
        worker_seed = seed + rank
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    # WORKER_DEVICE: put workers on a different GPU than the trainer (e.g. cuda:1
    # when the trainer fills the first card). Default: cuda:0 if available, else CPU.
    device = os.environ.get("WORKER_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Log of every attempted problem and when it was attempted.
    stats_df = pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
            "model_time",
            "worker_id",
            "problem",
            "nodes_expanded",
            "goal_path_length",
            "ratio",
            "difficulty",
            "termination_reason",
        ]
    )

    print(rank, "hello")

    # Shared-memory model:
    # The full Model cannot cross spawn boundaries because it holds
    # xmimir.XDomain (unpicklable pymimir C-ext).  Instead each worker loads
    # its own Model from disk (replicates the object graph locally with
    # its own pymimir refs), then re-points each parameter's `.data` to the
    # trainer's shared CUDA tensors.  After rebinding, `local_model.forward`
    # reads the trainer's latest weights directly: no staleness, no disk
    # reload.  Torn-read risk: workers may forward through params mid-
    # optimizer.step(); given CUDA kernel serialization and the small mismatch
    # window vs. bwqs duration this is rare and not catastrophic.
    local_model = Model.load_from_folder(save_path)
    local_model.to(device)

    if shared_param_tensors is not None:
        rebinds = 0
        for name, p in local_model.named_parameters():
            if name in shared_param_tensors:
                p.data = shared_param_tensors[name]
                rebinds += 1
        if shared_buffer_tensors:
            for name, b in local_model.named_buffers():
                if name in shared_buffer_tensors:
                    # Buffers aren't Parameter; set_() swaps storage in place.
                    with torch.no_grad():
                        b.set_(shared_buffer_tensors[name])
        print(f"[worker {rank}] rebound {rebinds} params + "
              f"{len(shared_buffer_tensors or {})} buffers to shared tensors")
    else:
        print(f"[worker {rank}] WARN: shared_param_tensors is None, "
              "falling back to reloading the model from disk")

    # problem set
    data_loader = InputData(data_path, train_dir=train_dir, domain_file=domain_file)
    problem_sampler = ProblemSampler(
        data_loader.train_problems,
        search_budget=search_budget,
        search_batching=search_batching,
        budget_growth_threshold=budget_growth_threshold,
        budget_max_multiplier=budget_max_multiplier,
        budget_growth_increment=budget_growth_increment,
        budget_growth_mode=budget_growth_mode,
        easy_first=easy_first,
    )

    # Cumulative counters.  Pushed to wandb_queue every 10 chunks for trainer-
    # side aggregation; the trainer emits workers/* + sampler/* aggregates
    # across all workers.  Per-worker and windowed metrics are not pushed:
    # they are noisy and unscoped aliases would race between workers.
    attempts_cum = 0
    goals_cum = 0
    path_length_cum = 0

    # Worker runs until it is killed, until the trainer signals shutdown
    # (process.terminate from training.main), or until a safety cap of 10M
    # iterations that in practice never trips.
    worker_start_time = time.time()
    for i in itertools.count():
        if i >= 10_000_000:
            print(f"[worker {rank}] safety cap reached at {i} iters, exiting")
            break
        # With shared_model the local_model reference is always current
        # (trainer updates in place).  model_time is recorded only to keep the
        # column in the per-worker stats CSV; its value carries no information.
        model_time = datetime.now()

        # collect batch
        batch_states = []
        batch_targets = []
        batch_target_map = torch.tensor([], dtype=torch.long)
        batch_bound = []

        se = StateEmbedder(local_model)
        while len(batch_states) < 128:
            se.clean()
            attempts_cum += 1
            problem, idx = problem_sampler.sample()

            start_time = datetime.now()

            gen = xmimir.XSuccessorGenerator(problem["problem"])

            # First call: batched search for speed.  gather_training=True
            # yields training data even when goal is None (the dead_end
            # nodes from failed searches feed DEAD_END_REWARD targets to the
            # trainer, see search.py:bwqs).  When a goal is found, a second
            # batch_size=1 call below overrides this with canonical-Q
            # training data.
            states, targets, target_map, bound, dif_ratio, states_expanded, goal, reason, root = (
                bwqs(
                    gen,
                    gen.initial_state,
                    se,
                    max_expansion=problem["budget"],
                    batch_size=problem["batch"],
                    time_limit_s=worker_time_limit_s,
                    gather_training=True,
                    action_cost=action_cost,
                    max_infeasible_retries=max_infeasible_retries,
                    random_exploration=random_exploration,
                )
            )
            _ex = getattr(root, "explore", None)
            if _ex is not None:
                # One line per search attempt with the exploration statistics.
                # min_unsat == init_unsat means the search never got a single
                # goal atom closer -- pure retracing.
                print("[explore] problem=%s budget=%d goal=%d init_unsat=%d min_unsat=%d "
                      "max_depth=%d uniq=%d" % (
                          os.path.basename(problem["problem"].filepath), problem["budget"],
                          goal is not None, _ex["init_unsat"], _ex["min_unsat"],
                          _ex["max_depth"], _ex["uniq"]), flush=True)
            if goal is not None:
                # Rerun the search with batch_size = 1 to collect the tree that
                # unbatched search would build; most states are cached, so this is fast.
                states, targets, target_map, bound, dif_ratio, states_expanded, goal, reason, root = (
                    bwqs(
                        gen,
                        gen.initial_state,
                        se,
                        max_expansion=problem["budget"],
                        batch_size=1,
                        time_limit_s=worker_time_limit_s,
                        action_cost=action_cost,
                        max_infeasible_retries=max_infeasible_retries,
                    )
                )

            # get goal length and add meta information
            path_length = 0
            g = goal
            while g is not None:
                g = g.p
                path_length += 1
            new_entry = {
                "start_time": start_time,
                "end_time": datetime.now(),
                "model_time": model_time,
                "worker_id": rank,
                "problem": problem["problem"].filepath,
                "nodes_expanded": states_expanded,
                "goal_path_length": path_length,
                "ratio": dif_ratio,
                "difficulty": problem["status"],
                "termination_reason": reason,
            }
            stats_df.loc[len(stats_df)] = new_entry

            if goal is None:
                problem_sampler.update(idx, True)
                # Fall through and append the (possibly small) dead_end-only
                # training data of the failed search to the batch.  Stuck
                # attempts contribute ~1 dead_end mark per (state, action)
                # pair that filter_action rejected.
            else:
                problem_sampler.update(idx, False, dif_ratio <= 1.0)

            batch_target_map = _extend_batch(
                batch_states, batch_targets, batch_target_map, batch_bound,
                states, targets, target_map, bound,
            )

        batch_states = batch_states[:256]  # cap the states to bound memory
        max_length = len(se.get_q_values(batch_states)[0])
        batch_target_map = batch_target_map[:max_length]
        batch_targets = batch_targets[:max_length]
        batch_bound = batch_bound[:max_length]

        batched_states = se.get_batch(batch_states, device="cpu")
        batched_targets = se.get_batch(batch_targets, device="cpu")

        c = Chunk(
            batched_states.cpu().detach(),
            batched_targets.cpu().detach(),
            batch_target_map.cpu().detach(),
            torch.tensor(batch_bound).cpu().detach(),
        )

        buffer_io = io.BytesIO()
        torch.save(c, buffer_io)
        chunk_queue.put(buffer_io.getvalue(), block=True)
        if i % 10 == 0:
            del se
            stats_df.to_csv(save_path / f"training.{rank}.csv", index=False)
            gc.collect()
            torch.cuda.empty_cache()

            # Push cumulative counters + sampler snapshot, tagged with _rank.
            # Trainer drains wandb_queue, accumulates per-rank latest, and
            # emits workers/* + sampler/* aggregates on its own step axis.
            wandb_queue.put({
                "_rank": rank,
                "attempts": attempts_cum,
                "goals": goals_cum,
                "path_length_sum": path_length_cum,
                "elapsed_s": time.time() - worker_start_time,
                "sampler_stats": problem_sampler.stats(),
            })


def validation_process(save_path, data_path, wandb_queue, max_infeasible_retries=2,
                       val_time_limit_s=60 * 5, val_sleep_s=60,
                       domain_file="domain.pddl", val_dir="validation", seed=None):
    # Spawn-started child process: re-seed.  Validator uses a fixed offset
    # so its RNG is distinct from any worker's but still deterministic.
    if seed is not None:
        val_seed = seed + 1_000_003
        torch.manual_seed(val_seed)
        np.random.seed(val_seed)
        random.seed(val_seed)
    # Track progress on the validation set during training.  Each pass
    # records the number of problems solved by greedy rollout, the total plan
    # length over the solved problems, and the value RMSE on them.  A pass
    # that beats all previous passes is saved (see checkpoint selection below).

    device = os.environ.get("VAL_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")  # see WORKER_DEVICE
    print("Starting Validation")

    # local_model

    # problem set
    data_loader = InputData(data_path, val_dir=val_dir, val_instances="all",
                            domain_file=domain_file)
    from reap.domains import checker_info
    print(checker_info.log_line(checker_info.describe_data(data_loader.pddl_domains_dir / domain_file,
                                                           [data_loader.val_dir]), "validation"), flush=True)

    n_val = len(data_loader.val_problems)
    print(f"Validation: {n_val} problems, time_limit={val_time_limit_s}s/problem, sleep={val_sleep_s}s between passes")
    max_solved = 0
    best_score = -1.0
    min_steps = 10**100
    min_rmse = 10**100  # default very high

    # loop over validation problems and keep track of the best model so far
    for checkpoint in range(100_000):
        local_model = Model.load_from_folder(save_path)
        local_model.to(device)

        df_output = rollouts(
            data_loader.val_problems,
            local_model,
            1_000,
            time_limit_s=val_time_limit_s,
            closed=True,
            stochastic=False,
            compute_rmse=True,
            max_infeasible_retries=max_infeasible_retries,
        )
        solved_df = df_output[df_output["solved"] == 1].copy()
        rmse = solved_df["rmse"].mean()

        gc.collect()
        torch.cuda.empty_cache()

        wandb_queue.put(
            {
                "val/solve_rate": len(solved_df),
                "val/solve_rate_pct": len(solved_df) / n_val * 100,
                "val/steps": solved_df["steps"].sum(),
                "val/rmse": rmse,
            }
        )

        # --- checkpoint selection ----------------------------------------
        # A new best is a pass that solves more validation problems (ties: fewer steps, then lower rmse).
        # Nothing is written while the model plateaus, so a crash can
        # cost a long stretch of training. CKPT_PERIOD=N also writes an
        # unconditional periodic checkpoint every N validation passes.
        _period = int(os.environ.get("CKPT_PERIOD", "0") or 0)
        score = float(len(solved_df))

        new_best = False
        if score > best_score + 1e-9:
            new_best = True
        elif abs(score - best_score) <= 1e-9:
            if solved_df["steps"].sum() < min_steps:
                new_best = True
            elif solved_df["steps"].sum() == min_steps and rmse < min_rmse:
                new_best = True

        periodic = _period > 0 and checkpoint > 0 and checkpoint % _period == 0

        if new_best or periodic:
            if new_best:
                best_score = score
                max_solved = max(max_solved, len(solved_df))
                min_steps = solved_df["steps"].sum()
                min_rmse = rmse
            tag = "best" if new_best else "periodic"
            local_model.save_to_folder(save_path / f"checkpoint/checkpoint_{checkpoint}")
            print(f"saving {tag} model (solved={len(solved_df)}/{n_val})")
            time.sleep(5 * 60)  # 5 minutes after saving
        else:
            time.sleep(val_sleep_s)  # always throttle between passes


def _micro_backward(se, batched_states, q_target, bounds, mbs, w_policy):
    """MICRO_BATCH_STATES: the chunk loss of the trainer -- MSE over the valid q rows + w_policy x the
    on-trace policy cross-entropy, each a MEAN over rows -- and its
    gradient, accumulated over micro-batches of at most `mbs` states instead of one forward/backward over the chunk.

    Exact (up to float summation order): each mean is decomposed as sum over micro-batches / the CHUNK-wide count,
    and the network acts within one state's graph (message passing, pooling, readout and the per-state softmaxes
    group by state; no batch statistics), so a state's outputs do not depend on which other states share its batch.
    The chunk-wide counts need the per-state q-row layout first: one no-grad q-only forward per micro-batch (rows
    are laid out state by state, so micro-batch rows concatenate to the full forward's rows). Gradients accumulate
    in .grad; the caller clips and steps as without micro-batching. Returns (q_target truncated, q_pred detached, mask,
    q_loss, policy_loss) for the caller's stats.
    """
    from torch_geometric.data import Batch
    datas = batched_states.to_data_list()
    parts = [Batch.from_data_list(datas[i:i + mbs]) for i in range(0, len(datas), mbs)]
    with torch.no_grad():
        rows = [se.run_model_no_cache(b, q_only=True)[1].shape[0] for b in parts]
    n_rows = sum(rows)
    q_target = q_target[:n_rows]
    mask = ~torch.isnan(q_target)
    n_mask = mask.sum()                                  # 0 -> nan loss, as MSELoss on an empty selection
    on_trace = ~torch.isnan(bounds[:n_rows])
    n_on = on_trace.sum()
    dev = q_target.device
    q_loss = torch.zeros((), device=dev); policy_loss = torch.zeros((), device=dev)
    preds, r0 = [], 0
    for b, r in zip(parts, rows):
        policy_m, q_m, _ = se.run_model_no_cache(b, q_only=False)
        assert q_m.shape[0] == r, "micro-batch rows changed between passes"
        m, ot = mask[r0:r0 + r], on_trace[r0:r0 + r]
        q_part = ((q_m[m] - q_target[r0:r0 + r][m]) ** 2).sum() / n_mask
        pol_part = (-torch.log(policy_m[ot].clamp_min(1e-8))).sum() / n_on if n_on > 0 \
            else torch.zeros((), device=dev)
        loss = q_part + w_policy * pol_part
        policy_loss = policy_loss + pol_part.detach()
        loss.backward()
        q_loss = q_loss + q_part.detach()
        preds.append(q_m.detach())
        r0 += r
    return q_target, torch.cat(preds), mask, q_loss, policy_loss


def trainer(shared_model, old_model, save_path, chunk_queue, wandb_queue,
            dead_end_reward=-30, lr_readout=1e-3, lr_gnn=1e-4,
            max_grad_norm=1.0, max_updates_per_iter=10, buffer_size=40,
            policy_loss_weight=0.1, max_outer_iters=None, seed=None):
    """Main training loop.

    max_outer_iters caps the number of outer iterations. None means run
    effectively forever (100_000 iterations); pass an integer to cap.

    seed: if not None, re-seeds torch/numpy/random with seed + 2_000_003 so
    the trainer's stochastic ops (e.g. buffer sampling) are deterministic
    relative to the main seed.  Trainer runs in-process, but seed control is
    still useful for downstream randomness.
    """
    if seed is not None:
        trainer_seed = seed + 2_000_003
        torch.manual_seed(trainer_seed)
        np.random.seed(trainer_seed)
        random.seed(trainer_seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MAX_GRAD_NORM = max_grad_norm
    MAX_UPDATES_PER_ITER = max_updates_per_iter
    DEAD_END_REWARD = dead_end_reward
    # Trace-imitation policy head: weight of the policy_loss in the total loss.
    POLICY_LOSS_WEIGHT = policy_loss_weight
    # Memory-only options: MICRO_BATCH_STATES=N accumulates the gradient of a chunk with more
    # than N states over micro-batches of N states (same loss and gradient, bounded peak memory; _micro_backward);
    # MEM_LOG=1 prints the chunk's graph size and the peak GPU memory of every update. Both default off.
    MICRO_BATCH_STATES = int(os.environ.get("MICRO_BATCH_STATES", "0") or 0)
    MEM_LOG = os.environ.get("MEM_LOG") == "1" and torch.cuda.is_available()

    shared_model = shared_model.to(device)
    old_model = old_model.to(device)

    se = StateEmbedder(shared_model)

    loss_fn = torch.nn.MSELoss()
    value_optimizer = torch.optim.Adam(
        itertools.chain(
            shared_model.q_readout.parameters(), shared_model.q_pooling.parameters()
        ),
        lr=lr_readout,
    )
    gnn_optimizer = torch.optim.Adam(shared_model.q_gnn.parameters(), lr=lr_gnn)
    buffer = []
    total_data = 0
    enter = 0

    iter_cap = int(max_outer_iters) if max_outer_iters is not None else 100_000
    # trainer_iter is the wandb x-axis.  Every wandb.log() call below passes step=trainer_iter
    # so worker/validator messages drained from wandb_queue and the
    # trainer's own metrics all land on the SAME monotonic axis.
    total_backprops = 0
    # Per-worker latest snapshot (rank -> stats dict).  Updated on every
    # drain of wandb_queue; used to emit aggregated workers/* + sampler/*
    # metrics that reflect the whole process rather than racing per-worker
    # values.  Validator messages don't have _rank and are logged as-is.
    worker_latest: dict = {}
    for i in range(iter_cap):
        trainer_iter = i
        old_model.load_state_dict(shared_model.state_dict())
        se_old = StateEmbedder(old_model)

        maxi = 0
        enter += 1
        while (
            enter >= MAX_UPDATES_PER_ITER or chunk_queue.qsize() > 0 or len(buffer) < 5
        ) and not maxi > 40:
            enter = 0
            maxi += 1

            chunk_bytes = chunk_queue.get()
            buffer_io = io.BytesIO(chunk_bytes)
            c = torch.load(
                buffer_io, weights_only=False
            )  # Loads directly into RAM, no mmap overhead

            buffer.insert(0, c)
            buffer = buffer[:buffer_size]
            total_data += 1

        # Drain wandb_queue: validator messages (no _rank) go straight to
        # wandb; worker messages are aggregated across ranks and emitted as
        # a single workers/* + sampler/* dict per trainer iter.
        while wandb_queue.qsize() > 0:
            msg = wandb_queue.get()
            rank = msg.pop("_rank", None)
            if rank is None:
                wandb.log(msg, step=trainer_iter)
            else:
                worker_latest[rank] = msg

        if worker_latest:
            n_workers = len(worker_latest)
            attempts = sum(s["attempts"] for s in worker_latest.values())
            goals = sum(s["goals"] for s in worker_latest.values())
            path_length = sum(s["path_length_sum"] for s in worker_latest.values())
            elapsed_max = max(s["elapsed_s"] for s in worker_latest.values())
            samplers = [s["sampler_stats"] for s in worker_latest.values()]

            wandb.log({
                "workers/active":             n_workers,
                "workers/attempts_total":     attempts,
                "workers/goals_total":        goals,
                "workers/goal_rate":          100.0 * goals / max(1, attempts),
                "workers/attempts_per_s":     attempts / max(1.0, elapsed_max),
                "workers/goals_per_s":        goals / max(1.0, elapsed_max),
                "workers/avg_plan_len":       path_length / max(1, goals),
                # Sampler aggregates: each worker has its own
                # sampler; status counts are means across workers, budgets are
                # population stats with the natural reducer per quantity.
                "sampler/solved_problems":    float(np.mean([s["solved_problems"]   for s in samplers])),
                "sampler/medium_problems":    float(np.mean([s["medium_problems"]   for s in samplers])),
                "sampler/unsolved_problems":  float(np.mean([s["unsolved_problems"] for s in samplers])),
                "sampler/avg_fail_streak":    float(np.mean([s["average_fail_streak"] for s in samplers])),
                "sampler/median_iter_in_status_medium": float(np.mean([s["median_iter_in_status_medium"] for s in samplers])),
                "sampler/budget_mean":        float(np.mean([s["budget_mean"] for s in samplers])),
                "sampler/budget_max":         int(max(s["budget_max"] for s in samplers)),
                "sampler/budget_min":         int(min(s["budget_min"] for s in samplers)),
            }, step=trainer_iter)

        # --- stats accumulators ---
        stats = {
            "q_pred_mean": 0.0,
            "q_pred_max": 0.0,
            "q_pred_min": 0.0,
            "q_target_mean": 0.0,
            "q_target_max": 0.0,
            "q_target_min": 0.0,
            "grad_norm": 0.0,
            "num_bounds": 0,
            "q_loss": 0,
            "policy_loss": 0,
        }
        num_updates = 0

        for chunk in buffer:
            batched_states = chunk.batched_states.to(device)
            batch_targets = chunk.batched_targets.to(device)
            target_map = chunk.target_map.to(device)
            bounds = chunk.bounds.to(device)
            # q_only=False also computes the policy head (model_output[0]) so it
            # can be trained as a trace-imitation signal.
            _micro = MICRO_BATCH_STATES > 0 and batched_states.num_graphs > MICRO_BATCH_STATES
            if MEM_LOG:
                torch.cuda.reset_peak_memory_stats(device)
            if _micro:
                model_output = (None, None, None)   # the trainable forward runs per micro-batch (_micro_backward)
            else:
                model_output = se.run_model_no_cache(batched_states, q_only=False)

            policy_pred = model_output[0]
            q_pred = model_output[1]

            with torch.no_grad():
                target_output = se_old.run_model_no_cache(batch_targets, q_only=True)
                q_target = -1 + global_max_pool(target_output[1], target_output[2])
                q_target = torch.cat(
                    [
                        torch.tensor([torch.nan, DEAD_END_REWARD], device=device),
                        q_target,
                    ]
                )

                t = q_target[target_map]
                mask = ~torch.isnan(bounds)
                t[mask] = torch.max(bounds[mask], t[mask])

                q_target = torch.min(
                    torch.max(t, torch.tensor(DEAD_END_REWARD, device=device)),
                    torch.tensor(0.0, device=device),
                )

            if _micro:
                q_target, q_pred, mask, q_loss, policy_loss = _micro_backward(
                    se, batched_states, q_target, bounds, MICRO_BATCH_STATES, POLICY_LOSS_WEIGHT)
                max_len = q_pred.shape[0]
            else:
                max_len = q_pred.shape[0]
            q_target = q_target[:max_len]
            if not _micro:
                mask = ~torch.isnan(q_target)
                q_loss = loss_fn(q_pred[mask], q_target[mask])

            # --- trace-imitation policy head -------------------------------
            # policy_pred is a per-state softmax distribution over actions
            # (_run_policy: softmax(scalar, action_batch)).  The on-trace
            # actions are the (state, action) pairs whose bound is finite
            # (they lead to a goal-path state).  The imitation objective is the
            # categorical cross-entropy that maximises the probability mass the
            # head puts on those on-trace actions, i.e. learn to *pick* the
            # actions that lie on a solved trace, including put-back moves with
            # negative q-progress that the monotone value head does not choose.
            # States with no on-trace action (off the solution path) carry no
            # target and drop out naturally.
            if _micro:
                pass  # the policy term is part of _micro_backward's loss, already backpropagated
            else:
                on_trace = ~torch.isnan(bounds)
                if on_trace.any():
                    policy_loss = -torch.log(
                        policy_pred[on_trace].clamp_min(1e-8)
                    ).mean()
                else:
                    policy_loss = torch.zeros((), device=device)
                loss = q_loss + POLICY_LOSS_WEIGHT * policy_loss

            if not _micro:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                shared_model.parameters(), MAX_GRAD_NORM
            )

            # ---- stats accumulation ----
            with torch.no_grad():
                if mask.any():
                    qp = q_pred[mask]
                    qt = q_target[mask]

                    stats["q_pred_mean"] += qp.mean().item()
                    stats["q_pred_max"] += qp.max().item()
                    stats["q_pred_min"] += qp.min().item()

                    stats["q_target_mean"] += qt.mean().item()
                    stats["q_target_max"] += qt.max().item()
                    stats["q_target_min"] += qt.min().item()
                stats["q_loss"] += q_loss.item()
                stats["policy_loss"] += policy_loss.item()
                stats["grad_norm"] += (
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )
                num_updates += 1

            value_optimizer.step()
            gnn_optimizer.step()
            value_optimizer.zero_grad()
            gnn_optimizer.zero_grad()
            if MEM_LOG:
                print("[mem] iter=%d upd=%d states=%d nodes=%d edges=%d q_rows=%d targets=%d target_nodes=%d "
                      "micro=%d peak_alloc_GB=%.2f peak_reserved_GB=%.2f" % (
                          trainer_iter, num_updates, batched_states.num_graphs, batched_states.num_nodes,
                          batched_states.num_edges, q_pred.shape[0], batch_targets.num_graphs,
                          batch_targets.num_nodes, MICRO_BATCH_STATES if _micro else 0,
                          torch.cuda.max_memory_allocated(device) / 1e9,
                          torch.cuda.max_memory_reserved(device) / 1e9), flush=True)

        if num_updates > 0:
            for k in [
                "q_pred_mean",
                "q_pred_max",
                "q_pred_min",
                "q_target_mean",
                "q_target_max",
                "q_target_min",
                "grad_norm",
                "q_loss",
                "policy_loss",
            ]:
                stats[k] /= num_updates
        total_backprops += num_updates
        wandb.log(
            {
                "trainer/iter": trainer_iter,
                "trainer/total_backprops": total_backprops,
                "trainer/updates_this_iter": num_updates,
                "q_loss": stats["q_loss"],
                "policy_loss": stats["policy_loss"],
                "q_pred/mean": stats["q_pred_mean"],
                "q_pred/max": stats["q_pred_max"],
                "q_pred/min": stats["q_pred_min"],
                "q_target/mean": stats["q_target_mean"],
                "q_target/max": stats["q_target_max"],
                "q_target/min": stats["q_target_min"],
                "grad_norm": stats["grad_norm"],
                "q_gap": stats["q_pred_mean"] - stats["q_target_mean"],
                "q_range": stats["q_pred_max"] - stats["q_pred_min"],
            },
            step=trainer_iter,
        )
        print(
            i,
            total_data,
            f"q_loss: {stats['q_loss']}, policy_loss: {stats['policy_loss']}",
        )
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            shared_model.save_to_folder(save_path)
            print("saving model")
