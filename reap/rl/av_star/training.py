# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

"""
Training entry point for AV-Star.

Imported by scripts/train.py without triggering any module-level side-effects.
"""
import multiprocessing as mp_std
import os
import random
import socket
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as mp

from reap.rl.av_star import Model
from reap.rl.av_star.worker import trainer, validation_process, worker_process
from reap.rl.data_layout import InputData


def main(
    # Data
    # Data: the dataset directory under $REAP_DATA (data/<env>), its training and validation subdirectories
    data_path="tabletop",
    train_dir="train",
    val_dir="validation",
    domain_file="domain.pddl",
    write_save_path=False,
    write_save_loc="",
    # Workers
    num_workers=10,
    pool_workers=5,
    validation=True,
    # Model architecture
    embedding_size=32,
    num_layers=30,
    attn_pooling=False,
    encoding="default",
    # Search
    search_budget=4096,
    search_batching=1,
    action_cost=0.5,
    # Retries on infeasible actions during expansion. The training-side value
    # drives bwqs in the worker processes; the validation-side value drives
    # rollouts in validation_process. Setting them to different values lets
    # you "train hard, deploy generously" (e.g. train_retries=0, val_retries=1).
    # If val_max_infeasible_retries is None it falls back to
    # max_infeasible_retries.
    max_infeasible_retries=2,
    val_max_infeasible_retries=None,
    # Adaptive budget
    budget_growth_threshold=3,
    budget_max_multiplier=8,
    # How much to add per per-problem failure past threshold. None => init = search_budget
    # (full init bump per failure → cap reached after ~9 consecutive failures
    # with multiplier=8).  Smaller values produce a gentler ramp that keeps
    # mean budgets from pegging near the cap; search_budget // 4 is a good choice.
    budget_growth_increment=None,
    # Budget growth mode: "additive" (default) adds budget_growth_increment
    # per failure past threshold; "exponential" doubles the budget each failure
    # past threshold.  Designed for use with a small init_budget (e.g. 15) so the
    # schedule is 15 → 30 → 60 → 120 → 240 → 480, capped at init × multiplier.
    budget_growth_mode="additive",
    easy_first=False,
    # Per-bwqs-attempt wall budget in training-side workers.  0 = no time limit
    # (the search is purely expansion-budget-bound).  A finite limit also caps
    # the adaptive budget growth.
    # Validation-side rollout time limit is configured separately via
    # val_time_limit_s and is unaffected by this flag.
    worker_time_limit_s=60,
    # ε-exploration in bwqs: probability of overriding a newly-pushed
    # child's heap priority to 0 (=high-priority).  0.0 is pure Q-greedy
    # search.  Higher (e.g. 0.1–0.3) helps the random-Q
    # bootstrap by injecting non-Q-best actions into the heap front.
    random_exploration=0.0,
    # Trainer
    dead_end_reward=-30,
    lr_readout=1e-3,
    lr_gnn=1e-4,
    max_grad_norm=1.0,
    max_updates_per_iter=10,
    buffer_size=40,
    # Weight of the auxiliary cross-entropy loss on the actions of solved plans (0 = off).
    policy_loss_weight=0.1,
    chunk_queue_size=20,
    # Cap on trainer outer iterations. None = run effectively forever
    # (100_000).
    max_outer_iters=None,
    # Validation
    val_time_limit_s=60 * 5,
    val_sleep_s=60,
    # Logging
    wandb_project="reap",
    # Reproducibility
    # If None (default), a random 32-bit int is drawn at startup; it's printed
    # and logged to wandb so the value can be reused for an exact-init replication.
    # When set, torch / numpy / python RNGs are all seeded at process entry,
    # and each spawned worker receives seed + rank for distinct deterministic
    # initial states.  Note: CUDA non-determinism still allows runs to diverge
    # slightly; this gives same-basin reproducibility, not bit-identical.
    seed=None,
    # ActionPoolingModule mode: "mask" (default, pools non-action
    # positions only, requires bool action input), "no_mask" (pool all
    # symbol positions; output 2× emb), or "separate" (pool
    # actions and non-actions separately; output 3× emb).  See
    # reap/utils/object_embeddings.py:ActionPoolingModule docstring.
    pooling_mode="mask",
    # Override the wandb run name and the save_path prefix.  Default (None):
    # the prefix is the basename of data_path and the run name is the
    # save_path basename.
    wandb_run_name=None,
):
    mp.set_start_method("spawn", force=True)

    # ---- seeding ----
    import numpy as np
    import torch
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Hostname:", socket.gethostname())
    print(f"Seed:      {seed}")
    print("=" * 50)
    print("Hyperparameters:")
    print(f"  data_path={data_path}, train_dir={train_dir}, val_dir={val_dir}")
    print(f"  num_workers={num_workers}, pool_workers={pool_workers}, validation={validation}")
    print(f"  embedding_size={embedding_size}, num_layers={num_layers}, attn_pooling={attn_pooling}, encoding={encoding}, pooling_mode={pooling_mode}")
    print(f"  search_budget={search_budget}, search_batching={search_batching}, action_cost={action_cost}")
    # Resolve the validation-side retries: None means "use the training value".
    if val_max_infeasible_retries is None:
        val_max_infeasible_retries = max_infeasible_retries
    print(f"  max_infeasible_retries={max_infeasible_retries} (val={val_max_infeasible_retries})")
    print(f"  budget_growth_threshold={budget_growth_threshold}, budget_max_multiplier={budget_max_multiplier}, budget_growth_increment={budget_growth_increment}, budget_growth_mode={budget_growth_mode}")
    print(f"  worker_time_limit_s={worker_time_limit_s} (0 = no time limit)")
    print(f"  random_exploration={random_exploration}")
    print(f"  dead_end_reward={dead_end_reward}, lr_readout={lr_readout}, lr_gnn={lr_gnn}")
    print(f"  max_grad_norm={max_grad_norm}, max_updates_per_iter={max_updates_per_iter}, buffer_size={buffer_size}")
    print(f"  policy_loss_weight={policy_loss_weight}")
    print(f"  chunk_queue_size={chunk_queue_size}, wandb_project={wandb_project}")
    print("=" * 50)

    os.environ["COLLISION_POOL_WORKERS"] = str(pool_workers)

    # wandb is optional: offline by default (WANDB_MODE=online to sync, =disabled to turn it off), logs under
    # $REAP_OUT/wandb. No entity is set; an online run goes to the logged-in user's default entity.
    from reap.paths import out_root
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(out_root()))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    import wandb

    if os.environ["WANDB_MODE"] == "online":
        wandb.login()

    data_loader = InputData(data_path, train_dir=train_dir, domain_file=domain_file)
    # which feasibility checker this run trains against (raises if a shelf checker is not selected explicitly)
    from reap.domains import checker_info
    checker = checker_info.describe_data(data_loader.pddl_domains_dir / domain_file,
                                         [data_loader.train_dir, data_loader.pddl_domains_dir / val_dir])
    print(checker_info.log_line(checker, "training"), flush=True)

    # Save path naming must be collision-resistant even when several runs use the same `--seed` (which makes
    # random.randint deterministic): an OS-entropy random draw, not the seeded random module.
    import secrets
    suffix = str(secrets.randbelow(1_000_000))
    save_path = out_root() / "model" / (
        (wandb_run_name or Path(data_path).name)
        + "_"
        + datetime.now().strftime("%d-%m_%H-%M-%S")
        + "_"
        + suffix
    )
    # Print the save_path basename so it can be found in the log.
    print(f"SAVE_PATH={str(save_path).split('/')[-1]}", flush=True)

    if write_save_path:
        path = Path(write_save_loc)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{str(save_path).split('/')[-1]}\n")

    wandb.init(
        project=wandb_project,
        name=wandb_run_name if wandb_run_name else str(save_path).split("/")[-1],
        config={
            "data_path": data_path,
            "train_dir": train_dir,
            "val_dir": val_dir,
            "domain_file": domain_file,
            "checker": checker["label"],
            "checker_name": checker["checker"],
            "checker_domain": checker["domain"],
            "checker_library": checker["library"],
            "checker_library_sha256": checker["library_sha256"],
            "tampanda_commit": checker["tampanda_commit"],
            "code_commit": checker["code_commit"],
            "num_workers": num_workers,
            "pool_workers": pool_workers,
            "embedding_size": embedding_size,
            "num_layers": num_layers,
            "attn_pooling": attn_pooling,
            "encoding": encoding,
            "search_budget": search_budget,
            "search_batching": search_batching,
            "action_cost": action_cost,
            "max_infeasible_retries": max_infeasible_retries,
            "val_max_infeasible_retries": val_max_infeasible_retries,
            "budget_growth_threshold": budget_growth_threshold,
            "budget_max_multiplier": budget_max_multiplier,
            "budget_growth_increment": budget_growth_increment,
            "dead_end_reward": dead_end_reward,
            "lr_readout": lr_readout,
            "lr_gnn": lr_gnn,
            "max_grad_norm": max_grad_norm,
            "max_updates_per_iter": max_updates_per_iter,
            "policy_loss_weight": policy_loss_weight,
            "buffer_size": buffer_size,
            "chunk_queue_size": chunk_queue_size,
            "val_time_limit_s": val_time_limit_s,
            "val_sleep_s": val_sleep_s,
            "seed": seed,
            "pooling_mode": pooling_mode,
            "budget_growth_mode": budget_growth_mode,
            "worker_time_limit_s": worker_time_limit_s,
            "random_exploration": random_exploration,
        },
    )
    wandb_queue = mp_std.Queue()
    chunk_queue = mp_std.Queue(maxsize=chunk_queue_size)

    import torch as _torch
    _device = "cuda:0" if _torch.cuda.is_available() else "cpu"

    model = Model.create_new(
        data_loader.domain,
        attn_pooling=attn_pooling,
        embedding_size=embedding_size,
        num_layer=num_layers,
        encoding=encoding,
        pooling_mode=pooling_mode,
    )
    # Warm-start / resume: when WARMSTART_CKPT is set, load the trained
    # weights from a prior checkpoint instead of random init (continues a
    # timed-out run). Off by default. Same architecture is rebuilt
    # by load_from_folder; the model.to(_device) below handles placement.
    _ws = os.environ.get("WARMSTART_CKPT")
    if _ws:
        print(f"WARM-START: loading model weights from {_ws}", flush=True)
        model = Model.load_from_folder(_ws)

    # Move model to GPU and prepare param tensors for cross-process sharing.
    # The full Model object is NOT picklable (it holds xmimir.XDomain, a
    # pymimir C-extension type), so it cannot be passed directly to spawn-
    # started children.  Instead only the param/buffer tensors are shipped
    # (Dict[str, Tensor] is fully picklable and CUDA tensors get IPC handles
    # under torch.multiprocessing).  Workers load their own Model instance
    # from disk to get the object graph, then re-point each parameter's
    # `.data` to the corresponding shared CUDA tensor.  Result: workers see
    # in-place updates from optimizer.step() with zero staleness, and no
    # disk reload.
    model.to(_device)
    model.share_memory()  # no-op for CUDA; sets shm for CPU fallback
    model.save_to_folder(save_path)  # workers load their own Model from this

    shared_param_tensors = {name: p for name, p in model.named_parameters()}
    shared_buffer_tensors = {name: b for name, b in model.named_buffers()}

    old_model = Model.load_from_folder(save_path)

    worker_kwargs = dict(
        shared_param_tensors=shared_param_tensors,
        shared_buffer_tensors=shared_buffer_tensors,
        search_budget=search_budget,
        search_batching=search_batching,
        action_cost=action_cost,
        budget_growth_threshold=budget_growth_threshold,
        budget_max_multiplier=budget_max_multiplier,
        budget_growth_increment=budget_growth_increment,
        budget_growth_mode=budget_growth_mode,
        max_infeasible_retries=max_infeasible_retries,
        domain_file=domain_file,
        train_dir=train_dir,
        seed=seed,
        worker_time_limit_s=worker_time_limit_s,
        random_exploration=random_exploration,
        easy_first=easy_first,
    )

    processes = []
    for rank in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(rank, save_path, data_path, wandb_queue, chunk_queue),
            kwargs=worker_kwargs,
        )
        p.start()
        processes.append(p)

    if validation:
        p = mp.Process(
            target=validation_process,
            args=(save_path, data_path, wandb_queue),
            kwargs=dict(
                max_infeasible_retries=val_max_infeasible_retries,
                val_time_limit_s=val_time_limit_s,
                val_sleep_s=val_sleep_s,
                domain_file=domain_file,
                val_dir=val_dir,
                seed=seed,
            ),
        )
        p.start()
        processes.append(p)

    try:
        trainer(
            model, old_model, save_path, chunk_queue, wandb_queue,
            dead_end_reward=dead_end_reward,
            lr_readout=lr_readout,
            lr_gnn=lr_gnn,
            max_grad_norm=max_grad_norm,
            max_updates_per_iter=max_updates_per_iter,
            buffer_size=buffer_size,
            policy_loss_weight=policy_loss_weight,
            max_outer_iters=max_outer_iters,
            seed=seed,
        )
    finally:
        # Tear down the child processes when the trainer returns (max_outer_iters reached) and when it raises (for
        # example CUDA out of memory). Otherwise the workers keep producing chunks and the validator keeps doing
        # rollouts, holding the GPU, and after an exception the non-daemon children keep the run from exiting.
        print(f"[main] Trainer finished, terminating {len(processes)} child processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                print(f"[main] WARN process {p.pid} did not exit on SIGTERM, killing")
                p.kill()
                p.join(timeout=5)
        print(f"[main] All child processes cleaned up.")
