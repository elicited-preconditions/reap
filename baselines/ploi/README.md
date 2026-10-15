# PLOI on the open tabletop

PLOI learns a GNN that scores the importance of each object from the initial state and the goal. It plans on the
subproblem with the objects that score at least gamma^N, N = 1, 2, ..., until the plan also solves the full problem.

- T. Silver, R. Chitnis, A. Curtis, J. B. Tenenbaum, T. Lozano-Pérez and L. P. Kaelbling. Planning with learned object
  importance in large problem instances using graph neural networks. AAAI 2021.

The network, the labelling and the graph encoding are ported or adapted from github.com/tomsilver/ploi (MIT license,
`LICENSES/ploi-MIT.txt`).

## Adaptation

- The graph has a node per cell and per cylinder, with node features for the object type and for emptiness and holding
  in the initial state and in the goal, and edges for occupancy and adjacency.
- Only cylinders are scored and pruned. Goal objects are always kept.
- The planner is REAP's breadth-first search (PLOI + BrFS) or greedy best-first search with hFF (PLOI + hFF), with
  REAP's feasibility check, in place of the paper's planners.
- Labels as in the authors' code: a cylinder is important if a plan for the full problem (breadth-first search with an
  hFF fallback) touches it. Training problems without such a plan are dropped, which leaves 436 of the 600.
- Encode-process-decode network (3 message-passing rounds, width 16, weighted cross-entropy), gamma = 0.9, one
  training seed, early stopping on a validation set. Outputs go to `outputs/baselines/`.

## Commands

```bash
python baselines/ploi/run.py labels                   # optional: regenerate the shipped labels (data/baselines/ploi/)
python baselines/ploi/run.py train                    # writes outputs/baselines/ploi.pt
python baselines/ploi/run.py eval --mode ploi_hff     # default model: checkpoints/baselines/ploi/ploi.pt
python baselines/ploi/run.py eval --mode ploi_bfs
python scripts/report.py table1_baselines --only ploi
```

## Reported results (Table 1)

One training run. PLOI + hFF: 72.3% solved, MP / plan 210.7 [2.0], checks/exp. 4.15, exp./steps 8.26 [1.00], 36.2 s
per solved problem. PLOI + BrFS: 53.3%, 97.3 [2.0], 3.18, 10.70 [1.00], 18.0 s.
