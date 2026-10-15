# BT-h on the open tabletop

BT-h refines a given plan skeleton by backtracking over the parameters of its actions. A transformer classifier
predicts whether a partially refined skeleton can be completed; candidates with low predicted feasibility are
discarded, and the classifier's confidence decides when to jump back.

- B. Cieślar, L. P. Kaelbling, T. Lozano-Pérez and J. Mendez-Mendez. Learning long-horizon action dependencies in
  sampling-based bilevel planning. CoRL 2024.

## Adaptation

- REAP's domain fixes the target cell of every action, so the domain is extended with a `relocate` action whose target
  cell is chosen during refinement, uniformly among the free cells (`bt_h/domain/relocate_domain.pddl`).
- At test time BT-h is given the order in which the reference plan of the problem moves the cylinders, and chooses the
  target cells. The cost of the search that found the reference plans is not counted.
- The classifier is trained with the paper's data-collection procedure on 600 synthetic problems (2 to 7 thin
  cylinders, 1 to 6 blocking the target), of which the demonstrations solve 575. Data collection uses 30 search trees
  per level that keep all their nodes as examples.
- Smaller classifier than in the paper (width 32, 4 heads, 2 blocks), batches of 32 for at most 400 steps per round,
  veto threshold set by hand, one training seed.
- The check runs in one process with a smaller planner step than REAP's and re-checks an infeasible verdict up to two
  times.

## Commands

```bash
python baselines/bt_h/run.py data     # instances and seed demonstrations (about 45 min)
python baselines/bt_h/run.py train    # data collection loop and classifier (about 30 min)
python baselines/bt_h/run.py eval     # the 300 Table 1 problems (default: checkpoints/baselines/bt_h/bt_h.pt)
python scripts/report.py table1_baselines --only bt_h
```

Outputs go to `outputs/baselines/` (the trained classifier: `outputs/baselines/bt_h/checkpoints/bt_h.pt`).

## Reported result (Table 1)

One training run, given the plan ordering: 81.7% solved, MP / plan 28.5 [3.0], 4.5 s per solved problem.
