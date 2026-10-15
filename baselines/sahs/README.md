# SAHS on the open tabletop

SAHS learns two relational GNNs, one for picks and one for puts, that score abstract actions over a state abstraction
with geometric predicates (whether an object can be grasped or placed, and which objects occlude these motions). A
best-first search over abstract actions uses these scores.

- B. Kim and L. Shimanuki. Learning value functions with relational state representations for guiding
  task-and-motion planning. CoRL 2019.
- B. Kim, L. Shimanuki, L. P. Kaelbling and T. Lozano-Pérez. Representation, learning, and planning algorithms for
  geometric task and motion planning. IJRR 41(2), 2022.

## Adaptation

We follow the IJRR version, with its full edge encoding and its large-margin ranking loss.

- The objects are the cylinders. A put can target any empty cell, about 63 per state.
- The geometric predicates are computed with inverse kinematics and motion planning through REAP's feasibility
  checker. They account for almost all motion-planner queries of SAHS.
- Training data: one example per step of the reference plans of REAP's 600 training problems (2,019 examples from the
  573 plans that could be replayed). Adam, learning rate 1e-3, full batch, 300 epochs.
- Put candidates with the same input features as the chosen one are excluded from the ranking losses of both networks.
  Without this, the put network does not learn.
- The arm has a fixed base, the heuristic has no goal-region terms, and each pick uses a single grasp. The check
  retries IK from up to three seeds, runs 16 processes per problem and retries a failed action up to three times.

## Commands

```bash
python baselines/sahs/run.py data     # training examples from the reference plans
python baselines/sahs/run.py train    # writes outputs/baselines/sahs_training/results/models/sahs_{pick,put}.pt
python baselines/sahs/run.py eval     # the 300 Table 1 problems, 300 s each (default: checkpoints/baselines/sahs/)
python scripts/report.py table1_baselines --only sahs
```

Outputs go to `outputs/baselines/`. The evaluation loads `checkpoints/baselines/sahs/sahs_pick.pt` and `sahs_put.pt`
(other weights: `--q-pick`, `--q-put`; `--help` lists the options).

## Reported result (Table 1)

One training run, evaluated once with the final weights: 51.0% solved, MP / plan 2937.4 [2857.0], exp./steps 1.37
[1.00], 68.8 s per solved problem. MP / plan includes the queries that build the predicates.
