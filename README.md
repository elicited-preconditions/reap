# REAP: Learning General Rearrangement Heuristics from Motion-Planner Feedback

![REAP in the five environments](media/teaser.webp)

Robotic rearrangement tasks require planning sequences of actions that fulfill high-level specifications while also
being geometrically feasible. Symbolic abstractions make the combinatorial structure of rearrangement tasks tractable,
but they usually do not fully capture the geometric constraints that determine which actions a robot can actually
realize. These constraints are hard to express in the symbolic model. We propose REAP (Rearrangement with Elicited
Action Preconditions), a framework in which a relational graph neural network heuristic is learned over a simple
symbolic abstraction. A motion planner serves as a supervisor that elicits the hidden geometric preconditions of the
problem during training. Deployed with greedy best-first search, the learned heuristic generalizes out-of-distribution
along object count, plan length, and object arrangement, and can be trained jointly on several domains.

Project page and paper: https://elicited-preconditions.github.io

## Install

Linux, Python 3.12 (the reported numbers were produced with Python 3.12.7).

```bash
conda create -y -n reap python=3.12
conda activate reap
pip install -e . -r requirements-tampanda.txt
export MUJOCO_GL=egl
```

- `pip install -e .` installs this repository with the pinned dependencies of `requirements.txt` (the versions behind
  the reported numbers).
- `requirements-tampanda.txt` installs TAMPanda v1.1.0, the simulator and feasibility checkers:
  `tampanda @ git+https://github.com/snoato/TAMPanda.git@v1.1.0`.
- `MUJOCO_GL=egl` runs MuJoCo headless.
- wandb is optional. Training logs offline by default (`WANDB_MODE=offline`); `WANDB_MODE=online` syncs,
  `WANDB_MODE=disabled` turns it off.

**3D stacking only: frantik and VAMP.** The 3D-stacking checker needs two native packages, `frantik` (analytic Franka
IK, https://github.com/CoMMALab/frantik) and `vamp-planner` 0.6.4 (https://github.com/KavrakiLab/vamp). Install them
into directories of their own and name these directories in `A19V2_PYLIB` (TAMPanda's variable for this). The other
environments do not need them.

```bash
PYLIB=$HOME/reap_pylib                                                   # any directory
conda create -y -p "$PYLIB/eigen" -c conda-forge eigen                   # Eigen3 headers for frantik
git clone https://github.com/CoMMALab/frantik "$PYLIB/frantik"
git -C "$PYLIB/frantik" checkout 4a23df38d291cb7f562b75091af3a3ccfc11f46b
CMAKE_PREFIX_PATH="$PYLIB/eigen" pip install --no-deps --target "$PYLIB/frantik_pkg" "$PYLIB/frantik"
pip install --no-deps --target "$PYLIB/vamp_pkg" vamp-planner==0.6.4
export A19V2_PYLIB="$PYLIB/frantik_pkg:$PYLIB/vamp_pkg"
```

Use `--no-deps` for both. Without it, `--target` also installs a second numpy, which then shadows the pinned one.

**PDDLStream baseline only.** pddlstream and Fast Downward are git submodules:

```bash
git submodule update --init --recursive
```

`baselines/pddlstream/README.md` describes the Fast Downward build.

## Download assets

Trained models and datasets are assets of the GitHub release v1.0. The script downloads them, checks each file
against its sha256 and extracts it at the repository root.

```bash
bash scripts/download_assets.sh                      # all groups
bash scripts/download_assets.sh tabletop datasets    # only the named groups
```

| Group | Extracts to | Size |
|---|---|---|
| `tabletop`, `confined`, `multilevel_shelf`, `stacking`, `access19` | `checkpoints/<group>/seed_<k>/` | 9.3 to 190 MB each |
| `baselines` | `checkpoints/baselines/{sahs,ploi,bt_h}/` | 0.3 MB |
| `datasets` | `data/<env>/` (problem sets), `data/baselines/` (PLOI labels) | 2.0 MB |

The tarballs stay in `assets/`. A tarball that is already there with the right sha256 is not downloaded again.

## Evaluate a pretrained open-tabletop model

```bash
bash scripts/download_assets.sh tabletop datasets
python scripts/eval.py --config configs/tabletop.yaml --seed 1
```

This runs REAP-GBFS with seed 1 on the 300 Table 1 problems (300 s per problem) and writes per-problem results to
`outputs/tabletop/reap_gbfs/seed_1/`. `python scripts/report.py table1 --seeds seed_1` prints the row.
`python scripts/eval.py --config configs/tabletop.yaml --list` lists the other protocols.

## Train

```bash
mkdir -p logs
python scripts/train.py --config configs/tabletop.yaml --seed 1 2>&1 | tee logs/tabletop_s1.log
```

- One config per environment: `configs/tabletop.yaml`, `confined.yaml`, `multilevel_shelf.yaml`, `stacking.yaml`,
  `access19.yaml`, `joint.yaml`. The header of each config lists its variants (for example the three Access-19
  policies) and its checkpoint rule.
- `--<parameter> <value>` overrides a training value of the config; `python scripts/train.py --config <config> --help`
  lists them.
- From a cold start, the first training update can take up to about an hour on confined and half an hour on joint with
  16 workers; fewer workers take longer. The reported runs used one GPU (24 GB for confined, 48 GB for joint and
  Access-19), 24 CPU cores and 120 GB of RAM.
- Access-19: the reported single-policy runs used 16 workers on a 48 GB GPU (peak about 31 GB); 8 workers need about
  18 GB; on smaller cards use 4 workers and `MICRO_BATCH_STATES=32` (`--num_workers 4 --env MICRO_BATCH_STATES=32`).
- Training stops at the config's `walltime` (24 h; 60 h for 3D stacking) or at `--walltime`. The tabletop config has
  none; the reported tabletop runs were stopped after 27 to 30 h.
- A run stopped by `--walltime` exits with code 143; its folder keeps the weights from the last save, every 10th
  trainer iteration.
- The run directory is `outputs/model/<name>`, where `<name>` is printed as `SAVE_PATH=<name>` at the start of the log.
  The trainer writes the final weights there, and the validator writes `checkpoint/checkpoint_<N>/`.
- Keep the log (`2>&1 | tee ...`): `scripts/select_checkpoint.py best_val` reads the validation results from it.

Several seeds are a shell loop over `--seed`:

```bash
for k in 1 2 3; do
  python scripts/train.py --config configs/confined.yaml --seed $k 2>&1 | tee logs/confined_s$k.log
done
```

Pick the checkpoint to report with the rule of the environment:

| Environment | Reported checkpoint | How to get it |
|---|---|---|
| Open tabletop, sample efficiency, noise | last best validation save | `python scripts/select_checkpoint.py best_val --log logs/tabletop_s1.log` |
| Confined shelf, multi-level shelf, joint training | final weights | the run directory `outputs/model/<name>` |
| 3D stacking `best_val/` | last best validation save | `python scripts/select_checkpoint.py best_val --log <log>` |
| 3D stacking `final/` | final weights | the run directory |
| Access-19 `single/` | most verified solves on the selection set | `python scripts/select_checkpoint.py selection_set --run_dir outputs/model/<name>` |
| Access-19 `extraction/` | final weights | the run directory |
| Access-19 `reload/` | best validation save over a chain of runs | `python scripts/select_checkpoint.py best_val --log <log 1> --log <log 2> ...` |

To evaluate a new checkpoint, put it at `checkpoints/<env>/seed_<k>/` (the layout of the checkpoint assets) or pass
`--checkpoint <dir>` to `scripts/eval.py` (Access-19: `--single`, `--extraction`, `--reload`).

**Note on the open-tabletop runs.** The open-tabletop runs of Table 1 and the sample-efficiency runs used an earlier
training loop. In it, failed searches give no targets and the problem sampler uses the simpler scheme described in the
appendix "Training Details". The released loop is the one used for the other environments. Retraining with
`configs/tabletop.yaml` therefore matches the logged hyperparameters but not that code. The shipped Table 1
checkpoints reproduce the reported numbers.

The single-axis ablations (appendix "Open-Tabletop Ablations and Sample Efficiency") can be retrained with the flags
`--random_exploration`, `--dead_end_reward`, `--worker_time_limit_s`, `--budget_max_multiplier` and `--buffer_size`;
their logs and checkpoints are not released.

## Reproduce the reported results

The motion planners are not seeded, so reruns can deviate slightly from the reported numbers.

| Result | Command (`<k>` is the seed) | Checkpoint | Report |
|---|---|---|---|
| Table 1, REAP-GBFS | `python scripts/eval.py --config configs/tabletop.yaml --seed <k>` (k = 1 to 5) | `checkpoints/tabletop/seed_<k>` | `python scripts/report.py table1` |
| Table 1, REAP-greedy | same, with `--protocol greedy` | `checkpoints/tabletop/seed_<k>` | same |
| Table 1, hFF and BrFS | same, with `--protocol hff` or `--protocol brfs` | none (no learned model) | same |
| Table 1, SAHS, PLOI, BT-h, PDDLStream | `baselines/<name>/README.md` | `checkpoints/baselines/<name>` | `python scripts/report.py table1_baselines` |
| Fig. 3; appendix "PDDLStream Baseline" | the Table 1 commands above | as above | `python scripts/report.py figure3`; `python scripts/report.py pddlstream_appendix` |
| Table 2, confined shelf | `python scripts/eval.py --config configs/confined.yaml --seed <k>` (k = 1 to 3) | `checkpoints/confined/seed_<k>` | `python scripts/report.py table2` |
| Table 2, multi-level shelf | `python scripts/eval.py --config configs/multilevel_shelf.yaml --seed <k>` | `checkpoints/multilevel_shelf/seed_<k>` | `python scripts/report.py table2` |
| Table 2, 3D stacking | `python scripts/eval.py --config configs/stacking.yaml --seed <k> --kind best_val`, then the same with `--kind final` | `checkpoints/stacking/seed_<k>/{best_val,final}` | `python scripts/report.py stacking` |
| Table 2, Access-19 | `python scripts/eval.py --config configs/access19.yaml --seed <k>` | `checkpoints/access19/seed_<k>/{single,extraction,reload}` | `python scripts/report.py access19` |
| Joint training | `python scripts/train.py --config configs/joint.yaml --seed <k>` (k = 1, 2), then `python scripts/eval.py --config configs/joint.yaml --seed <k> --checkpoint <run directory>` | not shipped: the run directory (final weights) | `python scripts/report.py joint` |

- Sample efficiency (appendix "Open-Tabletop Ablations and Sample Efficiency"), checkpoints not shipped: train with
  `python scripts/train.py --config configs/tabletop.yaml --seed <k> --train_dir train_n100 --max_infeasible_retries 0 --val_max_infeasible_retries 1 --worker_time_limit_s 0`
  (sizes n100, n200, n400; k = 1 to 3), then
  `python scripts/eval.py --config configs/tabletop.yaml --protocol sample_efficiency --size n100 --seed <k> --checkpoint <checkpoint>`,
  where `<checkpoint>` is the pick of `scripts/select_checkpoint.py best_val`.
- Noise (appendix "Execution Under Perturbed Observations"), checkpoint not shipped: train with
  `python scripts/train.py --config configs/tabletop.yaml --seed <k> --worker_time_limit_s 0`, then
  `python scripts/eval.py --config configs/tabletop.yaml --protocol noise --checkpoint <checkpoint>` (`<checkpoint>`
  as above). The tables are written to `outputs/noise/REPORT.md`, and `python scripts/report.py noise outputs/noise`
  prints them again.

## Acknowledgement

The training loop and the relational GNN build on AV* and the plangolin code base of Aichmüller et al. [Learning to
Search and Searching to Learn for Generalization in Planning, ICML 2026]; the files adapted from it are released with
the authors' permission.

M. Aichmüller, Y. Hesse, H. Geffner. Learning to Search and Searching to Learn for Generalization in Planning.
International Conference on Machine Learning (ICML), 2026. arXiv:2605.25720 (https://arxiv.org/abs/2605.25720).

## Citation

```bibtex
@inproceedings{swoboda2026reap,
  title     = {{REAP}: Learning General Rearrangement Heuristics from Motion-Planner Feedback},
  author    = {Swoboda, Daniel and Geffner, Hector},
  booktitle = {Proceedings of the Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

MIT, RWTH Aachen University (`LICENSE`). The third-party notices are in `LICENSES/`. pddlstream and Fast Downward are
GPL-3.0 git submodules; they are not vendored. The dependencies mifrost and pymimir (installed from `requirements.txt`)
are GPL-3.0, and quadprog (installed with TAMPanda) is GPL-2.0-or-later; none of them is vendored.

This release was cleaned up and documented with the help of a large language model (Claude). The methods, experiments and results are those of the paper.
