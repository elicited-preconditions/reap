# PDDLStream on the open tabletop

PDDLStream is a task-and-motion planner that interleaves symbolic search with calls to streams, procedures that sample
or test the continuous parameters of actions.

- C. R. Garrett, T. Lozano-Pérez and L. P. Kaelbling. PDDLStream: Integrating symbolic planners and blackbox samplers
  via optimistic adaptive planning. ICAPS 2020.

pddlstream and Fast Downward stay unvendored git submodules under their own licenses (GPL-3.0). The scripts in
`bench/` copy no pddlstream code and are covered by the repository's MIT license.

## Adaptation

We run the adaptive algorithm on the 300 Table 1 problems, with the same scenes, feasibility checks and 300 s cap (scene
construction, about 10 s, not included). The streams only check whether a pick or a put is feasible, so no continuous
parameter is sampled, and a motion-planner query is one such check. All four configurations use unit action costs,
the first plan found, a search-to-sample ratio of 1 and 30 s per Fast Downward call:

| Configuration | `CONFIG` | Output directory | Solved | MP / plan |
|---|---|---|---|---|
| test-stream, ff-astar2: all cells given, a fluent stream tests each put | `test_ffastar2` | `test_stream_ffastar2` | 41.7% | 1.7 |
| test-stream, ff-eager | `test_ffeager` | `test_stream_ffeager` | 42.7% | 2.1 |
| cell-generator, ff-eager: cells proposed lazily | `gen_ffeager` | `cell_generator_ffeager` | 43.7% | 9.8 |
| informed cells + occlusion axiom, ff-eager | `informed_cells_occlusion` | `informed_cells_occlusion` | 64.3% | 23.6 |

The last one orders the generated cells by the initial layout and adds a precondition to `pick`: no other object may
occlude the picked one (derived `occluder` predicate). Solved and MP / plan are the Table 1 numbers.

## Building upstream

```bash
git submodule update --init --recursive     # pddlstream 2c7d6f5, Fast Downward 11e98c5
cd baselines/pddlstream/pddlstream/downward && CMAKE_POLICY_VERSION_MINIMUM=3.5 ./build.py && cd -
```

Fast Downward needs CMake, a C++ compiler and make. The main install covers the Python dependencies. It also puts
CMake 4 into the environment (a dependency of pymimir), which rejects the old minimum version that Fast Downward
11e98c5 declares; `CMAKE_POLICY_VERSION_MINIMUM=3.5` lets it build.

## Commands

PDDLStream is not trained. Each configuration is a single evaluation run, one Python process per problem:

```bash
python baselines/pddlstream/bench/run.py informed_cells_occlusion outputs/baselines/pddlstream/informed_cells_occlusion
python scripts/report.py table1_baselines --only pddlstream
```
