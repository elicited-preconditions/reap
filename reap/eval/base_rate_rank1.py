"""Base rate of geometrically infeasible actions, and rank-1 infeasibility of uniform / hFF / REAP Q.
Loads the five released Table 1 checkpoints from $REAP_CKPT/tabletop/.

Per state s, every applicable action exactly as the searches enumerate it
(sorted(generate_actions(s), key=str)) gets:
  infeasible_p1, infeasible_p2 -- sim.filter_action(s, a) (True = infeasible), two independent passes
                                  (feasibility cache cleared between them; the planner is unseeded)
  q_<seed>                     -- REAP Q for each of the five Table 1 checkpoints, one forward pass,
                                  aligned with the sorted actions (as gbfs_value zips them)
  hff                          -- _hff_value(ff, pm_problem, a.apply(s)), as gbfs_hff pushes it
Rank 1:  Q -> what gbfs_value pops first: priority -v (weight=0, weight2=1) => argmax v (ties: the
         code's heap order). hFF -> lowest h; reported tie-averaged over the argmin set AND under the
         code's own tie-break (children pushed in sorted order into a heap, popped once, as gbfs_hff).
States:  PRIMARY = the 300 initial states (gripper empty => picks only).
         SECONDARY = a random walk from each initial state: k = 1..6 uniformly random FEASIBLE actions
         (pass-1 verdicts), numpy seed = SEED + problem index; every visited non-goal state recorded,
         as its action prefix, so pass 2 re-checks the identical state.

    python -m reap.eval.base_rate_rank1 --shard <dataset label> --out <dir>   # one of the 15 cells
    python -m reap.eval.base_rate_rank1 --sanity --out <dir>                 # rank-1 == first pop?
"""
import argparse, contextlib, csv, heapq, json, os, re, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
os.environ.setdefault("MUJOCO_GL", "egl")
from reap.eval import eval_all_datasets as E                                              # noqa: E402 (chdirs to REPO)
import numpy as np                                                         # noqa: E402
import pymimir as pm                                                       # noqa: E402
import torch                                                               # noqa: E402
import reap.xmimir as xmi                                                       # noqa: E402
import reap.rl.av_star.search as S                                    # noqa: E402
import reap.rl.av_star.sim as SIM                                     # noqa: E402
from reap.rl.av_star.agent import Model                               # noqa: E402
from reap.rl.av_star.utils import StateEmbedder as StateEmbedder2     # noqa: E402

# label -> checkpoint directory: the five Table 1 seeds (--checkpoints label=dir ...; default seed_k ->
# $REAP_CKPT/tabletop/seed_k)
SEEDS = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_5"]
CHECKPOINTS = {}
SEED = 20260929
WALK = 6


def load_models():
    out = {}
    for lab in SEEDS:
        ck = CHECKPOINTS.get(lab) or os.path.join(os.environ.get("REAP_CKPT") or "checkpoints", "tabletop", lab)
        m = Model.load_from_folder(str(ck)); m.to("cpu"); m.eval()
        out[lab] = (StateEmbedder2(m, low_memory=True), str(ck))
    return out


def qvals(emb, s, n):
    v, _ = emb.get_q_values([s])
    v = [float(x) for x in v.detach().cpu().reshape(-1).tolist()]
    assert len(v) == n, (len(v), n)
    return v


def code_first_pop(priorities):
    """Index popped first when priorities are pushed in order into a heap (as the searches do)."""
    h = []
    for i, p in enumerate(priorities):
        heapq.heappush(h, S.PrioritizedItem(p, i))
    return heapq.heappop(h).item


def measure_state(gen, s, models, ff, pm_problem, verdict_pass):
    acts = sorted(gen.action_generator.generate_actions(s), key=str)
    rows = []
    q = {lab: qvals(emb, s, len(acts)) for lab, (emb, _) in models.items()}
    hf = [S._hff_value(ff, pm_problem, S.next_real(a.apply(s), gen)) for a in acts]
    for i, a in enumerate(acts):
        rows.append({"action": str(a), "verb": str(a).strip("()").split()[0],
                     "infeasible_p%d" % verdict_pass: bool(SIM.filter_action(s, a)),
                     "hff": hf[i], **{"q_" + lab: q[lab][i] for lab in q}})
    return acts, rows


def shard(a):
    dsl = dict(E.EVAL_DATASETS)
    probs = E.load_problems(E.REPO_ROOT / dsl[a.shard], "domain.pddl")
    models = load_models()
    out_rows, states_meta = [], []
    for pi, prob in enumerate(probs):
        gen = xmi.XSuccessorGenerator(prob)
        pm_problem = gen.problem.base
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            ff = pm.FFHeuristic(pm_problem)
        pid = "%s/%s" % (a.shard, os.path.basename(str(prob.filepath)))
        # ---- pass 1: initial state + seeded random walk over pass-1-feasible actions
        rng = np.random.default_rng(SEED + pi)
        s, prefix, visited = gen.initial_state, [], []
        for k in range(WALK + 1):
            acts, rows = measure_state(gen, s, models, ff, pm_problem, 1)
            sid = "s0" if k == 0 else "walk%d" % k
            visited.append((sid, list(prefix), s, acts, rows))
            feas = [i for i, r in enumerate(rows) if not r["infeasible_p1"]]
            if k == WALK or not feas:
                break
            j = int(rng.choice(feas))
            nxt = S.next_real(acts[j].apply(s), gen)
            if nxt.is_goal():
                break                     # goal states are not recorded
            prefix.append(str(acts[j])); s = nxt
        # ---- pass 2: identical states, feasibility cache cleared first
        # (clear_cache=True empties the verdict cache, so pass 2 runs every check again)
        SIM.reset_filter_stats(clear_cache=True)
        for sid, pre, st, acts, rows in visited:
            for r, act in zip(rows, acts):
                r["infeasible_p2"] = bool(SIM.filter_action(st, act))
        SIM.reset_filter_stats(clear_cache=True)
        for sid, pre, st, acts, rows in visited:
            holding = any(str(x).startswith("(holding") for x in st.fluent_atoms)
            states_meta.append({"problem": pid, "state": sid, "prefix": pre, "holding": holding, "n_actions": len(acts)})
            for r in rows:
                out_rows.append({"problem": pid, "state": sid, "holding": holding, **r})
        print("%-44s states %d  actions %d" % (pid, len(visited), sum(len(v[3]) for v in visited)), flush=True)
    os.makedirs(a.out, exist_ok=True)
    cols = ["problem", "state", "holding", "action", "verb", "infeasible_p1", "infeasible_p2", "hff"] + \
           ["q_" + l for l in SEEDS]
    with open(os.path.join(a.out, "rows_%s.csv" % a.shard), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    json.dump({"shard": a.shard, "n_problems": len(probs), "states": states_meta,
               "checkpoints": {l: p for l, (_, p) in models.items()}},
              open(os.path.join(a.out, "meta_%s.json" % a.shard), "w"), indent=1)
    print("SHARD_DONE", a.shard, len(probs), len(out_rows))


def sanity(a):
    """On a few initial states, the action gbfs_value actually pops first must be argmax Q."""
    models = load_models()
    dsl = dict(E.EVAL_DATASETS)
    res = []
    for lab_ds in ("obj10", "obj14_plan9_biased3"):
        probs = E.load_problems(E.REPO_ROOT / dsl[lab_ds], "domain.pddl")[:3]
        for prob in probs:
            for lab, (emb, _) in models.items():
                gen = xmi.XSuccessorGenerator(prob)
                s0 = gen.initial_state
                acts = sorted(gen.action_generator.generate_actions(s0), key=str)
                v = qvals(emb, s0, len(acts))
                seen = []
                orig = StateEmbedder2.get_q_values

                def hook(self, states, *aa, **kk):
                    seen.append(states[0]); return orig(self, states, *aa, **kk)
                StateEmbedder2.get_q_values = hook
                try:
                    S.gbfs_value(gen, s0, StateEmbedder2(emb.model, low_memory=True), max_expansion=2,
                                 time_limit_s=0, weight=0.0, weight2=1, filter_actions=False)
                finally:
                    StateEmbedder2.get_q_values = orig
                child = seen[1] if len(seen) > 1 else None
                if child is not None:
                    popped = next((str(x) for x in acts if S.next_real(x.apply(s0), gen) == child), None)
                else:   # the first pop reached the goal, so gbfs_value stopped before expanding it
                    popped = next((str(x) for x in acts if S.next_real(x.apply(s0), gen).is_goal()), None)
                am = str(acts[int(np.argmax(v))])
                res.append({"problem": os.path.basename(str(prob.filepath)), "seed": lab, "argmax_q": am,
                            "gbfs_first_pop": popped, "match": am == popped})
                print(res[-1], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "sanity.json"), "w"), indent=1)
    print("SANITY", sum(r["match"] for r in res), "/", len(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=None, metavar="LABEL=DIR",
                    help="checkpoints to evaluate (default: seed_1..seed_5 under $REAP_CKPT/tabletop)")
    a = ap.parse_args()
    if a.checkpoints:
        CHECKPOINTS.update(x.split("=", 1) for x in a.checkpoints)
        SEEDS[:] = list(CHECKPOINTS)
    (sanity if a.sanity else shard)(a)
