"""Access-19 rows of Table 2, 100 s per problem (--tl).

Two deployment modes, both REAP-GBFS = eval_protocol.gbfs_value_eval (search.gbfs_value's loop:
priority -q, weight 0, feasibility checked at POP time, infeasible -> skipped, no retry cap):

  single     one goal-conditioned policy, the problem's full goal
  composed   the extraction policy (--a) on the OOI atom only, then the reload policy (--b) on the FULL goal
             from the extraction policy's final state (sub-problems written by write_problem). An ooi-only
             problem is finished by the extraction policy. The time cap is for the WHOLE problem: the reload
             policy gets what the extraction policy left.

MP/plan: the distinct (state, action) queries issued to the feasibility checker during search, per
problem, with the checker's verdict cache cleared at problem start. The state key is the
set of fluent atoms, so an identical query in the extraction and the reload phase counts once. Undeduplicated calls
are reported alongside. The independent replay of the final plan (from the ORIGINAL init, against the
ORIGINAL goal) is NOT counted and is NOT inside the timed window.

    python -m reap.eval.access19_eval --mode composed --a <extraction ckpt> --b <reload ckpt> --label <l> \
        --shard 0 --nshards 10 --out <dir>
    python -m reap.eval.access19_eval --mode single --a <ckpt> --label <l> --shard 0 --nshards 10 --out <dir>
"""
import argparse, csv, json, os, re, socket, subprocess, sys, time
# repository root = two levels up; run from there (relative paths below, e.g. checkpoint arguments, resolve against it)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_REPO)
from pathlib import Path

import torch
import reap.xmimir as xmi
from reap.eval import eval_protocol as EP
from reap.rl.av_star import sim
from reap.rl.av_star.agent import Model
from reap.rl.av_star.utils import StateEmbedder

# A19_ROOT: the dataset directory (default $REAP_DATA/access19)
ROOT = os.environ.get("A19_ROOT", (os.environ.get("REAP_DATA") or "data") + "/access19")
DOM = ROOT + "/domain.pddl"
# A19_SETS: name=dir[/glob] pairs, comma-separated; the set name goes into the "set" column, the optional glob selects
# problem files (default *.pddl), e.g. eval_18blocker_a=eval_18blocker/obj18_a_*.pddl
SETS = {"validation": ROOT + "/eval", "eval_ood": ROOT + "/eval_ood"}
if os.environ.get("A19_SETS"):
    SETS = {k: ROOT + "/" + v for k, v in (x.split("=", 1) for x in os.environ["A19_SETS"].split(","))}


def set_files(sname):
    """Problem files of a set: <dir>/*.pddl, or <dir>/<glob> when the spec ends in a glob."""
    d = SETS[sname]
    return sorted(Path(d).parent.glob(Path(d).name)) if "*" in Path(d).name else sorted(Path(d).glob("*.pddl"))
ATOM = re.compile(r"\([a-z][a-z0-9_-]*(?:\s+[^\s()]+)+\)")
OCC_BLK = re.compile(r"\(occupied \S+ (blocker_\d+)\)")
FLUENT = re.compile(r"\((occupied|empty|holding|gripper-empty)[\s)]")
STATIC = re.compile(r"\((adjacent|same-column-front)\s")


def write_problem(src_text: str, atoms, goal_atoms, name: str, dest: Path) -> Path:
    """Materialise a sub-problem: `atoms` as :init, `goal_atoms` as :goal."""
    gi, ii = src_text.find("(:goal"), src_text.find("(:init")
    statics = [l.strip() for l in src_text[ii:gi].splitlines() if STATIC.search(l)]
    fluents = [a for a in atoms if FLUENT.match(a)]
    body = "\n    ".join(sorted(set(statics)) + sorted(set(fluents)))
    head = re.sub(r"\(problem\s+[^)]*\)", f"(problem {name})", src_text[:ii], count=1)
    goal = "  (:goal (and\n    " + "\n    ".join(goal_atoms) + "\n  ))\n)\n"
    dest.write_text(f"{head}(:init\n    {body}\n  )\n{goal}")
    return dest


# ---------------- MP/plan accounting: wrap the name gbfs_value_eval calls ----------------
_ORIG_FILTER = EP.filter_action


class Acc:
    keys: set = set()
    raw = 0
    infeasible = 0
    check_s = 0.0      # wall-clock inside the feasibility checker (per-call cost diagnostic)


def _counted_filter(state, action):
    Acc.raw += 1
    Acc.keys.add((frozenset(str(x) for x in state.atoms() if FLUENT.match(str(x))), str(action)))
    t = time.perf_counter()
    r = _ORIG_FILTER(state, action)
    Acc.check_s += time.perf_counter() - t
    Acc.infeasible += bool(r)
    return r


EP.filter_action = _counted_filter


def reset_accounting(state):
    Acc.keys, Acc.raw, Acc.infeasible, Acc.check_s = set(), 0, 0, 0.0
    ck = sim._get_checker(state)
    if hasattr(ck, "init"):
        ck.init()
    if hasattr(ck, "_cache"):
        ck._cache.clear()
    sim.reset_filter_stats()


def load(path, dev):
    m = Model.load_from_folder(path); m.to(dev); m.eval()
    return StateEmbedder(m, low_memory=True)


def verify(gen0, plan):
    """Replay from the original init through the (unwrapped) checker; must end at the original goal."""
    s = gen0.initial_state
    for i, a in enumerate(plan):
        acts = {str(x): x for x in gen0.action_generator.generate_actions(s)}
        act = acts.get(str(a))
        if act is None:
            return "step%d:inapplicable" % i
        if _ORIG_FILTER(s, act):
            return "step%d:infeasible" % i
        s = act.apply(s)
    return "ok" if s.is_goal() else "final_not_goal"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("single", "composed"), required=True)
    ap.add_argument("--a", required=True, help="checkpoint: the single policy, or the extraction policy (composed)")
    ap.add_argument("--b", help="checkpoint of the reload policy (composed)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--sets", default="validation,eval_ood",
                    help="comma list of <set>[:<filename prefix>], e.g. validation,eval_ood:obj19")
    ap.add_argument("--only", help="file of '<set>/<problem>' lines: evaluate just these")
    ap.add_argument("--tl", type=float, default=100.0)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6, help="torch threads (the reported runs: 6)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-plans", default=None,
                    help="dir: for each solved problem write <problem>.plan.json with planA, planB, the "
                         "phase boundary index (= len(planA)) and the independent verification verdict")
    a = ap.parse_args()
    from reap.domains import checker_info
    checker = checker_info.describe_data(DOM, [str(Path(d).parent) if "*" in Path(d).name else d for d in SETS.values()])
    print(checker_info.log_line(checker, "access19_eval"), flush=True)
    torch.set_num_threads(a.threads)
    dev = "cpu"
    embA = load(a.a, dev)
    embB = load(a.b, dev) if a.mode == "composed" else None

    pending = []
    only = {l.strip() for l in open(a.only) if l.strip()} if a.only else None
    for spec in a.sets.split(","):
        sname, pre = (spec.split(":", 1) + [""])[:2]
        for pf in set_files(sname):
            if pf.name.startswith("domain") or (pre and not pf.name.startswith(pre)):
                continue
            if only is not None and "%s/%s" % (sname, pf.name) not in only:
                continue
            pending.append((sname, pf))
    pending = pending[a.shard::a.nshards]
    os.makedirs(a.out, exist_ok=True)
    tmp = Path(a.out) / ("tmp_%s_%d" % (a.label, a.shard)); tmp.mkdir(exist_ok=True)
    rows = []
    for sname, pf in pending:
        src = pf.read_text()
        gb = src.split("(:goal", 1)[1]
        gb = gb[gb.find("(and") + 4:].split("))", 1)[0] if "(and" in gb else gb
        goal_atoms = ATOM.findall(gb)
        occ_goals = [g for g in goal_atoms if g.lstrip().startswith("(occupied")]
        task = "ooi-only" if len(occ_goals) <= 1 else "_return"
        nblk = len(set(OCC_BLK.findall(src.split("(:init", 1)[1].split("(:goal", 1)[0])))
        _, prob0 = xmi.parse(DOM, str(pf)); gen0 = xmi.XSuccessorGenerator(prob0)
        reset_accounting(gen0.initial_state)
        row = {"set": sname, "problem": pf.name, "checker": checker["label"], "blockers": nblk, "tier": "18" if nblk >= 18 else "<=14",
               "task": task, "label": a.label, "mode": a.mode}
        t0 = time.time()
        expA = expB = 0; planA, planB = [], []; reasonB = ""
        if a.mode == "single":
            expA, planA, ok, reason, _ = EP.gbfs_value_eval(gen0, gen0.initial_state, embA,
                                                          max_expansion=a.budget, time_limit_s=a.tl)
        else:
            ooi_goal = [g for g in goal_atoms if g.rstrip().endswith("ooi)")] or goal_atoms[:1]
            pa = write_problem(src, [str(x) for x in gen0.initial_state.atoms()], ooi_goal, "pA", tmp / "a.pddl")
            _, probA = xmi.parse(DOM, str(pa)); genA = xmi.XSuccessorGenerator(probA)
            expA, planA, okA, reason, _ = EP.gbfs_value_eval(genA, genA.initial_state, embA,
                                                           max_expansion=a.budget, time_limit_s=a.tl)
            ok = False
            if okA:
                s = genA.initial_state
                for x in planA:
                    s = x.apply(s)
                pb = write_problem(src, [str(x) for x in s.atoms()], goal_atoms, "pB", tmp / "b.pddl")
                _, probB = xmi.parse(DOM, str(pb)); genB = xmi.XSuccessorGenerator(probB)
                if genB.initial_state.is_goal():
                    ok, reason = True, "solved_by_A"
                else:
                    left = a.tl - (time.time() - t0)
                    if left <= 0.5:
                        reason = "B_no_time_left"
                    else:
                        expB, planB, ok, reasonB, _ = EP.gbfs_value_eval(
                            genB, genB.initial_state, embB, max_expansion=max(1, a.budget - expA),
                            time_limit_s=left)
                        reason = "B:" + reasonB
            else:
                reason = "A:" + reason
        wall = time.time() - t0
        plan = list(planA) + list(planB)
        ver = verify(gen0, plan) if ok else ""
        solved = ok and ver == "ok"
        if a.dump_plans and solved:
            os.makedirs(a.dump_plans, exist_ok=True)
            json.dump({"problem": pf.name, "set": sname, "label": a.label, "mode": a.mode,
                       "ckpt_a": a.a, "ckpt_b": a.b, "time_limit_s": a.tl, "budget": a.budget,
                       "planA": [str(x) for x in planA], "planB": [str(x) for x in planB],
                       "boundary": len(planA), "verify": ver, "expanded_A": expA, "expanded_B": expB,
                       "mp_calls_dedup": len(Acc.keys), "mp_calls_raw": Acc.raw, "time_s": round(wall, 3)},
                      open(os.path.join(a.dump_plans, pf.stem + ".plan.json"), "w"), indent=1)
        st = sim.get_filter_stats()
        row.update({"solved": int(solved), "expanded": expA + expB, "expanded_A": expA, "expanded_B": expB,
                    "steps": len(plan) if ok else "", "mp_calls_dedup": len(Acc.keys), "mp_calls_raw": Acc.raw,
                    "infeasible_raw": Acc.infeasible, "filter_timeouts": st.get("timeouts", 0),
                    "check_time_s": round(Acc.check_s, 3),
                    "s_per_call": round(Acc.check_s / Acc.raw, 4) if Acc.raw else "",
                    "time_s": round(wall, 3), "termination_reason": reason, "verify": ver,
                    "plan": ";".join(str(x) for x in plan) if solved else ""})
        rows.append(row)
        print("%-14s %-30s %-5s %-8s solved=%d exp=%d steps=%s mp=%d raw=%d t=%.1f %s %s" % (
            sname, pf.name, row["tier"], task, solved, expA + expB, row["steps"], len(Acc.keys), Acc.raw,
            wall, reason, ver), flush=True)
    base = os.path.join(a.out, "%s__shard%02d" % (a.label, a.shard))
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    rev = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    json.dump({"label": a.label, "mode": a.mode, "ckpt_a": a.a, "ckpt_b": a.b, "time_limit_s": a.tl,
               "time_limit_scope": "whole problem (composed: shared by the two policies)", "budget": a.budget,
               "sets": a.sets, "only": a.only, "root": ROOT,
               "access19_no_side_by_side": os.environ.get("ACCESS19_NO_SIDE_BY_SIDE", "(default: rule ON)"),
               "a_goal": "ooi",
               "tampanda_file": __import__("tampanda").__file__, "shard": a.shard, "nshards": a.nshards, "n": len(rows), "domain": DOM,
               "search": "eval_protocol.gbfs_value_eval score=q (search.gbfs_value loop, filter at pop)",
               "mp_definition": "MP/plan: distinct (fluent-state, action) queries per problem, checker cache cleared",
               "filter_timeout_s": sim.FILTER_TIMEOUT_S, "code_rev": rev("."),
               "tampanda_rev": rev(os.path.dirname(os.path.dirname(__import__("tampanda").__file__))),
               "host": socket.gethostname(), "device": dev, "checker": checker},
              open(base + ".json", "w"), indent=1)
    print("ACCESS19_EVAL_DONE", a.label, a.shard, len(rows))


if __name__ == "__main__":
    main()
