"""Continuous verification of 3D-stacking plans under the GRIP-AWARE check (MLB_CHECKER=honest_grip).

The verifier records the committed motion of a plan, plays it back in an independent world and checks it:
1. RECORD: the plan runs ONCE, in order, on one world with tampanda multilevel_blocks/honest_grip.py's executor
   (GripExecutor.run_action back to back, no per-action restore; the initial state restored once). Recorded: the
   committed motion only (every dense sample of every accepted stage, finger sweeps, the release / grasp samples), each
   with the block it carries.
2. PLAY BACK in a second, independent world (the base_panda_v2 scene of tampanda multilevel_blocks.honest, initial
   state restored ONCE):
   * arm + fingers follow the recorded samples; a finger-width change is swept at the previous arm pose (<= 0.5 mm);
   * a block becomes held where the record says so (grasp): its in-hand pose is measured in the playback world at that
     instant and kept rigid until the release; blocks not held never move;
   * at a release the block falls straight down (gravity) from its in-air pose to the support height of the symbolic
     state (its x, y and orientation are kept, so any misplacement shows up in check (a)).
Also: (w) WALK: every action is also checked PER ACTION exactly as training / search see it (reap.domains
MultilevelBlocksFeasibilityChecker with MLB_CHECKER=honest_grip, xmimir state of the grip-aware domain, canonical
restore); (a+) at the start of every action that holds a block, the continuous in-hand pose must equal the canonical
pose of the symbolic grip (grip-centre / grip-end) up to the box symmetries (<= 0.01 mm / 0.01 deg) -- this is what makes
the per-action verdicts chain; (c+) the arm keeps >= 2 mm (exact) at EVERY sample ("c:arm-below-2mm"); (r) the release
height of every put (block above its support when the fingers open) is recorded.
3. CHECKS
   (a) boundary: after every action every block not held matches the symbolic state (centre <= 2 mm, orientation
       <= 2 deg modulo the box symmetries); the held block is the symbolically held one; at the start of every action
       the certifier's in-hand pose equals the playback's carried in-hand pose and the pose the previous action ended
       with (<= 2 mm / 2 deg); at a grasp both finger pads touch the block (<= 1 mm);
   (b) motion: between consecutive certifier samples the arm moves <= 5 mrad per joint (no teleport), no block that is
       not held moves (the released block's straight-down snap is the release drop, reported separately), the held
       block keeps its in-hand pose (no slip); the end-of-action state equals the last sample;
   (c) exact clearance (robust_contacts rule: mid-phase off; every box-box pair by the exact oriented-box distance --
       enumerated here independently of the collider, the collider's pairs cross-checked against
       clearance.Clearance on sampled configurations) at every playback sample incl. finger sweeps and drops:
       arm (link1-7, hand, fingers) vs everything except gripper vs the action's block (clearance.Clearance
       pairs and exclusions); the held block vs every other block / table; the falling block along its drop.
       Fail: any value < 0 (penetration; blocks sliding face to face are 0, tolerance 0.01 mm).

    python -m reap.eval.stacking_verify list --out OUT --spec train=<dataset dir>/train --spec idf=<plans dir> ...
        (<dataset dir>: data/<env>/<sub> with <problem>.pddl + .plan; <plans dir>: *.json with problem, plan)
    python -m reap.eval.stacking_verify run --out OUT --shard k/n
    python -m reap.eval.stacking_verify summarize --out OUT
"""
from __future__ import annotations

import argparse, collections, glob, itertools, json, math, os, re, sys, tempfile, time, traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disabled")
RG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
for p in (RG, RG + "/src"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import mujoco

OUT = None                              # --out
DD = (os.environ.get("REAP_DATA") or os.path.join(RG, "data")) + "/"   # $REAP_DATA (reap/paths.py)
INHAND_TOL_M, INHAND_TOL_DEG = 1e-5, 0.01   # (a+) continuous in-hand pose vs the canonical pose of the symbolic grip
ARM_MIN = 0.002                              # (c+) the arm must keep >= 2 mm at every sample
POS_TOL, ROT_TOL = 0.002, 2.0          # (a) boundary / in-hand tolerances (m, deg)
PAD_TOL = 0.001                        # (a) grasp: both finger pads within 1 mm of the block ...
PAD_PEN = 0.0005                       # ... and the kinematic closure overlapping it by <= 0.5 mm (off-centre block)
DQ_TOL = 0.005 + 1e-6                  # (b) max joint step between samples (DQ of tampanda multilevel_blocks.honest)
STATIC_TOL = 1e-6                      # (b) a block not held "moves" if its pose changes by more (m / rad)
SLIP_TOL = 1e-6                        # (b) held block in-hand pose change between samples (m / rad)
PEN_TOL = 1e-4                         # (c) block vs block / table: penetration if < -0.1 mm (30 mm blocks on the 30 mm
                                       #     grid slide face to face at 0; IK residual tilts of ~0.02 deg reach ~0.02 mm)
FING_STEP = 0.0005                     # finger sweep step (m)
DROP_STEP = 0.0005                     # release drop sampling (m)
RANGE = 0.05
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
NAN = float("nan")


# ----------------------------------------------------------------------------------------------------------- geometry
def _sperm():
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, p[i]] = s[i]
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


SPERM = _sperm()


def sym_angle(Ra, Rb, h):
    """Smallest rotation (deg) taking box orientation Ra to Rb modulo the symmetries of a box with half sizes h."""
    D = Ra.T @ Rb
    best = 180.0
    for S in SPERM:
        if np.allclose(np.abs(S) @ h, h, atol=1e-9):
            c = (np.trace(D @ S.T) - 1) / 2
            best = min(best, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return best


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def q2m(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def m2q(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


def sat_many(c1, R1, h1, C2, R2, H2):
    """robust_contacts.obb_sat of box 1 vs each of N boxes, vectorised: > 0 separated (lower bound of the distance),
    <= 0 overlapping (minus the penetration depth)."""
    N = len(C2)
    if N == 0:
        return np.zeros(0)
    A1 = R1.T
    A2 = np.transpose(R2, (0, 2, 1))
    cr = np.cross(A1[None, :, None, :], A2[:, None, :, :]).reshape(N, 9, 3)
    nrm = np.linalg.norm(cr, axis=2)
    ok = nrm > 1e-6
    cr = cr / np.where(ok, nrm, 1.0)[..., None]
    ax = np.concatenate([np.broadcast_to(A1, (N, 3, 3)), A2, cr], axis=1)
    t = C2 - c1
    proj = np.abs(np.einsum("nka,na->nk", ax, t))
    r1 = np.abs(np.einsum("nka,ab->nkb", ax, R1)) @ h1
    r2 = np.einsum("nkb,nb->nk", np.abs(np.einsum("nka,nab->nkb", ax, R2)), H2)
    gap = proj - r1 - r2
    valid = np.concatenate([np.ones((N, 6), bool), ok], axis=1)
    return np.where(valid, gap, -np.inf).max(1)


def sat_pairs(C1, R1, H1, C2, R2, H2):
    """sat_many for N independent pairs (box i of set 1 vs box i of set 2)."""
    N = len(C1)
    if N == 0:
        return np.zeros(0)
    A1 = np.transpose(R1, (0, 2, 1)); A2 = np.transpose(R2, (0, 2, 1))
    cr = np.cross(A1[:, :, None, :], A2[:, None, :, :]).reshape(N, 9, 3)
    nrm = np.linalg.norm(cr, axis=2)
    ok = nrm > 1e-6
    cr = cr / np.where(ok, nrm, 1.0)[..., None]
    ax = np.concatenate([A1, A2, cr], axis=1)
    t = C2 - C1
    proj = np.abs(np.einsum("nka,na->nk", ax, t))
    r1 = np.einsum("nkb,nb->nk", np.abs(np.einsum("nka,nab->nkb", ax, R1)), H1)
    r2 = np.einsum("nkb,nb->nk", np.abs(np.einsum("nka,nab->nkb", ax, R2)), H2)
    gap = proj - r1 - r2
    valid = np.concatenate([np.ones((N, 6), bool), ok], axis=1)
    return np.where(valid, gap, -np.inf).max(1)


def quat_angle(Qa, Qb):
    """rotation angle (rad) between unit quaternions (rows), numerically stable for tiny angles."""
    Qa = Qa / np.linalg.norm(Qa, axis=1, keepdims=True); Qb = Qb / np.linalg.norm(Qb, axis=1, keepdims=True)
    w = np.sum(Qa * Qb, axis=1)
    v = (Qa[:, :1] * Qb[:, 1:] - Qb[:, :1] * Qa[:, 1:] - np.cross(Qa[:, 1:], Qb[:, 1:]))
    return 2 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w))


# ----------------------------------------------------------------------------------------------------------- items
def level(label, name, prob):
    if label.endswith("eval_ood"):
        return "L5"
    m = re.match(r"(L\d)_", name)
    if m:
        return m.group(1)
    for src in ("stacking/train", "stacking/validation"):
        f = os.path.join(DD, src, os.path.basename(prob) + ".plan")
        if os.path.exists(f):
            mm = re.search(r"; difficulty: (\d+)", open(f).read())
            if mm:
                return "L" + mm.group(1)
    return "?"


def items_of(src):
    """(problem file, plan, name) triples: a dataset dir (<problem>.pddl + .plan; name = problem) or a plans dir (*.json
    with problem, plan[, goal_required]; name = the json file name)."""
    out = []
    if glob.glob(os.path.join(src, "*.json")) and not glob.glob(os.path.join(src, "*.pddl")):
        for jf in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = json.load(open(jf))
            out.append((r["problem"], r.get("plan") or None, os.path.basename(jf)[:-5], r.get("goal_required", True)))
    else:
        for f in sorted(glob.glob(os.path.join(src, "*.pddl"))):
            plan = [l.strip() for l in open(f + ".plan") if l.strip().startswith("(")] if os.path.exists(f + ".plan") else None
            out.append((f, plan, os.path.basename(f)[:-5], True))
    return out


def domain_of(prob_file):
    """domain.pddl of a problem: <problem dir>/../domain.pddl (data/<env>/<sub>/<problem>.pddl)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(prob_file))), "domain.pddl")


def _jobs_from_spec(a):
    jobs, counts = [], {}
    for spec in a.spec:
        lab, src = spec.split("=", 1)
        src = src if os.path.isabs(src) else os.path.join(DD, src)
        its = items_of(src)
        counts[lab] = [len(its), sum(x[1] is not None for x in its)]
        for prob, plan, name, goal in its:
            jobs.append({"label": lab, "problem": prob, "name": name, "plan": plan, "level": level(lab, os.path.basename(prob)[:-5], prob),
                         "goal_required": goal})
    jobs.sort(key=lambda j: (-len(j["plan"] or []), j["label"], j["name"]))
    return jobs, counts


def cmd_list(a):
    jobs, counts = _jobs_from_spec(a)
    os.makedirs(OUT, exist_ok=True)
    json.dump(jobs, open(os.path.join(OUT, "jobs.json"), "w"))
    print(json.dumps(counts), len(jobs), "jobs")


def _jobs(a):
    """--spec given: the job list is built in memory (deterministic order, so the shards agree); else jobs.json."""
    if a.spec:
        return _jobs_from_spec(a)[0]
    return json.load(open(os.path.join(OUT, "jobs.json")))


# ----------------------------------------------------------------------------------------------------------- recording
def _att(env):
    a = env._attached
    if a is None:
        return None
    return {"body": a["body_name"], "rel_pos": np.asarray(a["rel_pos"], float).copy(),
            "rel_mat": np.asarray(a["rel_mat"], float).copy()}


def record(prob_file, plan):
    """The continuity certificate with committed-motion recording: honest_grip GripExecutor.run_action back to back on
    ONE world (initial state restored once). Returns (env, cfg, prob, q_init, acts, recs, atts, ends)."""
    import reap.xmimir as xmi
    from tampanda.symbolic.domains.multilevel_blocks import honest as H
    from tampanda.symbolic.domains.multilevel_blocks import honest_grip as GR
    from reap.domains.multilevel_blocks import MultilevelBlocksFeasibilityChecker as C
    _, prob = xmi.parse(domain_of(prob_file), prob_file)
    cfg0 = C._infer_config(prob)
    env, ws, cfg = GR.build_grip_env(Path(tempfile.mkdtemp(prefix="vg_cert_")), config=cfg0)
    ex = GR.make_grip_executor(env, ws, cfg)
    ex.restore(C._state_to_pddl_dict_grip(xmi.XSuccessorGenerator(prob).initial_state))
    q_init = env.data.qpos.copy()
    acts, recs, atts, ends = [], [], [], []
    for i, s in enumerate(plan):
        name, *args = s.strip("()").split()
        a0 = _att(env)
        ex.record = rec = []
        t = time.perf_counter()
        try:
            ok = bool(ex.run_action(name, tuple(args)))
            err = None if ok else "no motion: %s" % ";".join((ex.last_stats or {}).get("fail", [])[:4])
        except H.HonestViolation as e:
            ok, err = False, "honest: %s" % e
        except Exception as e:
            ok, err = False, "%s: %s" % (type(e).__name__, e)
        dt = time.perf_counter() - t
        ex.record = None
        st = {k: v for k, v in (ex.last_stats or {}).items() if k in ("candidate", "release_mm", "release_width_mm", "line_top_mm")}
        norm = [(np.asarray(x[0], float), int(x[1])) for x in rec]
        acts.append({"i": i, "action": s, "type": name.lower(), "obj": args[0], "ok": ok, "error": err, "s": round(dt, 3),
                     "lib": st, "samples": len(norm)})
        recs.append(norm); atts.append((a0, _att(env))); ends.append(env.data.qpos.copy())
        if not ok:
            break
    return env, cfg, prob, q_init, acts, recs, atts, ends


_CHECKER = None


def walk(prob_file, plan, goal_required=True):
    """Per-action verdicts as training / search see them: reap.domains MultilevelBlocksFeasibilityChecker (MLB_CHECKER=
    honest_grip) on the xmimir states of the grip-aware domain along the plan. Returns (rows, grip-aware state dicts)."""
    global _CHECKER
    import reap.xmimir as xmi
    from reap.rl.av_star.search import next_real
    from reap.domains.multilevel_blocks import MultilevelBlocksFeasibilityChecker as C
    assert os.environ.get("MLB_CHECKER") == "honest_grip", "run with MLB_CHECKER=honest_grip"
    _, prob = xmi.parse(domain_of(prob_file), prob_file)
    gen = xmi.XSuccessorGenerator(prob); s = gen.initial_state
    if _CHECKER is None:
        _CHECKER = C()
    rows, dicts = [], [C._state_to_pddl_dict_grip(s)]
    for i, txt in enumerate(plan):
        name, *args = txt.strip("()").split()
        cand = [x for x in gen.action_generator.generate_actions(s)
                if x.name.lower() == name.lower() and [o.name.lower() for o in x.objects] == [y.lower() for y in args]]
        if not cand:
            rows.append({"i": i, "action": txt, "type": name.lower(), "applicable": False, "ok": False}); break
        act = cand[0]                      # disjunctive put-long preconditions ground twice (equivalent actions)
        first = not _CHECKER._initialized   # the first call builds the env (excluded from the timing summary)
        t = time.perf_counter()
        _CHECKER._cache.clear()            # time every check (no verdict cache across plans)
        inf = _CHECKER.check(s, act)
        dt = time.perf_counter() - t
        lst = dict(_CHECKER._executor.last_stats or {})
        rows.append({"i": i, "action": txt, "type": name.lower(), "applicable": True, "ok": not inf, "s": round(dt, 4), "init": first,
                     "release_mm": lst.get("release_mm"), "fail": None if not inf else (lst.get("fail") or [])[:6]})
        s = next_real(act.apply(s), gen)
        dicts.append(C._state_to_pddl_dict_grip(s))
        if inf:
            break
    return rows, dicts, (bool(s.is_goal()) or not goal_required) and len(rows) == len(plan) and all(r["ok"] for r in rows)


# ----------------------------------------------------------------------------------------------------------- measuring
class Meter:
    """Exact clearance on a twin of the tampanda multilevel_blocks.honest scene (clearance.Clearance pairs +
    robust_contacts distances)."""

    def __init__(self, cfg, q_home, obj_names):
        from tampanda.symbolic.domains.multilevel_blocks import honest as H
        from reap.eval import clearance as RV
        from reap.eval import robust_contacts as RC
        self.RC, self.RV = RC, RV
        tw, _, _ = H.build_honest_env(Path(tempfile.mkdtemp(prefix="vc_twin_")), config=cfg)
        self.tw = tw
        self.cl = RV.Clearance(tw, q_home)            # the fixed tool: margins, mid-phase off, structural pairs
        m = self.m = tw.model; self.d = tw.data
        ng = m.ngeom
        self.gbody = np.asarray(m.geom_bodyid)
        self.gtype = np.asarray(m.geom_type)
        self.isarm = np.zeros(ng, bool); self.isgrip = np.zeros(ng, bool); self.link = {}
        for g, link in self.cl.arm.items():
            self.isarm[g] = True; self.link[g] = link; self.isgrip[g] = link in RV.GRIPPER
        self.struct = np.zeros((ng, ng), bool)
        for (link, sg) in self.cl.structural:
            for g, l2 in self.cl.arm.items():
                if l2 == link:
                    self.struct[g, sg] = True
        col = (np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0)
        self.armbox = np.array([g for g in self.cl.arm if self.gtype[g] == BOX])
        self.scenebox = np.array([g for g in range(ng) if col[g] and not self.isarm[g] and self.gtype[g] == BOX])
        PA, PS = np.meshgrid(self.armbox, self.scenebox, indexing="ij")
        keep = ~self.struct[PA.ravel(), PS.ravel()]
        self.pa, self.ps = PA.ravel()[keep], PS.ravel()[keep]       # arm box x scene box pairs (structural removed)
        self.pa_grip = self.isgrip[self.pa]
        self.rad = np.linalg.norm(np.asarray(m.geom_size)[:, :3], axis=1)
        self.gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
        self.bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in obj_names}
        self.pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_finger_pad", "right_finger_pad")]
        self.n_qp = 0
        self.selfcheck = {"n": 0, "max_abs_diff_mm": 0.0, "enumeration_lower": 0}

    def _boxes(self, gs):
        d, m = self.d, self.m
        return d.geom_xpos[gs].copy(), d.geom_xmat[gs].reshape(-1, 3, 3).copy(), m.geom_size[gs][:, :3].copy()

    def _dist(self, g, others, gate):
        """exact distances of box g to the boxes `others` (gate: QP only where the SAT lower bound < gate)."""
        d, m = self.d, self.m
        c1, R1, h1 = d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), m.geom_size[g][:3].copy()
        C2, R2, H2 = self._boxes(others)
        s = sat_many(c1, R1, h1, C2, R2, H2)
        out = s.copy()
        for k in np.where((s > 0) & (s < gate))[0]:
            out[k] = self.RC.obb_exact(c1, R1, h1, C2[k], R2[k], H2[k]); self.n_qp += 1
        return out

    def set(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)

    def _pairs_dist(self, ga, gb, gate):
        """exact distances for the box pairs (ga[i], gb[i]); pairs farther than RANGE (bounding spheres) get RANGE."""
        d, m = self.d, self.m
        out = np.full(len(ga), RANGE)
        cen = np.linalg.norm(d.geom_xpos[ga] - d.geom_xpos[gb], axis=1) - self.rad[ga] - self.rad[gb]
        near = np.where(cen < RANGE)[0]
        if len(near) == 0:
            return out
        a, b = ga[near], gb[near]
        C1, R1, H1 = d.geom_xpos[a], d.geom_xmat[a].reshape(-1, 3, 3), m.geom_size[a][:, :3]
        C2, R2, H2 = d.geom_xpos[b], d.geom_xmat[b].reshape(-1, 3, 3), m.geom_size[b][:, :3]
        s = sat_pairs(C1, R1, H1, C2, R2, H2)
        v = np.minimum(s, RANGE)
        for k in np.where((s > 0) & (s < gate))[0]:
            v[k] = self.RC.obb_exact(C1[k], R1[k], H1[k], C2[k], R2[k], H2[k]); self.n_qp += 1
        out[near] = v
        return out

    def arm(self, q, obj_bid, gate):
        """(min, pair) of the arm vs scene with the Clearance exclusions; box-box pairs enumerated + exact."""
        m, d = self.m, self.d
        self.set(q); mujoco.mj_collision(m, d)
        best, pair = RANGE, None
        n = d.ncon
        if n:
            g1 = np.asarray(d.contact.geom1[:n]); g2 = np.asarray(d.contact.geom2[:n]); dist = np.asarray(d.contact.dist[:n])
            a1, a2 = self.isarm[g1], self.isarm[g2]
            ag = np.where(a1, g1, g2); sg = np.where(a1, g2, g1)
            sel = (a1 != a2) & ~((self.gtype[ag] == BOX) & (self.gtype[sg] == BOX))
            sel &= ~(self.isgrip[ag] & (self.gbody[sg] == obj_bid)) & ~self.struct[ag, sg]
            if sel.any():
                k = np.where(sel)[0][int(np.argmin(dist[sel]))]
                if dist[k] < best:
                    best, pair = float(dist[k]), (self.link[int(ag[k])], self.gname(int(sg[k])), self.bname(int(sg[k])))
        sel = ~(self.pa_grip & (self.gbody[self.ps] == obj_bid))
        ga, gb = self.pa[sel], self.ps[sel]
        v = self._pairs_dist(ga, gb, gate)
        if len(v):
            k = int(np.argmin(v))
            if v[k] < best:
                best, pair = float(v[k]), (self.link[int(ga[k])], self.gname(int(gb[k])), self.bname(int(gb[k])))
        return best, pair

    def block(self, q, gb, gate, kin=True):
        """(min, pair) of the block geom gb vs every other scene box (blocks, tables) and the floor."""
        if kin:
            self.set(q)
        others = self.scenebox[self.gbody[self.scenebox] != self.gbody[gb]]
        v = self._pairs_dist(np.full(len(others), gb), others, gate)
        k = int(np.argmin(v))
        best, pair = float(v[k]), (self.bname(gb), self.gname(int(others[k])), self.bname(int(others[k])))
        c, R, h = self.d.geom_xpos[gb], self.d.geom_xmat[gb].reshape(3, 3), self.m.geom_size[gb][:3]
        zf = float(c[2] - np.abs(R[2, :]) @ h)
        if zf < best:
            best, pair = zf, (self.bname(gb), "floor", "world")
        return min(best, RANGE), pair

    def pads_to(self, q, gb):
        self.set(q)
        return [float(self._dist(p, np.array([gb]), np.inf)[0]) for p in self.pads]

    def check_tool(self, q, obj_name, mine):
        """cross-check against the fixed tool (clearance.Clearance: mj_forward + robust_contacts.fix_boxbox)."""
        ref, _ = self.cl(q, obj_name)
        self.selfcheck["n"] += 1
        if mine < ref - 1e-7:
            self.selfcheck["enumeration_lower"] += 1          # a box pair the collider did not report
        elif abs(mine - ref) > 1e-7:
            self.selfcheck["max_abs_diff_mm"] = max(self.selfcheck["max_abs_diff_mm"], 1000 * abs(mine - ref))
        return ref


# ----------------------------------------------------------------------------------------------------------- verify
def verify(job, keep_full=False):
    from tampanda.symbolic.domains.multilevel_blocks import honest as H
    from tampanda.symbolic.domains.multilevel_blocks.feasibility import apply_action_effects
    from tampanda.symbolic.domains.multilevel_blocks.state import restore_state
    from tampanda.symbolic.domains.multilevel_blocks.executor import _HOME_NEUTRAL_Q
    from reap.domains.multilevel_blocks import MultilevelBlocksFeasibilityChecker as C
    import reap.xmimir as xmi
    from tampanda.symbolic.domains.multilevel_blocks import honest_grip as GR
    plan = job["plan"]
    t0 = time.time()
    wrows, wdicts, wok = walk(job["problem"], plan, job.get("goal_required", True))
    t_walk = time.time() - t0
    t0 = time.time()
    env, cfg, prob, q_init, acts, recs, atts, ends = record(job["problem"], plan)
    t_rec = time.time() - t0
    res = {"label": job["label"], "problem": job["problem"], "name": job["name"], "level": job["level"],
           "plan_len": len(plan), "plan": plan, "walk": wrows, "walk_ok": wok, "t_walk_s": round(t_walk, 1),
           "actions": acts, "t_record_s": round(t_rec, 1)}
    res["certified"] = len(acts) == len(plan) and all(x["ok"] for x in acts)
    if not res["certified"]:
        res["reproduced"] = False
        bad = acts[-1]
        res["not_reproducible"] = {"step": bad["i"], "action": bad["action"], "error": bad["error"]}
        return res, None
    res["reproduced"] = True
    m = env.model
    # blocks (free bodies with one box geom aligned with the body)
    blocks = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        b = int(m.jnt_bodyid[j]); nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        if nm.startswith(("cube_", "oblong_", "long_")):      # a block: one box geom aligned with its body
            assert len(gs) == 1 and m.geom_type[gs[0]] == BOX, nm
            assert np.allclose(m.geom_pos[gs[0]], 0) and np.allclose(m.geom_quat[gs[0]], [1, 0, 0, 0]), nm
            blocks.append({"name": nm, "bid": b, "qa": int(m.jnt_qposadr[j]), "gid": gs[0], "h": m.geom_size[gs[0]][:3].copy()})
        else:                                                  # a free fixture (the tables): must never move
            blocks.append({"name": nm, "bid": b, "qa": int(m.jnt_qposadr[j]), "gid": None, "h": None})
    B = {x["bid"]: x for x in blocks}
    byname = {x["name"]: x for x in blocks if x["gid"] is not None}
    QA = np.array([[x["qa"] + k for k in range(7)] for x in blocks])
    site = int(env._ee_site_id)
    dK = mujoco.MjData(m); dK.qpos[:] = q_init

    def ee(q9):
        dK.qpos[:9] = q9[:9]; mujoco.mj_kinematics(m, dK)
        T = np.eye(4); T[:3, :3] = dK.site_xmat[site].reshape(3, 3); T[:3, 3] = dK.site_xpos[site]; return T

    def bT(q, b):
        x = B[b]; T = np.eye(4); T[:3, :3] = q2m(q[x["qa"] + 3:x["qa"] + 7]); T[:3, 3] = q[x["qa"]:x["qa"] + 3]; return T

    def setT(q, b, T):
        x = B[b]; q[x["qa"]:x["qa"] + 3] = T[:3, 3]; q[x["qa"] + 3:x["qa"] + 7] = m2q(T[:3, :3])

    def Tdiff(Ta, Tb, h=None):
        dp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
        da = sym_angle(Ta[:3, :3], Tb[:3, :3], h) if h is not None else rot_angle(Ta[:3, :3], Tb[:3, :3])
        return dp, da

    # symbolic states and their canonical block poses (restore_state into a third world)
    envS, wsS, cfgS = H.build_honest_env(Path(tempfile.mkdtemp(prefix="vc_sym_")), config=cfg)
    st = C._state_to_pddl_dict_grip(xmi.XSuccessorGenerator(prob).initial_state)
    states = [st]
    for s in plan:   # xmimir folds case: the symbolic state's cell names are lower case
        st = apply_action_effects(st, tuple(x.lower() for x in s.strip("()").split())); states.append(st)

    def canon(state):
        restore_state(envS, wsS, cfgS, state, home_qpos=_HOME_NEUTRAL_Q, on_held="park")
        mujoco.mj_kinematics(envS.model, envS.data)
        held = [k[1] for k in state if str(k[0]).startswith("held")]
        return {x["bid"]: (envS.data.xpos[x["bid"]].copy(), envS.data.xmat[x["bid"]].reshape(3, 3).copy()) for x in blocks}, held

    goal_ok = all(states[-1].get(k, False) for k in [("in", b.lower(), c.lower()) for b, c in re.findall(r"\(in\s+(\S+)\s+(\S+)\)", open(job["problem"]).read().split("(:goal", 1)[1])])
    res["symbolic_goal_reached"] = bool(goal_ok)
    # playback world
    envP, wsP, cfgP = H.build_honest_env(Path(tempfile.mkdtemp(prefix="vc_play_")), config=cfg)
    restore_state(envP, wsP, cfgP, states[0], home_qpos=_HOME_NEUTRAL_Q, on_held="park")
    qp = envP.data.qpos.copy()
    res["init_world_match"] = bool(np.allclose(qp, q_init, atol=1e-9))
    meter = Meter(cfg, q_init, list(byname))
    fails = []                                         # first occurrence per (step, check, block) + count
    fidx = {}
    def fail(i, check, detail):
        key = (i, check, detail.get("block") if isinstance(detail, dict) else None)
        if key in fidx:
            fails[fidx[key]]["count"] += 1; return
        fidx[key] = len(fails)
        fails.append({"step": i, "action": plan[i] if 0 <= i < len(plan) else None, "check": check, "detail": detail, "count": 1})
    stats = {"samples": 0, "sweep_samples": 0, "drop_samples": 0}
    arm = {"min": RANGE, "at": None, "neg": 0, "lt2": 0}
    held = {"min": RANGE, "at": None, "neg": 0, "touch": 0, "lt1": 0, "lt2": 0}
    fall = {"min": RANGE, "at": None, "neg": 0}
    drops, grasps, boundary, inhand = [], [], [], []
    dev_cert = {"max_mm": 0.0, "at": None}
    seq_q, seq_a, seq_k = [], [], []                  # rendering / storage: playback configuration, action, kind

    def measure(i, obj, hb, kind):
        stats["samples"] += 1
        if kind == 1:
            stats["sweep_samples"] += 1
        gate = max(0.0025, arm["min"])
        v, pr = meter.arm(qp, meter.bid.get(obj, -1), gate)
        if stats["samples"] <= 12 or stats["samples"] % 997 == 0:     # cross-check vs the fixed tool (exact, ungated)
            vx, _ = meter.arm(qp, meter.bid.get(obj, -1), np.inf)
            meter.check_tool(qp, obj, vx)
        if v < arm["min"]:
            arm["min"], arm["at"] = v, {"step": i, "pair": pr}
        arm["neg"] += v < 0; arm["lt2"] += v < 0.002
        if hb >= 0:
            gate = max(0.0025, held["min"])
            v, pr = meter.block(qp, B[hb]["gid"], gate, kin=False)
            if v < held["min"]:
                held["min"], held["at"] = v, {"step": i, "pair": pr}
            held["neg"] += v < -PEN_TOL; held["touch"] += abs(v) <= PEN_TOL; held["lt1"] += v < 0.001; held["lt2"] += v < 0.002
            if v < -PEN_TOL:
                fail(i, "c:held-block-penetration", {"mm": round(1000 * v, 3), "pair": pr})
        if keep_full:
            seq_q.append(qp.copy()); seq_a.append(i); seq_k.append(kind)
        else:
            seq_q.append(qp[:9].astype(np.float32)); seq_a.append(i); seq_k.append(kind)

    hp, relp = -1, None                                # playback: held body, in-hand transform
    prev = q_init.copy(); hprev = -1
    last_released = None
    for i, a in enumerate(acts):
        obj = a["obj"]
        cposes, sheld = canon(states[i + 1])
        a0, a1 = atts[i]
        # (a+) the continuous in-hand pose == the canonical pose of the symbolic grip (what the per-action check restores)
        if a0 is not None and i < len(wdicts):
            fl_, hb_, grip_ = GR.held_and_grip(wdicts[i])
            if hb_ is None or hb_ != a0["body"].lower():
                fail(i, "a:in-hand-vs-symbolic-held", {"symbolic": hb_, "world": a0["body"]})
            else:
                cp_, cR_ = GR.canonical_grip(grip_, GR.shape_of(hb_), cfg.cube_size)
                Tk = np.eye(4); Tk[:3, :3] = cR_; Tk[:3, 3] = cp_
                Tw = np.eye(4); Tw[:3, :3] = a0["rel_mat"]; Tw[:3, 3] = a0["rel_pos"]
                dp_, da_ = Tdiff(Tk, Tw, byname[a0["body"]]["h"])
                if dp_ > INHAND_TOL_M or da_ > INHAND_TOL_DEG:
                    fail(i, "a:in-hand-not-canonical", {"grip": grip_, "dp_mm": round(1000 * dp_, 4), "da_deg": round(da_, 4)})
        # (a) in-hand pose at the start of the action: certifier vs the previous action's end vs the playback
        if a0 is not None:
            prev_end = atts[i - 1][1] if i > 0 else None
            if prev_end is None or prev_end["body"] != a0["body"] or np.abs(prev_end["rel_pos"] - a0["rel_pos"]).max() > 1e-9 or np.abs(prev_end["rel_mat"] - a0["rel_mat"]).max() > 1e-9:
                fail(i, "a:in-hand-changed-between-actions", {})
            b0 = byname[a0["body"]]["bid"]
            Tc = np.eye(4); Tc[:3, :3] = a0["rel_mat"]; Tc[:3, 3] = a0["rel_pos"]
            if hp != b0 or relp is None:
                fail(i, "a:held-block-mismatch", {"certifier": a0["body"], "playback": B[hp]["name"] if hp >= 0 else None})
            else:
                dp, da = Tdiff(Tc, relp, B[b0]["h"])
                inhand.append({"step": i, "dp_mm": round(1000 * dp, 3), "da_deg": round(da, 3)})
                if dp > POS_TOL or da > ROT_TOL:
                    fail(i, "a:in-hand-pose", {"dp_mm": round(1000 * dp, 3), "da_deg": round(da, 3)})
        for item in recs[i]:
            if item is None:
                continue
            q, h = item
            # ---------------- (b) on the certifier record
            darm = float(np.max(np.abs(q[:7] - prev[:7])))
            if darm > DQ_TOL:
                fail(i, "b:arm-teleport", {"rad": round(darm, 4)})
            Pp, Pc = prev[QA], q[QA]
            dpos = np.linalg.norm(Pc[:, :3] - Pp[:, :3], axis=1)
            dang = quat_angle(Pp[:, 3:], Pc[:, 3:])
            for k in np.where((dpos > STATIC_TOL) | (dang > STATIC_TOL))[0]:
                b = blocks[k]["bid"]
                if b in (h, hprev):
                    continue
                if b == last_released:                 # the release snap (library put): straight down only
                    lat = float(np.linalg.norm(Pc[k, :2] - Pp[k, :2]))
                    if lat > 1e-4 or dang[k] > 1e-4:
                        fail(i, "b:release-snap-not-vertical", {"block": blocks[k]["name"], "lat_mm": round(1000 * lat, 3), "deg": round(math.degrees(dang[k]), 3)})
                    last_released = None
                    continue
                fail(i, "b:non-held-block-moved", {"block": blocks[k]["name"], "mm": round(1000 * float(dpos[k]), 3), "deg": round(math.degrees(float(dang[k])), 3)})
            if h >= 0 and h == hprev:
                r0 = np.linalg.inv(ee(prev)) @ bT(prev, h); r1 = np.linalg.inv(ee(q)) @ bT(q, h)
                dp, da = Tdiff(r0, r1)
                if dp > SLIP_TOL or math.radians(da) > 1e-5:
                    fail(i, "b:in-hand-slip", {"mm": round(1000 * dp, 4), "deg": round(da, 4)})
            # ---------------- playback
            arm_prev = qp[:9].copy()
            if hprev >= 0 and h != hprev:              # release at the previous arm pose: the block drops
                if hp != hprev:
                    fail(i, "a:held-block-mismatch", {"release": B[hprev]["name"]})
                b = hprev
                Tair = bT(qp, b)
                cp, cR = cposes[b]
                hb_ = B[b]["h"]
                z_support = cp[2] - np.abs(cR[2, :]) @ hb_                     # canonical bottom = support height
                z_rest = z_support + np.abs(Tair[2, :3]) @ hb_                 # lowest point of the in-air pose on it
                lat = float(np.linalg.norm(Tair[:2, 3] - cp[:2])); drop = float(Tair[2, 3] - z_rest)
                da = sym_angle(cR, Tair[:3, :3], B[b]["h"])
                n = max(1, int(math.ceil(abs(drop) / DROP_STEP)))
                fmin = RANGE
                for k in range(1, n + 1):
                    T = Tair.copy(); T[2, 3] = Tair[2, 3] - drop * k / n
                    setT(qp, b, T)
                    v, pr = meter.block(qp, B[b]["gid"], max(0.0025, fall["min"]))
                    stats["drop_samples"] += 1
                    fmin = min(fmin, v)
                    if v < fall["min"]:
                        fall["min"], fall["at"] = v, {"step": i, "pair": pr}
                    if v < -PEN_TOL:
                        fall["neg"] += 1
                        fail(i, "c:drop-penetration", {"mm": round(1000 * v, 3), "pair": pr})
                    if keep_full:
                        seq_q.append(qp.copy()); seq_a.append(i); seq_k.append(2)
                drops.append({"step": i, "type": a["type"], "block": B[b]["name"], "drop_mm": round(1000 * drop, 3),
                              "lateral_mm": round(1000 * lat, 3), "rot_deg": round(da, 3), "fall_min_mm": round(1000 * fmin, 3)})
                if drop < -PEN_TOL:
                    fail(i, "c:released-below-support", {"mm": round(1000 * drop, 3)})
                hp, relp = -1, None
                last_released = b
            if abs(q[7] - qp[7]) > 1e-9 or abs(q[8] - qp[8]) > 1e-9:     # finger sweep at the previous arm pose
                f0, f1 = qp[7:9].copy(), q[7:9].copy()
                n = max(1, int(math.ceil(float(np.max(np.abs(f1 - f0))) / FING_STEP)))
                for k in range(1, n + 1):
                    qp[7:9] = f0 + (f1 - f0) * k / n
                    if k < n:
                        measure(i, obj, hp, 1)
            if h >= 0 and hprev < 0:                   # grasp at the previous arm pose (fingers closed)
                b = h
                pads = meter.pads_to(qp, B[b]["gid"])
                grasps.append({"step": i, "block": B[b]["name"], "pads_mm": [round(1000 * x, 3) for x in pads]})
                if max(pads) > PAD_TOL or min(pads) < -PAD_PEN:
                    fail(i, "a:grasp-pads-not-on-block", {"pads_mm": [round(1000 * x, 3) for x in pads]})
                relp = np.linalg.inv(ee(qp)) @ bT(qp, b)
                hp = b
                last_released = None
            qp[:9] = q[:9]
            if hp >= 0:
                setT(qp, hp, ee(qp) @ relp)
            measure(i, obj, hp, 0)
            prev, hprev = q, h
        last_released = None
        # certifier world vs playback world at the action end (information: the certifier leaves top-down puts 2 mm
        # above their support and snaps library puts down)
        Dp = np.linalg.norm(ends[i][QA][:, :3] - qp[QA][:, :3], axis=1)
        k = int(np.argmax(Dp))
        if Dp[k] > dev_cert["max_mm"] / 1000:
            dev_cert["max_mm"], dev_cert["at"] = float(1000 * Dp[k]), {"step": i, "block": blocks[k]["name"]}
        # end of the action: the certifier's end state == the last sample
        qe = ends[i]
        if float(np.max(np.abs(qe[:9] - prev[:9]))) > 1e-9:
            fail(i, "b:end-state-differs-from-last-sample", {"rad": float(np.max(np.abs(qe[:9] - prev[:9])))})
        if a1 is not None and (hprev < 0 or B[hprev]["name"] != a1["body"]):
            fail(i, "b:end-attachment-differs", {"end": a1["body"]})
        # (a) boundary vs the symbolic state
        worst = (0.0, 0.0, None)
        for x in blocks:
            b = x["bid"]
            if b == hp:
                continue
            Tp = bT(qp, b); cp, cR = cposes[b]
            dp = float(np.linalg.norm(Tp[:3, 3] - cp))
            da = sym_angle(cR, Tp[:3, :3], x["h"]) if x["h"] is not None else rot_angle(cR, Tp[:3, :3])
            if dp > worst[0] or da > worst[1]:
                worst = (max(dp, worst[0]), max(da, worst[1]), x["name"])
            if dp > POS_TOL or da > ROT_TOL:
                fail(i, "a:boundary-pose", {"block": x["name"], "dp_mm": round(1000 * dp, 3), "da_deg": round(da, 3)})
        hb_name = B[hp]["name"] if hp >= 0 else None
        if sorted(sheld) != ([hb_name] if hb_name else []):
            fail(i, "a:held-identity", {"symbolic": sheld, "playback": hb_name})
        boundary.append({"step": i, "max_dp_mm": round(1000 * worst[0], 3), "max_da_deg": round(worst[1], 3)})
    chk = {
        "a": not any(f["check"].startswith("a:") for f in fails),
        "b": not any(f["check"].startswith("b:") for f in fails),
        "c": arm["neg"] == 0 and held["neg"] == 0 and fall["neg"] == 0 and not any(f["check"].startswith("c:") for f in fails),
    }
    if arm["neg"]:
        fail(arm["at"]["step"], "c:arm-penetration", {"n": arm["neg"], "min_mm": round(1000 * arm["min"], 3), "pair": arm["at"]["pair"]})
        chk["c"] = False
    if arm["lt2"]:
        fail(arm["at"]["step"], "c:arm-below-2mm", {"n": arm["lt2"], "min_mm": round(1000 * arm["min"], 3), "pair": arm["at"]["pair"]})
        chk["c"] = False
    chk["walk"] = wok
    res.update({
        "checks": chk, "pass": all(chk.values()), "failures": fails, "n_failures": len(fails),
        "release_mm": [(a["i"], a["type"], a["lib"].get("release_mm")) for a in acts if a["type"].startswith("put")],
        "samples": stats, "arm": {"min_mm": round(1000 * arm["min"], 3), "at": arm["at"], "n_lt0": arm["neg"], "n_lt2mm": arm["lt2"]},
        "held": {"min_mm": round(1000 * held["min"], 3), "at": held["at"], "n_lt0": held["neg"], "n_touch": held["touch"],
                 "n_lt1mm": held["lt1"], "n_lt2mm": held["lt2"]},
        "fall": {"min_mm": round(1000 * fall["min"], 3), "at": fall["at"], "n_lt0": fall["neg"]},
        "drops": drops, "grasps": grasps, "boundary": boundary, "inhand_start": inhand,
        "cert_vs_playback_max_mm": round(dev_cert["max_mm"], 3), "cert_vs_playback_at": dev_cert["at"],
        "tool_crosscheck": meter.selfcheck, "exact_qp_calls": meter.n_qp, "wall_s": round(time.time() - t0, 1),
    })
    seq = {"q": np.array(seq_q), "act": np.array(seq_a, np.int16), "kind": np.array(seq_k, np.int8)}
    return res, seq


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def cmd_run(a):
    jobs = _jobs(a)
    k, n = (int(x) for x in a.shard.split("/"))
    for job in jobs[k::n]:
        if a.only and job["name"] not in a.only.split(","):
            continue
        od = os.path.join(OUT, "per_plan", job["label"]); os.makedirs(od, exist_ok=True)
        dst = os.path.join(od, job["name"] + ".json")
        if os.path.exists(dst) and not a.force:
            continue
        if job["plan"] is None:
            json.dump({"label": job["label"], "problem": job["problem"], "name": job["name"], "level": job["level"],
                       "no_plan": True}, open(dst, "w"), indent=1)
            continue
        t0 = time.time()
        try:
            res, seq = verify(job)
            if seq is not None and job["level"] in ("L4", "L5") and a.motion:
                sd = os.path.join(OUT, "motion", job["label"]); os.makedirs(sd, exist_ok=True)
                np.savez_compressed(os.path.join(sd, job["name"] + ".npz"), **seq)
        except Exception as e:
            res = {"label": job["label"], "problem": job["problem"], "name": job["name"], "level": job["level"],
                   "plan_len": len(job["plan"]), "error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc(),
                   "reproduced": False}
        res["wall_s"] = round(time.time() - t0, 1)
        json.dump(res, open(dst, "w"), indent=1, default=_json_default)
        print(job["label"], job["name"], job["level"], len(job["plan"]), "walk", res.get("walk_ok"),
              "pass" if res.get("pass") else "FAIL", res.get("error", ""), res.get("not_reproducible", ""), res["wall_s"], flush=True)


def cmd_summarize(a):
    import statistics as S
    jobs = _jobs(a)
    if a.spec:
        json.dump(jobs, open(os.path.join(OUT, "jobs.json"), "w"))
    rows = collections.OrderedDict()
    fails, errs = [], []
    agg = collections.defaultdict(lambda: {"arm_min": RANGE, "lt2": 0, "held_min": RANGE, "fall_min": RANGE})
    by_check = collections.Counter()
    wtype = collections.defaultdict(collections.Counter); wtime = collections.defaultdict(list)
    rel = collections.defaultdict(collections.Counter)
    xc = {"n": 0, "max_abs_diff_mm": 0.0, "enumeration_lower": 0}
    for j in jobs:
        f = os.path.join(OUT, "per_plan", j["label"], j["name"] + ".json")
        key = (j["label"], j["level"])
        r_ = rows.setdefault(key, collections.Counter())
        if not os.path.exists(f):
            r_["missing"] += 1; continue
        r = json.load(open(f))
        if r.get("no_plan"):
            r_["no_plan"] += 1; continue
        r_["plans"] += 1
        for w in r.get("walk", []):
            wtype[w["type"]]["checked"] += 1; wtype[w["type"]]["rejected"] += not w["ok"]
            if w.get("s") is not None and not w.get("init"):
                wtime[w["type"]].append(w["s"])
        r_["walk_ok"] += bool(r.get("walk_ok"))
        if "error" in r:
            errs.append((j["label"], j["name"], r["error"])); r_["error"] += 1; continue
        if not r.get("reproduced"):
            r_["not_continuous"] += 1
            fails.append((j["label"], j["level"], j["name"], r.get("not_reproducible", {}).get("step"),
                          r.get("not_reproducible", {}).get("action"), "continuous run failed", r.get("not_reproducible", {}).get("error")))
            continue
        r_["pass"] += bool(r["pass"])
        for c in "abc":
            r_["fail_" + c] += not r["checks"][c]
        g = agg[key]
        g["arm_min"] = min(g["arm_min"], r["arm"]["min_mm"] / 1000); g["lt2"] += r["arm"]["n_lt2mm"]
        g["held_min"] = min(g["held_min"], r["held"]["min_mm"] / 1000); g["fall_min"] = min(g["fall_min"], r["fall"]["min_mm"] / 1000)
        for (_, typ, mm) in r.get("release_mm", []):
            rel[typ][mm] += 1
        for x in r["failures"]:
            by_check[(x["check"], x["action"].strip("()").split()[0] if x.get("action") else None)] += 1
        if not r["pass"]:
            seen = set()
            for x in r["failures"]:
                kk = (x["step"], x["check"])
                if kk in seen:
                    continue
                seen.add(kk); fails.append((j["label"], j["level"], j["name"], x["step"], x["action"], x["check"], json.dumps(x["detail"])))
        for k2 in ("n", "enumeration_lower"):
            xc[k2] += r["tool_crosscheck"][k2]
        xc["max_abs_diff_mm"] = max(xc["max_abs_diff_mm"], r["tool_crosscheck"]["max_abs_diff_mm"])
    L = ["# Grip-aware continuous verification -- summary (stacking_verify.py summarize)", "",
         "| set | level | plans | per-action walk accepted | continuous run completes | pass all (a,b,c, walk) | fail (a) | fail (b) | fail (c) | arm exact min (mm) | arm samples < 2 mm | held-block min (mm) | drop-path min (mm) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    labs = list(dict.fromkeys(k[0] for k in rows))
    for (lab, lv) in sorted(rows, key=lambda k: (labs.index(k[0]), k[1])):
        c = rows[(lab, lv)]; g = agg.get((lab, lv))
        L.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %s | %s | %s | %s |" % (
            lab, lv, c["plans"], c["walk_ok"], c["plans"] - c["not_continuous"] - c["error"], c["pass"], c["fail_a"], c["fail_b"], c["fail_c"],
            "%.3f" % (1000 * g["arm_min"]) if g else "-", g["lt2"] if g else "-", "%.3f" % (1000 * g["held_min"]) if g else "-",
            "%.3f" % (1000 * g["fall_min"]) if g else "-"))
        for k in c:
            tot[k] += c[k]
    L += ["", "Totals: %s" % dict(tot), "Errors: %d %s" % (len(errs), errs[:3]),
          "Fixed-tool cross-check of the arm clearance (clearance.Clearance on %d sampled configurations): max |diff| %.6f mm, enumeration lower than the collider in %d" % (xc["n"], xc["max_abs_diff_mm"], xc["enumeration_lower"]), "",
          "## Per-action check (walk) by action type", "", "| action type | checked | rejected | time mean / median / p90 / max (s) |", "|---|---|---|---|"]
    for typ in sorted(wtype):
        v = sorted(wtime[typ])
        L.append("| %s | %d | %d | %s |" % (typ, wtype[typ]["checked"], wtype[typ]["rejected"],
                 "%.3f / %.3f / %.3f / %.3f" % (S.mean(v), S.median(v), v[int(0.9 * len(v))], v[-1]) if v else "-"))
    L += ["", "## Release height of the puts (block above its support when the fingers open)", "",
          "| put type | puts | release height (mm): count |", "|---|---|---|"]
    for typ in sorted(rel):
        L.append("| %s | %d | %s |" % (typ, sum(rel[typ].values()), ", ".join("%s: %d" % (k, v) for k, v in sorted(rel[typ].items(), key=lambda kv: kv[0] or 0))))
    L += ["", "## Failure counts by check and action type", "", "| check | action type | occurrences |", "|---|---|---|"]
    L += ["| %s | %s | %d |" % (k[0], k[1], v) for k, v in by_check.most_common()]
    L += ["", "## Failing plans: first occurrence of each (step, check) (%d rows)" % len(fails), "",
          "| set | level | problem | step | action | check | detail |", "|---|---|---|---|---|---|---|"]
    L += ["| %s | %s | %s | %s | %s | %s | %s |" % x for x in fails]
    open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:80]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("list", "run", "summarize"))
    ap.add_argument("--out", required=True); ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--only", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--motion", action="store_true", help="store L4/L5 playback motion (.npz)")
    a = ap.parse_args()
    OUT = a.out
    {"list": cmd_list, "run": cmd_run, "summarize": cmd_summarize}[a.mode](a)
