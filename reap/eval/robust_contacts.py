"""Exact contact distances for the clearance checks of the 3D-stacking verification (reap.eval.clearance,
reap.eval.stacking_verify).

MuJoCo 3.4 effects found on these scenes (reproductions in selftest()):
* margins raised at RUNTIME (after compile) are ignored by the mid-phase whenever either body of a pair has more than
  one geom -> pairs inside the runtime margin get no contact (held box 1.819 mm above the multi-level shelf top deck:
  none);
* the primitive box-box collider reports deep penetrations for SEPARATED boxes (finger pad vs shelf top: -720.3 mm,
  exact +27.4 mm) -- hidden while the mid-phase culls those pairs, exposed once it is off;
* mj_geomDistance returns a spurious 0.0 for some separated mesh/box and box/box pairs.
Rule: mid-phase OFF; every contact distance from the collision pipeline (convex CCD for meshes / cylinders) EXCEPT
box-box pairs: separated (separating-axis value > 0) -> the EXACT Euclidean distance of the two oriented boxes (convex
QP over the box coordinates, L-BFGS-B); overlapping -> minus the smallest separating-axis overlap (= penetration depth of
two convex polyhedra). mj_geomDistance and the separating-axis lower bound are not used as distances.

    install(scene)     wrap scene._pose (kinematics + collision) of a scene object, e.g. the tampanda access19_v2 Scene
    configure(m); fix_boxbox(m, d)    for own pose functions
    python -m reap.eval.robust_contacts   self-test (reproduction cases + exact vs brute force + collider check)
"""
import mujoco
import numpy as np

BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
MID = int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
CONTEXT = None
LOG = []            # box-box pairs with exact < 2.5 mm (or penetrating): collider / SAT / exact
BOGUS = []          # box-box pairs the collider reports penetrating (< -2 mm) while they are separated
STATS = {"boxbox_pairs": 0, "exact_qp": 0}


def configure(m):
    m.opt.disableflags |= MID


def _obb(m, d, g):
    return d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), m.geom_size[g][:3].copy()


def obb_sat(c1, R1, h1, c2, R2, h2):
    """Largest separating-axis gap over the 15 axes (> 0: separated, value is a lower bound on the distance;
    <= 0: overlapping, minus the value is the penetration depth)."""
    axes = [R1[:, i] for i in range(3)] + [R2[:, j] for j in range(3)]
    for i in range(3):
        for j in range(3):
            a = np.cross(R1[:, i], R2[:, j]); n = np.linalg.norm(a)
            if n > 1e-6:
                axes.append(a / n)
    t = c2 - c1
    return float(max(abs(t @ a) - np.abs(R1.T @ a) @ h1 - np.abs(R2.T @ a) @ h2 for a in axes))


def obb_exact(c1, R1, h1, c2, R2, h2):
    """Exact Euclidean distance of two oriented boxes: min |c1 + R1 u - c2 - R2 v| over |u| <= h1, |v| <= h2."""
    from scipy.optimize import minimize

    def f(x):
        r = c1 + R1 @ x[:3] - c2 - R2 @ x[3:]
        return float(r @ r), np.concatenate([2 * R1.T @ r, -2 * R2.T @ r])
    x0 = np.concatenate([np.clip(R1.T @ (c2 - c1), -h1, h1), np.clip(R2.T @ (c1 - c2), -h2, h2)])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(-a, a) for a in h1] + [(-a, a) for a in h2],
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500})
    return float(np.sqrt(max(res.fun, 0.0)))


def boxbox_distance(m, d, g1, g2):
    A, B = _obb(m, d, g1), _obb(m, d, g2)
    s = obb_sat(*A, *B)
    if s <= 0:
        return s, s
    STATS["exact_qp"] += 1
    return obb_exact(*A, *B), s


def fix_boxbox(m, d):
    n = d.ncon
    if n == 0:
        return 0
    g1 = np.array(d.contact.geom1[:n]); g2 = np.array(d.contact.geom2[:n])
    idx = np.where((m.geom_type[g1] == BOX) & (m.geom_type[g2] == BOX))[0]
    STATS["boxbox_pairs"] += len(idx)
    for k in idx:
        a, b = int(g1[k]), int(g2[k]); pipe = float(d.contact.dist[k])
        val, s = boxbox_distance(m, d, a, b)
        if pipe < -0.002 and s > 0:
            BOGUS.append({"ctx": CONTEXT, "g1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a),
                          "g2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b), "collider_mm": round(1000 * pipe, 3),
                          "exact_mm": round(1000 * val, 3)})
        if val < 0.0025:
            LOG.append({"ctx": CONTEXT, "g1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a),
                        "g2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b), "collider_mm": round(1000 * pipe, 3),
                        "sat_mm": round(1000 * s, 3), "exact_mm": round(1000 * val, 3)})
        d.contact.dist[k] = val
    return len(idx)


def install(sc):
    """Wrap sc._pose (kinematics + collision of a scene object, e.g. the tampanda access19_v2 Scene): every contact
    read afterwards is exact."""
    configure(sc.m)
    orig = sc._pose

    def _pose(*a, **k):
        r = orig(*a, **k)
        fix_boxbox(sc.m, sc.d)
        return r
    sc._pose = _pose
    return sc


def dump(path):
    import json
    json.dump({"stats": STATS, "n_logged_lt_2p5mm": len(LOG), "worst_logged": min(LOG, key=lambda r: r["exact_mm"], default=None),
               "logged": LOG[:300], "collider_penetrating_but_separated": len(BOGUS),
               "bogus_examples": sorted(BOGUS, key=lambda r: r["collider_mm"])[:20]}, open(path, "w"), indent=1)


def selftest():
    """(1) reproduction cases: runtime margin + multi-geom bodies -> the contact must be found at the right distance;
    (2) exact OBB distance vs brute force; (3) a separated box pair that the primitive collider reports penetrating."""
    ok = True
    for gap in (0.001, 0.005):
        for legs in (0, 4):
            for multi in (False, True):
                L = "".join('<geom type="box" size="0.01 0.01 0.15" pos="%f %f 0.15"/>' % (sx * 0.38, sy * 0.2)
                            for sx, sy in [(1, 1), (-1, 1), (1, -1), (-1, -1)][:legs])
                ex = '<geom type="sphere" size="0.01" pos="0 0 0.06"/>' if multi else ""
                xml = ('<mujoco><worldbody><geom name="deck" type="box" size="0.4 0.225 0.006" pos="0 0 0.3"/>%s'
                       '<body pos="0.1 0.1 %f"><freejoint/><geom name="held" type="box" size="0.03 0.03 0.045"/>%s</body>'
                       '</worldbody></mujoco>') % (L, 0.306 + 0.045 + gap, ex)
                m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); m.geom_margin[:] = 0.01   # runtime margin
                configure(m); mujoco.mj_forward(m, d); fix_boxbox(m, d)
                v = [c.dist for c in d.contact[:d.ncon] if {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                                             mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)} == {"deck", "held"}]
                good = bool(v) and abs(min(v) - gap) < 1e-6
                ok &= good
                print("repro gap %.0f mm world geoms %d body geoms %d runtime margin: %s %s" % (1000 * gap, 1 + legs, 2 if multi else 1,
                      "OK" if good else "FAIL", [round(1000 * x, 4) for x in v]))
    rng = np.random.default_rng(7); worst = 0.0; n = 0
    while n < 50:
        q = rng.normal(size=4); q /= np.linalg.norm(q); R2 = np.zeros(9); mujoco.mju_quat2Mat(R2, q); R2 = R2.reshape(3, 3)
        A = (np.zeros(3), np.eye(3), np.array([0.08, 0.05, 0.02])); B = (rng.uniform(-0.12, 0.12, 3) + [0, 0, 0.1], R2, np.array([0.03, 0.03, 0.045]))
        if obb_sat(*A, *B) <= 0:
            continue
        e = obb_exact(*A, *B)
        g = np.linspace(-1, 1, 121); U = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
        surf = U[np.any(np.abs(U) > 0.999, axis=1)] * B[2]                     # surface samples of box B
        P = B[0] + surf @ B[1].T; bf = float(np.linalg.norm(np.maximum(np.abs(P) - A[2], 0), axis=1).min())
        assert bf >= e - 1e-7, (bf, e)
        worst = max(worst, bf - e); n += 1
    ok &= worst < 0.0008
    print("exact vs brute force (50 random separated pairs, 0.75 mm surface grid): brute >= exact always, max excess %.3f mm %s"
          % (1000 * worst, "OK" if worst < 0.0008 else "FAIL"))
    xml = ('<mujoco><worldbody><geom name="top" type="box" size="0.2 0.15 0.01" pos="0 0 0.5"/>'
           '<body pos="0.19 0.0 0.465" euler="0 0.6 0.3"><freejoint/><geom name="pad" type="box" size="0.01 0.0045 0.009"/></body>'
           '</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); m.geom_margin[:] = 0.05; configure(m); mujoco.mj_forward(m, d)
    col = [c.dist for c in d.contact[:d.ncon]]; e, s = boxbox_distance(m, d, 0, 1)
    print("collider vs exact on a separated pad/top pair: collider %s mm, exact %.3f mm" % ([round(1000 * x, 2) for x in col], 1000 * e))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    selftest()
