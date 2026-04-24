"""Print a sanity-check report for a WE2 MJCF model.

Usage:
    python inspect_model.py            # WE2_3D.xml
    python inspect_model.py human      # WE2Human_3D.xml
    python inspect_model.py <path.xml>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent.resolve()
MODELS = {"exo": HERE / "WE2_3D.xml", "human": HERE / "WE2Human_3D.xml"}

JNT_TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
ACT_TRN  = {0: "joint", 1: "jointinparent", 2: "tendon", 3: "site", 4: "body"}


def resolve(arg):
    if arg is None:
        return MODELS["exo"]
    if arg in MODELS:
        return MODELS[arg]
    p = Path(arg)
    return p if p.is_absolute() else HERE / p


def name(model, adr):
    return model.names[adr:].split(b"\x00", 1)[0].decode() or "<unnamed>"


def section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = resolve(arg)
    print(f"Loading {path}")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    section("Model summary")
    print(f"  bodies:       {m.nbody}")
    print(f"  joints:       {m.njnt}    dofs: {m.nv}    qpos: {m.nq}")
    print(f"  actuators:    {m.nu}")
    print(f"  geoms:        {m.ngeom}    meshes: {m.nmesh}")
    print(f"  sensors:      {m.nsensor}")
    print(f"  contact pairs (predefined): {m.npair}")
    print(f"  exclude pairs: {m.nexclude}")
    print(f"  equality constraints: {m.neq}")
    print(f"  total mass:   {m.body_mass.sum():.3f} kg")
    print(f"  gravity:      {m.opt.gravity}")
    print(f"  timestep:     {m.opt.timestep} s")

    section("Joints")
    print(f"{'idx':>3}  {'name':<14} {'type':<6} {'limited':<7} "
          f"{'range (rad)':<22} {'range (deg)':<22}")
    for i in range(m.njnt):
        nm = name(m, m.name_jntadr[i])
        jt = JNT_TYPE.get(int(m.jnt_type[i]), str(m.jnt_type[i]))
        lim = bool(m.jnt_limited[i])
        rng = m.jnt_range[i]
        rng_d = np.degrees(rng) if lim else rng
        rng_s = f"[{rng[0]:+.3f}, {rng[1]:+.3f}]" if lim else "      —      "
        rng_ds = f"[{rng_d[0]:+7.1f}, {rng_d[1]:+7.1f}]" if lim else "      —      "
        print(f"{i:>3}  {nm:<14} {jt:<6} {str(lim):<7} {rng_s:<22} {rng_ds:<22}")

    section("Actuators")
    if m.nu == 0:
        print("  (none — model has NO actuators; control panel will be empty)")
    else:
        print(f"{'idx':>3}  {'name':<16} {'trn':<6} {'target':<14} "
              f"{'ctrl range':<22} {'force range':<22} gear")
        for i in range(m.nu):
            nm = name(m, m.name_actuatoradr[i])
            trn = ACT_TRN.get(int(m.actuator_trntype[i]), "?")
            tgt_id = int(m.actuator_trnid[i, 0])
            tgt = name(m, m.name_jntadr[tgt_id]) if trn == "joint" and tgt_id >= 0 else "—"
            cr = m.actuator_ctrlrange[i]
            fr = m.actuator_forcerange[i]
            cl = bool(m.actuator_ctrllimited[i])
            fl = bool(m.actuator_forcelimited[i])
            cr_s = f"[{cr[0]:+.2f}, {cr[1]:+.2f}]" if cl else "unlimited"
            fr_s = f"[{fr[0]:+.2f}, {fr[1]:+.2f}]" if fl else "unlimited"
            print(f"{i:>3}  {nm:<16} {trn:<6} {tgt:<14} {cr_s:<22} {fr_s:<22} {m.actuator_gear[i, 0]:.2f}")

    section("Bodies (mass & inertia)")
    print(f"{'idx':>3}  {'name':<18} {'mass (kg)':>10}  {'diag inertia (kg·m²)':<28}  warnings")
    warn_count = 0
    for i in range(m.nbody):
        nm = name(m, m.name_bodyadr[i])
        mass = float(m.body_mass[i])
        inertia = m.body_inertia[i]
        warn = []
        if i > 0:  # skip world
            if mass <= 0:
                warn.append("zero mass")
                warn_count += 1
            if np.any(inertia <= 0) and mass > 0:
                warn.append("non-positive inertia")
                warn_count += 1
        inertia_s = f"[{inertia[0]:.3e}, {inertia[1]:.3e}, {inertia[2]:.3e}]"
        warn_s = ", ".join(warn) if warn else ""
        print(f"{i:>3}  {nm:<18} {mass:>10.4f}  {inertia_s:<28}  {warn_s}")
    print(f"\n  inertia warnings: {warn_count}")

    section("Contacts (predefined pairs / exclusions)")
    if m.npair == 0:
        print("  (no <pair> entries — collisions rely on contype/conaffinity defaults)")
    else:
        for i in range(m.npair):
            g1 = name(m, m.name_geomadr[m.pair_geom1[i]])
            g2 = name(m, m.name_geomadr[m.pair_geom2[i]])
            print(f"  pair {i}: {g1}  <->  {g2}")
    if m.nexclude:
        print(f"  exclude pairs: {m.nexclude}")

    section("Equality constraints")
    if m.neq == 0:
        print("  (none)")
    else:
        for i in range(m.neq):
            nm = name(m, m.name_eqadr[i])
            print(f"  eq {i}: {nm}  type={m.eq_type[i]}  active={bool(m.eq_active0[i])}")

    section("Initial state (after mj_forward)")
    print(f"  qpos: {np.array2string(d.qpos, precision=3, suppress_small=True)}")
    print(f"  com z (root body 1): {d.xipos[1, 2]:.4f} m")
    if m.nkey:
        print(f"  keyframes defined: {m.nkey}")
    else:
        print("  no <keyframe> defined → model spawns at qpos0 (often midair)")

    print("\nDone.")


if __name__ == "__main__":
    main()
