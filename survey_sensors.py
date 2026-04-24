"""List the sensors in the WE2 model with their type, target, and initial reading."""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent.resolve()
MODELS = {"exo": HERE / "WE2_3D.xml"}

OBJ_TYPE = {
    mujoco.mjtObj.mjOBJ_UNKNOWN: "unknown",
    mujoco.mjtObj.mjOBJ_BODY: "body",
    mujoco.mjtObj.mjOBJ_XBODY: "xbody",
    mujoco.mjtObj.mjOBJ_JOINT: "joint",
    mujoco.mjtObj.mjOBJ_GEOM: "geom",
    mujoco.mjtObj.mjOBJ_SITE: "site",
    mujoco.mjtObj.mjOBJ_ACTUATOR: "actuator",
    mujoco.mjtObj.mjOBJ_TENDON: "tendon",
    mujoco.mjtObj.mjOBJ_CAMERA: "camera",
}

CATEGORY = {
    "jointpos": "kinematic",
    "jointvel": "kinematic",
    "tendonpos": "kinematic",
    "tendonvel": "kinematic",
    "actuatorpos": "kinematic",
    "actuatorvel": "kinematic",
    "actuatorfrc": "force",
    "jointactfrc": "force",
    "framepos": "kinematic",
    "framequat": "kinematic",
    "framexaxis": "kinematic",
    "frameyaxis": "kinematic",
    "framezaxis": "kinematic",
    "framelinvel": "kinematic",
    "frameangvel": "kinematic",
    "framelinacc": "inertial",
    "frameangacc": "inertial",
    "accelerometer": "inertial",
    "gyro": "inertial",
    "magnetometer": "inertial",
    "velocimeter": "inertial",
    "force": "force",
    "torque": "force",
    "touch": "force",
    "subtreecom": "kinematic",
    "subtreelinvel": "kinematic",
    "subtreeangmom": "kinematic",
}


def sensor_type_name(t):
    try:
        return mujoco.mjtSensor(t).name.replace("mjSENS_", "").lower()
    except ValueError:
        return f"type_{t}"


def obj_name(model, objtype, objid):
    if objid < 0:
        return "—"
    adr_map = {
        mujoco.mjtObj.mjOBJ_BODY: model.name_bodyadr,
        mujoco.mjtObj.mjOBJ_XBODY: model.name_bodyadr,
        mujoco.mjtObj.mjOBJ_JOINT: model.name_jntadr,
        mujoco.mjtObj.mjOBJ_GEOM: model.name_geomadr,
        mujoco.mjtObj.mjOBJ_SITE: model.name_siteadr,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.name_actuatoradr,
        mujoco.mjtObj.mjOBJ_TENDON: model.name_tendonadr,
    }
    if objtype not in adr_map:
        return f"<{OBJ_TYPE.get(objtype, '?')}:{objid}>"
    adr = adr_map[objtype][objid]
    return model.names[adr:].split(b"\x00", 1)[0].decode() or "<unnamed>"


def resolve(arg):
    if arg is None:
        return MODELS["exo"]
    if arg in MODELS:
        return MODELS[arg]
    p = Path(arg)
    return p if p.is_absolute() else HERE / p


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = resolve(arg)
    print(f"Loading {path}")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"\nTotal sensors: {m.nsensor}")
    if m.nsensor == 0:
        print("(none)")
        return

    rows = []
    for i in range(m.nsensor):
        nm_adr = m.name_sensoradr[i]
        nm = m.names[nm_adr:].split(b"\x00", 1)[0].decode() or f"<sensor_{i}>"
        stype = sensor_type_name(int(m.sensor_type[i]))
        objt = int(m.sensor_objtype[i])
        objid = int(m.sensor_objid[i])
        tgt = obj_name(m, objt, objid)
        dim = int(m.sensor_dim[i])
        adr = int(m.sensor_adr[i])
        vals = d.sensordata[adr:adr + dim]
        cat = CATEGORY.get(stype, "other")
        rows.append((i, nm, stype, cat, tgt, dim, vals))

    print("\n--- Count by sensor type ---")
    type_counts = {}
    for _, _, st, _, _, _, _ in rows:
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {st:<18} {c}")

    print("\n--- Count by category ---")
    cat_counts = {}
    for _, _, _, cat, _, _, _ in rows:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, c in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<12} {c}")

    for cat in ["kinematic", "inertial", "force", "other"]:
        sub = [r for r in rows if r[3] == cat]
        if not sub:
            continue
        print(f"\n--- {cat.upper()} sensors ({len(sub)}) ---")
        print(f"{'idx':>3}  {'name':<22} {'type':<16} {'target':<16} {'dim':>3}  initial value")
        for i, nm, st, _, tgt, dim, vals in sub:
            val_s = np.array2string(vals, precision=3, suppress_small=True, max_line_width=80)
            print(f"{i:>3}  {nm:<22} {st:<16} {tgt:<16} {dim:>3}  {val_s}")


if __name__ == "__main__":
    main()
