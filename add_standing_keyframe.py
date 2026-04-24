"""Compute a standing-pose keyframe by lifting the base until the foot meshes touch the floor."""
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).parent.resolve()
MODEL = HERE / "WE2_3D.xml"
MARGIN = 0.005

CONTACT_GEOMS = {"footsole_L", "footsole_R"}


def geom_name(m, i):
    adr = m.name_geomadr[i]
    return m.names[adr:].split(b"\x00", 1)[0].decode()


def world_min_z(m, d, geom_id):
    if m.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        center_z = float(d.geom_xpos[geom_id, 2])
        size_z = float(m.geom_size[geom_id, 2])
        return center_z - size_z

    mesh_id = int(m.geom_dataid[geom_id])
    v_adr = int(m.mesh_vertadr[mesh_id])
    v_num = int(m.mesh_vertnum[mesh_id])
    verts_local = m.mesh_vert[v_adr:v_adr + v_num]

    pos = np.asarray(d.geom_xpos[geom_id])
    rot = np.asarray(d.geom_xmat[geom_id]).reshape(3, 3)
    verts_world = verts_local @ rot.T + pos
    return float(verts_world[:, 2].min())


def main():
    print(f"Loading {MODEL}")
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)

    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    mujoco.mj_forward(m, d)

    print("\nFoot geom analysis:")
    foot_min_z = np.inf
    lowest_foot = None
    for i in range(m.ngeom):
        nm = geom_name(m, i)
        if nm in CONTACT_GEOMS:
            mz = world_min_z(m, d, i)
            print(f"  {nm:<14} lowest vertex z = {mz:+.4f} m")
            if mz < foot_min_z:
                foot_min_z = mz
                lowest_foot = nm

    print(f"\nLowest foot vertex: {lowest_foot} at z = {foot_min_z:.4f}")
    z_shift = -foot_min_z + MARGIN
    print(f"Shifting base up by {z_shift:.4f} m (target: bottom at z = {MARGIN:.4f})")

    d.qpos[2] = z_shift
    mujoco.mj_forward(m, d)

    print("\nAfter shift:")
    for i in range(m.ngeom):
        nm = geom_name(m, i)
        if nm in CONTACT_GEOMS:
            print(f"  {nm:<14} lowest vertex z = {world_min_z(m, d, i):+.4f} m")

    qpos_str = " ".join(f"{v:.6f}" for v in d.qpos)
    print("\nKeyframe qpos:")
    print(f'  {qpos_str}')
    print("\nReplace the existing <key> in WE2_3D.xml with:")
    print(f'  <key name="standing" qpos="{qpos_str}"/>')


if __name__ == "__main__":
    main()
