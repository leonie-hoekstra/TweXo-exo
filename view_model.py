"""Run the WE2 exoskeleton model in the MuJoCo viewer with CTC and an optional gait FSM."""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from gait_fsm import GaitFSM

HERE = Path(__file__).parent.resolve()
MODELS = {"exo": HERE / "WE2_3D.xml"}
DEFAULT_BWS_PCT = 0.5
DEFAULT_CTRL = "ctc"
DEFAULT_KP = 400.0
DEFAULT_KD = 30.0
DEFAULT_KP_PD = 150.0
DEFAULT_KD_PD = 8.0
DEFAULT_GAIT_ON = False
FOOT_SENSORS_L = ("LHI", "LTI", "LHO", "LTO")
FOOT_SENSORS_R = ("RHI", "RTI", "RHO", "RTO")


def parse_args():
    bws_pct = DEFAULT_BWS_PCT
    ctrl = DEFAULT_CTRL
    gait_on = DEFAULT_GAIT_ON
    kp = None
    kd = None
    positional = []
    for a in sys.argv[1:]:
        if a.startswith("--bws="):
            bws_pct = float(a.split("=", 1)[1])
        elif a.startswith("--ctrl="):
            ctrl = a.split("=", 1)[1].lower()
            if ctrl not in ("ctc", "pd", "none"):
                sys.exit(f"--ctrl must be ctc|pd|none, got {ctrl}")
        elif a.startswith("--pd="):
            ctrl = "pd" if bool(int(a.split("=", 1)[1])) else "none"
        elif a.startswith("--gait="):
            gait_on = bool(int(a.split("=", 1)[1]))
        elif a.startswith("--kp="):
            kp = float(a.split("=", 1)[1])
        elif a.startswith("--kd="):
            kd = float(a.split("=", 1)[1])
        else:
            positional.append(a)
    if kp is None:
        kp = DEFAULT_KP if ctrl == "ctc" else DEFAULT_KP_PD
    if kd is None:
        kd = DEFAULT_KD if ctrl == "ctc" else DEFAULT_KD_PD
    return positional[0] if positional else None, bws_pct, ctrl, gait_on, kp, kd


def resolve_model(arg):
    if arg is None:
        return MODELS["exo"]
    if arg in MODELS:
        return MODELS[arg]
    p = Path(arg)
    return p if p.is_absolute() else HERE / p


def find_site(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return sid if sid >= 0 else None


def sensor_addrs(model, names):
    out = []
    for nm in names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, nm)
        if sid >= 0:
            out.append((int(model.sensor_adr[sid]), int(model.sensor_dim[sid])))
    return out


def foot_force_mag(data, addrs):
    total = 0.0
    for adr, dim in addrs:
        v = data.sensordata[adr:adr + dim]
        total += float(np.linalg.norm(v))
    return total


def main():
    arg, bws_pct, ctrl, gait_on, kp, kd = parse_args()
    model_path = resolve_model(arg)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    print(f"Loading {model_path.name}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"  bodies:    {model.nbody}")
    print(f"  joints:    {model.njnt}")
    print(f"  dofs:      {model.nv}")
    print(f"  actuators: {model.nu}")
    print(f"  geoms:     {model.ngeom}")
    print(f"  keyframes: {model.nkey}")

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("  initial state: keyframe 0")

    bws_site_id = find_site(model, "bws_attach")
    bws_force = 0.0
    bws_body_id = -1
    if bws_site_id is not None and bws_pct > 0:
        bws_body_id = int(model.site_bodyid[bws_site_id])
        total_mass = float(model.body_mass.sum())
        bws_force = bws_pct * total_mass * abs(float(model.opt.gravity[2]))
        print(f"  BWS harness: {bws_pct * 100:.0f}% unload "
              f"({bws_force:.1f} N upward at site 'bws_attach')")
    elif bws_pct == 0:
        print("  BWS harness: disabled (--bws=0)")
    else:
        print("  BWS harness: site 'bws_attach' not found, disabled")

    act_qpos_adr = np.array(
        [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)],
        dtype=np.int32,
    )
    act_dof_adr = np.array(
        [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)],
        dtype=np.int32,
    )
    target_qpos = data.qpos.copy()
    qdd_ref = np.zeros(model.nu)

    M_full = np.zeros((model.nv, model.nv))
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
    has_ctrl_limits = model.actuator_ctrllimited.astype(bool)

    if ctrl == "ctc" and model.nu > 0:
        print(f"  CTC controller: kp={kp}, kd={kd} on {model.nu} actuators")
    elif ctrl == "pd" and model.nu > 0:
        print(f"  PD controller: kp={kp}, kd={kd} on {model.nu} actuators")
    else:
        print("  Controller: disabled")

    q_ref_hold = target_qpos[act_qpos_adr].copy()
    fsm = GaitFSM() if gait_on else None
    foot_L_addrs = sensor_addrs(model, FOOT_SENSORS_L) if gait_on else []
    foot_R_addrs = sensor_addrs(model, FOOT_SENSORS_R) if gait_on else []
    if gait_on:
        print(f"  Gait FSM: ON (foot sensors L={len(foot_L_addrs)}, "
              f"R={len(foot_R_addrs)})")
        last_state = None
    q_ref_prev = q_ref_hold.copy()

    print("Launching viewer (close the window to exit)")
    apply_bws = bws_site_id is not None and bws_force > 0
    F_world = np.array([0.0, 0.0, bws_force])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if apply_bws:
                site_pos = data.site_xpos[bws_site_id]
                body_com = data.xipos[bws_body_id]
                arm = site_pos - body_com
                torque = np.cross(arm, F_world)
                data.xfrc_applied[bws_body_id, :3] = F_world
                data.xfrc_applied[bws_body_id, 3:6] = torque

            if model.nu > 0 and ctrl != "none":
                q = data.qpos[act_qpos_adr]
                qd = data.qvel[act_dof_adr]

                if fsm is not None:
                    fL = foot_force_mag(data, foot_L_addrs)
                    fR = foot_force_mag(data, foot_R_addrs)
                    q_ref = fsm.step(model.opt.timestep, fL, fR)
                    if fsm.state != last_state:
                        print(f"  [t={fsm.t_total:6.2f}] gait state -> {fsm.state}")
                        last_state = fsm.state
                else:
                    q_ref = q_ref_hold

                qd_ref = (q_ref - q_ref_prev) / max(model.opt.timestep, 1e-6)
                q_ref_prev = q_ref.copy()
                e = q_ref - q
                ed = qd_ref - qd

                if ctrl == "ctc":
                    mujoco.mj_rne(model, data, 1, data.qfrc_bias)
                    h_act = data.qfrc_bias[act_dof_adr]
                    mujoco.mj_fullM(model, M_full, data.qM)
                    M_act = M_full[np.ix_(act_dof_adr, act_dof_adr)]
                    tau = M_act @ qdd_ref + h_act + kp * e + kd * ed
                else:
                    tau = kp * e + kd * ed

                tau = np.where(has_ctrl_limits, np.clip(tau, ctrl_lo, ctrl_hi), tau)
                data.ctrl[:] = tau

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            sleep = model.opt.timestep - elapsed
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
