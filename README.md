# TweXo

MuJoCo simulation of the WE2 lower-limb exoskeleton, controlled with computed-torque
control (CTC) and a finite-state-machine gait generator.

This repo contains the standalone exoskeleton model. The combined exo + paralyzed
human (MyoFullBody) integration lives in a separate workspace.

## Contents

| File | Purpose |
| --- | --- |
| `WE2_3D.xml` | MJCF description of the WE2 exoskeleton (8 actuated joints, foot-contact sensors, standing keyframe). |
| `STLS/` | Mesh files referenced by `WE2_3D.xml`. |
| `view_model.py` | Interactive viewer with CTC inner loop and optional gait FSM. |
| `gait_fsm.py` | Sagittal gait FSM: STAND -> DS_R -> SWING_R -> DS_L -> SWING_L. |
| `inspect_model.py` | Print the model's bodies, joints, actuators, sensors. |
| `add_standing_keyframe.py` | Utility to capture the current pose as the standing keyframe. |
| `survey_sensors.py` | List the foot-contact sensor layout. |

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

The MuJoCo Python bindings (`mujoco`) ship a self-contained simulator and viewer,
so no extra system dependencies are needed.

### Conda

```bash
conda create -n TweXo python=3.11
conda activate TweXo
pip install -r requirements.txt
```

## Usage

Hold the standing keyframe with CTC:

```bash
python view_model.py
```

Walk with the gait FSM:

```bash
python view_model.py --gait=1
```

Tune CTC gains:

```bash
python view_model.py --kp=400 --kd=30
```

Print model structure:

```bash
python inspect_model.py
```

## Control

`view_model.py` runs computed-torque control on the 8 exo actuators each step:

    tau = M(q) * qdd_ref + h(q, qd) + Kp * (q_ref - q) + Kd * (qd_ref - qd)

with `M` and `h` extracted from MuJoCo via `mj_fullM` and `mj_rne`. Torques are
clipped to the per-actuator `ctrlrange` declared in `WE2_3D.xml`.

When `--gait=1`, `q_ref` comes from `GaitFSM.step(dt, foot_force_L, foot_force_R)`
in `gait_fsm.py`; otherwise the controller holds the standing keyframe pose.

## License

MIT, see `LICENSE`.
