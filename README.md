# TweXo
A MuJoCo simulation of the WE2 lower-limb exoskeleton.

## What's in here

- `WE2_3D.xml` — the exoskeleton model (8 motors, foot sensors, a standing pose)
- `STLS/` — mesh files used by the model
- `view_model.py` — runs the simulation in MuJoCo's viewer
- `gait_fsm.py` — a simple state machine that makes the exo walk
- `inspect_model.py` — prints what's inside the model (joints, motors, etc.)
- `add_standing_keyframe.py` — saves the current pose as the standing pose
- `survey_sensors.py` — lists the foot sensors

## Setup

You need Python 3.11.

```bash
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n TweXo python=3.11
conda activate TweXo
pip install -r requirements.txt
```

## Running it

```bash
python view_model.py
```

Walk:

```bash
python view_model.py --gait=1
```

Close the viewer window to stop.

## How the control works

Each step the controller computes the torques the motors need to make the joints
follow a target pose. The target is either the standing pose, or — if `--gait=1`
— a walking trajectory from the FSM in `gait_fsm.py`. The FSM cycles through
stand → double support → swing right → double support → swing left, using the
foot sensors to detect heel-strike.

## License

MIT
