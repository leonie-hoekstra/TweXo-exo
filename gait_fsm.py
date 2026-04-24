"""Finite-state machine for sagittal gait of the WE2 exoskeleton.

States: STAND -> DS_R -> SWING_R -> DS_L -> SWING_L -> DS_R -> ...
  - STAND: hold standing pose until start_after seconds elapsed
  - DS_*:  brief double-support dwell before the named swing
  - SWING_R: right leg swings, left in stance; exit on R-foot heel-strike or timeout
  - SWING_L: mirror of SWING_R

Output: q_ref of length 8 in actuator order
        [LHA, LHF, LK, LA, RHA, RHF, RK, RA]
"""
import numpy as np

LHA, LHF, LK, LA, RHA, RHF, RK, RA = range(8)

STAND, DS_R, SWING_R, DS_L, SWING_L = "STAND", "DS_R", "SWING_R", "DS_L", "SWING_L"


class GaitParams:
    def __init__(self):
        self.swing_dur = 1.4          # s, slow swing keeps reaction torques low
        self.ds_dur = 0.40            # s, longer DS for re-stabilization
        self.hip_flex_amp = 0.25      # rad (~14 deg) modest swing hip flexion
        self.knee_flex_amp = 0.55     # rad (~32 deg) peak swing knee flexion
        self.ankle_clearance = 0.05   # rad dorsiflex during swing (toe-clearance)
        # Stance pose: keep hip at 0 — positive flex on a grounded leg pitches
        # the pelvis BACKWARD (foot is pinned). 0 is neutral.
        self.stance_hip = 0.0
        self.stance_knee = 0.0
        self.stance_ankle = 0.0
        self.heel_strike_N = 80.0     # |force| threshold for ground contact
        self.heel_strike_min_s = 0.6  # earliest phase for contact-driven exit


class GaitFSM:
    def __init__(self, params=None, start_after=1.5):
        self.p = params or GaitParams()
        self.state = STAND
        self.t_state = 0.0
        self.t_total = 0.0
        self.start_after = start_after

    def _swing(self, s):
        # half-sine bell for hip & knee; flat dorsiflex for ankle clearance
        hip = self.p.hip_flex_amp * np.sin(np.pi * s)
        knee = self.p.knee_flex_amp * np.sin(np.pi * s)
        ankle = self.p.ankle_clearance
        return hip, knee, ankle

    def _stance(self):
        return self.p.stance_hip, self.p.stance_knee, self.p.stance_ankle

    def step(self, dt, foot_force_L, foot_force_R):
        self.t_state += dt
        self.t_total += dt
        s = 0.0

        if self.state == STAND:
            if self.t_total >= self.start_after:
                self._go(DS_R)
        elif self.state == DS_R:
            if self.t_state >= self.p.ds_dur:
                self._go(SWING_R)
        elif self.state == SWING_R:
            s = min(self.t_state / self.p.swing_dur, 1.0)
            heel = foot_force_R > self.p.heel_strike_N
            if (heel and s > self.p.heel_strike_min_s) or s >= 1.0:
                self._go(DS_L)
        elif self.state == DS_L:
            if self.t_state >= self.p.ds_dur:
                self._go(SWING_L)
        elif self.state == SWING_L:
            s = min(self.t_state / self.p.swing_dur, 1.0)
            heel = foot_force_L > self.p.heel_strike_N
            if (heel and s > self.p.heel_strike_min_s) or s >= 1.0:
                self._go(DS_R)

        return self._qref(s)

    def _go(self, new_state):
        self.state = new_state
        self.t_state = 0.0

    def _qref(self, s):
        q = np.zeros(8)
        st_h, st_k, st_a = self._stance()
        sw_h, sw_k, sw_a = self._swing(s)
        q[LHF], q[LK], q[LA] = st_h, st_k, st_a
        q[RHF], q[RK], q[RA] = st_h, st_k, st_a
        if self.state == SWING_R:
            q[RHF], q[RK], q[RA] = sw_h, sw_k, sw_a
        elif self.state == SWING_L:
            q[LHF], q[LK], q[LA] = sw_h, sw_k, sw_a
        return q
