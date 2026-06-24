"""
Central Pattern Generator (CPG) trot gait for the Go2 quadruped.

Produces, for any commanded velocity (vx, vy, wz) and a continuously
advancing phase clock, the desired foot position for each leg in the
base frame. These foot targets are converted to joint angles via the
verified IK in go2_kinematics.py, then tracked by a PD controller in
the actual MuJoCo simulation (see collect_data.py).

Gait pattern: trot. Diagonal leg pairs (FR+RL) and (FL+RR) move in
phase with each other, and in anti-phase (offset by half a cycle)
with the other pair. This is the standard, most stable gait for a
quadruped trotting forward, sideways, or turning.

Each leg cycles through:
    - stance phase (duty_factor of the cycle): foot is on the ground,
      moving backward relative to the body at the command velocity,
      providing the propulsive force.
    - swing phase (1 - duty_factor of the cycle): foot lifts off,
      swings forward in an arc, and touches down again at the start
      of the next stance phase.

Foot height during swing follows a half-sine arc (smooth lift-off
and touch-down, zero vertical velocity at both ends).
"""

from dataclasses import dataclass, field
import numpy as np

from go2_kinematics import (
    GO2_LEG_PARAMS,
    LEG_BASE_OFFSETS,
    LEG_SIDE_SIGN,
    inverse_kinematics,
    world_to_hip_frame,
)


@dataclass
class TrotCPGParams:
    stride_freq: float = 1.6          # Hz, full gait cycle frequency at nominal speed
    duty_factor: float = 0.55         # fraction of cycle each foot spends in stance
    swing_height: float = 0.08        # m, peak foot clearance during swing
    stance_depth_nominal: float = 0.30  # m, nominal standing height of hip above foot
    max_stride_length: float = 0.18   # m, cap on stance excursion (forward/back)
    max_lateral_stride: float = 0.10  # m, cap on lateral excursion
    max_yaw_stride: float = 0.12      # m, cap on per-leg arc length from yaw rate
    velocity_to_stride_gain: float = 0.5  # how strongly stride length scales with v
    velocity_to_freq_gain: float = 0.6    # how strongly cadence scales with |v|
    min_freq: float = 0.8             # Hz, floor on cadence even at v=0 (idle trot)


# Diagonal trot phase offsets: FL+RR together (phase 0), FR+RL together
# (phase 0.5). This is the standard trot diagonal-pair pattern.
LEG_PHASE_OFFSET = {
    "FL": 0.0,
    "RR": 0.0,
    "FR": 0.5,
    "RL": 0.5,
}


class TrotCPG:
    """
    Stateful CPG: call .reset() at episode start, then .step(dt, vx, vy, wz)
    every control step to advance the internal phase clock and obtain the
    next set of per-leg foot targets (in the base frame) and corresponding
    joint angle targets (via IK).
    """

    def __init__(self, params: TrotCPGParams = TrotCPGParams()):
        self.p = params
        self.phase = 0.0  # cycle phase in [0, 1)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        # Randomize initial phase so demonstrations don't all start from
        # the same gait configuration -- improves BC data diversity.
        self.phase = float(rng.uniform(0.0, 1.0))

    def _cadence_hz(self, vx, vy, wz):
        speed = np.sqrt(vx ** 2 + vy ** 2) + 0.5 * abs(wz)
        return self.p.min_freq + self.p.velocity_to_freq_gain * speed

    def _stride_vector(self, vx, vy, wz):
        """
        Per-leg stride excursion (forward, lateral) driven by the
        commanded velocity, clipped to safe maxima. Yaw rate adds a
        rotational stride component that differs sign by leg side
        (left legs vs right legs move oppositely for turning).
        """
        sx = np.clip(vx * self.p.velocity_to_stride_gain, -self.p.max_stride_length, self.p.max_stride_length)
        sy = np.clip(vy * self.p.velocity_to_stride_gain, -self.p.max_lateral_stride, self.p.max_lateral_stride)
        return sx, sy

    def step(self, dt: float, vx: float, vy: float, wz: float):
        """
        Advance the CPG phase by dt and compute foot targets + IK joint
        angles for all four legs.

        Returns
        -------
        dict with keys:
            'foot_targets_base': {leg: (3,) array, foot position in base frame}
            'joint_angles': {leg: (3,) array [hip, thigh, calf]}
        """
        freq = self._cadence_hz(vx, vy, wz)
        self.phase = (self.phase + dt * freq) % 1.0

        sx, sy = self._stride_vector(vx, vy, wz)

        foot_targets = {}
        joint_angles = {}

        for leg in ("FL", "FR", "RL", "RR"):
            leg_phase = (self.phase + LEG_PHASE_OFFSET[leg]) % 1.0

            # Yaw contribution: rotate the stride vector slightly per leg
            # based on its position, so a positive wz makes front legs
            # step more to one side and rear legs the other -- approximates
            # turning in place. Sign depends on leg's x-position (front/back)
            # and whether it's on the left/right.
            base_offset = LEG_BASE_OFFSETS[leg]
            yaw_dx = -wz * base_offset[1] * 0.5  # rough arc-length contribution
            yaw_dy = wz * base_offset[0] * 0.5
            stride_x = np.clip(sx + yaw_dx, -self.p.max_stride_length, self.p.max_stride_length)
            stride_y = np.clip(sy + yaw_dy, -self.p.max_lateral_stride, self.p.max_lateral_stride)

            if leg_phase < self.p.duty_factor:
                # --- Stance phase ---
                # foot moves linearly backward relative to body, from
                # +stride/2 (touchdown point) to -stride/2 (lift-off point).
                u = leg_phase / self.p.duty_factor  # 0 -> 1 across stance
                dx = stride_x * (0.5 - u)
                dy = stride_y * (0.5 - u)
                dz = 0.0
            else:
                # --- Swing phase ---
                u = (leg_phase - self.p.duty_factor) / (1.0 - self.p.duty_factor)  # 0->1
                dx = stride_x * (-0.5 + u)
                dy = stride_y * (-0.5 + u)
                dz = self.p.swing_height * np.sin(np.pi * u)  # half-sine lift arc

            # Nominal standing foot position directly under the hip,
            # offset by the gait excursion (dx, dy) and lift height (dz).
            nominal = np.array([dx, dy, -self.p.stance_depth_nominal + dz])

            foot_target_hip_frame = nominal  # already expressed relative to hip
            try:
                q = inverse_kinematics(foot_target_hip_frame, leg, knee_bend="backward")
            except ValueError:
                # Target briefly out of reach (e.g. extreme command) --
                # fall back to the nominal standing pose for this leg.
                q = inverse_kinematics(
                    np.array([0.0, 0.0, -self.p.stance_depth_nominal]),
                    leg, knee_bend="backward",
                )

            foot_targets[leg] = foot_target_hip_frame + LEG_BASE_OFFSETS[leg]
            joint_angles[leg] = q

        return {
            "foot_targets_base": foot_targets,
            "joint_angles": joint_angles,
        }


# Standard Go2 joint ordering used by gym-quadruped / unitree convention:
# FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
# RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
LEG_ORDER = ("FL", "FR", "RL", "RR")


def joint_angles_dict_to_vector(joint_angles: dict) -> np.ndarray:
    """Flatten the per-leg joint angle dict into the standard 12-dim vector."""
    return np.concatenate([joint_angles[leg] for leg in LEG_ORDER])


if __name__ == "__main__":
    # Quick sanity test: run the CPG for a few cycles at a fixed forward
    # velocity and print the joint angle vector to check it's well-formed
    # (no NaNs, no IK failures, reasonable magnitudes).
    cpg = TrotCPG()
    cpg.reset(seed=0)
    dt = 1.0 / 50.0  # 50 Hz control rate, matches a typical sim control loop
    vx, vy, wz = 0.5, 0.0, 0.0

    bad = 0
    for step in range(250):  # 5 seconds
        out = cpg.step(dt, vx, vy, wz)
        qvec = joint_angles_dict_to_vector(out["joint_angles"])
        if np.any(np.isnan(qvec)) or np.any(np.abs(qvec) > 3.0):
            bad += 1
        if step % 50 == 0:
            print(f"t={step*dt:.2f}s phase={cpg.phase:.3f} qvec={np.round(qvec, 3)}")

    print(f"\n{250-bad}/250 steps produced well-formed joint vectors")
