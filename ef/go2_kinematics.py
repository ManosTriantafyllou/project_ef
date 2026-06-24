"""
Go2 leg kinematics: forward and inverse kinematics for a standard
3-DOF quadruped leg (hip abduction, hip flexion, knee flexion).

Design choice: FK is defined FIRST, as a simple, unambiguous chain of
rotations. IK is then derived algebraically as the exact inverse of
that same chain. This avoids the classic bug where FK and IK are
derived independently with inconsistent sign/axis conventions.

Leg model
---------
Each leg has 3 joints, in order from the body:
    1. hip_joint   (abduction/adduction, rotation about local x-axis)
    2. thigh_joint (hip flexion/extension, rotation about local y-axis,
                    applied AFTER the abduction rotation, i.e. about the
                    leg's own rotated y-axis)
    3. calf_joint  (knee flexion/extension, rotation about the same
                    rotated y-axis)

Link lengths (Unitree Go2, meters):
    l_hip   = 0.0955   # hip offset (hip joint -> thigh joint), along the
                        # abduction-rotated y-axis
    l_thigh = 0.213    # thigh length (thigh joint -> calf joint)
    l_calf  = 0.213    # calf length (calf joint -> foot)

Frame convention
-----------------
Hip frame: origin at the hip joint, x forward, y left, z up. All
IK/FK below is expressed in this frame. To get a foot target from the
base frame, subtract LEG_BASE_OFFSETS[leg] first (see
world_to_hip_frame).

Kinematic chain (FK), by construction
--------------------------------------
1. Start at the hip joint. Apply abduction rotation theta1 about the
   local x-axis. This rotates the y-z plane. The hip-link vector
   (length l_hip, signed by leg side) and the "leg-plane down" vector
   both live in this rotated frame.
2. From the end of the hip link, the thigh and calf form a standard
   2-link planar chain in the (x, leg-plane-down) plane, where
   theta2 (hip pitch) and theta3 (knee) are measured from straight-down.

Concretely:
    p0 = R_x(theta1) @ [0, l_hip_signed, 0]              (hip link)
    # planar chain in the rotated plane spanned by global-x and the
    # rotated "down" direction d_hat = R_x(theta1) @ [0, 0, -1]
    planar_fwd  = l_thigh*sin(theta2) + l_calf*sin(theta2+theta3)
    planar_down = l_thigh*cos(theta2) + l_calf*cos(theta2+theta3)
    p1 = planar_fwd * x_hat + planar_down * d_hat
    foot = p0 + p1
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Go2LegParams:
    l_hip: float = 0.0955
    l_thigh: float = 0.213
    l_calf: float = 0.213


GO2_LEG_PARAMS = Go2LegParams()

LEG_BASE_OFFSETS = {
    "FL": np.array([0.1934, 0.0465, 0.0]),
    "FR": np.array([0.1934, -0.0465, 0.0]),
    "RL": np.array([-0.1934, 0.0465, 0.0]),
    "RR": np.array([-0.1934, -0.0465, 0.0]),
}

# Sign of l_hip for each leg: hip link points toward +y (left) for left
# legs, -y (right) for right legs, when theta1 = 0.
LEG_SIDE_SIGN = {
    "FL": 1.0,
    "FR": -1.0,
    "RL": 1.0,
    "RR": -1.0,
}


def forward_kinematics(q: np.ndarray, leg: str,
                        params: Go2LegParams = GO2_LEG_PARAMS) -> np.ndarray:
    """
    q = [theta1 (abduction), theta2 (hip pitch), theta3 (knee)] in radians.
    Returns foot position [x, y, z] in the hip frame.
    """
    theta1, theta2, theta3 = q
    l_hip = params.l_hip * LEG_SIDE_SIGN[leg]
    l_thigh = params.l_thigh
    l_calf = params.l_calf

    # Hip link: starts pointing along +y (for left legs), rotated by
    # theta1 about the x-axis.
    # R_x(theta1) @ [0, l_hip, 0] = [0, l_hip*cos(theta1), l_hip*sin(theta1)]
    p0 = np.array([0.0, l_hip * np.cos(theta1), l_hip * np.sin(theta1)])

    # "Down" direction rotated by theta1 about x-axis:
    # R_x(theta1) @ [0, 0, -1] = [0, sin(theta1), -cos(theta1)]
    d_hat = np.array([0.0, np.sin(theta1), -np.cos(theta1)])
    x_hat = np.array([1.0, 0.0, 0.0])

    planar_fwd = l_thigh * np.sin(theta2) + l_calf * np.sin(theta2 + theta3)
    planar_down = l_thigh * np.cos(theta2) + l_calf * np.cos(theta2 + theta3)

    p1 = planar_fwd * x_hat + planar_down * d_hat

    return p0 + p1


def inverse_kinematics(foot_pos_hip_frame: np.ndarray, leg: str,
                        params: Go2LegParams = GO2_LEG_PARAMS,
                        knee_bend: str = "backward") -> np.ndarray:
    """
    Exact algebraic inverse of forward_kinematics above.

    Parameters
    ----------
    foot_pos_hip_frame : (3,) array [x, y, z] target in the hip frame.
    leg : one of "FL", "FR", "RL", "RR".
    knee_bend : "backward" (knee flexes backward; standard quadruped
        stance, theta3 < 0) or "forward" (theta3 > 0).

    Returns
    -------
    q = [theta1, theta2, theta3] in radians.
    """
    x, y, z = foot_pos_hip_frame
    l_hip = params.l_hip * LEG_SIDE_SIGN[leg]
    l_thigh = params.l_thigh
    l_calf = params.l_calf

    # --- Step 1: solve theta1 from (y, z) ---
    # From FK: y = l_hip*cos(theta1) + planar_down*sin(theta1)
    #          z = l_hip*sin(theta1) - planar_down*cos(theta1)
    # => y^2 + z^2 = l_hip^2 + planar_down^2  (planar_fwd doesn't appear)
    # So first solve for planar_down (call it pd) and planar_fwd (pf),
    # then theta1, in that order -- but pd depends on theta2/theta3 too.
    #
    # Better: treat (l_hip, pd) as an orthogonal basis rotated by theta1
    # that must reproduce (y, z). This is a standard "2-bar in a plane"
    # sub-problem:
    #   y = l_hip*cos(theta1) + pd*sin(theta1)
    #   z = l_hip*sin(theta1) - pd*cos(theta1)
    # Solving this pair for theta1 given pd requires knowing pd, and pd
    # depends on r = sqrt(x^2 + pd^2) via the law of cosines, which
    # depends on theta1 only through pd. We resolve this by first
    # computing pd directly from the constraint that the foot's distance
    # from the hip-link's far end, projected appropriately, must match
    # the leg length -- standard approach: compute pd from
    #   pd^2 = y^2 + z^2 - l_hip^2
    # (this holds regardless of theta1, since it's the projection of the
    # foot onto the plane perpendicular to the hip link's rotation axis
    # combined with the hip link itself -- see derivation note below).

    d_yz_sq = y ** 2 + z ** 2
    pd_sq = d_yz_sq - l_hip ** 2
    if pd_sq < 0:
        raise ValueError(
            f"Target unreachable for leg {leg}: y,z too close to hip "
            f"axis (y^2+z^2={d_yz_sq:.5f} < l_hip^2={l_hip**2:.5f})"
        )
    pd = np.sqrt(pd_sq)  # planar_down, defined >= 0 (foot below hip-pitch joint)

    # Now solve for theta1 from:
    #   y = l_hip*cos(theta1) + pd*sin(theta1)
    #   z = l_hip*sin(theta1) - pd*cos(theta1)
    # This is a rotation: [y, z] = R_x(theta1) applied to [l_hip, -pd]
    # in the (y,z) plane representation used by FK. Solve directly:
    #   theta1 = atan2(z, y) - atan2(-pd, l_hip)
    theta1 = np.arctan2(z, y) - np.arctan2(-pd, l_hip)

    # --- Step 2: planar 2-link IK for theta2, theta3 using (x, pd) ---
    r = np.sqrt(x ** 2 + pd ** 2)
    if r > (l_thigh + l_calf) + 1e-9:
        raise ValueError(
            f"Target unreachable for leg {leg}: r={r:.4f} exceeds max "
            f"reach {l_thigh + l_calf:.4f}"
        )
    if r < abs(l_thigh - l_calf) - 1e-9:
        raise ValueError(
            f"Target unreachable for leg {leg}: r={r:.4f} below min "
            f"reach {abs(l_thigh - l_calf):.4f}"
        )

    cos_knee_interior = (l_thigh ** 2 + l_calf ** 2 - r ** 2) / (2 * l_thigh * l_calf)
    cos_knee_interior = np.clip(cos_knee_interior, -1.0, 1.0)
    knee_interior = np.arccos(cos_knee_interior)  # in [0, pi], 0 = fully folded

    if knee_bend == "backward":
        theta3 = -(np.pi - knee_interior)
    else:
        theta3 = (np.pi - knee_interior)

    cos_alpha = (l_thigh ** 2 + r ** 2 - l_calf ** 2) / (2 * l_thigh * r)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    gamma = np.arctan2(x, pd)  # angle of hip->foot line from straight-down

    if knee_bend == "backward":
        theta2 = gamma + alpha
    else:
        theta2 = gamma - alpha

    return np.array([theta1, theta2, theta3])


def world_to_hip_frame(foot_pos_base: np.ndarray, leg: str) -> np.ndarray:
    """Translate a foot target from the base frame to the leg's hip frame."""
    return foot_pos_base - LEG_BASE_OFFSETS[leg]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for leg in ["FL", "FR", "RL", "RR"]:
        n_ok = 0
        n_tested = 0
        max_err = 0.0
        for _ in range(2000):
            theta1 = rng.uniform(-0.3, 0.3)
            theta2 = rng.uniform(-0.6, 1.2)
            theta3 = rng.uniform(-2.5, -0.3)
            q_true = np.array([theta1, theta2, theta3])
            foot = forward_kinematics(q_true, leg)
            n_tested += 1
            try:
                q_ik = inverse_kinematics(foot, leg, knee_bend="backward")
                foot_check = forward_kinematics(q_ik, leg)
                err = np.linalg.norm(foot_check - foot)
                max_err = max(max_err, err)
                if err < 1e-6:
                    n_ok += 1
                else:
                    print(f"{leg} FK->IK->FK mismatch err={err:.6f} "
                          f"q_true={q_true} q_ik={q_ik}")
            except ValueError as e:
                print(f"{leg} unreachable (unexpected): {e}")
        print(f"{leg}: {n_ok}/{n_tested} FK->IK->FK round-trip OK, max_err={max_err:.2e}")
