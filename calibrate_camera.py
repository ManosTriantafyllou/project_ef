import numpy as np


# ── Φυσικές διαστάσεις ρομπότ (πραγματικές μετρήσεις) ────────────────
#
#   Έδαφος
#     └─  8.00 cm  → Βάση (pedestal) — J1 ξεκινά εδώ
#   J1  (yaw,  z=0.000 m από base frame)
#     └─ 16.95 cm  → Upper arm
#   J2/J3 (pitch/yaw, z=0.1695 m)
#     └─ 11.55 cm  → Forearm
#   J4/J5 (pitch/yaw, z=0.2850 m)
#     └─ 12.78 cm  → Wrist
#   J6/J7 (pitch/yaw, z=0.4128 m)
#   EE→J6 = 0.1100m,  EE→J4 = 0.2378m,  EE→J2 = 0.3533m,  EE→J1 = 0.5228m
#
#   Συνολικό φυσικό ύψος από έδαφος: 8 + 52.28 = 60.28 cm

ROBOT_BASE_HEIGHT_M = 0.08    # m — ύψος βάσης/pedestal από έδαφος
ROBOT_EE_TOOL_M = 0.11    # m — μήκος end-effector tool (vacuum sucker)
ROBOT_UPPER_ARM_M = 0.1695  # m — J1  → J2  (από space screws S2)
ROBOT_FOREARM_M = 0.1155  # m — J2  → J4  (0.2850 - 0.1695)
ROBOT_WRIST_M = 0.1278  # m — J4  → J6  (0.4128 - 0.2850)
ROBOT_WRIST_Z_M = 0.4128  # m — ύψος J6/J7 από base frame
ROBOT_TOTAL_HEIGHT_M = ROBOT_BASE_HEIGHT_M + ROBOT_WRIST_Z_M + ROBOT_EE_TOOL_M
# = 0.08 + 0.4128 + 0.11 = 0.6028 m ≈ 60.28 cm

# number of joints
N = 7

# initialize screws
screws = {}

# ── Space frame screws ────────────────────────────────────────────────
# Format: [ω_x, ω_y, ω_z, v_x, v_y, v_z]  (ω = axis, v = -ω × q_joint)
# Z=0 είναι πλέον το τραπέζι (Z_base = 0.08m)
S1 = np.array([0.,  0., 1.,  0.,       0.,   0.])
S2 = np.array([0.,  1., 0., -(0.1695 + ROBOT_BASE_HEIGHT_M), 0., 0.])
S3 = np.array([0.,  0., 1.,  0.,       0.,   0.])
S4 = np.array([0., -1., 0.,  (0.2850 + ROBOT_BASE_HEIGHT_M), 0., 0.])
S5 = np.array([0.,  0., 1.,  0.,       0.,   0.])
S6 = np.array([0., -1., 0.,  (0.4128 + ROBOT_BASE_HEIGHT_M), 0., 0.])
S7 = np.array([0.,  0., 1.,  0.,       0.,   0.])
screws["space"] = [S1, S2, S3, S4, S5, S6, S7]

# ── Body frame screws ─────────────────────────────────────────────────
# Αποστάσεις από EE (z_ee = 0.5228m) προς κάθε joint:
#   EE→J6 = 0.1100m,  EE→J4 = 0.2378m,  EE→J2 = 0.3533m,  EE→J1 = 0.5228m
_EE_Z = ROBOT_WRIST_Z_M + ROBOT_EE_TOOL_M   # = 0.5628 m
B1 = np.array([0.,  0., 1.,  0.,              _EE_Z,   0.])
B2 = np.array([0.,  1., 0.,  _EE_Z - 0.1695, 0.,     0.])
B3 = np.array([0.,  0., 1.,  0.,              _EE_Z,   0.])
B4 = np.array([0., -1., 0., -(0.4128 - 0.2850 + ROBOT_EE_TOOL_M), 0., 0.])
B5 = np.array([0.,  0., 1.,  0.,              _EE_Z,   0.])
B6 = np.array([0., -1., 0., -ROBOT_EE_TOOL_M, 0.,     0.])
B7 = np.array([0.,  0., 1.,  0.,              _EE_Z,   0.])
screws["body"] = [B1, B2, B3, B4, B5, B6, B7]

# ── T_space → body  (T_table → end-effector, home config q=0) ────────
# Z = ύψος βάσης (0.08) + ύψος wrist (0.4128) + μήκος EE (0.11) = 0.6028m
# X = 0 (χωρίς οριζόντιο offset)
Tsb = np.array([[1.0, 0.0, 0.0,  0.0],
                [0.0, 1.0, 0.0,  0.0],
                [0.0, 0.0, 1.0,  _EE_Z + ROBOT_BASE_HEIGHT_M],   # 0.6028 m
                [0.0, 0.0, 0.0,  1.0]])

# ── Joint limits ──────────────────────────────────────────────────────
q_lb = np.deg2rad([-160,  -70, -170, -113, -170, -115, -180])
q_ub = np.deg2rad([160,  115,  170,   75,  170,  -50,  180])

# ── T_world → space  (T_world → base frame) ──────────────────────────
# Base frame origin = J1 = 8cm πάνω από το έδαφος.
# Το Tws δεν αλλάζει εδώ γιατί οι IK στόχοι δίνονται ήδη στο base frame.
z_rotation = np.deg2rad(0)
translation = np.array([0.0, 0.0, 0.0])
Tws = np.array([[np.cos(z_rotation), -np.sin(z_rotation), 0.0, translation[0]],
                [np.sin(z_rotation),  np.cos(z_rotation), 0.0, translation[1]],
                [0.0,                 0.0,                 1.0, translation[2]],
                [0.0,                 0.0,                 0.0, 1.0]])
