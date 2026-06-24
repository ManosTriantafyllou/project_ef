"""
laptop_ik_node.py
=================
Λογική (3 cases):

1. /object_pose  → κόκκινο ΟΛΟΚΛΗΡΟ    → hover → grasp → RED_DROP
2. /red_occluded → κόκκινο ΜΙΣΟΚΡΥΜΜΕΝΟ → αφαίρεσε blocker → BLOCKER_DROP
3. Τίποτα για N sec → αφαίρεσε ψηλότερο → BLOCKER_DROP

Gripper geometry:
  - Το IK στοχεύει το ENDEFFECTOR (άκρο URDF)
  - hover_height_m=0.0  → βεντούζα ακουμπά τον κύβο
  - ΔΕΝ χρειάζεται GRIPPER_OFFSET_Z

ΣΗΜΑΝΤΙΚΟ: Τα task counters (_red_grasped κλπ) αρχικοποιούνται
ΜΟΝΟ στο __init__. ΔΕΝ πρέπει να μηδενίζονται πουθενά αλλού
(ούτε στο _red_full_cb που τρέχει σε κάθε frame, ούτε στο _reset
που τρέχει μετά από κάθε grasp) — αλλιώς χάνεται η μέτρηση.

OBSTACLE AVOIDANCE (κουτί αντικειμένων):
  Χρησιμοποιούνται δύο ΣΤΑΘΕΡΑ ενδιάμεσα σημεία WAYPOINT_A και
  WAYPOINT_B (πάνω από το κουτί, εκτός τοιχωμάτων):
    - Κίνηση ΠΡΟΣ μέσα στο κουτί (grasp):   ... → A → B → approach → target
    - Κίνηση ΑΠΟ μέσα στο κουτί (drop έξω): target → B → A → ...
  Κάθε κίνηση μεταξύ διαδοχικών waypoints γίνεται ως ξεχωριστή IK
  λύση+resend (όχι ένα μεγάλο "πήδημα"), οπότε ο βραχίονας περνά πάντα
  από τα ίδια γνωστά-ασφαλή σημεία, ανεξάρτητα από το από πού ξεκίνησε.

APPROACH HEIGHT (2-σταδιακό κάθετο grasp):
  Μετά το obstacle-avoidance routing (A→B), ο βραχίονας πηγαίνει ΠΡΩΤΑ
  στο (bx,by,APPROACH_Z_M) — το ΙΔΙΟ XY με τον κύβο, σε σταθερό ύψος
  APPROACH_Z_M=0.25m — και ΑΜΕΣΩΣ ΜΕΤΑ (χωρίς να περιμένει gripper_done)
  κατεβαίνει κάθετα στο πραγματικό target_z = bz + hover_height_m.
  Εκεί σταματά και περιμένει gripper_done. Το APPROACH_Z_M είναι πάντα
  το ίδιο, ανεξάρτητα από το ύψος του εκάστοτε κύβου — μόνο το ΤΕΛΙΚΟ
  ύψος (target_z) εξαρτάται από bz+hover_height_m όπως πριν.
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Empty, String

from cube_follow_pkg.cube_utils.solve_inv_kine import solve_ik


# ── Drop positions (robot base frame) ────────────────────────────────
RED_DROP_X,     RED_DROP_Y,     RED_DROP_Z     =  0.00, -0.15, 0.20
BLOCKER_DROP_X, BLOCKER_DROP_Y, BLOCKER_DROP_Z =  0.00, +0.15, 0.20

# ── Σταθερά ενδιάμεσα σημεία διαδρομής (obstacle avoidance) ───────────
# Κάθε κίνηση ΠΡΟΣ το κουτί (για grasp) ή ΑΠΟ το κουτί (μετά grasp,
# προς ένα drop point) περνά πάντα από αυτά τα δύο fixed waypoints,
# με αυτή τη σειρά:
#   πηγαίνοντας ΠΡΟΣ το κουτί:    ... → A → B → approach → target
#   επιστρέφοντας ΑΠΟ το κουτί:   target → B → A → ...
WAYPOINT_A = (0.00, -0.16, 0.26)
WAYPOINT_B = (-0.11, -0.11, 0.26)

# ── Approach height (2-σταδιακό κάθετο grasp) ─────────────────────────
# Σταθερό ενδιάμεσο ύψος, ΙΔΙΟ XY με τον κύβο, πριν την τελική κάθετη
# κατάβαση στο πραγματικό hover ύψος πάνω από τον κύβο. Δεν εξαρτάται
# από το ύψος του εκάστοτε κύβου — πάντα 0.25m.
APPROACH_Z_M = 0.25

# ── Camera → robot base transform ────────────────────────────────────
_t_cb = np.array([-0.243235, -0.069028, 0.499280])
_R_cb = np.array([
    [-0.055439, -0.998210,  0.022423],
    [-0.998399,  0.055673,  0.009973],
    [-0.011203, -0.021834, -0.999699],
])

Ry180 = np.array([[-1., 0.,  0.],
                   [ 0., 1.,  0.],
                   [ 0., 0., -1.]])

CUBE_SIZE_M = 0.024

ST_SEARCHING        = "searching"
ST_HOVERING_RED     = "hovering_red"
ST_WAITING_RED      = "waiting_red"
ST_DROPPING_RED     = "dropping_red"
ST_HOVERING_BLOCKER = "hovering_blocker"
ST_WAITING_BLOCKER  = "waiting_blocker"
ST_DROPPING_BLOCKER = "dropping_blocker"

_FALLBACK_ANGLES_DEG = [0, 15, -15, 30, -30, 45, -45]

NO_RED_TIMEOUT_S = 2.0

# ── Task parameters ───────────────────────────────────────────────────
GRASP_TARGET = 3    # πόσα κόκκινα αντικείμενα να πιάσει συνολικά


def _make_target(x, y, z, yaw_deg=0.0):
    cy = math.cos(math.radians(yaw_deg))
    sy = math.sin(math.radians(yaw_deg))
    Rz = np.array([[cy, -sy, 0.],[sy, cy, 0.],[0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = Ry180 @ Rz
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


class LaptopIKNode(Node):

    RESEND_COUNT      = 20
    RESEND_INTERVAL_S = 0.3
    POSE_ALPHA        = 0.1
    STABLE_FRAMES     = 3

    def __init__(self):
        super().__init__("laptop_ik_node")

        self.declare_parameter("hover_height_m", 0.153)

        # ── State machine ─────────────────────────────────────────────
        self.state             = ST_SEARCHING
        self.q0_list           = None
        self._pending_msg      = None
        self._send_count       = 0
        self._send_timer       = None
        self._on_done          = None
        self._last_ik_time     = 0.0
        self._filtered_pos     = None
        self._stable_count     = 0
        self._last_red_time    = time.time()
        self._last_markers     = []
        self._occluded_red_cam = None
        self.current_real_q    = None

        # ── Task counters — ΜΟΝΟ ΕΔΩ αρχικοποιούνται ──────────────────
        # ΔΕΝ ξαναμηδενίζονται πουθενά αλλιώς (ούτε σε _red_full_cb,
        # ούτε σε _reset) ώστε να μετράνε σωστά σε όλη τη διάρκεια.
        self._red_grasped     = 0
        self._red_attempts    = 0
        self._ik_failures     = 0
        self._blocker_removed = 0
        self._task_start_time = time.time()

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, "/object_pose",   self._red_full_cb, 10)
        self.create_subscription(
            PoseStamped, "/red_occluded",  self._red_occluded_cb, 10)
        self.create_subscription(
            MarkerArray, "/detected_objects", self._markers_cb, 10)
        self.create_subscription(
            Empty, "/gripper_done", lambda _: self._go_next(), 10)
        self.create_subscription(
            Empty, "/reset_target", lambda _: self._reset(), 10)
        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10)

        # ── Publishers ────────────────────────────────────────────────
        self.joint_pub   = self.create_publisher(JointState, "/joint_commands", 10)
        self.state_pub   = self.create_publisher(String, "/arm_state", 10)
        self.pose_pub    = self.create_publisher(PoseStamped, "/cube_pose_robot_frame", 10)
        self.gripper_pub = self.create_publisher(String, "/gripper_cmd", 10)

        # Bulletproof gripper publishing: Στέλνει συνεχώς την επιθυμητή κατάσταση!
        self._target_gripper_state = "OFF"
        self.create_timer(0.5, self._continuous_gripper_pub)

        self.create_timer(0.5, self._check_no_red)

        self.get_logger().info("=" * 50)
        self.get_logger().info("LaptopIKNode έτοιμο")
        self.get_logger().info(
            f"GRASP_TARGET={GRASP_TARGET}  "
            f"hover_height={self.get_parameter('hover_height_m').value}m  "
            f"approach_z={APPROACH_Z_M}m")
        self.get_logger().info(
            f"Obstacle avoidance waypoints: A={WAYPOINT_A}  B={WAYPOINT_B}")
        self.get_logger().info("=" * 50)

    # ── Helpers ───────────────────────────────────────────────────────

    def _pub_state(self, s):
        msg = String(); msg.data = s
        self.state_pub.publish(msg)
        self.get_logger().info(f"[STATE] {s.upper()}")

    def _continuous_gripper_pub(self):
        if self._target_gripper_state == "ON":
            msg = String()
            msg.data = self._target_gripper_state
            self.gripper_pub.publish(msg)

    def _pub_gripper(self, cmd: str):
        self._target_gripper_state = cmd
        if cmd == "ON":
            self.get_logger().info(f"[GRIPPER TARGET] -> ON (στέλνεται συνεχώς)")
        else:
            self.get_logger().info(f"[GRIPPER TARGET] -> OFF (στέλνεται 5 φορές και σταματάει)")
            def _send_multiple_off():
                msg = String()
                msg.data = "OFF"
                for _ in range(5):
                    self.gripper_pub.publish(msg)
                    time.sleep(0.1)
            import threading
            threading.Thread(target=_send_multiple_off, daemon=True).start()

    def _make_joint_msg(self, q_rad):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ["joint1","joint2","joint3","joint4",
                        "joint5","joint6","joint7"]
        msg.position = [round(q, 5) for q in q_rad]
        return msg

    def _start_resend(self, msg_j, on_done=None):
        if self._send_timer:
            self._send_timer.cancel()
        self._pending_msg = msg_j
        self._send_count  = 0
        self._on_done     = on_done
        self._send_timer  = self.create_timer(
            self.RESEND_INTERVAL_S, self._resend_cb)

    def _resend_cb(self):
        if self._pending_msg is None:
            return
        self._send_count += 1
        self.joint_pub.publish(self._pending_msg)
        self.get_logger().info(
            f"  publish #{self._send_count}/{self.RESEND_COUNT} [{self.state}]")
        if self._send_count >= self.RESEND_COUNT:
            self._send_timer.cancel()
            self._send_timer  = None
            self._pending_msg = None
            if self._on_done:
                cb = self._on_done
                self._on_done = None
                cb()

    def _solve_ik_with_fallback(self, x, y, z, avoid_box=False):
        """Δοκιμάζει IK με 7 fallback orientations + χαλαρότερο tolerance."""
        for yaw in _FALLBACK_ANGLES_DEG:
            T = _make_target(x, y, z, yaw)
            q_rad, ok = solve_ik(
                T, "space", self.q0_list, 1e-5,
                max_ikine_iter=20, max_q0_search=16, max_q0_tries=4, avoid_box=avoid_box)
            if ok:
                if yaw != 0:
                    self.get_logger().info(f"  IK επιτυχές με yaw={yaw}°")
                return q_rad, True

        self.get_logger().warn(
            f"IK 1η προσπάθεια απέτυχε — δοκιμάζω χωρίς warm start")
        for yaw in _FALLBACK_ANGLES_DEG:
            T = _make_target(x, y, z, yaw)
            q_rad, ok = solve_ik(
                T, "space", None, 1e-3,
                max_ikine_iter=30, max_q0_search=32, max_q0_tries=8, avoid_box=avoid_box)
            if ok:
                self.get_logger().info(
                    f"  IK επιτυχές (2η προσπάθεια) yaw={yaw}°")
                self._ik_failures += 1
                return q_rad, True

        self._ik_failures += 1
        self.get_logger().error(
            f"IK απέτυχε εντελώς — "
            f"target=({x:.3f},{y:.3f},{z:.3f}) εκτός workspace")
        return None, False

    def _cam_to_base(self, x, y, z):
        p = _R_cb @ np.array([x, y, z]) + _t_cb
        return float(p[0]), float(p[1]), float(p[2])

    # ── Obstacle avoidance: εκτέλεση κίνησης ──────────────────────────

    def _issue_ik_move(self, target_x, target_y, target_z, callback, avoid_box=False):
        """
        Πρωτογενής κίνηση: λύνει IK για ΕΝΑ (x,y,z) endeffector target
        και ξεκινά το resend. Καλεί το callback μετά το τέλος του resend.
        """
        self.get_logger().info(
            f"  IK waypoint target=({target_x:.3f},{target_y:.3f},{target_z:.3f}) [avoid_box={avoid_box}]")

        # Χρησιμοποιούμε την ΤΡΕΧΟΥΣΑ ΠΡΑΓΜΑΤΙΚΗ θέση του ρομπότ ως αρχική μαντεψιά 
        # για να αποφύγουμε τεράστιες άσκοπες περιστροφές στην αρχή!
        if self.q0_list is None and self.current_real_q is not None:
            self.q0_list = [np.array(self.current_real_q)]

        q_rad, ok = self._solve_ik_with_fallback(target_x, target_y, target_z, avoid_box=avoid_box)
        if not ok:
            self._reset()
            return

        self.q0_list = [np.copy(q_rad),
            np.array([-1.557, 1.177, -0.532, -1.209, 0.612, -0.940, -2.180])]

        q_deg = [math.degrees(q) for q in q_rad]
        self.get_logger().info(
            f"  IK OK → deg={[f'{d:.1f}' for d in q_deg]}")

        self._start_resend(self._make_joint_msg(q_rad), on_done=callback)

    def _move_through_waypoints(self, waypoints, callback, waypoint_hooks=None):
        """
        Εκτελεί διαδοχικά μια λίστα από (x,y,z,avoid_box) endeffector waypoints.
        To waypoint_hooks είναι ένα dictionary: π.χ. {1: func} εκτελεί την func
        πριν ξεκινήσει η κίνηση για το waypoint 1.
        """
        if waypoint_hooks is None:
            waypoint_hooks = {}

        def _step(idx):
            if idx in waypoint_hooks:
                waypoint_hooks[idx]()

            if idx >= len(waypoints):
                if callback:
                    callback()
                return
            x, y, z, avoid_box = waypoints[idx]
            self._issue_ik_move(x, y, z, lambda: _step(idx + 1), avoid_box=avoid_box)

        _step(0)

    def _move_into_box(self, target_x, target_y, target_z, callback):
        """
        Μετακίνηση ΠΡΟΣ ένα σημείο μέσα στο κουτί (π.χ. grasp).

        Σειρά waypoints (x, y, z, avoid_box):
          0: WAYPOINT_A (False)
          1: WAYPOINT_B (False)
          2: approach (False) -> Πάνω από τον κύβο, στο ίδιο XY, ύψος APPROACH_Z_M
          3: target (True) -> Εδώ ενεργοποιείται το CBF για ασφαλή κάθετη κατάβαση!
        """
        waypoints = [
            (*WAYPOINT_A, False),
            (*WAYPOINT_B, False),
            (target_x, target_y, APPROACH_Z_M, False),
            (target_x, target_y, target_z, True),
        ]
        self.get_logger().info(
            f"Obstacle avoidance: A→B→approach(z={APPROACH_Z_M})→target "
            f"({target_x:.3f},{target_y:.3f},{target_z:.3f})")
        
        # Ζητήθηκε από τον χρήστη να ανάβει η βεντούζα όταν πάει στο 2ο waypoint (WAYPOINT_B)
        hooks = {
            1: lambda: self._pub_gripper("ON")
        }
        self._move_through_waypoints(waypoints, callback, waypoint_hooks=hooks)

    def _move_out_of_box(self, target_x, target_y, target_z, callback):
        """
        Μετακίνηση ΑΠΟ ένα σημείο μέσα στο κουτί προς έναν εξωτερικό
        στόχο (π.χ. drop point). Περνά ΠΑΝΤΑ πρώτα από B → A.
        """
        # Το πρώτο βήμα προς το WAYPOINT_B είναι η άνοδος μέσα από το κουτί, άρα avoid_box=True!
        waypoints = [
            (*WAYPOINT_B, True),
            (*WAYPOINT_A, False), 
            (target_x, target_y, target_z, False)
        ]
        self.get_logger().info(
            f"Obstacle avoidance: B→A→target "
            f"({target_x:.3f},{target_y:.3f},{target_z:.3f})")
        self._move_through_waypoints(waypoints, callback)

    def _hover_and_wait(self, x_cam, y_cam, z_cam, next_state):
        hover_h = self.get_parameter("hover_height_m").value
        bx, by, bz = self._cam_to_base(x_cam, y_cam, z_cam)

        target_x = bx
        target_y = by
        target_z = bz + hover_h

        self.get_logger().info(
            f"cube_base=({bx:.3f},{by:.3f},{bz:.3f})  "
            f"endeff_target=({target_x:.3f},{target_y:.3f},{target_z:.3f})")

        def _on_arrived():
            self.state = next_state
            self._pub_state(next_state)
            
            # Η βεντούζα έχει ήδη ανάψει από το WAYPOINT_B (μέσω του hook στο _move_into_box).
            self.get_logger().info(">>> Έφτασα στο αντικείμενο! (Η βεντούζα ρουφάει ήδη). Περιμένω 2s... <<<")
            
            # Αφήνουμε 2.0 δευτερόλεπτα χρόνο για να ρουφήξει καλά τον κύβο (αν και ρουφάει ήδη)
            import threading
            threading.Timer(2.0, self._go_next).start()

        # Πηγαίνουμε ΜΕΣΑ στο κουτί για grasp:
        # A → B → approach(ίδιο XY, z=0.25) → target(τελικό hover ύψος)
        self._move_into_box(target_x, target_y, target_z, _on_arrived)

    def _drop_and_then(self, dx, dy, dz, callback):
        self.get_logger().info(
            f"Drop endeff target=({dx:.3f},{dy:.3f},{dz:.3f})")
        # Φεύγουμε ΑΠΟ το κουτί προς εξωτερικό drop point → B → A → target
        self._move_out_of_box(dx, dy, dz, callback)

    def _smooth_pos(self, pos):
        if self._filtered_pos is None:
            self._filtered_pos = pos.copy()
            return pos
        a = self.POSE_ALPHA
        self._filtered_pos = (1 - a) * self._filtered_pos + a * pos
        return self._filtered_pos.copy()

    def _find_blocker_near(self, red_x_cam, red_y_cam, red_z_cam):
        candidates = []
        for mk in self._last_markers:
            if mk.ns == "red":
                continue
            mx = mk.pose.position.x
            my = mk.pose.position.y
            mz = mk.pose.position.z
            if mz >= red_z_cam - 0.02:
                continue
            dist = np.sqrt((mx - red_x_cam)**2 + (my - red_y_cam)**2)
            if dist > 0.15:
                continue
            candidates.append((mz, mx, my, mz, mk.ns))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        _, bx, by, bz, bname = candidates[0]
        self.get_logger().info(f"Blocker: [{bname}] z={bz:.3f}m")
        return bx, by, bz, bname

    def _find_highest_object(self):
        candidates = [mk for mk in self._last_markers if mk.ns != "red"]
        if not candidates:
            return None
        highest = min(candidates, key=lambda mk: mk.pose.position.z)
        return (highest.pose.position.x,
                highest.pose.position.y,
                highest.pose.position.z,
                highest.ns)

    # ── Subscribers ───────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState):
        if len(msg.position) >= 7:
            self.current_real_q = list(msg.position[:7])

    def _markers_cb(self, msg: MarkerArray):
        self._last_markers = msg.markers

    def _red_full_cb(self, msg: PoseStamped):
        """
        Case 1: κόκκινο ΟΛΟΚΛΗΡΟ.
        ΠΡΟΣΟΧΗ: Αυτή η callback τρέχει σε ΚΑΘΕ frame (~30x/sec)
        όσο ο detector βλέπει κόκκινο. ΜΗΝ προσθέσεις εδώ καμία
        αρχικοποίηση μεταβλητής που πρέπει να διατηρηθεί.
        """
        self._last_red_time    = time.time()
        self._occluded_red_cam = None

        if self.state != ST_SEARCHING:
            return

        x_cam = msg.pose.position.x
        y_cam = msg.pose.position.y
        z_cam = msg.pose.position.z
        p_cam = self._smooth_pos(np.array([x_cam, y_cam, z_cam]))

        self._stable_count += 1
        if self._stable_count < self.STABLE_FRAMES:
            return

        bx, by, bz = self._cam_to_base(p_cam[0], p_cam[1], p_cam[2])
        self.get_logger().info(
            f"[RED FULL] cam=({x_cam:.3f},{y_cam:.3f},{z_cam:.3f}) "
            f"base=({bx:.3f},{by:.3f},{bz:.3f})")

        now = time.time()
        if now - self._last_ik_time < 0.5:
            return
        self._last_ik_time = now

        self.get_logger().info("Κόκκινο ολόκληρο → hover RED")
        self.state = ST_HOVERING_RED
        self._pub_state(ST_HOVERING_RED)
        self._hover_and_wait(p_cam[0], p_cam[1], p_cam[2], ST_WAITING_RED)

    def _red_occluded_cb(self, msg: PoseStamped):
        self._last_red_time = time.time()
        self._occluded_red_cam = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z)
        if self.state != ST_SEARCHING:
            return
        now = time.time()
        if now - self._last_ik_time < 0.5:
            return

        x_cam, y_cam, z_cam = self._occluded_red_cam
        blocker = self._find_blocker_near(x_cam, y_cam, z_cam)
        if blocker is None:
            self.get_logger().warn("Κόκκινο μισοκρυμμένο — δεν βρέθηκε blocker")
            return

        self._last_ik_time = now
        bx_cam, by_cam, bz_cam, blocker_color = blocker
        self.get_logger().info(
            f"Κόκκινο μισοκρυμμένο → αφαιρώ [{blocker_color}] blocker "
            f"cam=({bx_cam:.3f},{by_cam:.3f},{bz_cam:.3f})")
        self.state = ST_HOVERING_BLOCKER
        self._pub_state(ST_HOVERING_BLOCKER)
        self._hover_and_wait(bx_cam, by_cam, bz_cam, ST_WAITING_BLOCKER)

    def _check_no_red(self):
        if self.state != ST_SEARCHING:
            return
        elapsed = time.time() - self._last_red_time
        if elapsed < NO_RED_TIMEOUT_S:
            return
        highest = self._find_highest_object()
        if highest is None:
            return
        bx_cam, by_cam, bz_cam, highest_color = highest
        self.get_logger().info(
            f"Δεν βλέπω κόκκινο για {elapsed:.1f}s → "
            f"αφαιρώ [{highest_color}] (ψηλότερο) z={bz_cam:.3f}m")
        self._last_red_time = time.time()
        self._stable_count  = 0
        self._filtered_pos  = None
        self.state = ST_HOVERING_BLOCKER
        self._pub_state(ST_HOVERING_BLOCKER)
        self._hover_and_wait(bx_cam, by_cam, bz_cam, ST_WAITING_BLOCKER)

    # ── State transitions ─────────────────────────────────────────────

    def _go_next(self):
        if self.state == ST_WAITING_RED:
            self.get_logger().info("Gripper done → drop RED")
            self._red_attempts += 1
            self.state = ST_DROPPING_RED
            self._pub_state(ST_DROPPING_RED)
            def _after_red():
                self._pub_gripper("OFF")
                self._red_grasped += 1
                elapsed = time.time() - self._task_start_time
                self.get_logger().info(
                    f"✓ Grasp #{self._red_grasped}/{GRASP_TARGET} "
                    f"επιτυχές! ({elapsed:.0f}s)")
                self.get_logger().info(
                    ">>> Βεντούζα OFF — αντικείμενο αφέθηκε εκτός κουτιού <<<")
                if self._red_grasped >= GRASP_TARGET:
                    self._task_complete()
                else:
                    self.get_logger().info("Αυτόματο reset σε 3 δευτερόλεπτα για το επόμενο...")
                    import threading
                    threading.Timer(3.0, self._reset).start()
            self._drop_and_then(RED_DROP_X, RED_DROP_Y, RED_DROP_Z,
                                callback=_after_red)

        elif self.state == ST_WAITING_BLOCKER:
            self.get_logger().info("Gripper done → drop BLOCKER")
            self._blocker_removed += 1
            self.state = ST_DROPPING_BLOCKER
            self._pub_state(ST_DROPPING_BLOCKER)
            def _after_blocker():
                self._pub_gripper("OFF")
                self.get_logger().info(
                    f"Blocker #{self._blocker_removed} αφαιρέθηκε.")
                self.get_logger().info(
                    ">>> Βεντούζα OFF — αντικείμενο αφέθηκε εκτός κουτιού <<<")
                self.get_logger().info("Αυτόματο reset σε 3 δευτερόλεπτα για το επόμενο...")
                import threading
                threading.Timer(3.0, self._reset).start()
            self._drop_and_then(BLOCKER_DROP_X, BLOCKER_DROP_Y, BLOCKER_DROP_Z,
                                callback=_after_blocker)
        else:
            self.get_logger().warn(
                f"/gripper_done αγνοείται — state={self.state}")

    def _task_complete(self):
        elapsed = time.time() - self._task_start_time
        rate = self._red_grasped / max(self._red_attempts, 1) * 100
        self.get_logger().info("=" * 50)
        self.get_logger().info("TASK ΟΛΟΚΛΗΡΩΘΗΚΕ!")
        self.get_logger().info(f"  Κόκκινα πιασμένα:     {self._red_grasped}/{GRASP_TARGET}")
        self.get_logger().info(f"  Απόπειρες:            {self._red_attempts}")
        self.get_logger().info(f"  Success rate:         {rate:.0f}%")
        self.get_logger().info(f"  Blockers αφαιρέθηκαν: {self._blocker_removed}")
        self.get_logger().info(f"  IK failures:          {self._ik_failures}")
        self.get_logger().info(f"  Συνολικός χρόνος:     {elapsed:.0f}s")
        self.get_logger().info("=" * 50)
        self._pub_state("task_complete")

    def _reset(self):
        """
        Reset της state machine για επόμενη αναζήτηση.
        ΔΕΝ μηδενίζει τους task counters (_red_grasped κλπ) —
        αυτοί μετράνε ΣΥΝΟΛΙΚΑ σε όλη τη διάρκεια εκτέλεσης
        και αρχικοποιούνται ΜΟΝΟ μία φορά στο __init__.
        """
        if self._send_timer:
            self._send_timer.cancel()
            self._send_timer = None
        self.state             = ST_SEARCHING
        self._pending_msg      = None
        self._send_count       = 0
        self._last_ik_time     = 0.0
        self._filtered_pos     = None
        self._stable_count     = 0
        self._on_done          = None
        self._last_red_time    = time.time()
        self._occluded_red_cam = None
        self._pub_state(ST_SEARCHING)
        self.get_logger().info(
            f"--- Reset: αναζητώ κόκκινο "
            f"(grasped so far: {self._red_grasped}/{GRASP_TARGET}) ---")


def main(args=None):
    rclpy.init(args=args)
    node = LaptopIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()