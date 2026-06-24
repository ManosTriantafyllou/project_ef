import math
import rclpy
import tf2_ros
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from cube_follow_pkg.cube_utils.solve_inv_kine import solve_ik
from cube_follow_pkg.cube_utils.myarm_connect import connect
from cube_follow_pkg.cube_utils.math_utils import quat_to_rot


# ── State machine states ───────────────────────────────────────────────────
STATE_IDLE      = "IDLE"       # αναμονή για κύβο
STATE_HOVER     = "HOVER"      # πήγαινε πάνω από τον κύβο
STATE_DESCEND   = "DESCEND"    # κατέβα στον κύβο
STATE_GRASP     = "GRASP"      # κλείσε gripper
STATE_LIFT      = "LIFT"       # σήκωσε τον κύβο
STATE_PLACE     = "PLACE"      # πήγαινε στη θέση τοποθέτησης
STATE_RELEASE   = "RELEASE"    # άνοιξε gripper
STATE_RETURN    = "RETURN"     # επέστρεψε σε home


class MyArmCubeNode(Node):
    """
    Εκτεταμένος myarm_node για Project 3.
    Παίρνει pose κύβου από /object_pose (από blue_cube_detector_node)
    και εκτελεί ακολουθία: hover → descend → grasp → lift → place → release.
    """

    def __init__(self):
        super().__init__("myarm_cube_node")

        # ── TF ────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("port",           "/dev/ttyAMA0")
        self.declare_parameter("baudrate",        115200)
        self.declare_parameter("speed",           60)
        self.declare_parameter("hover_height_m",  0.01)   # ύψος hover πάνω από κύβο
        self.declare_parameter("descend_height_m", 0.01)  # ύψος κατά grasp
        self.declare_parameter("lift_height_m",   0.15)   # ύψος μετά από grasp
        # Θέση τοποθέτησης κύβου (σε world frame, τροποποίησε ανάλογα)
        self.declare_parameter("place_x", 0.20)
        self.declare_parameter("place_y", 0.10)
        self.declare_parameter("place_z", 0.05)

        port     = self.get_parameter("port").value
        baudrate = self.get_parameter("baudrate").value
        self.speed = self.get_parameter("speed").value

        # ── Σύνδεση με βραχίονα ───────────────────────────────────────
        self.get_logger().info(f"Σύνδεση MyArm → port:{port} baud:{baudrate}")
        self.myarm = connect(port, baudrate, 1.0)

        # ── Publisher: joint states για RViz ──────────────────────────
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.timer     = self.create_timer(0.05, self.publish_joint_states)

        # ── Subscriber: pose κύβου από detector ───────────────────────
        self.cube_sub = self.create_subscription(
            PoseStamped,
            "/object_pose",
            self.cube_pose_callback,
            10
        )

        # ── Εσωτερική κατάσταση ───────────────────────────────────────
        self.state     = STATE_IDLE
        self.q0_list   = None
        self.last_cube_pose = None   # τελευταία γνωστή 3D θέση κύβου

        # Ανοιξε gripper κατά την εκκίνηση
        self._set_gripper(open=True)

        self.get_logger().info("MyArmCubeNode έτοιμο — περιμένει κύβο...")

    # ──────────────────────────────────────────────────────────────────
    # Joint state publisher (ίδιος με lab6)
    # ──────────────────────────────────────────────────────────────────
    def publish_joint_states(self):
        try:
            angles_deg = self.myarm.get_angles()
        except Exception:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ["joint1","joint2","joint3","joint4",
                        "joint5","joint6","joint7"]
        msg.position = [round(math.radians(a), 4) for a in angles_deg]
        self.joint_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    # Κύρια callback — έρχεται κάθε φορά που ο detector βλέπει κύβο
    # ──────────────────────────────────────────────────────────────────
    def cube_pose_callback(self, msg: PoseStamped):
        self.speed = self.get_parameter("speed").value

        # ── Α: Μετασχηματισμός camera frame → myarm_base_frame ────────
        try:
            transform = self.tf_buffer.lookup_transform(
                "myarm_base_frame",
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF lookup απέτυχε: {e}")
            return

        t  = transform.transform.translation
        t_cb = np.array([t.x, t.y, t.z])

        rq   = transform.transform.rotation
        R_cb = quat_to_rot(np.array([rq.x, rq.y, rq.z, rq.w]))

        p_cam  = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        p_base = R_cb @ p_cam + t_cb   # κύβος σε base frame

        cube_x, cube_y, cube_z = p_base
        self.last_cube_pose = p_base
        self.get_logger().info(
            f"cube sto robot frame:x={cube_x:.3f}, y={cube_y:.3f}, z={cube_z:.3f}"
        )

        # ── Β: State machine ───────────────────────────────────────────
        if self.state == STATE_IDLE:
            self.get_logger().info(
                f"Κύβος εντοπίστηκε στο ({cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}) m"
            )
            self.state = STATE_HOVER
            self._move_to(cube_x, cube_y,
                          cube_z + self.get_parameter("hover_height_m").value,
                          next_state=STATE_DESCEND)

        elif self.state == STATE_DESCEND:
            self._move_to(cube_x, cube_y,
                          cube_z + self.get_parameter("descend_height_m").value,
                          next_state=STATE_GRASP)

        elif self.state == STATE_GRASP:
            self._set_gripper(open=False)
            self.get_logger().info("Gripper έκλεισε — πιάσιμο κύβου")
            self.state = STATE_LIFT
            lift_z = cube_z + self.get_parameter("lift_height_m").value
            self._move_to(cube_x, cube_y, lift_z,
                          next_state=STATE_PLACE)

        elif self.state == STATE_PLACE:
            px = self.get_parameter("place_x").value
            py = self.get_parameter("place_y").value
            pz = self.get_parameter("place_z").value
            self._move_to(px, py, pz, next_state=STATE_RELEASE)

        elif self.state == STATE_RELEASE:
            self._set_gripper(open=True)
            self.get_logger().info("Gripper άνοιξε — κύβος τοποθετήθηκε")
            self.state = STATE_RETURN
            self._go_home()

    # ──────────────────────────────────────────────────────────────────
    # Βοηθητικές μέθοδοι
    # ──────────────────────────────────────────────────────────────────
    def _move_to(self, x: float, y: float, z: float, next_state: str):
        """
        Κατασκευάζει Twbd και καλεί IK — ίδια λογική με lab6 follow_obstacle.
        Στροφή 180° γύρω από y: end-effector κοιτά προς τα κάτω.
        """
        Ry180 = np.array([[-1., 0.,  0.],
                           [ 0., 1.,  0.],
                           [ 0., 0., -1.]])
        Twbd = np.eye(4)
        Twbd[:3, :3] = Ry180
        Twbd[0, 3]   = x
        Twbd[1, 3]   = y
        Twbd[2, 3]   = z

        self.get_logger().info(
            f"IK → ({x:.3f}, {y:.3f}, {z:.3f})  state={next_state}"
        )

        q_rad, success = solve_ik(Twbd, "space", self.q0_list, 1e-5)

        if success:
            q_deg = [math.degrees(q) for q in q_rad]
            self.myarm.send_angles(q_deg, self.speed)
            self.q0_list = [
                np.copy(q_rad),
                np.array([-1.557, 1.177, -0.532,
                          -1.209, 0.612, -0.940, -2.180])
            ]
            self.state = next_state
            self.get_logger().info(f"IK επιτυχία → state={self.state}")
        else:
            self.get_logger().error(
                f"IK απέτυχε για ({x:.3f},{y:.3f},{z:.3f}) — "
                "επέστρεψα σε IDLE"
            )
            self.state = STATE_IDLE

    def _set_gripper(self, open: bool):
        """Άνοιγμα/κλείσιμο gripper."""
        value = 100 if open else 0
        try:
            self.myarm.set_gripper_value(value, 50)
        except Exception as e:
            self.get_logger().warn(f"Gripper σφάλμα: {e}")

    def _go_home(self):
        """Επιστροφή σε ασφαλή home θέση."""
        home_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            self.myarm.send_angles(home_angles, self.speed)
        except Exception as e:
            self.get_logger().warn(f"Home σφάλμα: {e}")
        self.state   = STATE_IDLE
        self.q0_list = None
        self.get_logger().info("Επέστρεψα σε home — STATE_IDLE")


def main(args=None):
    rclpy.init(args=args)
    node = MyArmCubeNode()
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
