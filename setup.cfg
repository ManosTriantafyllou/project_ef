"""
robot_receiver_node.py
======================
Τρέχει ΜΟΝΟ στο ρομπότ (Pi).

Subscribes:  /joint_commands  (sensor_msgs/JointState)
Publishes:   /joint_states    (sensor_msgs/JointState)  — για RViz

Timestamp-based debounce: εκτελεί κάθε μήνυμα μόνο μία φορά
(βάσει header.stamp) — έτσι ο laptop μπορεί να στέλνει πολλές φορές
για αξιοπιστία χωρίς ο βραχίονας να ακυρώνει την κίνησή του.
"""

import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from cube_follow_pkg.cube_utils.myarm_connect import connect

# ── GPIO Pins για Vacuum Sucker ───────────────────────────────────
PIN_PUMP  = 26
PIN_VALVE = 20

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    # Το test έδειξε ότι τα pins ανάβουν με LOW (Active-LOW).
    # Άρα για να είναι σβηστά στην αρχή, τα αρχικοποιούμε σε HIGH.
    GPIO.setup(PIN_PUMP,  GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(PIN_VALVE, GPIO.OUT, initial=GPIO.HIGH)
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    # Στο laptop / simulation δεν υπάρχει RPi.GPIO
    GPIO_AVAILABLE = False


class RobotReceiverNode(Node):

    def __init__(self):
        super().__init__("robot_receiver_node")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("port",     "/dev/ttyAMA0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("speed",    60)

        port          = self.get_parameter("port").value
        baudrate      = self.get_parameter("baudrate").value
        self.speed    = self.get_parameter("speed").value

        # ── Σύνδεση με MyArm ─────────────────────────────────────────
        self.get_logger().info(f"Σύνδεση MyArm → port={port}  baud={baudrate}")
        self.myarm = connect(port, baudrate, 1.0)
        self.get_logger().info("MyArm συνδέθηκε!")

        # ── Timestamp του τελευταίου μηνύματος που εκτελέστηκε ────────
        # Χρησιμοποιούμε (sec, nanosec) από header.stamp ως μοναδικό ID.
        # Αν το ίδιο timestamp έρθει ξανά → duplicate → αγνοούμε.
        self._last_stamp = (-1, -1)

        # ── Subscribers ────────────────────────────────────────────────
        self.joint_sub = self.create_subscription(
            JointState, "/joint_commands",
            self.joint_cmd_callback, 10)

        self.gripper_sub = self.create_subscription(
            String, "/gripper_cmd",
            self._gripper_cmd_callback, 10)

        # ── Publisher: joint states για RViz ──────────────────────────
        self.joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10)
        self.create_timer(0.05, self._publish_joint_states)

        self.get_logger().info(
            "RobotReceiverNode έτοιμο — περιμένει /joint_commands & /gripper_cmd ...")
        if GPIO_AVAILABLE:
            self.get_logger().info(f"GPIO ενεργό: pump=GPIO{PIN_PUMP}, valve=GPIO{PIN_VALVE}")

    # ─────────────────────────────────────────────────────────────────
    def _gripper_cmd_callback(self, msg: String):
        self.get_logger().info(f">>> ΕΛΑΒΑ ΜΗΝΥΜΑ ΣΤΟ /gripper_cmd: '{msg.data}' <<<")
        cmd = msg.data.strip().upper()
        if cmd == "ON":
            self._set_vacuum(True)
        elif cmd == "OFF":
            self._set_vacuum(False)
        else:
            self.get_logger().warn(f"Άγνωστη εντολή: '{cmd}'")

    def _set_vacuum(self, active: bool):
        state_str = "ON" if active else "OFF"
        if GPIO_AVAILABLE:
            if active:
                # GRASP (Πιάσιμο): Αντλία ON (LOW), Βαλβίδα Κλειστή (HIGH)
                GPIO.output(PIN_PUMP,  GPIO.LOW)
                GPIO.output(PIN_VALVE, GPIO.HIGH)
            else:
                # DROP (Άφημα): Αντλία OFF (HIGH), Βαλβίδα Ανοιχτή (LOW)
                GPIO.output(PIN_PUMP,  GPIO.HIGH)
                GPIO.output(PIN_VALVE, GPIO.LOW)
                
                # Κλείνουμε τη βαλβίδα μετά από 2 δευτερόλεπτα (γίνεται HIGH)
                import threading
                threading.Timer(2.0, lambda: GPIO.output(PIN_VALVE, GPIO.HIGH)).start()

            self.get_logger().info(f"[VACUUM] {state_str}")
        else:
            self.get_logger().info(f"[VACUUM] {state_str} (simulation)")

    # ─────────────────────────────────────────────────────────────────
    def joint_cmd_callback(self, msg: JointState):
        # Timestamp-based deduplication
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self._last_stamp:
            self.get_logger().debug("Duplicate μήνυμα — αγνοώ")
            return
        self._last_stamp = stamp

        if len(msg.position) < 7:
            self.get_logger().warn(
                f"Έλαβα {len(msg.position)} joints, αναμένω 7 — αγνοώ")
            return

        q_deg = [round(math.degrees(q), 2) for q in msg.position[:7]]
        self.get_logger().info(f"Εκτέλεση joints (deg): {q_deg}")

        try:
            self.myarm.send_angles(q_deg, self.speed)
            time.sleep(0.1)   # χρόνος για serial buffer
        except Exception as e:
            self.get_logger().error(f"send_angles σφάλμα: {e}")

    # ─────────────────────────────────────────────────────────────────
    def _publish_joint_states(self):
        try:
            angles_deg = self.myarm.get_angles()
        except Exception:
            return

        # get_angles() μπορεί να επιστρέψει int (error code) αντί για list
        if not isinstance(angles_deg, (list, tuple)):
            return
        if len(angles_deg) < 7:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = ["joint1","joint2","joint3","joint4",
                            "joint5","joint6","joint7"]
        msg.position     = [round(math.radians(a), 4) for a in angles_deg]
        self.joint_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if GPIO_AVAILABLE:
            GPIO.output(PIN_PUMP, GPIO.HIGH)
            GPIO.output(PIN_VALVE, GPIO.HIGH)
            GPIO.cleanup()


if __name__ == "__main__":
    main()