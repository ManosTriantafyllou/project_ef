import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Bool


RED_RANGES = [
    (np.array([0,   90,  50]), np.array([10,  255, 255])),
    (np.array([170, 90,  50]), np.array([180, 255, 255])),
]
YELLOW_RANGES = [
    (np.array([20, 100, 100]), np.array([35, 255, 255])),
]
GREEN_RANGES = [
    (np.array([80,  55,  50]), np.array([100, 255, 255])),
]

ALL_COLORS = [
    ("red",    RED_RANGES,    (0,   0,   255)),
    ("yellow", YELLOW_RANGES, (0,   230, 230)),
    ("green",  GREEN_RANGES,  (0,   200, 0  ))
]

# Ελάχιστο εμβαδό για "ολόκληρο" κόκκινο κύβο σε pixels
RED_MIN_AREA = 900
CUBE_SIZE_M  = 0.024


class BlueCubeDetectorNode(Node):

    def __init__(self):
        super().__init__("blue_cube_detector_node")

        self.bridge = CvBridge()

        self.declare_parameter("min_contour_area", 300)
        self.declare_parameter("min_depth_m",      0.05)
        self.declare_parameter("max_depth_m",      1.00)

        self.fx = self.fy = self.cx = self.cy = None
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.camera_info_received = False
        self.depth_image   = None
        # Temporal smoothing για τη binary HSV mask ανά χρώμα — βλ.
        # _get_smoothed_mask. Κρατά ένα "running average" mask σε
        # float32 (0..255) ώστε μικρές pixel-level αλλαγές της raw
        # μάσκας (λόγω lighting flicker/sensor noise) να αποσβένονται
        # σε χρόνο, αντί να αλλάζουν το contour shape απότομα κάθε frame.
        self._mask_ema = {}   # color_name -> float32 array (0..255)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info",
            self.camera_info_callback, 10)
        self.create_subscription(
            Image, "/camera/camera/color/image_raw",
            self.image_callback, 10)
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback, 10)

        # ── Publishers ────────────────────────────────────────────────
        # Κόκκινο ΟΛΟΚΛΗΡΟ — laptop_ik_node πηγαίνει κατευθείαν εκεί
        self.pose_pub         = self.create_publisher(
            PoseStamped, "/object_pose", 10)
        # Κόκκινο ΜΙΣΟΚΡΥΜΜΕΝΟ — laptop_ik_node αφαιρεί blocker πρώτα
        self.occluded_pub     = self.create_publisher(
            PoseStamped, "/red_occluded", 10)
        # Όλα τα αντικείμενα για blocker selection + RViz
        self.markers_pub      = self.create_publisher(
            MarkerArray, "/detected_objects", 10)
        self.colors_pub       = self.create_publisher(
            String, "/detected_colors", 10)
        self.debug_pub        = self.create_publisher(
            Image, "/camera/debug/detection", 10)
        self.marker_pub       = self.create_publisher(
            Marker, "/blue_marker", 10)

        self.get_logger().info("DetectorNode (RGB-D multi-color) έτοιμο")

    # ─────────────────────────────────────────────────────────────────
    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs   = np.array(msg.d, dtype=np.float64)
        self.camera_info_received = True
        self.get_logger().info(
            f"Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}  "
            f"distortion_coeffs={self.dist_coeffs.tolist()}")

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def image_callback(self, msg: Image):
        if not self.camera_info_received or self.depth_image is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.detect_objects(frame)

    # ─────────────────────────────────────────────────────────────────
    def _get_depth(self, u, v, cube_radius_px=8):
        """
        Εκτιμά το depth στο κέντρο ενός κύβου από το depth image.

        Αλλαγές σε σχέση με την παλιά εκδοχή (±5px patch):
          1) Adaptive patch: χρησιμοποιεί cube_radius_px (εκτιμώμενο
             μισό-πλάτος κύβου σε pixels) για να κλιμακώσει το sampling
             window στο πραγματικό μέγεθος του κύβου στην εικόνα — σε
             μεγάλο depth (μικρός κύβος) μικρότερο patch, σε κοντινό
             depth (μεγάλος κύβος) μεγαλύτερο.
          2) Outlier rejection: αφαιρεί τα pixels που είναι >10mm πιο
             ΚΟΝΤΑ από τον median — αυτά αντιστοιχούν σε άκρες/σκιές
             που η κάμερα μετρά ψευδώς πιο κοντά από την επάνω επιφάνεια
             του κύβου, και θα σπρούσαν το median προς τα κάτω.
        """
        h, w = self.depth_image.shape
        r = max(4, min(cube_radius_px, 20))   # clamp σε [4, 20] px
        v0, v1 = max(0, v - r), min(h, v + r)
        u0, u1 = max(0, u - r), min(w, u + r)
        roi = self.depth_image[v0:v1, u0:u1]
        valid = roi[roi > 0].astype(np.float32)
        if len(valid) == 0:
            return -1.0
        median_mm = float(np.median(valid))
        # Κρατά μόνο pixels εντός 10mm από τον median (outlier rejection)
        valid = valid[np.abs(valid - median_mm) < 10.0]
        if len(valid) == 0:
            return median_mm / 1000.0
        return float(np.median(valid)) / 1000.0

    def _undistort_pixel(self, u, v):
        """
        Διορθώνει ένα pixel coordinate (u,v) για ραδιακή/εφαπτομενική
        παραμόρφωση φακού, πριν εφαρμοστεί ο τύπος pinhole projection
        (x_cam = depth*(u-cx)/fx). Χωρίς αυτή τη διόρθωση, η μετατροπή
        pixel→3D υποθέτει ιδανικό pinhole μοντέλο, με αποτέλεσμα σφάλμα
        θέσης που ΜΕΓΑΛΩΝΕΙ ακτινικά όσο το pixel απομακρύνεται από το
        κέντρο της εικόνας (cx,cy) — ακριβώς το σύμπτωμα "μικρή απόκλιση
        στο κέντρο, μεγαλύτερη στις άκρες" που παρατηρήθηκε στην πράξη.

        Επιστρέφει (u', v') — ισοδύναμο σημείο σε ΙΔΑΝΙΚΟ pinhole χώρο,
        έτοιμο να μπει στον υπάρχοντα τύπο x_cam = depth*(u'-cx)/fx.
        """
        if self.dist_coeffs is None or not np.any(self.dist_coeffs):
            # Μηδενικά distortion coefficients (π.χ. ήδη rectified
            # stream) — τίποτα να διορθωθεί, επέστρεψε ως έχει.
            return float(u), float(v)

        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(
            pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])

    # ─────────────────────────────────────────────────────────────────
    def _split_blob_into_cubes(self, cnt, mask, expected_single_px, n_estimate):
        """
        Σπάει ένα blob πολλαπλών ενωμένων κύβων στα πραγματικά κέντρα τους.

        Μέθοδος: distance transform πάνω στη μάσκα του blob (κάθε pixel
        παίρνει τιμή = απόσταση από το πλησιέστερο όριο). Τα κέντρα των
        κύβων αντιστοιχούν σε τοπικά μέγιστα αυτής της εικόνας — δηλαδή
        στα "πιο εσωτερικά" σημεία κάθε υπο-σχήματος. Σε αντίθεση με το
        grid-split, αυτό ανταποκρίνεται στην πραγματική γεωμετρία του
        blob ανεξάρτητα από τυχαία περιστροφή/διάταξη των κύβων.

        Επιστρέφει: λίστα από (u, v, area_px) — ένα ανά εντοπισμένο κύβο.
        """
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        if w_b <= 0 or h_b <= 0:
            return []

        # Μάσκα μόνο αυτού του blob (γέμισμα contour σε τοπικό ROI)
        pad = 3
        rx0, ry0 = max(0, x_b - pad), max(0, y_b - pad)
        rx1 = min(mask.shape[1], x_b + w_b + pad)
        ry1 = min(mask.shape[0], y_b + h_b + pad)

        local = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint8)
        shifted_cnt = cnt - np.array([[rx0, ry0]])
        cv2.drawContours(local, [shifted_cnt], -1, 255, thickness=cv2.FILLED)

        # Distance transform: κάθε pixel → απόσταση από το όριο
        dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)

        # Ελάχιστη απόσταση μεταξύ peaks ώστε να μη βρει 2 peaks στον ίδιο κύβο
        cube_half = max(3, int(math.sqrt(expected_single_px) * 0.35))

        # Διαστολή για εύρεση τοπικών μεγίστων: pixel == max στη γειτονιά του
        kernel_size = 2 * cube_half + 1
        local_max = cv2.dilate(dist, np.ones((kernel_size, kernel_size), np.uint8))
        peak_mask = (dist == local_max) & (dist > cube_half * 0.5)

        ys, xs = np.where(peak_mask)
        if len(xs) == 0:
            return []

        # Ομαδοποίηση κοντινών peaks (αν η dilation άφησε πλατό > 1 pixel)
        peaks = list(zip(xs.tolist(), ys.tolist(), dist[ys, xs].tolist()))
        peaks.sort(key=lambda p: -p[2])  # ισχυρότερα peaks πρώτα

        accepted = []
        min_sep = cube_half * 1.4
        for (px, py, pval) in peaks:
            too_close = False
            for (ax, ay, _) in accepted:
                if math.hypot(px - ax, py - ay) < min_sep:
                    too_close = True
                    break
            if not too_close:
                accepted.append((px, py, pval))
            if len(accepted) >= n_estimate:
                break

        centers = []
        for (px, py, _) in accepted:
            u = int(px + rx0)
            v = int(py + ry0)
            centers.append((u, v, int(expected_single_px)))
        return centers

    def _get_smoothed_mask(self, color_name, raw_mask, alpha=0.4):
        """
        Εφαρμόζει exponential moving average ΣΤΗΝ ΙΔΙΑ τη binary mask
        (πριν τον υπολογισμό contour), για να μειωθεί το frame-to-frame
        "τρεμόπαιγμα" του blob shape (και άρα του υπολογιζόμενου
        κέντρου) που προκαλείται από lighting flicker / sensor noise
        στο HSV thresholding.

        Λειτουργία: η raw_mask (0/255 binary) μετατρέπεται σε float,
        γίνεται EMA με την προηγούμενη smoothed μάσκα, και μετά
        ξαναγίνεται binary με threshold στο 127. Ένα pixel που
        "τρεμοπαίζει" (μπαίνει/βγαίνει από το threshold κάθε frame)
        θα παραμείνει στο >127 της smoothed μάσκας μόνο αν είναι
        ΣΥΝΗΘΩΣ μέσα, σταθεροποιώντας έτσι το συνολικό σχήμα.

        alpha=0.4 σημαίνει ότι η μάσκα προσαρμόζεται αρκετά γρήγορα σε
        ΠΡΑΓΜΑΤΙΚΗ κίνηση του κύβου (όχι σαν να είναι "κολλημένη"),
        αλλά αποσβένει το pixel-level jitter που είναι ασύμμετρο και
        τυχαίο frame-to-frame.
        """
        raw_f = raw_mask.astype(np.float32)
        if color_name not in self._mask_ema:
            self._mask_ema[color_name] = raw_f.copy()
        else:
            prev = self._mask_ema[color_name]
            if prev.shape != raw_f.shape:
                # Αλλαγή ανάλυσης εικόνας — ξεκίνα από την αρχή
                self._mask_ema[color_name] = raw_f.copy()
            else:
                self._mask_ema[color_name] = (1 - alpha) * prev + alpha * raw_f

        smoothed = self._mask_ema[color_name]
        _, binary = cv2.threshold(
            smoothed.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        return binary

    # ─────────────────────────────────────────────────────────────────
    def detect_objects(self, frame):
        min_area  = self.get_parameter("min_contour_area").value
        min_depth = self.get_parameter("min_depth_m").value
        max_depth = self.get_parameter("max_depth_m").value

        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Μεγαλύτερο kernel (ήταν 5x5) — πιο αποτελεσματικό OPEN/CLOSE
        # ώστε μικρά "κομμάτια" στα όρια του blob (λόγω lighting
        # flicker) να γεμίζουν/καθαρίζουν πιο αξιόπιστα, σταθεροποιώντας
        # το contour shape frame-to-frame.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        debug  = frame.copy()

        marker_array    = MarkerArray()
        detected_colors = []
        marker_id       = 0
        now_stamp       = self.get_clock().now().to_msg()

        # Καλύτερο κόκκινο
        best_red = None  # (depth, u, v, x, y, z, area_px)

        for color_name, ranges, dbg_bgr in ALL_COLORS:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lo, hi) in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # Temporal smoothing ΠΑΝΩ στη μάσκα — βλ. _get_smoothed_mask.
            # Αυτό μειώνει το frame-to-frame "τρεμόπαιγμα" του blob
            # shape, που μεταφράζεται σε σταθερότερο centroid (x,y).
            mask = self._get_smoothed_mask(color_name, mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color_found = False
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                if h_b == 0 or not (0.35 < w_b / float(h_b) < 3.0):
                    continue

                # ── Εκτίμηση depth πρόχειρα (στο κέντρο bounding box) ──
                # για να ξέρουμε πόσα px είναι ΕΝΑΣ κύβος σε αυτό depth
                u_rough = x_b + w_b // 2
                v_rough = y_b + h_b // 2
                depth_rough = self._get_depth(u_rough, v_rough)
                if depth_rough < 0:
                    continue

                expected_single_px = (self.fx * CUBE_SIZE_M / depth_rough) ** 2
                # Εκτιμώμενο μισό-πλάτος κύβου σε pixels — χρησιμοποιείται
                # για adaptive depth sampling patch στο _get_depth
                cube_radius_px = max(4, int(self.fx * CUBE_SIZE_M / depth_rough * 0.4))

                rect    = cv2.minAreaRect(cnt)
                box     = np.int32(cv2.boxPoints(rect))
                rw, rh  = rect[1]
                area_px = int(rw * rh)

                # ── Αν το contour είναι πολύ μεγάλο → πολλαπλοί ενωμένοι κύβοι ──
                # Χρήση distance-transform + local-maxima (mini watershed)
                # για να βρεθεί το ΠΡΑΓΜΑΤΙΚΟ κέντρο κάθε κύβου μέσα στο blob,
                # ανεξάρτητα από τη χωρική διάταξή τους (grid δεν δουλεύει
                # καλά όταν οι κύβοι είναι σε γωνία/τρίγωνο).
                n_cubes_est = max(1, round(area_px / expected_single_px))
                n_cubes_est = min(n_cubes_est, 6)  # ασφάλεια

                if n_cubes_est <= 1:
                    # minAreaRect center αντί για moments centroid:
                    # το minAreaRect προσαρμόζει ορθογώνιο στο ΣΥΝΟΛΙΚΟ
                    # σχήμα του contour (βασισμένο σε convex hull/
                    # ακραία σημεία), άρα είναι πολύ λιγότερο ευαίσθητο
                    # σε ένα μικρό "κομμάτι" που λείπει/περισσεύει σε
                    # μία γωνία του blob (π.χ. λόγω lighting flicker)
                    # απ' ό,τι το moments centroid, που σταθμίζει ΚΑΘΕ
                    # pixel του contour εξίσου.
                    cu, cv_ = rect[0]
                    cu, cv_ = int(round(cu)), int(round(cv_))
                    centers = [(cu, cv_, area_px)]
                    n_cubes = 1
                else:
                    centers = self._split_blob_into_cubes(
                        cnt, mask, expected_single_px, n_cubes_est)
                    n_cubes = len(centers) if centers else 1
                    if not centers:
                        M = cv2.moments(cnt)
                        cu = int(M["m10"]/M["m00"]) if M["m00"] else u_rough
                        cv_ = int(M["m01"]/M["m00"]) if M["m00"] else v_rough
                        centers = [(cu, cv_, area_px)]

                for (u, v, this_area_px) in centers:
                    depth_m = self._get_depth(u, v, cube_radius_px)
                    if depth_m < 0 or not (min_depth < depth_m < max_depth):
                        continue

                    # Διόρθωση lens distortion ΠΡΙΝ την pinhole projection.
                    # Το depth lookup παραπάνω χρησιμοποιεί το ωμό (u,v)
                    # γιατί το depth image είναι ακόμα στο distorted grid·
                    # μόνο η μετατροπή σε 3D χρειάζεται undistorted pixel.
                    u_ud, v_ud = self._undistort_pixel(u, v)

                    x_cam = depth_m * (u_ud - self.cx) / self.fx
                    y_cam = depth_m * (v_ud - self.cy) / self.fy
                    # Το depth_m μετρά την ΚΟΡΥΦΗ του κύβου (το πιο κοντινό
                    # σημείο στην bird-eye κάμερα). Για να στοχεύουμε το
                    # ΚΕΝΤΡΟ του κύβου (όπου ορίζεται η θέση του στο base
                    # frame), προσθέτουμε μισό ύψος κύβου στο z_cam — έτσι
                    # το z_cam αντιστοιχεί στο κέντρο μάζας του κύβου,
                    # και η μετατροπή camera→base δίνει το σωστό bz.
                    z_cam = depth_m + CUBE_SIZE_M / 2.0

                    color_found = True

                    if color_name == "red":
                        if best_red is None or depth_m < best_red[0]:
                            best_red = (depth_m, u, v, x_cam, y_cam, z_cam, this_area_px)

                    mk = Marker()
                    mk.header.stamp    = now_stamp
                    mk.header.frame_id = "camera_color_optical_frame"
                    mk.ns     = color_name
                    mk.id     = marker_id
                    mk.action = Marker.ADD
                    mk.type   = Marker.CUBE
                    mk.pose.position.x = float(x_cam)
                    mk.pose.position.y = float(y_cam)
                    mk.pose.position.z = float(z_cam)
                    mk.pose.orientation.w = 1.0
                    mk.scale.x = mk.scale.y = mk.scale.z = CUBE_SIZE_M
                    mk.color.r = dbg_bgr[2] / 255.0
                    mk.color.g = dbg_bgr[1] / 255.0
                    mk.color.b = dbg_bgr[0] / 255.0
                    mk.color.a = 0.9
                    mk.lifetime.sec = 1
                    marker_array.markers.append(mk)
                    marker_id += 1

                    cv2.circle(debug, (u, v), 5, dbg_bgr, -1)
                    cv2.putText(debug,
                        f"{color_name} {depth_m:.2f}m A={this_area_px}px",
                        (max(0,u-20), max(15,v-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dbg_bgr, 2)

                if n_cubes <= 1:
                    # Ένας κύβος → σχεδίασε το πραγματικό rotated rect του
                    cv2.drawContours(debug, [box], 0, dbg_bgr, 2)
                else:
                    # Πολλαπλοί ενωμένοι κύβοι → ΞΕΧΩΡΙΣΤΟ μικρό τετράγωνο
                    # γύρω από κάθε εντοπισμένο κέντρο (όχι ένα μεγάλο
                    # περίγραμμα που τα "ενώνει" οπτικά σε ένα σχήμα).
                    half = max(6, int(math.sqrt(expected_single_px) / 2))
                    for (u, v, _) in centers:
                        cv2.rectangle(debug,
                            (u - half, v - half), (u + half, v + half),
                            dbg_bgr, 2)
                    cv2.putText(debug, f"[{n_cubes}x split]",
                        (x_b, max(15, y_b - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            if color_found:
                detected_colors.append(color_name)

        # ── Κατάσταση κόκκινου ────────────────────────────────────────
        if best_red is None:
            # Δεν φαίνεται καθόλου κόκκινο
            cv2.putText(debug, "! ΔΕΝ ΕΝΤΟΠΙΖΕΤΑΙ ΚΟΚΚΙΝΟΣ",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            self.get_logger().debug("Δεν φαίνεται κόκκινο")

        else:
            depth_m, u, v, x_cam, y_cam, z_cam, area_px = best_red
            # Χωρίς επιπλέον smoothing εδώ — το variance-based stability
            # detection στο laptop_ik_node αναλαμβάνει το φιλτράρισμα.
            tvec = np.array([x_cam, y_cam, z_cam])

            pose = PoseStamped()
            pose.header.stamp    = now_stamp
            pose.header.frame_id = "camera_color_optical_frame"
            pose.pose.position.x = float(tvec[0])
            pose.pose.position.y = float(tvec[1])
            pose.pose.position.z = float(tvec[2])
            pose.pose.orientation.w = 1.0

            if area_px >= RED_MIN_AREA:
                # ── ΟΛΟΚΛΗΡΟ κόκκινο → /object_pose ─────────────────
                self.pose_pub.publish(pose)
                cv2.putText(debug, f"RED OK A={area_px}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
                self.get_logger().debug(
                    f"[red] cam=({tvec[0]:.3f},{tvec[1]:.3f},{tvec[2]:.3f}) "
                    f"depth={depth_m:.3f}m area={area_px}px")

                # Marker κόκκινου
                mk = Marker()
                mk.header = pose.header
                mk.ns = "red"; mk.id = 999
                mk.action = Marker.ADD; mk.type = Marker.CUBE
                mk.pose = pose.pose
                mk.scale.x = mk.scale.y = mk.scale.z = CUBE_SIZE_M
                mk.color.r = 1.0; mk.color.a = 0.9
                mk.lifetime.sec = 1
                self.marker_pub.publish(mk)

            else:
                # ── ΜΙΣΟΚΡΥΜΜΕΝΟ κόκκινο → /red_occluded ────────────
                self.occluded_pub.publish(pose)
                cv2.putText(debug,
                    f"RED OCCLUDED A={area_px}px < {RED_MIN_AREA}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 140, 255), 2)
                cv2.putText(debug, "Αφαιρεσε κυβο απο πανω",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 140, 255), 2)
                self.get_logger().debug(
                    f"[RED OCCLUDED] area={area_px}px < {RED_MIN_AREA}px")

        # ── Publish ───────────────────────────────────────────────────
        self.markers_pub.publish(marker_array)
        self.colors_pub.publish(String(data=str(detected_colors)))
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))


def main(args=None):
    try:
        rclpy.init(args=args)
        node = BlueCubeDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()