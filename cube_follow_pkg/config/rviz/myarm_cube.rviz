Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views

Visualization Manager:
  Displays:

    # ── 1. Βραχίονας (URDF) ───────────────────────────────────────────
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: true
      Description Topic:
        Value: /robot_description

    # ── 2. TF frames ──────────────────────────────────────────────────
    # Δείχνει τα axes: myarm_base_frame, camera_link, joint1..7
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
      Show Axes: true
      Show Names: true
      Marker Scale: 0.3

    # ── 3. Κόκκινο αντικείμενο (Marker) ──────────────────────────────
    # Εμφανίζεται ως κόκκινος κύβος/σφαίρα στη θέση που το βλέπει η κάμερα
    # ΣΗΜΑΝΤΙΚΟ: φαίνεται σε σχέση με το camera_link frame
    # Μετά το TF μετατρέπεται σε myarm_base_frame
    - Class: rviz_default_plugins/Marker
      Name: Κόκκινο Αντικείμενο
      Enabled: true
      Topic:
        Value: /blue_marker

    # ── 4. PoseStamped — θέση αντικειμένου ───────────────────────────
    # Δείχνει ένα βέλος από τη βάση προς το αντικείμενο
    - Class: rviz_default_plugins/Pose
      Name: Θέση Αντικειμένου
      Enabled: true
      Topic:
        Value: /object_pose
      Shape: Arrow
      Color:
        R: 255
        G: 0
        B: 0

    # ── 5. Εικόνα κάμερας με detection overlay ────────────────────────
    # Δείχνει τι βλέπει η κάμερα με πράσινο/κόκκινο περίγραμμα
    - Class: rviz_default_plugins/Image
      Name: Detection (κάμερα)
      Enabled: true
      Topic:
        Value: /camera/debug/detection

    # ── 6. Αρχεία εικόνας raw (προαιρετικό) ──────────────────────────
    - Class: rviz_default_plugins/Image
      Name: Camera Raw
      Enabled: false
      Topic:
        Value: /camera/image_raw

    # ── 7. Grid ───────────────────────────────────────────────────────
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Plane Cell Count: 10
      Cell Size: 0.1
      Color:
        R: 160
        G: 160
        B: 160

  Global Options:
    Fixed Frame: myarm_base_frame   # όλα φαίνονται σε σχέση με τη βάση
    Background Color:
      R: 48
      G: 48
      B: 48
    Frame Rate: 30

  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Orbit
      Distance: 1.5
      Pitch: 0.5
      Yaw: 0.8