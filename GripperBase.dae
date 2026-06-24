"""
laptop_launch.py — RealSense D435i + ROS2 Jazzy
"""
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import ExecuteProcess

pkg_name = "cube_follow_pkg"

def generate_launch_description():

    urdf_file   = PathJoinSubstitution(
        [FindPackageShare(pkg_name), "urdf", "myarm_300_pi.urdf"])
    rviz_config = PathJoinSubstitution(
        [FindPackageShare(pkg_name), "config", "rviz", "myarm_cube.rviz"])

    return LaunchDescription([

        # ── 1. RealSense D435i ────────────────────────────────────────
        Node(
            package    = "realsense2_camera",
            executable = "realsense2_camera_node",
            name       = "camera",
            output     = "screen",
            parameters = [{
                "enable_color":  True,
                "enable_depth":  True,
                "enable_infra1": False,
                "enable_infra2": False,
                # Σωστά profile names για v4.x
                "rgb_camera.color_profile":   "848x480x30",
                "depth_module.depth_profile": "848x480x30",
                # Align depth στο color frame
                "align_depth.enable": True,
                # IMU off
                "enable_gyro":  False,
                "enable_accel": False,
                "pointcloud.enable": False,
            }],
        ),

        # ── 2. Detector Node ──────────────────────────────────────────
        Node(
            package    = pkg_name,
            executable = "detector_node",
            name       = "detector_node",
            output     = "screen",
        ),

        # ── 3. Laptop IK Node ─────────────────────────────────────────
        Node(
            package    = pkg_name,
            executable = "laptop_ik_node",
            name       = "laptop_ik_node",
            output     = "screen",
            parameters = [{
                "hover_height_m": 0.153,
            }],
        ),

        # ── 4. Robot State Publisher ──────────────────────────────────
        Node(
            package    = "robot_state_publisher",
            executable = "robot_state_publisher",
            output     = "screen",
            parameters = [{
                "robot_description": ParameterValue(
                    Command(["cat ", urdf_file]), value_type=str),
            }],
        ),


        Node(
            package    = "tf2_ros",
            executable = "static_transform_publisher",
            name       = "camera_to_base_tf",
            arguments  = [
                # x       y       z      roll    pitch   yaw
                "-0.20", "-0.10", "0.45", "3.1416", "0", "-1.5708",
                "myarm_base_frame",            # parent
                "camera_color_optical_frame"   # child
            ],
        ),

        # ── 5. RViz με σωστό config ───────────────────────────────────
        Node(
            package    = "rviz2",
            executable = "rviz2",
            name       = "rviz2",
            arguments  = ["-d", rviz_config],
            output     = "screen",
        ),

    ])

