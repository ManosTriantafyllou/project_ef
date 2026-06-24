import os
from glob import glob
from setuptools import setup, find_packages

package_name = "cube_follow_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        (os.path.join("share", package_name, "config", "rviz"),
            glob(os.path.join("config", "rviz", "*.rviz"))),
        (os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "urdf"),
            glob(os.path.join("urdf", "*.urdf"))),
        (os.path.join("share", package_name, "meshes"),
            glob(os.path.join("meshes", "*.dae"))),
        # JSON με παραμέτρους κάμερας
        (os.path.join("share", package_name, "config"),
            glob(os.path.join("cube_follow_pkg", "camera_params.json"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Project 3 — Blue cube detection + MyArm grasping",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"detector_node  = {package_name}.detector_node:main",
            f"myarm_cube_node         = {package_name}.myarm_cube_node:main",
            # ── Νέοι nodes για laptop ↔ robot over WiFi ──────────────
            f"laptop_ik_node          = {package_name}.laptop_ik_node:main",
            f"robot_receiver_node     = {package_name}.robot_receiver_node:main",
        ],
    },
)
