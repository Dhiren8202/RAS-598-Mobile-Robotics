import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    home = os.path.expanduser("~")
    map_yaml_path     = os.path.join(home, "ros_ws/src/ras598_assignment_2/map.yaml")
    scout_script_path = os.path.join(home, "ros_ws/src/ras598_assignment_2/grading_scout.py")
    rviz_config_path  = os.path.join(home, "ros_ws/src/ras598_assignment_2/planning.rviz")
    world_dir         = os.path.join(home, "ros_ws/src/stage_ros2/world")

    return LaunchDescription([

        # Stage Simulator
        ExecuteProcess(
            cmd=["ros2", "run", "stage_ros2", "stage_ros2", "cave.world"],
            cwd=world_dir,
            output="screen"
        ),

        # Map Server
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            parameters=[{"yaml_filename": map_yaml_path}]
        ),

        # Lifecycle Manager
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager",
            output="screen",
            parameters=[{"autostart": True, "node_names": ["map_server"]}]
        ),

        # RViz
        ExecuteProcess(
            cmd=["rviz2", "-d", rviz_config_path],
            output="screen"
        ),

        # Grading Scout
        ExecuteProcess(
            cmd=["python3", scout_script_path],
            output="screen"
        ),

        # Our Planner Node
        Node(
            package="ras598_assignment_2",
            executable="planner",
            name="path_planner",
            output="screen"
        ),
    ])
