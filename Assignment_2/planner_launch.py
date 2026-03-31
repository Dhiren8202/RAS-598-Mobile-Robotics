import sys
import os
from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node
from launch.actions import ExecuteProcess  # <--- Added this

def main():
    # Helper to handle the home directory path
    home = os.path.expanduser('~')
    map_yaml_path = os.path.join(home, 'ros_ws/src/ras598_assignment_2/map.yaml')
    scout_script_path = os.path.join(home, 'ros_ws/src/ras598_assignment_2/grading_scout.py')

    ld = LaunchDescription([
        # Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{
                'yaml_filename': map_yaml_path,
                'use_sim_time': True
            }]
        ),

        # Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            parameters=[{'autostart': True, 'node_names': ['map_server']}]
        ),

        # Grading Scout (External Python Script)
        ExecuteProcess(
            cmd=['python3', scout_script_path],
            output='screen'
        ),
    ])

    # Initialize and run
    ls = LaunchService()
    ls.include_launch_description(ld)
    
    print("--- Starting ROS 2 Launch Service ---")
    return ls.run()

if __name__ == '__main__':
    sys.exit(main())