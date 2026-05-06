import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('ras598_assignment_3'),
        'rviz', 'bayes.rviz'
    )

    return LaunchDescription([
        # Launch Stage simulator
        ExecuteProcess(
            cmd=['ros2', 'launch', 'stage_ros2', 'demo.launch.py',
                 'world:=cave', 'use_stamped_velocity:=false'],
            output='screen'
        ),

        # Launch RViz with our config
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config],
            output='screen'
        ),

        # Launch our Bayes filter node
        Node(
            package='ras598_assignment_3',
            executable='bayes_filter',
            name='bayes_filter_3d_node',
            output='screen'
        ),
    ])
