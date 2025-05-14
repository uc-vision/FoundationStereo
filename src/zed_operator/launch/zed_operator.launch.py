from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='zed_operator',
            executable='zed_operator',
            name='zed_operator',
            output='screen',
            parameters=[{}]
        )
    ])