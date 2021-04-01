from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bag = LaunchConfiguration('bag')

    return LaunchDescription([
        Node(
            package='seif_ros',
            executable='seif',
            name='seif',
            namespace='seif',
            output='screen'
        ),
        ExecuteProcess(cmd=['ros2', 'bag', 'play', bag], output='screen'),
    ])

