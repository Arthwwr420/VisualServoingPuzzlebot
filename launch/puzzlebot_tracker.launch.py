"""
puzzlebot_tracker.launch.py  (LQR Edition)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('puzzlebot_tracker'), 'config', 'params.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('mode',     default_value='hybrid'),
        DeclareLaunchArgument('debug',    default_value='true'),
        DeclareLaunchArgument('aruco_id', default_value='0'),
        LogInfo(msg='=== Puzzlebot LQR Visual Tracker iniciando ==='),
        Node(
            package='puzzlebot_tracker', executable='vision_tracker',
            name='vision_tracker', output='screen',
            parameters=[params_file, {
                'detection_mode':  LaunchConfiguration('mode'),
                'publish_debug':   LaunchConfiguration('debug'),
                'aruco_target_id': LaunchConfiguration('aruco_id'),
            }],
            remappings=[('/image_raw', '/camera/image_raw')],
        ),
        Node(
            package='puzzlebot_tracker', executable='lqr_controller',
            name='lqr_visual_controller', output='screen',
            parameters=[params_file],
            remappings=[('/cmd_vel', '/puzzlebot/cmd_vel')],
        ),
    ])
