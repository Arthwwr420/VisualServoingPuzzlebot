"""
puzzlebot_tracker.launch.py
============================
Lanza los dos nodos principales del sistema de seguimiento:
  1. vision_tracker   — percepción y detección
  2. visual_servo_controller — control PID

Uso:
  ros2 launch puzzlebot_tracker puzzlebot_tracker.launch.py
  ros2 launch puzzlebot_tracker puzzlebot_tracker.launch.py mode:=aruco
  ros2 launch puzzlebot_tracker puzzlebot_tracker.launch.py debug:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Argumentos en línea de comandos ──
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='hybrid',
        description='Modo de detección: aruco | hsv | hybrid'
    )
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Publicar imagen de debug en /detection/debug'
    )
    aruco_id_arg = DeclareLaunchArgument(
        'aruco_id',
        default_value='0',
        description='ID del marcador ArUco a seguir'
    )

    # ── Ruta al archivo de parámetros ──
    params_file = PathJoinSubstitution([
        FindPackageShare('puzzlebot_tracker'),
        'config',
        'params.yaml'
    ])

    # ── Nodo de visión ──
    vision_node = Node(
        package='puzzlebot_tracker',
        executable='vision_tracker',
        name='vision_tracker',
        output='screen',
        parameters=[
            params_file,
            {
                'detection_mode': LaunchConfiguration('mode'),
                'publish_debug':  LaunchConfiguration('debug'),
                'aruco_target_id': LaunchConfiguration('aruco_id'),
            }
        ],
        remappings=[
            ('/image_raw', '/camera/image_raw'),   # Ajusta al topic real de tu cámara
        ]
    )

    # ── Nodo de control ──
    controller_node = Node(
        package='puzzlebot_tracker',
        executable='pid_controller',
        name='visual_servo_controller',
        output='screen',
        parameters=[params_file],
        remappings=[
            ('/cmd_vel', '/puzzlebot/cmd_vel'),    # Ajusta al topic de tu Puzzlebot
        ]
    )

    return LaunchDescription([
        mode_arg,
        debug_arg,
        aruco_id_arg,
        LogInfo(msg='=== Puzzlebot Visual Tracker iniciando ==='),
        vision_node,
        controller_node,
    ])
