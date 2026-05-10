from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'puzzlebot_tracker'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Puzzlebot Team',
    maintainer_email='team@puzzlebot.com',
    description='Closed-loop visual servoing controller for Puzzlebot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_tracker    = puzzlebot_tracker.vision_tracker:main',
            'pid_controller    = puzzlebot_tracker.pid_controller:main',
        ],
    },
)
