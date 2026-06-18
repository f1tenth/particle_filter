# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    pkg_share = get_package_share_directory('particle_filter')
    localize_config = os.path.join(pkg_share, 'config', 'localize.yaml')
    maps_dir = os.path.join(pkg_share, 'maps')

    try:
        cfg = yaml.safe_load(open(localize_config, 'r'))
        default_map = (cfg.get('map_server', {})
                          .get('ros__parameters', {})
                          .get('map', 'levine'))
    except Exception:
        default_map = 'levine'

    localize_la = DeclareLaunchArgument(
        'localize_config',
        default_value=localize_config,
        description='Path to localization config YAML')
    map_name_la = DeclareLaunchArgument(
        'map_name',
        default_value=default_map,
        description='Map name (no extension) inside particle_filter/maps/; '
                    'overrides the map_server.ros__parameters.map entry in localize.yaml')

    ld = LaunchDescription([localize_la, map_name_la])

    pf_node = Node(
        package='particle_filter',
        executable='particle_filter',
        name='particle_filter',
        parameters=[LaunchConfiguration('localize_config')]
    )

    # PythonExpression concatenates the maps directory (resolved at launch-file load
    # time) with the overridable map_name launch argument and the .yaml suffix.
    map_yaml_path = PythonExpression(
        ["'", maps_dir + '/', "' + '", LaunchConfiguration('map_name'), "' + '.yaml'"]
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_yaml_path},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'},
                    {'use_sim_time': True}]
    )

    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    ld.add_action(pf_node)

    return ld
