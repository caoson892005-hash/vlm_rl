# social_navigation

ROS 2 Humble package for Gazebo Classic and Nav2. It provides:

- ground-truth tracking of `person_*` Gazebo models;
- arbitrary time-stamped waypoint motion;
- proximity-based group detection;
- O-P-R group markers and asymmetric individual-space markers;
- a Nav2 costmap plugin that writes social costs into global/local costmaps.

## Build and run

```bash
cd ~/ninorobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select social_navigation
source install/setup.bash
```

Start Gazebo and Nav2 using the existing project launch files, then start people:

```bash
ros2 launch social_navigation social_sim.launch.py
```

Set `spawn_people:=false` when the models already exist, or `move_people:=false`
to keep their current Gazebo poses. Edit `config/people_paths.yaml` to define paths.

In RViz add a `MarkerArray` display with topic `/social_spaces`. The actual planning
cost is part of `/global_costmap/costmap` and `/local_costmap/costmap`.

## Topics

- `/people` (`social_navigation/msg/People`)
- `/people_groups` (`social_navigation/msg/Groups`)
- `/social_spaces` (`visualization_msgs/msg/MarkerArray`)

The included group detector is a simulation baseline based on distance. For research
evaluation, replace it with an F-formation detector using orientation and temporal
tracking while keeping the same `/people_groups` interface.
