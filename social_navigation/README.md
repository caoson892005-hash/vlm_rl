# social_navigation

ROS 2 Humble package for Gazebo Classic and Nav2. It provides:

- a Nav2 costmap plugin that writes social costs into global/local costmaps.
- a last-line velocity safety filter consuming localized people;
- simulation actors used to exercise social navigation.

Camera perception is intentionally owned by the separate `social_perception`
package. This package consumes its stable `/people` and `/people_groups` topics;
it does not load YOLO, depth images, or the VLM.

## Build and run

```bash
cd ~/ninorobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to social_navigation
source install/setup.bash
```

Four terminals, in this order. Releasing the actors is deliberately its own
command rather than part of perception: on real hardware there is nothing to
spawn, and in simulation the moment people walk in is worth keeping in your own
hands instead of tying it to model loading.

```bash
# 1. Gazebo. The default lirs_test.world is person-free.
#    Do not add run_ekf:=false here. The diff_drive plugin sets
#    <publish_odom_tf>false</publish_odom_tf>, so the EKF is the only thing
#    publishing odom -> base_footprint; without it the `odom` frame never
#    exists and step 4 cannot navigate at all.
ros2 launch linorobot2_gazebo gazebo.launch.py

# 2. Perception only: YOLO + depth + VLM. Loads no actors.
#    Wait for "SẴN SÀNG: camera + YOLO hoạt động, VLM đã nạp xong".
ros2 launch social_navigation social_bringup.launch.py rviz:=true

# 3. Put people into the scene, whenever you want them.
ros2 launch social_navigation social_sim.launch.py scenario:=talking

# 4. Nav2 with the social costmap layer. Do not pass rviz:=true here as well.
ros2 launch linorobot2_navigation navigation.launch.py sim:=true \
    map:=<path to map.yaml>
```

`scenario:=talking` (the default) releases `m_sweater` and `m_mechanic` into the
already-running Gazebo world. They stand 1.6 m apart, face each other, and loop
their `talk.dae` skeletal animations for the whole session.

`scenario:=gathering` runs the same two actors through a repeating cycle
instead: they walk in from outside the camera's view, hold the conversation,
walk back out, and stay hidden for a while before returning. A static pair only
proves the costmap can *add* a social region; this scenario also proves it
removes it once the conversation ends.

Keep terminal 3 running while the actors should be visible; `Ctrl+C` there hides
them again without disturbing perception, so you can release the other scenario
straight afterwards without reloading the VLM. Add `wait_for_perception:=true`
if you would rather start terminals 2 and 3 together and let the actors hold
back until perception reports ready.

This package publishes nothing on `/people`; terminal 2 is what produces it.
On real hardware terminal 3 disappears and terminal 2 becomes
`social_bringup.launch.py sim:=false` plus the camera's pose in the map frame.

`lirs_test.world` provides the `/dataset_camera` RGB-D camera for the
`social_perception` node, and defines each actor's pose and animation. Edit that
world file to add people or change where they stand.

In RViz add a `MarkerArray` display with topic `/social_spaces`. The actual planning
cost is part of `/global_costmap/costmap` and `/local_costmap/costmap`.

## Topics

- `/people` (`social_perception/msg/People`)
- `/people_groups` (`social_perception/msg/Groups`)
- `/social_spaces` (`visualization_msgs/msg/MarkerArray`)

Only regions confirmed as conversations by `social_perception` are emitted on
`/people_groups`. The costmap applies each region's supplied center and O/P/R
radii directly.
