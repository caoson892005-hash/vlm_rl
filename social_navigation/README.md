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

The RGB-D camera feeding `social_perception` rides on the robot: the
`depth_sensor` macro in `linorobot2_description/urdf/robots/2wd.urdf.xacro`,
publishing `/camera/color/image_raw` and `/camera/depth/image_rect_raw` in
frame `camera_depth_link`. It replaced the static `/dataset_camera` model that
used to sit on the wall in `lirs_test.world`; that world file now only defines
each actor's pose and animation. Edit it to add people or change where they
stand.

Because the camera moves with the robot, people are only detected while the
robot is facing them, and `target_frame: map` needs the `map -> odom` edge that
AMCL publishes — run `navigation.launch.py` alongside perception.

In RViz add a `MarkerArray` display with topic `/social_spaces`. The actual planning
cost is part of `/global_costmap/costmap` and `/local_costmap/costmap`.

## Topics

- `/people` (`social_perception/msg/People`)
- `/people_groups` (`social_perception/msg/Groups`)
- `/social_spaces` (`visualization_msgs/msg/MarkerArray`)

Only regions confirmed as conversations by `social_perception` are emitted on
`/people_groups`. The costmap applies each region's supplied center and O/P/R
radii directly.

## Automatic Nav2/RL handoff

For “select a Nav2 goal, let RL drive, and avoid any tracked people”, launch
`social_rl/nav2_rl_handoff.launch.py` instead of plain navigation. It isolates
Nav2's final smoother output on `/cmd_vel_nav_filtered`, goal-tagged RL commands
on `/cmd_vel_rl_stamped`, then puts the goal-triggered selector before this
package's velocity filter. A fresh `/people` stream is required even when its
list is empty. The filter remains the only final bridge to `/cmd_vel_safe` in
Gazebo or `/cmd_vel` on hardware. Nav2 recovery commands are routed through the
same smoother and selector, so `Spin`/`BackUp` cannot bypass that ownership.

The filter also publishes zero if its input disappears for 0.35 s, so a dead
mux cannot leave the base executing its last non-zero command.
