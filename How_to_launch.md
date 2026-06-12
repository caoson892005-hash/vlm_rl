#### Bring up Lidar (Pi 4)

ros2 run sllidar_ros2 sllidar_node --ros-args -p serial_port:=/dev/rplidar -p serial_baudrate:=115200 -p scan_mode:=Standard

ros2 launch linorobot2_bringup sensors.launch.py

ros2 service call /stop_motor std_srvs/srv/Empty {}

ros2 service call /start_motor std_srvs/srv/Empty {}

## URDF

### 1. Define robot properties
Build the robot computer's workspace to load the new URDF:

    cd <robot_computer_ws>
    colcon build

The same changes must be made on the host machine's <robot_type>.properties.urdf.xacro if you're simulating the robot in Gazebo.

    cd <host_machine_ws>
    colcon build

### 2. Visualize the newly created URDF
#### Visualize the robot from the host machine:

  ros2 launch linorobot2_description description.launch.py rviz:=true

## Quickstart

### 1. Booting up the robot

#### 1.1a Using a real robot:

    export LINOROBOT2_LASER_SENSOR=""
    
    ros2 launch linorobot2_bringup bringup.launch.py base_serial_port:=/dev/ttyUSB0 lidar_serial_port:=/dev/ttyUSB1 micro_ros_baudrate:=921600

#### 1.1b Using Gazebo:
    
    ros2 launch linorobot2_gazebo gazebo.launch.py


    
### 
    ros2 launch linorobot2_gazebo gazebo.launch.py world:=worlds/empty.world

        ros2 launch linorobot2_navigation navigation.launch.py sim:=true rviz:=true map:=/home/hung/ninorobot2/linorobot2_navigation/maps/hungmap.yaml
#####




    ros2 launch linorobot2_gazebo gazebo.launch.py world:=worlds/empty.world
rviz:=true
    ros2 launch linorobot2_gazebo gazebo.launch.py paused:=true rviz:=true world:=worlds/empty.world spawn_x:=1.1 spawn_y:=0.8 spawn_yaw:=0.0

    ros2 service call /reset_simulation std_srvs/srv/Empty {}

    ros2 service call /reset_world std_srvs/srv/Empty {}

    ros2 service call /set_pose gazebo_msgs/srv/SetEntityState '{state: {name: "linorobot2", pose: {position: {x: 0.0, y: 0.0, z: 0.1}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}'

    ros2 service call /set_pose robot_localization/srv/SetPose "{pose: {header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}"

    ros2 run tf2_tools view_frames

linorobot2_bringup.launch.py or gazebo.launch.py must always be run on a separate terminal before creating a map or robot navigation when working on a real robot or gazebo simulation respectively.

### 2. Controlling the robot

    ros2 run teleop_twist_keyboard teleop_twist_keyboard

### 3. Creating a map

#### 3.1 Run [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox):

    ros2 launch linorobot2_navigation slam.launch.py rviz:=true sim:=true

- **sim** - Set to true for simulated robots on the host machine. Default value is false.
- **rviz** - Set to true to visualize the robot in RVIZ. Default value is false.

#### 3.2 Move the robot to start mapping

Drive the robot manually until the robot has fully covered its area of operation. Alternatively, you can use the `2D Goal Pose` tool in RVIZ to set an autonomous goal while mapping.

#### 3.3 Save the map

    cd ~/ninorobot2/linorobot2_navigation/maps
    ros2 run nav2_map_server map_saver_cli -f hungmap --ros-args -p save_map_timeout:=10000.0

### 4. Autonomous Navigation

#### 4.1 Load the map you created:

Open linorobot2/linorobot2_navigation/launch/navigation.launch.py and change *MAP_NAME* to the name of the newly created map. Build the robot computer's workspace once done:
    
    cd ~/linorobot2_ws
    colcon build

Alternatively, `map` argument can be used when launching Nav2 (next step) to dynamically load map files. For example:

    ros2 launch linorobot2_navigation navigation.launch.py map:=linorobot2_navigation/maps/hungmap.yaml


#### 4.2 Run [Nav2](https://navigation.ros.org/tutorials/docs/navigation2_on_real_turtlebot3.html) package:

    ros2 launch linorobot2_navigation navigation.launch.py sim:=true rviz:=true map:=/home/hung/ninorobot2/linorobot2_navigation/maps/hungmap.yaml

Optional parameter for loading maps:
- **map** - Path to newly created map <map_name.yaml>.

Optional parameters for simulation on host machine:
- **sim** - Set to true for simulated robots on the host machine. Default value is false.
- **rviz** - Set to true to visualize the robot in RVIZ. Default value is false.


-------------------------------------------------------------------------------------


# 1. Xóa các thư mục build cũ để tránh rác cấu hình
cd ~/linorobot2_ws
rm -rf build/ install/ log/

# 2. Build lại toàn bộ
cd ~/linorobot2_ws
colcon build
source install/setup.bash

killall -9 gzserver gzclient
killall -9 rviz2
pkill -f rviz

ros2 daemon stop
ros2 daemon start
