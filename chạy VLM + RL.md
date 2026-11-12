# terminal 1:
source install/setup.bash
ros2 launch linorobot2_gazebo gazebo.launch.py run_ekf:=true


source /opt/ros/humble/setup.bash
source install/setup.bash

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch linorobot2_gazebo gazebo.launch.py \
  run_ekf:=true \
  spawn_x:=-3.0 \
  spawn_y:=0.0 \
  spawn_z:=0.05 \
  spawn_yaw:=0.0 \
  publish_odom_tf:=false


source /opt/ros/humble/setup.bash
source /home/hung/ninorobot2/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
  --frame-id world --child-frame-id odom


# terminal 2:
source install/setup.bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch social_navigation social_bringup.launch.py \
  sim:=true \
  perception:=true \
  vlm_interaction:=true \
  vlm_interaction_config:=/home/son/social_ninorobot/social_perception/config/social_vlm_interaction.yaml \
  yolo_model_path:=/home/son/ninorobot2/yolov8n.pt \
  rviz:=false


__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch social_navigation social_bringup.launch.py \
  sim:=true \
  perception:=true \
  vlm_interaction:=true \
  vlm_interaction_config:=/home/son/social_ninorobot/social_perception/config/social_vlm_interaction.yaml \
  yolo_model_path:=/home/son/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt \
  rviz:=false


  source /opt/ros/humble/setup.bash
source install/setup.bash

source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py yolo_model_path:=/home/hung/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt

#### 
source /opt/ros/humble/setup.bash
source install/setup.bash


## terminal 3:

source install/setup.bash
ros2 launch social_navigation social_sim.launch.py \
  scenario:=talking \
  wait_for_perception:=true


source install/setup.bash
ros2 topic pub --once /animated_people/scenario std_msgs/msg/String "{data: waiting}"
ros2 topic pub --once /animated_people/scenario std_msgs/msg/String "{data: talking}"

##### terminal 4:
#
source install/setup.bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch social_rl rl_agent.launch.py \
  model_path:=/home/son/social_rl_runs/20261008_155530/checkpoints/recurrent_ppo_427500_steps.zip \
  sim:=true \
  localization:=true \
  rviz:=true \
  people:=perception \
  zones:=true

#Talking
  source install/setup.bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch social_rl rl_agent.launch.py \
  model_path:=/home/son/social_rl_runs/20261016_085455/final_model.zip \
  sim:=true \
  localization:=true \
  rviz:=true \
  people:=perception \
  zones:=true





source install/setup.bash

ros2 run social_rl rl_agent --ros-args \
  -p use_sim_time:=true \
  -p model_path:=/home/hung/social_rl_runs/20260922_235954/checkpoints/recurrent_ppo_495352_steps.zip \
  -p people_topic_override:=/people/tracks \
  -p cmd_vel_topic:=/cmd_vel


  ros2 launch social_rl rl_agent.launch.py \
  model_path:=/home/hung/social_rl_runs/20260922_235954/checkpoints/recurrent_ppo_495352_steps.zip \
  people:=ground_truth \
  localization:=true \
  rviz:=true



# Terminal 3: chay slam
source install/setup.bash
ros2 launch linorobot2_navigation slam.launch.py rviz:=true

cd ~/social_ninorobot
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0

## Chay thuc


# Pi
# Terminal 1:
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch linorobot2_bringup bringup.launch.py base_serial_port:=/dev/ttyUSB1

# Terminal 2: Chayj lidar
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ldlidar_stl_ros2 ld19.launch.py port_name:=/dev/ttyUSB0

# Terminal 3 — Chay depth cam

ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true enable_sync:=true


### Trên laptop

# Terminal laptop 1 — nhận RGB-D, thay IP ví dụ bằng IP Pi:

PI_IP=192.168.1.10
ros2 run social_camera_zmq zmq_camera_receiver.py --ros-args \
  --params-file "$(ros2 pkg prefix social_camera_zmq)/share/social_camera_zmq/config/laptop_rgbd.yaml" \
  -p connect_endpoint:="tcp://${PI_IP}:5555"

# Terminal laptop 2 — YOLO Pose, depth và tracking người:

ros2 run social_perception social_vlm_perception.py --ros-args \
  --params-file "$(ros2 pkg prefix social_perception)/share/social_perception/config/social_vlm_perception_real.yaml" \
  -p target_frame:=odom \
  -p yolo_model_path:=/home/son/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt

target_frame:=odom phù hợp với bài test chưa chạy AMCL. Camera gắn trên robot phải có TF từ base_link tới frame camera; không dùng pose camera cố định theo map.

# Terminal laptop 3 — bật VLM thật sự:

ros2 run social_perception social_vlm_interaction.py --ros-args \
  --params-file "$(ros2 pkg prefix social_perception)/share/social_perception/config/social_vlm_interaction_real.yaml" \
  -p enable_vlm:=true \
  -p vlm_adapter_path:=/home/son/social_ninorobot/Saved_Model

Tham số enable_vlm:=true là bắt buộc ở đây: profile robot thật (social_perception/config/social_vlm_interaction_real.yaml:4) đang để false. Terminal này sẽ in dòng VLM raw: ... sau khi nhận đủ 4 frame có người được track.

# Terminal laptop 4 — trường ràng buộc và policy RL:

ros2 launch social_rl rl_agent.launch.py \
  model_path:=/home/son/social_rl_runs/20261016_085455/final_model.zip \
  sim:=false localization:=false people:=perception rviz:=false





Xem kết quả:

ros2 topic echo /people/tracks
ros2 topic echo /social_perception/vlm_person_states


ros2 run rqt_image_view rqt_image_view
