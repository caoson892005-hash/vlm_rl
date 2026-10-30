# VLM + RL social navigation for linorobot2

Workspace ROS 2 Humble cho robot linorobot2: mô phỏng Gazebo Classic, nhận biết
người từ camera RGB-D, suy luận trạng thái tương tác bằng VLM và điều khiển robot
bằng chính sách Recurrent PPO. Mã nguồn cũng có một nhánh Nav2 với social
costmap để thử nghiệm điều hướng xã hội độc lập với chính sách RL.

## Luồng dữ liệu

```text
RGB-D + TF → YOLO Pose, depth, tracking → /people/tracks
                                    └→ VLM Qwen (tùy chọn) → /social_perception/vlm_person_states
/people/tracks + trạng thái VLM → constraint field → quan sát của Recurrent PPO
đích + odometry + laser scan ────────────────────────────────┘
Recurrent PPO → /cmd_vel
```

VLM chạy ở node riêng, được bật bằng `vlm_interaction:=true`. Khi chạy agent RL,
chính sách xuất lệnh vận tốc trực tiếp; Nav2 không điều khiển cùng tuyến đường.
`social_navigation` cung cấp thêm social costmap cho luồng Nav2 riêng.

## Cấu trúc repo

| Thư mục | Vai trò |
| --- | --- |
| `linorobot2_*` | Robot, URDF, Gazebo, bản đồ, cấu hình Nav2 và bringup. |
| `social_perception/` | RGB-D, YOLO Pose, tracking, VLM và ROS messages. |
| `social_navigation/` | Social costmap, kịch bản người trong mô phỏng và RViz. |
| `social_rl/` | Constraint field, môi trường huấn luyện, Recurrent PPO và agent chạy model. |
| `social_camera_zmq/` | Truyền ảnh camera qua ZMQ khi dùng camera ở máy khác. |
| `docs/`, `recordings/` | Tài liệu và dữ liệu ghi hình đã được đưa vào repo. |

Hướng dẫn chi tiết nằm trong
[`social_perception/README.md`](social_perception/README.md),
[`social_navigation/README.md`](social_navigation/README.md) và
[`social_rl/README.md`](social_rl/README.md). Cấu hình train và tham số robot
nằm trong các thư mục `config/` của từng package.

## Yêu cầu và build

- Ubuntu với ROS 2 Humble, Gazebo Classic và `colcon`.
- Các ROS dependency của từng package theo `package.xml`.
- Python dependency trong `social_perception/requirements.txt` và
  `social_rl/requirements.txt`; chọn bản PyTorch phù hợp với GPU riêng.
- Checkpoint YOLO Pose cục bộ để lấy keypoint và body yaw. Khi bật VLM, cần
  thêm Qwen base model và adapter cục bộ theo
  `social_perception/config/social_vlm_interaction.yaml`.
- Checkpoint Recurrent PPO `.zip` và `env_config.yaml` đi cùng checkpoint để
  chạy agent. Không có bước tải model trong luồng chạy robot.

Đặt repo trong `src/` của một ROS workspace, rồi build từ thư mục cha:

```bash
mkdir -p ~/vlm_rl_ws/src
git clone https://github.com/caoson892005-hash/vlm_rl.git ~/vlm_rl_ws/src/vlm_rl
cd ~/vlm_rl_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to social_rl social_navigation
source install/setup.bash
```

Cài Python dependency trên môi trường dùng để chạy perception/RL trước khi
launch. Các gói VLM chỉ cần thiết khi bật node Qwen. Adapter, checkpoint train,
file ZIP và cache sinh ra trong quá trình chạy không được đồng bộ vào Git.

## Chạy mô phỏng với VLM và RL

Mở mỗi lệnh trong một terminal đã `source /opt/ros/humble/setup.bash` và
`source ~/vlm_rl_ws/install/setup.bash`. Chuẩn bị sẵn đường dẫn cục bộ cho
YOLO Pose, adapter VLM và checkpoint RL; các đường dẫn trong YAML cần được đặt
theo máy đang chạy.

```bash
# 1. Robot và Gazebo
ros2 launch linorobot2_gazebo gazebo.launch.py gui:=false run_ekf:=true

# 2. RGB-D, tracking và VLM
ros2 launch social_navigation social_bringup.launch.py \
  sim:=true vlm_interaction:=true \
  yolo_model_path:=/absolute/path/to/yolo-pose.pt

# 3. Thả người vào mô phỏng
ros2 launch social_navigation social_sim.launch.py scenario:=talking

# 4. Chính sách RL và RViz
ros2 launch social_rl rl_agent.launch.py \
  model_path:=/absolute/path/to/final_model.zip \
  sim:=true localization:=true rviz:=true people:=perception
```

Trong RViz, gửi đích tới `/rl_goal_pose`. Agent đọc cấu hình quan sát từ
`env_config.yaml` nằm cạnh model; nếu dùng một file khác, truyền
`env_config:=/absolute/path/to/env_config.yaml`. Có thể chạy riêng perception
bằng cách bỏ `vlm_interaction:=true`, nhưng khi đó agent không nhận nhãn VLM.
`people:=ground_truth` chỉ dùng trong Gazebo để kiểm tra policy bằng người thật
trong mô phỏng; không chạy cùng `localization:=true`.

## Huấn luyện và kiểm tra

Training mặc định dùng người ground truth từ Gazebo theo
`social_rl/config/rl_train.yaml`; không cần khởi động perception hoặc VLM.
Sau khi mở Gazebo, chạy:

```bash
ros2 launch social_rl rl_train.launch.py
```

Xem [`social_rl/RUN_RL.txt`](social_rl/RUN_RL.txt) và
[`social_rl/TRAIN_RL.txt`](social_rl/TRAIN_RL.txt) để biết cách đặt kịch bản,
resume checkpoint và đánh giá policy. Các kiểm tra mã nguồn có thể chạy bằng:

```bash
python3 -m unittest discover -s ~/vlm_rl_ws/src/vlm_rl/social_perception/test
python3 -m pytest ~/vlm_rl_ws/src/vlm_rl/social_rl/test
```

Các file cấu hình mô phỏng và robot thật khác nhau ở thời gian ROS, camera,
TF và topic vận tốc. Chọn `sim:=false` cùng profile cấu hình tương ứng khi chạy
trên robot thật; kiểm tra lại model, map và topic trước khi cấp lệnh di chuyển.
