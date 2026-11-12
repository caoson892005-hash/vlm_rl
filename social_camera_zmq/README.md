# social_camera_zmq

This ROS 2 Humble package sends a Pi camera stream to a laptop with ZeroMQ.
The laptop publishes the regular ROS camera topics that `social_perception`
expects. It supports a USB webcam or an RGB-D ROS driver.

| Mode | Pi source | Data sent | Laptop topics |
| --- | --- | --- | --- |
| `rgb` | USB webcam (`v4l2`) | JPEG RGB | `/camera/color/image_raw` |
| `rgbd` | Camera vendor ROS driver | JPEG RGB, lossless PNG `16UC1` depth, calibrations | `/camera/color/image_raw`, `/camera/depth/image_raw`, both `camera_info` topics |

Webcam mode does not produce depth or 3-D `/people` positions. It is suitable
for testing stream latency, YOLO 2-D detection and tracking. Do not feed
monocular detections into the social navigation costmap as measured distances.

## Setup on Pi and laptop

Both machines need ROS 2 Humble and this package in a ROS workspace. Install
runtime packages on each machine:

```bash
sudo apt update
sudo apt install python3-zmq python3-opencv python3-numpy python3-yaml
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select social_camera_zmq
source install/setup.bash
```

Connect both machines to the same trusted network. Prefer Ethernet. Give the
Pi a stable IP address and synchronize both clocks with NTP or chrony. ZeroMQ
uses TCP port 5555 in the example; allow the laptop to connect to that port.
ZeroMQ PUB/SUB has no built-in authentication in this configuration, so keep
this endpoint off public networks.

### Pi: current USB webcam

Find the webcam path (`v4l2-ctl --list-devices` if available). Edit
`config/pi_webcam.yaml` for its `device`, width, height and frame rate. A stable
`/dev/v4l/by-id/...` path is preferable if multiple USB cameras are present.
Then run:

```bash
ros2 launch social_camera_zmq zmq_camera_sender.launch.py
```

The sender binds `tcp://0.0.0.0:5555`, captures RGB at 640×480/15 FPS by
default, JPEG encodes each frame and includes the capture timestamp. Dropped
frames are logged; the sender does not queue old video indefinitely. If the
webcam disconnects, the sender retries opening it every `reopen_interval_s`
(seconds, default 2.0, positive values only).

### Laptop: receive and publish ROS images

Edit `config/laptop_receiver.yaml`: replace `10.136.10.188` in
`connect_endpoint` with the actual Pi IP address. Then run:

```bash
ros2 launch social_camera_zmq zmq_camera_receiver.launch.py
ros2 topic hz /camera/color/image_raw
ros2 topic echo /camera/color/image_raw --once --field header
```

To inspect the image, use RViz `Image` display on
`/camera/color/image_raw` with a sensor data / best effort QoS subscription.

If a calibrated webcam intrinsics YAML is available, set its path in the
laptop's `camera_info_file`. Its resolution must match the actual webcam image.
Otherwise `/camera/color/camera_info` stays absent. Calibration alone does not
provide depth.

## RealSense D435i on the Pi

The ready-to-use configuration files are `config/pi_realsense_d435i.yaml` and
`config/laptop_rgbd.yaml`. Install `realsense2_camera` on the Pi, then run its
driver with RGB, depth aligned to RGB pixels, and synchronized frames:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true enable_sync:=true
ros2 topic list -t | grep camera
ros2 topic echo /camera/camera/aligned_depth_to_color/image_raw --once --field encoding
```

The sender requires `16UC1` depth in millimetres and uses the RealSense ROS
default `/camera/camera/...` topics. If the driver uses another namespace,
update the four `ros_*_topic` entries in `pi_realsense_d435i.yaml` to match
the topics on the Pi. The aligned depth image must match the RGB resolution.
On the laptop, set `connect_endpoint` in `laptop_rgbd.yaml` to the Pi's current
IP address; it currently mirrors the existing receiver configuration.

Launch each side with its RGB-D configuration after building and sourcing this
package on both machines:

```bash
# Pi
ros2 launch social_camera_zmq zmq_camera_sender.launch.py config_file:=$(ros2 pkg prefix social_camera_zmq)/share/social_camera_zmq/config/pi_realsense_d435i.yaml

# Laptop
ros2 launch social_camera_zmq zmq_camera_receiver.launch.py config_file:=$(ros2 pkg prefix social_camera_zmq)/share/social_camera_zmq/config/laptop_rgbd.yaml
```

In `require_depth` mode the receiver publishes an RGB/depth pair only when
both calibrations are present, image sizes match, and capture timestamps
differ by at most `max_pair_delta_s` (seconds, default 0.15). Check:

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_raw
ros2 topic echo /camera/color/camera_info --once
ros2 topic echo /camera/depth/camera_info --once
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

The ROS driver/robot must also publish the camera TF; ZeroMQ transports images
and calibration, not TF. `social_perception` needs camera calibration, aligned
depth, TF into its target frame, and synchronized Pi/laptop clocks to produce
valid 3-D `/people` positions. Different RGB-D models can use this package
once their drivers are mapped to the input ROS topics, but USB plug-in alone
cannot install or start an unknown vendor driver.

## Parameters and limits

The Pi-side parameters are in `config/pi_webcam.yaml` or
`config/pi_realsense_d435i.yaml`; the laptop-side parameters are in
`config/laptop_receiver.yaml` or `config/laptop_rgbd.yaml`. `jpeg_quality` is 1–100
(default 80). `send_hwm` and `receive_hwm` bound the ZeroMQ queues to two
messages. `stale_timeout_s` is 1.0 s: the receiver warns after no new RGB
arrives, without publishing the previous image again. `max_payload_bytes`
defaults to 8 MiB. The image publishers use the ROS sensor data QoS profile
(best effort, volatile). Both sides must agree on `camera.*.v1` ZeroMQ topics.
