# social_camera_zmq

This ROS 2 Humble package sends a Pi camera stream to a laptop with ZeroMQ.
The laptop publishes the regular ROS camera topics that `social_perception`
expects. It supports a USB webcam now and an RGB-D ROS driver later.

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

Edit `config/laptop_receiver.yaml`: replace `192.168.1.50` in
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

## Switch to an RGB-D camera later

The transport and laptop ROS topic names stay the same. Install the camera's
ROS driver on the Pi and configure it to publish RGB, **depth aligned to the
RGB pixels**, and CameraInfo for both streams. The depth image must be
`16UC1` in millimetres. Copy `config/pi_webcam.yaml` to a local configuration
file and change these entries to the vendor driver's real topic names:

```yaml
source: ros
camera_mode: rgbd
ros_rgb_topic: /camera/color/image_raw
ros_depth_topic: /camera/aligned_depth_to_color/image_raw
ros_camera_info_topic: /camera/color/camera_info
ros_depth_camera_info_topic: /camera/aligned_depth_to_color/camera_info
```

On the laptop, copy `config/laptop_receiver.yaml` to a local configuration
file and set:

```yaml
receive_depth: true
require_depth: true
```

Launch each side with its custom configuration:

```bash
ros2 launch social_camera_zmq zmq_camera_sender.launch.py config_file:=/absolute/path/pi_rgbd.yaml
ros2 launch social_camera_zmq zmq_camera_receiver.launch.py config_file:=/absolute/path/laptop_rgbd.yaml
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

The Pi-side parameters are in `config/pi_webcam.yaml`; the laptop-side
parameters are in `config/laptop_receiver.yaml`. `jpeg_quality` is 1–100
(default 80). `send_hwm` and `receive_hwm` bound the ZeroMQ queues to two
messages. `stale_timeout_s` is 1.0 s: the receiver warns after no new RGB
arrives, without publishing the previous image again. `max_payload_bytes`
defaults to 8 MiB. The image publishers use the ROS sensor data QoS profile
(best effort, volatile). Both sides must agree on `camera.*.v1` ZeroMQ topics.
