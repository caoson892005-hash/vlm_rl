# social_perception

ROS 2 package that owns the complete camera-side social-perception pipeline:

1. YOLO detects people in the RGB image.
2. Registered depth and `CameraInfo` localize each person in `target_frame`.
3. Lightweight tracking assigns stable IDs and estimates velocity.
4. Nearby person pairs are cropped and passed to the supplied Qwen2-VL LoRA
   adapter with the Vietnamese "đang nói chuyện?" prompt.
5. VLM-confirmed conversations are published as social regions. Their `center`
   is the mean world position of the participating people.

The detector runs independently of the VLM worker. If the VLM dependencies or
base model are unavailable, `/people` continues to be published and the node
logs the VLM error instead of stopping camera perception.

Both profiles run YOLO and Qwen2-VL on CUDA: sharing the card costs ~50 MiB
next to the VLM's ~2.0 GiB, while leaving YOLO on the CPU makes it contend with
the decode loop in this one process (10.1 s per inference against 3.6 s). VLM
pair crops are capped at 256 merged visual tokens, only the nearest pair is
processed, and an unchanged pair is refreshed every 15 seconds. A new pair or
a position change of at least 0.25 m triggers an earlier inference. Each run
prints `VLM latency ...` with its elapsed time and crop size.
With `vlm_require_cuda: true`, loss of CUDA disables only the VLM worker rather
than silently falling back to very slow CPU generation.

`vlm_offline: true` loads the checkpoint straight from `~/.cache/huggingface`
instead of asking huggingface.co whether the cached copy is still current.
Measured on this machine, startup drops from ~32.5 s to ~27.8 s; on a robot
that is offline or behind a captive portal the saving is however long the HTTP
timeout takes. A machine with an empty cache still downloads the model once,
with a warning, rather than failing.

### Where the time goes

Measured on a Quadro T1000 with the 4-bit base model, a single inference is
~1.8 s of prefill plus ~0.36 s per generated token — decoding, not the image,
is the bottleneck. Two consequences shape the defaults:

- `vlm_force_answer_prefix: true` starts the assistant turn at
  `{"talking":"` so only the decisive word is generated instead of the whole
  7-token JSON object. Measured 3.97 s → 2.51 s per inference, with the same
  answer, and it also rules out the malformed replies (`{"talking":"có}`,
  `{"talking":true}`) that previously cost an entire inference to recover
  from. `vlm_max_new_tokens` applies only when this is `false`.
- The model is warmed up with one throwaway inference at load time. The first
  generate() call costs ~9.4 s against ~4.0 s for every call after it, so
  paying it during startup keeps the first real conversation at steady-state
  latency.

Lowering `vlm_max_pixels` helps far less than it looks: 256 → 128 → 64 merged
visual tokens moved a full inference only from 4.05 s to 3.55 s to 2.98 s, so
it costs image detail for a modest gain. Try it only after the two settings
above.

Queued and running VLM work carries a scene-generation ID. Losing a tracked
person or receiving `/animated_people/hide` advances that ID, so an answer from
an older image is discarded even if GPU generation finishes later.

## Model setup

`Saved_Model` is a PEFT/LoRA adapter, not a standalone model. Its
`adapter_config.json` points to
`unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`, so the base model must also be present
in the Hugging Face cache (or downloadable on the first run).

Install the Python inference dependencies in the Python environment used by
ROS 2:

```bash
python3 -m pip install -r social_perception/requirements.txt
```

The default config resolves `Saved_Model` and `yolov8n.pt` from the workspace
root. Set `vlm_adapter_path` and `yolo_model_path` explicitly for an installed
deployment. Set `enable_vlm: false` to test only RGB-D localization on a machine
without the VLM stack.

## Build and launch

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to social_perception social_navigation
source install/setup.bash
ros2 launch social_perception social_vlm_perception.launch.py
```

## Splitting the model off onto another machine

The robot carries the camera but not the GPU, so the model can run elsewhere.
Set `vlm_remote: true` in the profile (already the case in
`social_vlm_perception_real.yaml`) and start the other half on the workstation:

```bash
# workstation
ros2 launch social_perception social_vlm_worker.launch.py

# robot
ros2 launch social_perception social_vlm_perception.launch.py \
    config_file:=<...>/social_vlm_perception_real.yaml
```

Only the crop of a candidate pair crosses the network, measured at 13-14 KB per
inference, against the ~12 MB/s that shipping raw RGB-D at 8 Hz would cost:

```text
robot        /social_perception/vlm_request   (VlmRequest: crop + prompt)
workstation  /social_perception/vlm_response  (VlmResponse: raw answer)
workstation  /social_perception/vlm_worker_ready  (latched, gates "SẴN SÀNG")
```

The split is deliberately lopsided. Every rule about what an answer *means* —
which pair is worth asking about, what a newer camera frame invalidates, how
much a contrary reply weighs against a region already on the costmap — stays in
the perception node. The worker is stateless: it holds no track ids and no
cache, so restarting it costs one inference rather than a rebuilt world model.
`vlm_backend.py` is imported by whichever side loads the model, so the robot
never needs `transformers`/`peft` and the workstation never needs
`ultralytics`.

With `vlm_remote: false` the model runs inside the perception node exactly as
before; that is what the simulation profile does.

## Parameters

`config/social_vlm_perception.yaml` (simulation) and
`config/social_vlm_perception_real.yaml` (camera on the workstation) are the
only place parameters are defined; the node carries no defaults of its own and
declares whatever the profile it was launched with contains. A parameter
missing from the profile therefore raises at startup instead of running on a
hidden value. Consequently the node must always be started with a profile —
`social_vlm_perception.launch.py` passes one through its `config_file`
argument, and `social_bringup.launch.py` picks the right one from `sim:=`.
Keep the two profiles in step: a setting added to one belongs in the other.

## Topics

- `/people` (`social_perception/msg/People`): localized people and velocity.
- `/people_groups` (`social_perception/msg/Groups`): VLM-confirmed social
  regions, including center and O/P/R radii for the Nav2 layer.
- `/social_perception/talking_interactions`
  (`social_perception/msg/TalkingInteractions`): `talking`/`not_talking`
  decision, member IDs, each person's pose and velocity, center, confidence,
  source-image timestamp, inference-completion timestamp, and raw VLM response.
- `/social_perception/vlm_request` (`social_perception/msg/VlmRequest`) and
  `/social_perception/vlm_response` (`social_perception/msg/VlmResponse`): the
  JPEG pair crop and the model's raw answer, only when `vlm_remote: true`.
- `/social_perception/detections_2d`: YOLO boxes.
- `/social_perception/annotated_image`: RGB image with detections and track IDs.
- `/social_perception/depth_visualization`: colored depth image.
- `/social_perception/person_markers`: localized people for RViz.
- `/social_spaces`: O/P/R outlines for each VLM-confirmed region in RViz.

The social-navigation costmap consumes only `/people` and `/people_groups`, so
the VLM implementation can evolve without changing the costmap plugin.

With `log_vlm_results: true` (the default), each completed inference is also
printed to the launch terminal, for example:

```text
VLM RESULT [NÓI CHUYỆN] pair=(person_1, person_2) confidence=0.90 ...
```

Inspect the complete timestamped output with:

```bash
ros2 topic echo /social_perception/talking_interactions
```
