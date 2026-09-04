# Run the whole stack without touching the network. Source it, do not execute:
#
#     source ~/ninorobot2/offline_env.sh
#
# The Hugging Face variables are already applied by
# social_perception/launch/social_vlm_perception.launch.py, so sourcing this
# only matters for scripts that start the model outside that launch file.

# Use the cached Qwen2-VL checkpoint instead of asking the hub for the current
# revision on every start.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Gazebo must not reach for the online model database. `cafe`, `ground_plane`
# and `sun` come from ~/.gazebo/models; the actors come from
# share/social_navigation/models, which gazebo.launch.py puts on the path.
export GAZEBO_MODEL_DATABASE_URI=
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
export GAZEBO_IP=127.0.0.1

# Keep DDS on the loopback interface. Do NOT source this on the robot or on a
# laptop driving it over WiFi: it hides every node running on the other
# machine.
export ROS_LOCALHOST_ONLY=1

# The first load after boot reads 1.5 GB of weights off the NVMe. Pulling them
# into the page cache here means the node itself starts against warm memory.
# With 15 GB of RAM a heavy Gazebo session can evict them again, so re-source
# this before a run that must start fast.
preload_vlm_weights() {
    local weights
    weights=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen2-VL-2B-Instruct-bnb-4bit/snapshots/*/model.safetensors 2>/dev/null | head -1)
    if [ -n "$weights" ]; then
        cat "$weights" > /dev/null
        echo "VLM weights preloaded into the page cache"
    else
        echo "VLM weights not found in the Hugging Face cache" >&2
    fi
}
