string id
string[] member_ids
geometry_msgs/Point center
float32 o_radius
float32 p_radius
float32 r_radius

# Freshness of the decision behind this region, per region rather than per
# message. Groups.header carries the current camera frame, so it says nothing
# about how old the conversation verdict is: a consumer reading only that would
# treat a three-minute-old decision as if it had just arrived. These two let it
# tell an actively reconfirmed region from a stale one, and let one region go
# stale without implying anything about the others in the same message.
builtin_interfaces/Time inference_stamp      # when the VLM last ruled on it
builtin_interfaces/Time observation_stamp    # camera frame that ruling used
float32 confidence
