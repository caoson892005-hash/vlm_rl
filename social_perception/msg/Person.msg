string id
geometry_msgs/Pose pose
geometry_msgs/Twist velocity

# --- block B: short-horizon prediction -------------------------------------
#
# Constant-velocity extrapolation of `pose`, one entry per element of
# `prediction_times`. The two arrays are the same length; the first time is
# 0.0, so `predicted_poses[0]` repeats `pose`. Block D charges the reward at
# t+0.0 and rasterises the later samples, and having both read the same array
# is what stops the field the policy SEES from drifting away from the depth it
# is CHARGED for.
#
# Orientation carries the predicted facing: the direction of travel once
# somebody is walking, and the reported facing while they stand still. A
# standing person keeping their facing is the whole point for a conversation.
#
# Constant velocity even in simulation, where the plugin knows the actual
# waypoints and could publish the true future. It deliberately does not: a
# policy conditioned on a future only the simulator can supply is a policy that
# stops working the day it moves to the robot. What ground truth buys here is
# an exact pose and velocity to extrapolate FROM, not knowledge of what happens
# next.
float32[] prediction_times            # seconds ahead, ascending, starts at 0.0
geometry_msgs/Pose[] predicted_poses  # same length as prediction_times

# --- block B: motion relative to the robot ---------------------------------
#
# Unlike everything above, these two are measured against the robot, so a
# consumer standing somewhere else cannot re-derive them from this message.
# They are filled by the producer rather than by the observation bridge so that
# block F and any logging consumer read the same labels the policy does; two
# copies of this arithmetic drift apart the first time one of them is tuned.
#
# Component of `velocity` along the robot -> person direction, in m/s.
# Negative means closing on the robot. Zero when the producer cannot see the
# robot's pose.
float32 radial_velocity

# How this person is moving across the robot's view. Geometry only: this is
# block B's judgement and it stays filled when block C is switched off.
#   static         speed below the still threshold
#   head_on        closing, inside the head-on cone
#   receding       opening, inside the head-on cone
#   left_to_right  crossing, moving towards the robot's right
#   right_to_left  crossing, moving towards the robot's left
# Empty means the producer could not see the robot's pose.
string motion_type

# Which social situation this person is in, as far as the producer knows.
#
# Block D reads it to choose the constraint-field geometry: a person walking
# across the robot's path and a person standing in a conversation call for
# different regions, and no amount of position and velocity says which is
# which. Filled by block C (the video VLM) on the robot, and by the Gazebo
# ground-truth publisher during RL training, which knows the scenario it is
# playing rather than having to infer it.
#
# Values used by social_rl/constraint_field.py:
#   talking       standing in a conversation with somebody else
#   backs_turned  facing away from the robot; passing behind is acceptable
#   passing       walking past: across the robot's path or roughly head on
#                 (crossing and approaching merged 10-09-2026)
#   walking       moving, but none of the above
#
# Empty means "unknown". Block D then infers it from the relative geometry,
# which is what a run without block C gets.
string scene_type

# --- block C: how much that ruling is worth --------------------------------
#
# Confidence in `scene_type`, in [0, 1]. 1.0 is the Gazebo ground-truth
# publisher, which is not guessing -- it knows the scenario it was asked to
# play. 0.0 means nothing ruled at all: block C is switched off, or it answered
# and the answer was "unknown". Those two are told apart by `scene_type` being
# empty in the first case and set in the second, which is why an empty
# scene_type must never be published with a confidence above zero.
float32 scene_confidence

# When block C last ruled on `scene_type`. Header.stamp says when the person
# was SEEN, which is a different and much newer time: a conversation verdict is
# refreshed every few seconds while the camera runs at frame rate, so a
# consumer reading only the header would treat a three-minute-old ruling as if
# it had just arrived. Zero while scene_confidence is 0.0.
#
# Deliberately per person rather than per message, for the same reason Group
# carries its own pair: one person's situation can go stale without saying
# anything about anybody else's in the same frame.
builtin_interfaces/Time scene_stamp
