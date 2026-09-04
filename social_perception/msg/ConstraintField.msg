# K_soc: everything block D says about one instant. Geometry, not an image.
#
# The raster the policy's CNN reads is made from this by block E; this message
# is what block F consumes, and it carries what a float per cell cannot -- the
# track ids behind a region, how much its scene_type ruling is worth, and
# whether it is a wall or a cost.
#
# header.stamp is the "timestamp" of the plan document, header.frame_id its
# "frame" -- base_link, so every centre is already egocentric and no consumer
# needs TF to use it.
std_msgs/Header header

# INTERVALS, not samples. A 2.0 s horizon at 0.5 s spacing is 4 steps and 5
# samples, the extra one being t = 0.0 -- the present, which is the instant the
# reward is charged at and the one block F acts on.
uint16 horizon_steps
float32 dt

Zone[] zones
