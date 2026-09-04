# One social region at one instant of block D's prediction horizon.

float32 t                  # seconds ahead of the header stamp; first is 0.0
string shape               # ellipse
float32[2] center          # metres, in the field's frame

# (front, side, rear) reach, NOT the two semi-axes an ellipse normally needs.
# These regions are asymmetric front to back and that asymmetry is the point:
# `backs_turned` cuts the rear reach to 0.45 of nominal precisely so that
# passing behind somebody is affordable, and two numbers cannot hold that.
float32[3] size

# Inner radius. The region is solid inside it and ramps linearly to nothing at
# the boundary. Where the boundary in some direction is nearer than this -- the
# short axis of a group zone -- the region is solid out to the boundary there.
#
# This contour, not the outer one, is what a safety shield should treat as the
# thing to stay out of: the outer reach is a cost, the core is a wall.
float32 core

float32 orientation        # radians, of the +front axis
