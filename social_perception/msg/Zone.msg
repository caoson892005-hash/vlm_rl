# One social region tracked across block D's whole horizon.

string zone_id             # stable while the tracks behind it are
string[] track_ids         # block B ids this region was built from
string scene_type          # block C's ruling that shaped it
float32 confidence         # block C's confidence; a group zone takes the lower

# What the region MEANS, not how it is drawn -- both kinds rasterise the same.
#   hard   the shared space of a conversation. The one place the robot must
#          not pass through, and the one block F takes as a CBF constraint.
#   soft   one person's personal space. Expensive to enter, sometimes worth
#          it, which is why block F only enforces these when asked to.
string hardness

float32 valid_from
float32 valid_to
ZoneSample[] trajectory_of_zone
