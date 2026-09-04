# The model's answer to one VlmRequest.

std_msgs/Header header
string request_id            # copied from the request this answers

# Exactly what the model produced, unparsed. Perception applies the same
# talking_from_response() to it as it does to a local answer, so a remote and a
# local run cannot disagree about what a reply means.
string raw_response

# Inference failures travel as data rather than as silence: perception is
# blocked waiting for this message, and a dropped answer would cost it the full
# request timeout before it could ask again.
bool failed
string error

# The same breakdown the local backend records, measured on the worker. The gap
# between their sum and the round trip perception measures is what the network
# and the JPEG encode cost.
float32 prepare_seconds
float32 lock_seconds
float32 generate_seconds
uint32 token_count
