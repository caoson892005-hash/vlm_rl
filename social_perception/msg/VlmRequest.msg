# One conversation-pair crop sent to the VLM worker, which normally runs on
# another machine. Perception keeps every decision around this call -- which
# pair is worth asking about, what invalidates the answer, how much a contrary
# reply counts -- so this message carries only what the model itself needs.

std_msgs/Header header       # stamp = when this request was sent

# Echoed back on VlmResponse. The exchange is one question at a time, so this
# is all that is needed to recognise a late reply to an abandoned question.
string request_id

# The two tracked people in the crop. Carried for logging on the worker side;
# perception does not read them back.
string[] member_ids

# Sent rather than read from a second config file: two copies of the wording
# drift apart silently, and the model keeps answering the older question.
string prompt

sensor_msgs/CompressedImage crop
