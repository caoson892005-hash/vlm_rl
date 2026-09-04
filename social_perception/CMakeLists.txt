cmake_minimum_required(VERSION 3.8)
project(social_perception)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ZoneSample.msg"
  "msg/Zone.msg"
  "msg/ConstraintField.msg"
  "msg/Person.msg"
  "msg/People.msg"
  "msg/Group.msg"
  "msg/Groups.msg"
  "msg/TalkingInteraction.msg"
  "msg/TalkingInteractions.msg"
  "msg/VlmRequest.msg"
  "msg/VlmResponse.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs std_msgs
)

# vlm_backend.py is imported by both nodes rather than run, but it has to land
# in the same directory they are launched from for that import to resolve.
install(PROGRAMS
  scripts/social_vlm_perception.py
  scripts/social_vlm_worker.py
  scripts/vlm_backend.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(FILES README.md requirements.txt DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
