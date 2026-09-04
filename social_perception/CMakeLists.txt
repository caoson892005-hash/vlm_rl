cmake_minimum_required(VERSION 3.8)
project(social_perception)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Person.msg"
  "msg/People.msg"
  "msg/Group.msg"
  "msg/Groups.msg"
  "msg/TalkingInteraction.msg"
  "msg/TalkingInteractions.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

install(PROGRAMS
  scripts/social_vlm_perception.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(FILES README.md requirements.txt DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
