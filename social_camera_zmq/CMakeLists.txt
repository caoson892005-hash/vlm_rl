cmake_minimum_required(VERSION 3.8)
project(social_camera_zmq)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/zmq_camera_sender.py
  scripts/zmq_camera_receiver.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
