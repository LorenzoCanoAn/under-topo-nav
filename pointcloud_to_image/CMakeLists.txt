cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_to_image)

find_package(catkin REQUIRED COMPONENTS
  rospy
)
#catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

