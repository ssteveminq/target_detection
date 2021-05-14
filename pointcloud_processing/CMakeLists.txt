cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_processing)

add_compile_options(-std=c++11)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  darknet_ros_msgs
  message_filters
  visualization_msgs
  geometry_msgs
  pointcloud_processing_msgs
  tf
  tf2_geometry_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
#find_package(PCL REQUIRED)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs
#  tf2_geometry_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(pcl_processing src/pcl_processing.cpp)
add_executable(sep_processing src/sep_processing.cpp)
#add_executable(people_processing src/people_processing.cpp)

target_link_libraries(pcl_processing ${catkin_LIBRARIES})
target_link_libraries(sep_processing ${catkin_LIBRARIES})
#target_link_libraries(people_processing ${catkin_LIBRARIES})


## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
