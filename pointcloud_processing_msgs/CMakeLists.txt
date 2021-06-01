cmake_minimum_required(VERSION 2.8.12)

project(pointcloud_processing_msgs)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED
  COMPONENTS
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    std_srvs
    message_generation
)

add_message_files(
  FILES
    handle_position.msg
    ObjectInfo.msg
    ObjectInfoArray.msg
    fov_positions.msg
    ClusterArray.msg
#    BoundingBoxes.msg
)

#add_action_files(
#  FILES
#    CheckForObjects.action
#)

add_service_files(
    FILES
    register_scene.srv
    get_targetpose.srv
    )

generate_messages(
  DEPENDENCIES
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    std_srvs
    geometry_msgs 
)

catkin_package(
  CATKIN_DEPENDS
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    message_runtime
    std_msgs
    geometry_msgs 
)
