#!/usr/bin/env python

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Int8
import std_msgs.msg
import rospy
import math

topic = 'yolo_object'
topic_array = 'yolo_objects'
publisher = rospy.Publisher(topic, Marker,queue_size=10)
array_publisher = rospy.Publisher(topic_array, MarkerArray,queue_size=10)
# pub = rospy.Publisher('/Int_cmd_trackhuman', Int8, queue_size=10)

rospy.init_node('yolo_markers')

markerArray = MarkerArray()

count = 0
MARKERS_MAX = 100

while not rospy.is_shutdown():

   del markerArray.markers[:]
   marker = Marker()
   marker.header.frame_id = "/map"
   marker.type = marker.SPHERE
   marker.action = marker.ADD
   marker.scale.x = 0.5
   marker.scale.y = 0.5
   marker.scale.z = 0.5
   marker.color.a = 1.0
   marker.color.r = 1.0
   marker.color.g = 0.0
   marker.color.b = 0.0
   marker.pose.orientation.w = 1.0
   
   # marker.pose.position.x = 4.5
   # marker.pose.position.y = 0.95
   # marker.pose.position.z = 1

   marker.pose.position.x = 2.1
   marker.pose.position.y = 0.3
   marker.pose.position.z = 1

   marker2 = Marker()
   marker2.header.frame_id = "/map"
   marker2.type = marker.SPHERE
   marker2.action = marker.ADD
   marker2.scale.x = 0.5
   marker2.scale.y = 0.5
   marker2.scale.z = 0.5
   marker2.color.a = 1.0
   marker2.color.r = 0.0
   marker2.color.g = 1.0
   marker2.color.b = 0.0
   marker2.pose.orientation.w = 1.0
   
   marker2.pose.position.x = 1.5
   marker2.pose.position.y = -0.45
   marker2.pose.position.z = 1


   marker3 = Marker()
   marker3.header.frame_id = "/map"
   marker3.type = marker.SPHERE
   marker3.action = marker.ADD
   marker3.scale.x = 0.5
   marker3.scale.y = 0.5
   marker3.scale.z = 0.5
   marker3.color.a = 1.0
   marker3.color.r = 0.0
   marker3.color.g = 0.0
   marker3.color.b = 1.0
   marker3.pose.orientation.w = 1.0
   
   marker3.pose.position.x = 0.8
   marker3.pose.position.y = 0.4
   marker3.pose.position.z = 1
        
   #moving human
   #marker.pose.position.x = 2.2+0.0015*count
   #marker.pose.position.y = -0.3+0.2*math.sin(count/100.0)
   #marker.pose.position.z = 1

   # marker.pose.position.y = 2.7+0.2*math.sin(count/100.0)
   # marker.pose.position.y = +0.2*math.sin(count / 40.0) 
   
   # marker.pose.position.x = 3.0
   # marker.pose.position.y = 0.5
   # marker.pose.position.z = 1

   int_msg=std_msgs.msg.Int8()
   int_msg.data=1

   # We add the new marker to the MarkerArray, removing the oldest
   # marker from it when necessary
   # if(count > MARKERS_MAX):
   #     markerArray.markers.pop(0)

   markerArray.markers.append(marker)
   markerArray.markers.append(marker2)
   markerArray.markers.append(marker3)
   # Renumber the marker IDs
   id = 0
   for m in markerArray.markers:
       m.id = id
       id += 1

   # Publish the MarkerArray
   # publisher.publish(marker)
   array_publisher.publish(markerArray)
   # pub.publish(int_msg)

   count += 1

   rospy.sleep(0.1)
