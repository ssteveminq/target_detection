#!/usr/bin/env python
import roslib
import rospy
import roslib
import math
import sys
import rospy
import actionlib
import control_msgs.msg
import controller_manager_msgs.srv
import trajectory_msgs.msg
import geometry_msgs.msg
import controller_manager_msgs.srv
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from std_msgs.msg import Int8
from geometry_msgs.msg import PoseStamped, Point
from villa_yolocloud.msg import ObjectInfoArray

topic_array = 'yolo_objects'

class Marker_manager(object):
    def __init__(self, wait=0.0):
        # initialize action client
        # publisher for delvering command for base move
        rospy.Subscriber("/objects_info", ObjectInfoArray, self.objinfo_callback)
        self.array_publisher = rospy.Publisher(topic_array, MarkerArray,queue_size=10)
        self.markerArray = MarkerArray()
        self.obj_info= ObjectInfoArray()

    def listener(self,wait=0.0):
        # r=rospy.Rate(1)
        rospy.spin()            
        # rospy.sleep(0.1)
        # fill ROS message
    def objinfo_callback(self, data):
        rospy.loginfo("callback")
        self.obj_info=data

        del self.markerArray.markers[:]
        for i, objectinfo in enumerate(data.objectinfos):
           marker = Marker()
           marker.header.frame_id = "/map"
           marker.type = marker.TEXT_VIEW_FACING 
           marker.action = marker.ADD
           marker.id = i
           marker.scale.x = 0.5
           marker.scale.y = 0.5
           marker.scale.z = 0.5
           marker.color.a = 1.0
           marker.color.r = 1.0
           marker.color.g = 0.0
           marker.color.b = 0.0
           marker.pose.orientation.w = 1.0
           marker.pose.position=objectinfo.point
           marker.text = objectinfo.label
           marker.lifetime = rospy.Duration()
           self.markerArray.markers.append(marker)

        self.array_publisher.publish(self.markerArray)
        rospy.sleep(1.0)
        
        # rospy.loginfo(rospy.get_caller_id()+"I heard %d",data.data)
        # if len(data.markers)>0:
            # print data.markers[0].pose
       
if __name__ == '__main__':
    rospy.init_node('gaze_action_client_test')
    yolo_manager = Marker_manager(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
    yolo_manager.listener()

