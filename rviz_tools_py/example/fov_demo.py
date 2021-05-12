#!/usr/bin/env python

# Copyright (c) 2015, Carnegie Mellon University
# All rights reserved.
# Authors: David Butterworth <dbworth@cmu.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of Carnegie Mellon University nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
This is a demo of Rviz Tools for python which tests all of the
available functions by publishing lots of Markers in Rviz.
modified by Minkyu Kim 20210511
"""

# Python includes
import numpy
import random
import math
import sys
import numpy as np
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
# ROS includes
import roslib
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3, Polygon, PointStamped
from tf import transformations # rotation_matrix(), concatenate_matrices()
from tf import TransformListener 
import tf

from pointcloud_processing_msgs.msg import ObjectInfo, ObjectInfoArray, fov_positions
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
import rviz_tools_py as rviz_tools
from rviz_tools_py.geometry_tools import Frustum
from rviz_tools_py.geometry_tools import LinePlaneInterSection
from rviz_tools_py.geometry_tools import tangent_point_circle_extpoint 
from rviz_tools_py.geometry_tools import projection_point_to_plane 

# WIDTH_RES = 512
# HEIGHT_RES = 512

WIDTH_RES = 1920
HEIGHT_RES = 1080


_MAP_TF='odom'
_SESNSOR_TF ='camera_rgb_frame'
# _BASE_TF = 'base_link'

# Initialize the ROS Node
# Define exit handler


class DrawManager(object):
        def __init__(self, wait=0.0):
            self.markers = rviz_tools.RvizMarkers(_SESNSOR_TF, 'visualization_marker')
            samples_markers = rviz_tools.RvizMarkers(_SESNSOR_TF, 'sample_marker')
            rospy.Subscriber("point_3d", PointStamped,self.point_callback)
            rospy.Subscriber("imagepoint", PointStamped,self.imagepoint_callback)
            darknet_topic = "/darknet_ros/bounding_boxes"
            rospy.Subscriber(darknet_topic, BoundingBoxes, self.yolo_callback)
            objinfo_topic = "/ObjectInfos"
            rospy.Subscriber(objinfo_topic, ObjectInfoArray, self.object_callback)
            self.occ_pose_pub = rospy.Publisher("occluder_pose",PoseStamped,queue_size=10)
            self.fovpoints_pub = rospy.Publisher("fov_regions",fov_positions,queue_size=10)
            self.fovpoints= fov_positions()
            # rospy.Publisher()

            self.object_poseStamped= PoseStamped()
            self.object_poseStamped2= PoseStamped()
            self.object_poseStamped.pose  = Pose(Point(0.2, -0.1, 1.0), Quaternion())
            self.object_poseStamped.pose.orientation.x=0.7071068
            self.object_poseStamped.pose.orientation.y=0
            self.object_poseStamped.pose.orientation.z=0
            self.object_poseStamped.pose.orientation.w=0.7071068

            # self.object_poseStamped.header.frame_id=_MAP_TF
            self.object_poseStamped.header.frame_id=_SESNSOR_TF 
            self.object_poseStamped2.header.frame_id=_SESNSOR_TF 

            self.tflistener = tf.TransformListener()
            self.tflistener.waitForTransform(_MAP_TF, _SESNSOR_TF, rospy.Time(), rospy.Duration(10.0))

           #Frustum
            self.Frustum = Frustum()
            #this pose are represented w.r.t head_rgbd_sensor_frame
            # usually z is the direction of depth
            self.x_pose=1.00
            self.y_pose=0.00
            self.z_pose=0.00

            self.image_u=0.00
            self.image_v=0.00
            self.image_d=0.88
            self.xmin=0
            self.ymin=0
            self.xmax=0
            self.ymax=0

            self.is_target=False

            self.infoarray = ObjectInfoArray()
            self.object_height=0.0
            self.object_radius=0.0
            # self.object_poseStamped= PoseStamped()
            # self.object_poseStamped.header.frame_id=_MAP_TF

        def point_callback(self,msg):
            # rospy.loginfo("hello")
            self.x_pose=msg.point.x
            self.y_pose=msg.point.y
            self.z_pose=msg.point.z

        def imagepoint_callback(self,msg):
            # rospy.loginfo("hello")
            self.image_u=int(msg.point.x)
            self.image_v=int(msg.point.y)
            self.image_d=msg.point.z

        def object_callback(self,msg):
            self.infoarray=msg
            for index in range(len(self.infoarray.objectinfos)):
                if self.infoarray.objectinfos[index].label == "bottle":
                    self.z_pose= self.infoarray.objectinfos[index].average_depth
                    object_point = self.infoarray.objectinfos[index].center

                    self.object_poseStamped.header.frame_id = _MAP_TF 
                    self.object_poseStamped.header.stamp=rospy.Time.now()

                    self.object_poseStamped.pose  = Pose(object_point, Quaternion())
                    self.object_radius =0.2;
                    self.object_height =0.25;
                    # self.occ_pose_pub.publish(self.object_poseStamped)

        def yolo_callback(self,msg):
            # rospy.loginfo("yolo_callback")
            self.bounding_boxes = msg.bounding_boxes
            if len(self.bounding_boxes)>0:
                self.is_target=True
                self.xmin=self.bounding_boxes[0].xmin
                self.xmax=self.bounding_boxes[0].xmax
                self.ymin=self.bounding_boxes[0].ymin
                self.ymax=self.bounding_boxes[0].ymax
                self.distance =0.5
                # rospy.loginfo("xmin: %.2lf, xmax: %.2lf, ymin: %.2lf, ymax: %.2lf", self.xmin, self.xmax, self.ymin, self.ymax)


        def listener(self):
            self.drawing()

        def drawing(self):
            while not rospy.is_shutdown():

                  # self.is_target=True
                  #parameters for test
                  # self.x_pose=-0.09
                  # self.y_pose=1.18
                  # self.z_pose=0.88
                  FOV_horizontal=62.0*2*math.pi/360.0 #kinect camera info
                  FOV_vertical=48.6*2*math.pi/360.0   #kinect camera info
                  self.x_pose=1.0 #min_depth 
                  test_point=Point(self.x_pose,self.y_pose,self.z_pose)

                  #plane at the air 
                  plane_height = self.x_pose*2*math.tan(0.5*FOV_vertical)
                  plane_width = self.x_pose*2*math.tan(0.5*FOV_horizontal)
                  originpoint = Point(0,0,0)

                  # Publish a plane using a ROS Pose Msg
                  height= plane_height
                  width = plane_width
                  vertical_resolution=plane_height/HEIGHT_RES
                  horizontal_resolution=plane_width/WIDTH_RES
                  # offset from sensor
                  x_offset=(WIDTH_RES*0.5-self.image_u)*horizontal_resolution;
                  y_offset=(HEIGHT_RES*0.5-self.image_v)*vertical_resolution;
                  # P = Pose(Point(self.x_pose+x_offset, self.y_pose+y_offset,self.z_pose),Quaternion(0,0,0,1))
                  P = Pose(Point(self.x_pose, self.y_pose,self.z_pose),Quaternion(0,0,0,1))
                  P2 = Pose(Point(self.x_pose, self.y_pose,self.z_pose),Quaternion(0,1,0,1))
                  d=math.sqrt(P2.orientation.x**2+P2.orientation.y**2+P2.orientation.z**2+P2.orientation.w**2)
                  P2.orientation.x=P2.orientation.x/d
                  P2.orientation.y=P2.orientation.y/d
                  P2.orientation.z=P2.orientation.z/d
                  P2.orientation.w=P2.orientation.w/d

                  test_point_ur=Point(self.x_pose, self.y_pose+0.5*plane_width,self.z_pose-0.5*plane_height)
                  test_point_ul=Point(self.x_pose,self.y_pose-0.5*plane_width,self.z_pose-0.5*plane_height)
                  test_point_ll=Point(self.x_pose,self.y_pose-0.5*plane_width,self.z_pose+0.5*plane_height)
                  test_point_lr=Point(self.x_pose,self.y_pose+0.5*plane_width,self.z_pose+0.5*plane_height)

                  #FOV plane
                  # self.markers.publishPlane(P, width, height, 'translucent_light', 3.0) # pose, depth, width, color, lifetime
                  self.markers.publishPlane(P2, plane_height, plane_width, 'yellow', 3.0) # pose, depth, width, color, lifetime

                  mod_xmin_offset =-self.xmin*horizontal_resolution+0.5*plane_width
                  mod_xmax_offset =-self.xmax*horizontal_resolution+0.5*plane_width
                  mod_ymin_offset =-self.ymin*vertical_resolution+0.5*plane_height
                  mod_ymax_offset =-self.ymax*vertical_resolution+0.5*plane_height
                  # print("self.xmin", self.xmin)
                  # print("self.mod_xmin_offset", mod_xmin_offset)
                  # print("self.mod_xmax_offset", mod_xmax_offset)
                  # print("self.mod_ymin_offset", mod_ymin_offset)
                  # print("self.mod_ymax_offset", mod_ymax_offset)

                  # rospy.loginfo("bbxmin: %.2lf, bbxmax: %.2lf, bbymin: %.2lf,bbymax: %.2lf", mod_xmin_offset,mod_xmax_offset,
                                    # mod_ymin_offset,mod_ymax_offset)

                  #bounding box plane
                  # bb_point_ul=Point(self.x_pose-0.25, mod_xmin_offset, mod_ymin_offset)
                  # bb_point_ur=Point(self.x_pose-0.25,mod_xmax_offset, mod_ymin_offset)
                  # bb_point_dl=Point(self.x_pose-0.25, mod_xmin_offset, mod_ymax_offset)
                  # bb_point_dr=Point(self.x_pose-0.25, mod_xmax_offset, mod_ymax_offset)

                  # self.markers.publishLine(bb_point_ul, bb_point_ur, 'blue', 0.2, 3) # point1, point2, color, width, lifetime
                  # self.markers.publishLine(bb_point_ur, bb_point_dr, 'blue', 0.2, 3) # point1, point2, color, width, lifetime
                  # self.markers.publishLine(bb_point_dl, bb_point_dr, 'blue', 0.2, 3) # point1, point2, color, width, lifetime
                  # self.markers.publishLine(bb_point_dl, bb_point_ul, 'blue', 0.2, 3) # point1, point2, color, width, lifetime

                  # bb_point_ul_off=Point(self.x_pose+0.25, mod_xmin_offset, mod_ymin_offset)
                  # bb_point_ur_off=Point(self.x_pose+0.25,mod_xmax_offset, mod_ymin_offset)
                  # bb_point_dl_off=Point(self.x_pose+0.25, mod_xmin_offset, mod_ymax_offset)
                  # bb_point_dr_off=Point(self.x_pose+0.25, mod_xmax_offset, mod_ymax_offset)

                  # self.markers.publishLine(bb_point_ul, bb_point_ul_off, 'blue', 0.2, 3) # point1, point2, color, width, lifetime
                  # self.markers.publishLine(bb_point_ur, bb_point_ur_off, 'blue', 0.2, 3) # point1, point2, color, width, lifetime
                  # self.markers.publishLine(bb_point_dl, bb_point_dl_off, 'blue', 0.2, 3) # point1, point2, color, width, lifetime
                  # self.markers.publishLine(bb_point_dr, bb_point_dr_off, 'blue', 0.2, 3) # point1, point2, color, width, lifetime

                  #plane 2
                  # self.x_pose=0.09
                  # self.y_pose=0.014
                  self.x_pose2=3.00 #max depth

                  plane_height2 = self.x_pose2*2*math.tan(0.5*FOV_vertical)
                  plane_width2 = self.x_pose2*2*math.tan(0.5*FOV_horizontal)

                  vertical_resolution2=plane_height2/HEIGHT_RES
                  horizontal_resolution2=plane_width2/WIDTH_RES

                  # projected target bounding box to the rear plane
                  mod_xmin_offset2 =-self.xmin*horizontal_resolution2+0.5*plane_width2
                  mod_xmax_offset2 =-self.xmax*horizontal_resolution2+0.5*plane_width2
                  mod_ymin_offset2 =-self.ymin*vertical_resolution2+0.5*plane_height2
                  mod_ymax_offset2 =-self.ymax*vertical_resolution2+0.5*plane_height2
 

                  # target_pose
                  P = Pose(Point(self.x_pose2,self.y_pose,self.z_pose),Quaternion(0,1,0,1))
                  d=math.sqrt(P.orientation.x**2+P.orientation.y**2+P.orientation.z**2+P.orientation.w**2)
                  P.orientation.x=P.orientation.x/d
                  P.orientation.y=P.orientation.y/d
                  P.orientation.z=P.orientation.z/d
                  P.orientation.w=P.orientation.w/d

                  self.markers.publishPlane(P, plane_height2, plane_width2, 'yellow', 3.0) # pose, depth, width, color, lifetime

                  #lines connecting origin to the vertexes of plane 2

                  test_point_ur2=Point(self.x_pose2, self.y_pose+0.5*plane_width2,self.z_pose-0.5*plane_height2)
                  test_point_ul2=Point(self.x_pose2, self.y_pose-0.5*plane_width2,self.z_pose-0.5*plane_height2)
                  test_point_ll2=Point(self.x_pose2, self.y_pose-0.5*plane_width2,self.z_pose+0.5*plane_height2)
                  test_point_lr2=Point(self.x_pose2, self.y_pose+0.5*plane_width2,self.z_pose+0.5*plane_height2)

                  # Drawlines
                  self.markers.publishLine(originpoint, test_point_ur, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(originpoint, test_point_ul, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(originpoint, test_point_ll, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(originpoint, test_point_lr, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  # Publish a set of spheres using

                  self.markers.publishLine(test_point_ur2, test_point_ur, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(test_point_ul2, test_point_ul, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(test_point_ll2, test_point_ll, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(test_point_lr2, test_point_lr, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime


                  #fake cylinder
                  if self.is_target==True:
                   
                      self.object_radius =0.2
                      self.object_height =0.25
                      self.object_poseStamped.pose.position.x=self.x_pose
                      self.object_poseStamped.pose.position.y=(mod_xmin_offset+mod_xmax_offset)*0.5
                      self.object_poseStamped.pose.position.z=(mod_ymin_offset+mod_ymax_offset)*0.5
                      self.object_poseStamped.pose.orientation.x=0.0
                      self.object_poseStamped.pose.orientation.y=0.0
                      self.object_poseStamped.pose.orientation.z=0.0
                      self.object_poseStamped.pose.orientation.w=1.0
                      self.markers.publishCylinder(self.object_poseStamped.pose,'green', self.object_height, self.object_radius, 6.0)
                      self.object_poseStamped.header.frame_id=_SESNSOR_TF
                      self.object_poseStamped.header.stamp=rospy.Time.now()
                      self.occ_pose_pub.publish(self.object_poseStamped)

                      self.object_poseStamped2.pose.position=Point(self.x_pose2, self.y_pose, self.z_pose)
                      self.object_poseStamped2.pose.position.y=(mod_xmin_offset2+mod_xmax_offset2)*0.5
                      self.object_poseStamped2.pose.position.z=(mod_ymin_offset2+mod_ymax_offset2)*0.5
                      self.object_poseStamped2.pose.orientation.x=0.0
                      self.object_poseStamped2.pose.orientation.y=0.0
                      self.object_poseStamped2.pose.orientation.z=0.0
                      self.object_poseStamped2.pose.orientation.w=1.0
                      self.markers.publishCylinder(self.object_poseStamped2.pose,'blue', self.object_height*2, self.object_radius*1.5, 6.0)
                  

                      #Fill fov messeges
                      self.fovpoints.positions=[]
                      self.fovpoints.header.stamp=rospy.Time.now()
                      self.fovpoints.header.frame_id = _SESNSOR_TF
                      self.fovpoints.positions.append(self.object_poseStamped.pose.position)
                      self.fovpoints.positions.append(self.object_poseStamped2.pose.position)

                      Margin=1.5
                      # fl_point = Point()
                      # fl_point.x =self.object_poseStamped.pose.position.x
                      # fl_point.y=self.object_poseStamped.pose.position.y-Margin
                      # fl_point.z=self.object_poseStamped.pose.position.z-Margin

                      # fr_point = Point()
                      # fr_point.x =self.object_poseStamped.pose.position.x
                      # fr_point.y=self.object_poseStamped.pose.position.y+Margin
                      # fr_point.z=self.object_poseStamped.pose.position.z-Margin

                      # fu_point = Point()
                      # fu_point.x =self.object_poseStamped.pose.position.x
                      # fu_point.y=self.object_poseStamped.pose.position.y-Margin
                      # fu_point.z=self.object_poseStamped.pose.position.z+Margin

                      # fd_point = Point()
                      # fd_point.x =self.object_poseStamped.pose.position.x
                      # fd_point.y=self.object_poseStamped.pose.position.y+Margin
                      # fd_point.z=self.object_poseStamped.pose.position.z+Margin

                      # fl_point2 = Point()
                      # fl_point2.x =self.object_poseStamped2.pose.position.x
                      # fl_point2.y=self.object_poseStamped2.pose.position.y-Margin
                      # fl_point2.z=self.object_poseStamped2.pose.position.z-Margin

                      # fr_point2 = Point()
                      # fr_point2.x =self.object_poseStamped2.pose.position.x
                      # fr_point2.y=self.object_poseStamped2.pose.position.y+Margin
                      # fr_point2.z=self.object_poseStamped2.pose.position.z-Margin

                      # fu_point2 = Point()
                      # fu_point2.x =self.object_poseStamped2.pose.position.x
                      # fu_point2.y=self.object_poseStamped2.pose.position.y-Margin
                      # fu_point2.z=self.object_poseStamped2.pose.position.z+Margin

                      # fd_point2 = Point()
                      # fd_point2.x =self.object_poseStamped2.pose.position.x
                      # fd_point2.y=self.object_poseStamped2.pose.position.y+Margin
                      # fd_point2.z=self.object_poseStamped2.pose.position.z+Margin

                      # self.fovpoints.positions.append(fl_point)
                      # self.fovpoints.positions.append(fr_point)
                      # self.fovpoints.positions.append(fu_point)
                      # self.fovpoints.positions.append(fd_point)
                      # self.fovpoints.positions.append(fl_point2)
                      # self.fovpoints.positions.append(fr_point2)
                      # self.fovpoints.positions.append(fu_point2)
                      # self.fovpoints.positions.append(fd_point2)
                      # for point in self.fovpoints.positions:
                          # print(point)

                      self.fovpoints_pub.publish(self.fovpoints)
                      # self.fovpoints.positions.resize(8); #Frustum points
                      # self.fovpoints.positions.push_back(test_point_clear())


                  # rospy.loginfo("published")
                  # self.listener = tf.TransformListener()
                  # print trans_object_pose
                  rospy.Rate(10).sleep() #1 Hz


        def drawing_hoz(self):
            while not rospy.is_shutdown():

                  #parameters for test
                  FOV_horizontal=58.0*2*math.pi/360.0
                  FOV_vertical=45.0*2*math.pi/360.0

                  test_point=Point(self.x_pose,self.y_pose,self.z_pose)

                  # Axis:
                  # T = transformations.translation_matrix((1,0,0))
                  # axis_length = 0.4
                  # axis_radius = 0.05
                  # self.markers.publishAxis(T, axis_length, axis_radius, 5.0) # pose, axis length, radius, lifetime

                  # P = Pose(test_point,Quaternion(0,0,0,1))
                  # axis_length = 0.3
                  # axis_radius = 0.02
                  # samples_markers.publishAxis(P, axis_length, axis_radius, 5.0) # pose, axis length, radius, lifetime

                  # br=tf.TransformBroadcaster()
                  # br.sendTransform((self.x_pose,self.y_pose,self.z_pose),tf.transformations.quaternion_from_euler(0,0,0),rospy.Time.now(),"sample_axis","head_rgbd_sensor_rgb_frame")

                  #plane at the air 
                  plane_height = self.z_pose*2*math.tan(0.5*FOV_vertical)
                  plane_width = self.z_pose*2*math.tan(0.5*FOV_horizontal)
                  originpoint = Point(0,0,0)

                  # Publish a plane using a ROS Pose Msg
                  height= plane_height
                  width = plane_width
                  vertical_resolution=height/480
                  horizontal_resolution=width/640
                  # offset from sensor
                  x_offset=(320-self.image_u)*horizontal_resolution;
                  y_offset=(240-self.image_v)*vertical_resolution;
                  # P = Pose(Point(self.x_pose+x_offset, self.y_pose+y_offset,self.z_pose),Quaternion(0,0,0,1))
                  P = Pose(Point(self.x_pose, self.y_pose,self.z_pose),Quaternion(0,0,0,1))

                  test_point_ur=Point(self.x_pose+0.5*plane_width,self.y_pose-0.5*plane_height,self.z_pose)
                  test_point_ul=Point(self.x_pose-0.5*plane_width,self.y_pose-0.5*plane_height,self.z_pose)
                  test_point_ll=Point(self.x_pose-0.5*plane_width,self.y_pose+0.5*plane_height,self.z_pose)
                  test_point_lr=Point(self.x_pose+0.5*plane_width,self.y_pose+0.5*plane_height,self.z_pose)
                  # test_point_ur=Point(self.x_pose+0.5*plane_width+x_offset,self.y_pose-0.5*plane_height+y_offset,self.z_pose)
                  # test_point_ul=Point(self.x_pose-0.5*plane_width+x_offset,self.y_pose-0.5*plane_height+y_offset,self.z_pose)
                  # test_point_ll=Point(self.x_pose-0.5*plane_width+x_offset,self.y_pose+0.5*plane_height+y_offset,self.z_pose)
                  # test_point_lr=Point(self.x_pose+0.5*plane_width+x_offset,self.y_pose+0.5*plane_height+y_offset,self.z_pose)

                  #projection test
                  '''
                  temp_x = np.random.uniform(-2.0, 2.0)
                  temp_y = np.random.uniform(-1.0, 1.0)
                  test_query = Point(temp_x, temp_y, 1.6)
                  projection_beforepose= Pose(test_query, Quaternion())
                  self.markers.publishSphere(projection_beforepose, 'yellow',0.2,5)
                  fplane_coeffs = self.Frustum.getPlanewithThreepoints(test_point_ul, test_point_ll,test_point_ur)
                  ppoint = projection_point_to_plane(fplane_coeffs, test_query) 
                  print "projection point"
                  print ppoint
                  projection_pose= Pose(ppoint , Quaternion())
                  self.markers.publishSphere(projection_pose, 'black',0.2,5)
                  '''
                  #FOV plane
                  # self.markers.publishPlane(P, width, height, 'translucent_light', 3.0) # pose, depth, width, color, lifetime
                  self.markers.publishPlane(P, width, height, 'yellow', 3.0) # pose, depth, width, color, lifetime

                  mod_xmin_offset =self.xmin*horizontal_resolution-0.5*plane_width
                  mod_xmax_offset =self.xmax*horizontal_resolution-0.5*plane_width
                  mod_ymin_offset =self.ymin*vertical_resolution-0.5*plane_height
                  mod_ymax_offset =self.ymax*vertical_resolution-0.5*plane_height

                  # rospy.loginfo("bbxmin: %.2lf, bbxmax: %.2lf, bbymin: %.2lf,bbymax: %.2lf", mod_xmin_offset,mod_xmax_offset,
                                    # mod_ymin_offset,mod_ymax_offset)

                  #bounding box plane
                  bb_point_ul=Point(mod_xmin_offset, mod_ymin_offset, self.z_pose)
                  bb_point_ur=Point(mod_xmax_offset, mod_ymin_offset, self.z_pose)
                  bb_point_dl=Point(mod_xmin_offset, mod_ymax_offset, self.z_pose)
                  bb_point_dr=Point(mod_xmax_offset, mod_ymax_offset, self.z_pose)

                  self.markers.publishLine(bb_point_ul, bb_point_ur, 'red', 0.01, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(bb_point_ur, bb_point_dr, 'red', 0.01, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(bb_point_dl, bb_point_dr, 'red', 0.01, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(bb_point_dl, bb_point_ul, 'red', 0.01, 0.3) # point1, point2, color, width, lifetime

                  #plane 2
                  # self.x_pose=0.09
                  # self.y_pose=0.014
                  self.z_pose2=2.50

                  plane_height2 = self.z_pose2*2*math.tan(0.5*FOV_vertical)
                  plane_width2 = self.z_pose2*2*math.tan(0.5*FOV_horizontal)

                  P = Pose(Point(self.x_pose,self.y_pose,self.z_pose2),Quaternion(0,0,0,1))
                  height= plane_height2
                  width = plane_width2
                  self.markers.publishPlane(P, width, height, 'yellow', 3.0) # pose, depth, width, color, lifetime

                  #lines connecting origin to the vertexes of plane 2

                  test_point_ur2=Point(self.x_pose+0.5*plane_width2,self.y_pose-0.5*plane_height2,self.z_pose2)
                  test_point_ul2=Point(self.x_pose-0.5*plane_width2,self.y_pose-0.5*plane_height2,self.z_pose2)
                  test_point_ll2=Point(self.x_pose-0.5*plane_width2,self.y_pose+0.5*plane_height2,self.z_pose2)
                  test_point_lr2=Point(self.x_pose+0.5*plane_width2,self.y_pose+0.5*plane_height2,self.z_pose2)

                  # Drawlines
                  self.markers.publishLine(originpoint, test_point_ur, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(originpoint, test_point_ul, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(originpoint, test_point_ll, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(originpoint, test_point_lr, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  # Publish a set of spheres using

                  self.markers.publishLine(test_point_ur2, test_point_ur, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(test_point_ul2, test_point_ul, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(test_point_ll2, test_point_ll, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime
                  self.markers.publishLine(test_point_lr2, test_point_lr, 'green', 0.1, 0.3) # point1, point2, color, width, lifetime

                  # projected target bounding box to the rear plane
                  mod_xmin_offset2 =self.xmin*horizontal_resolution-0.5*plane_width2
                  mod_xmax_offset2 =self.xmax*horizontal_resolution-0.5*plane_width2
                  mod_ymin_offset2 =self.ymin*vertical_resolution-0.5*plane_height2
                  mod_ymax_offset2 =self.ymax*vertical_resolution-0.5*plane_height2





                  #fake cylinder

                  self.object_radius =0.2
                  self.object_height =0.25
                  self.markers.publishCylinder(self.object_poseStamped.pose,'green', self.object_height, self.object_radius, 6.0)
                  self.object_poseStamped.header.frame_id=_SESNSOR_TF
                  self.object_poseStamped.header.stamp=rospy.Time.now()
                  self.occ_pose_pub.publish(self.object_poseStamped)
                  # rospy.loginfo("published")

                  # self.listener = tf.TransformListener()
                  '''
                  self.tflistener.waitForTransform(_MAP_TF, _SESNSOR_TF, rospy.Time(), rospy.Duration(10.0))
                  try:
                      trans_object_pose= self.tflistener.transformPose(_MAP_TF, self.object_poseStamped)
                      trans_object_pose.header.stamp=rospy.Time.now()
                      self.occ_pose_pub.publish(trans_object_pose)
                      # self.markers.publishCylinder(trans_object_pose.pose,'green', self.object_height, self.object_radius, 6.0)
                  except:
                      rospy.loginfo("trans error")
                  '''

                  # print trans_object_pose
                  rospy.Rate(10).sleep() #1 Hz




        def cleanup_node(self):
            print "Shutting down node"
            self.markers.deleteAllMarkers()

if __name__=='__main__':
	rospy.init_node('plane_plot', anonymous=False, log_level=rospy.INFO, disable_signals=False)
        draw_manager=DrawManager()
        draw_manager.listener()
        rospy.on_shutdown(draw_manager.cleanup_node)



