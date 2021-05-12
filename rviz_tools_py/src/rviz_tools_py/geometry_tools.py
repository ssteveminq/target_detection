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


# Python includes
import math
import numpy
import numpy as np
from numpy import linalg as LA
import random # randint

# ROS includes
import roslib
import rospy
import tf # tf/transformations.py
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point, Point32
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Polygon
from visualization_msgs.msg import Marker


class Frustum(object):
    """
    A class for Frustum ( Field of View volume)
    """

    def __init__(self):
        self.plane_coeffset=[]
        self.pointset=[]
        # self.plane_coeffset=[]

    def getPlanewithThreepoints(self, point1,point2,point3):
        """
        get plane with threepoints
        ax+by+cz = d
        normal vector is (a,b,c)
        """
        p1= np.array([point1.x, point1.y, point1.z])
        p2= np.array([point2.x, point2.y, point2.z])
        p3= np.array([point3.x, point3.y, point3.z])
        #two vectors in the plane
        v1 = p3-p1
        v2 = p2-p1
        cp = np.cross(v1,v2)
        # print cp
        a,b,c = cp/(LA.norm(cp))
        normal = np.array([a,b,c])
        # normal_vec = normal/ norm(normal)
        # print normal

        d = -np.dot(normal,p3)
        coeff = np.array([a,b,c,d])

        return coeff
#
    def getFrustumwithEightPoints(self, flt, flb, frt, frb,
                                rlt, rlb, rrt, rrb):
        """
        make Frustum with eight points 
        Sequence is very important to define Frustum
        Front plane : flt, flb, frt, frb
        Rear plane : rlt, rlb, rrt, rrb
        make 6 planes with three points inside plane
        store coefficient of six planes
        """
        self.pointset.append(flt)
        self.pointset.append(flb)
        self.pointset.append(frt)
        self.pointset.append(frb)
        self.pointset.append(rlt)
        self.pointset.append(rlb)
        self.pointset.append(rrt)
        self.pointset.append(rrb)
        # flt = pointarray[0]
        # flb = pointarray[1]
        # frt = pointarray[2]
        # frb = pointarray[3]
        # rlt = pointarray[4]
        # rlb = pointarray[5]
        # rrt = pointarray[6]
        # rrb = pointarray[7]

        self.plane_coeffset=[]
        self.plane_coeffset.append(self.getPlanewithThreepoints(flt,frt,rlt)) #top
        self.plane_coeffset.append(self.getPlanewithThreepoints(frb,flb,rrb)) #bottom
        self.plane_coeffset.append(self.getPlanewithThreepoints(flt,rlt,flb))  #left
        self.plane_coeffset.append(self.getPlanewithThreepoints(frb,rrb, frt)) #right
        self.plane_coeffset.append(self.getPlanewithThreepoints(flt,flb,frt)) #front plane
        self.plane_coeffset.append(self.getPlanewithThreepoints(rrt,rlb,rlt)) #rear plane
        # print self.plane_coeffset
    def querypointInFrustum(self, query):
        """
        Check if query point is inside of Frustum 
        """
        InsideFrustum = True
        queryp = np.array([query.x, query.y, query.z])
        # for i in range(len(self.plane_coeffset)):
        for coeff in self.plane_coeffset:
            normal = coeff[0:3]
            print normal
            value = np.dot(normal, queryp)+coeff[3]
            if value <0:
                InsideFrustum = False
                return False

        return InsideFrustum 



            # normal = np.array([coeff[0], coeff[1], coeff[2]])





    def getFrustumwithPointset(self, pointarray):
        """
        make Frustum with eight points 
        Front plane : flt, flb, frt, frb
        Rear plane : rlt, rlb, rrt, rrb
        make 6 planes with three points inside plane
        store coefficient of six planes
        """
        flt = pointarray[0]
        flb = pointarray[1]
        frt = pointarray[2]
        frb = pointarray[3]
        rlt = pointarray[4]
        rlb = pointarray[5]
        rrt = pointarray[6]
        rrb = pointarray[7]

        plane_coeffset=[]
        self.plane_coeffset.append(self.getPlanewithThreepoints(frt,flt,rlt)) #top
        self.plane_coeffset.append(self.getPlanewithThreepoints(flb,frb,rrb)) #bottom
        self.plane_coeffset.append(self.getPlanewithThreepoints(flt,flb,rlb))  #left
        self.plane_coeffset.append(self.getPlanewithThreepoints(frb,frt,rrb)) #right
        self.plane_coeffset.append(self.getPlanewithThreepoints(flt,frt,frb)) #front plane
        self.plane_coeffset.append(self.getPlanewithThreepoints(rrt,rlt,rlb)) #rear plane
        print self.plane_coeffset


    def publishPolygon(self, polygon, color, width, lifetime=None):
        """
        Publish a polygon Marker.

        @param polygon (ROS Polygon)
        @param color name (string) or RGB color value (tuple or list)
        @param width line width (float)
        @param lifetime (float, None = never expire)

        a path with the start and end points connected
        """

        if (self.muted == True):
            return True

        # Check input
        if type(polygon) == Polygon:
            polygon_msg = polygon
        else:
            rospy.logerr("Path is unsupported type '%s' in publishPolygon()", type(polygon).__name__)
            return False

        # Copy points from ROS Polygon Msg into a list
        polygon_path = []
        for i in range(0, len(polygon_msg.points)):
            x = polygon_msg.points[i].x
            y = polygon_msg.points[i].y
            z = polygon_msg.points[i].z
            polygon_path.append( Point(x,y,z) )

        # Add the first point again
        x = polygon_msg.points[0].x
        y = polygon_msg.points[0].y
        z = polygon_msg.points[0].z
        polygon_path.append( Point(x,y,z) )

        return self.publishPath(polygon_path, color, width, lifetime)


 
#------------------------------------------------------------------------------#

def projection_point_to_plane(plane_coeff, querypoint):
    '''
    plane coefficient: [a,b,c,d] ax+by+cz+d=0
    query point : Point
    '''
    normal = np.array([plane_coeff[0], plane_coeff[1], plane_coeff[2]])
    query_point = np.array([querypoint.x,querypoint.y, querypoint.z])
    nominator =-(normal.dot(query_point)+plane_coeff[3])
    denominator = math.pow(plane_coeff[0],2)+math.pow(plane_coeff[1],2)+math.pow(plane_coeff[2],2)
    t = nominator/denominator 
    x = querypoint.x+plane_coeff[0]*t
    y = querypoint.y+plane_coeff[1]*t
    z = querypoint.z+plane_coeff[2]*t
    projection_point = Point(x,y,z)

    return projection_point


def tangent_point_circle_extpoint(center, radius, ext_point):
    '''
    http://www.ambrsoft.com/TrigoCalc/Circles2/CirclePoint/CirclePointDistance.htm
    circle : center, radius
    (x-a)^2+(y-b)^2=r^2
    external point :ext_point
    get tangent point on the circle 
    '''
    a = center.x
    b = center.y
    xp = ext_point.x
    yp = ext_point.y
    r = radius
    
    x1 = (math.pow(r,2)*(xp-a)+r*(yp-b)*math.sqrt(math.pow((xp-a),2)+math.pow((yp-b),2)-math.pow(r,2)))/(math.pow(xp-a,2)+math.pow(yp-b,2))+a
    x2 = (math.pow(r,2)*(xp-a)-r*(yp-b)*math.sqrt(math.pow((xp-a),2)+math.pow((yp-b),2)-math.pow(r,2)))/(math.pow(xp-a,2)+math.pow(yp-b,2))+a

    y1 = (math.pow(r,2)*(yp-b)-r*(xp-a)*math.sqrt(math.pow((xp-a),2)+math.pow((yp-b),2)-math.pow(r,2)))/(math.pow(xp-a,2)+math.pow(yp-b,2))+b
    y2 = (math.pow(r,2)*(yp-b)+r*(xp-a)*math.sqrt(math.pow((xp-a),2)+math.pow((yp-b),2)-math.pow(r,2)))/(math.pow(xp-a,2)+math.pow(yp-b,2))+b

    pointset=[]
    #left first
    if x1<x2:
        pointset.append(Point(x1,y1,0))
        pointset.append(Point(x2,y2,0))
    else:
        pointset.append(Point(x2,y2,0))
        pointset.append(Point(x1,y1,0))

    return pointset
    # print x1,y1
    # print x2,y2


    

def LinePlaneInterSection(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-4):

#   #converting to np array
    plane_normal=np.array([planeNormal.x, planeNormal.y,planeNormal.z])
    plane_point=np.array([planePoint.x, planePoint.y,planePoint.z])
    ray_direction=np.array([rayDirection.x, rayDirection.y,rayDirection.z])
    ray_point=np.array([rayPoint.x, rayPoint.y,rayPoint.z])

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray_direction + plane_point
    intersection = Point(Psi[0],Psi[1],Psi[2])

    return intersection





def point_inside_polygon(x, y, poly, include_edges=True):
    '''
    Test if point (x,y) is inside polygon poly.
    poly is N-vertices polygon defined as 
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)
    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times. 
    Works fine for non-convex polygons.
    '''
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    # print "p1x, p1y:"
    # print p1x, p1y
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        # print "p2x, p2y:"
        # print p2x, p2y
        if p1y == p2y:
            if y == p1y:
                # print "if y==p1y"
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                elif x < min(p1x, p2x):  # point is to the left from current edge
                    # print "else x< min(p1x,p2x)"
                    inside = not inside
        else:  # p1y!= p2y
            # print "else"
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                # print " mi()<Y<max"
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    # print " x==xinters"
                    inside = include_edges
                    break
                if x < xinters:  # point is to the left from current edge
                    # print " x<xinters"
                    inside = not inside

        p1x, p1y = p2x, p2y
    return inside


