#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Terrance Williams
@title: ss04_tf2Watcher
@date: 6 December 2023
@description:
    A node to send request transforms between relevant frames.
    Using Python 2.7 because ROS Melodic's tf2_ros package is not
    compiled for Python3. 
"""
from __future__ import print_function

import rospy
import tf2_ros
from geometry_msgs.msg import Transform, TransformStamped
from uav_follower.srv import TF2Poll, TF2PollResponse

"""
The idea of this node is to listen to the transform from jethexa/map to
jethexa/base_link. Upon request from ss03_DataProcessor, this node grabs the
latest transform, gets its Vector3 message, and sends it in response.
"""
class tf2Watcher:
    def __init__(self):        
        rospy.init_node('ss04_tf2Watcher', log_level=rospy.INFO)
        self.name = rospy.get_name()
        self.map_frame = 'jethexa/map'
        self.base_frame = 'jethexa/base_link'
        # tf2 setup
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf2_pub = rospy.Publisher(
            "/tf2Pub",
            TransformStamped,
            queue_size=10
        )
        timeout = 1
        rospy.loginfo('{}: Online.'.format(self.name))

        while not rospy.is_shutdown():
            try:
                # Get TransformStamped msg
                current_transf = self.tfBuffer.lookup_transform(
                    self.map_frame, 
                    self.base_frame,
                    rospy.Time.now(),
                    rospy.Duration(timeout)
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException
            ) as e:
                print("No frames.\n", e)
                # what are messages initialized as?            
            else:
                # print("<{}>: current_transf".format(self.name)
                self.tf2_pub.publish(current_transf)



if __name__ == "__main__":
    try:
        tf2Watcher()
    except rospy.ROSInterruptException:
        pass

