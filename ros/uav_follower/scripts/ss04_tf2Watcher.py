#!/usr/bin/env python2.7
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
from geometry_msgs.msg import Vector3
from uav_follower.srv import TF2Poll

"""
The idea of this node is to listen to the transform from jethexa/map to
jethexa/base_link. Upon request from ss03_DataProcessor, this node grabs the
latest transform, gets its Vector3 message, and sends it in response.
"""
class tf2Watcher:
    def __init__(self):        
        rospy.init_node('ss04_tf2Watcher', log_level=rospy.INFO)
        self.name = rospy.get_name()
        topics = rospy.get_param('topics')
        self.base_frame = rospy.get_param('~base_frame')
        self.map_frame = rospy.get_param('~map_frame')

        # tf2 setup
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf2_srv = rospy.Service(
                topics['tf2'],
                TF2Poll,
                self.response
        )
        rospy.loginfo(f'{self.name}: Online.')
        rospy.spin()

    def response(self, *args, **kwargs):
        """Get transform, extract Vector3, send in response"""
        current_transf = self.tfBuffer.lookup_transform(
                self.map_frame, 
                self.base_frame,
                rospy.Time.now(),
                rospy.Duration(1.25)
        )
        print("<{name}>: current_transf".format(self.name))
        translation = current_transf.transform.translation
        assert(isinstance(translation, Vector3))
        # print(translation)
        # print(type(translation))
        return translation


if __name__ == "__main__":
    try:
        tf2Watcher()
    except rospy.ROSInterruptException:
        pass
