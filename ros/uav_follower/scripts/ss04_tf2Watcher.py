#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function

import rospy
import tf2_ros
from geometry_msgs.msg import Vector3
from uav_follower.srv import TF2Poll

"""
The idea of this node is to listen to the transform from jethexa/base_link to jethexa_map.
Upon request from ss03_DataProcessor, this node grabs the latest transform,
gets its Vector3 message, and sends it in response.
"""
def response(*args, **kwargs):
    """Get transform, extract Vector3, send in response"""
    current_transf = tfBuffer.lookup_transform(
            "jethexa/map", 
            "jethexa/base_link",
            rospy.Time.now(),
            rospy.Duration(1.25)
    )
    print("<{}>: current_transf".format(name))
    translation = current_transf.transform.translation
    # print(translation)
    # print(type(translation))
    return translation



name = 'ss04_tf2Watcher'
rospy.init_node(name, log_level=rospy.INFO)
tfBuffer = tf2_ros.Buffer()
tfListener = tf2_ros.TransformListener(tfBuffer)
tf2_srv = rospy.Service(
        'tf2_poll',
        TF2Poll,
        response
)
rospy.spin()


