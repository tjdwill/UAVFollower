#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Terrance Williams
@title: tf2_pub
@date: 7 March 2024
@description:
    This is a node I wrote to enable getting a specific tf tranform. I couldn't figure out
    how to use the correct `rosbag filter` command that only selects for a specific tf transform, so 
    I run this node while replaying a bag file with the entire tf playback. 

    Play a rosbag while using a simulated clock and then run this node with the desired transform link.
    Then, save the results of this topic to another bag for later extraction. 
    Probably "inelegant" as a solution, but it worked.
"""
# from __future__ import print_function

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
            else:
                self.tf2_pub.publish(current_transf)


if __name__ == "__main__":
    try:
        tf2Watcher()
    except rospy.ROSInterruptException:
        pass
