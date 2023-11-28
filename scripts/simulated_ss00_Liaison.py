#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@date: 8 November 2023
@description:
    This program defines the node for simulating ss00_Liaison
    for testing purposes.

    May write a version that doesn't use the depth camera but instead
    simulates those images as well.
"""


import numpy as np
import rospy
from rosnp_msgs.rosnp_helpers import encode_rosnp_list
from std_srvs.srv import Empty, EmptyResponse
from sensor_msgs.msg import Image
from uav_follower.srv import DepthImgReq, DepthImgReqResponse
from geometry_msgs.msg import PointStamped


class NodeLiaison:
    def __init__(self) -> None:
        rospy.init_node('ss00_Liaison', log_level=rospy.INFO)
        self.name = rospy.get_name()
        self.collect = False
        self.imgs = []
        self.amount = -1

        topics = rospy.get_param('topics')
        self.depth_req = rospy.Service(
            topics['depth_req'],
            DepthImgReq,
            self.depth_callback
        )
        self.resume_trigger = rospy.ServiceProxy(
            topics['resume_trigger'],
            Empty
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_raw',
            Image,
            self.depth_img_handler
        )
        self.waypoint_sub = rospy.Subscriber(
            topics['waypoints'],
            PointStamped,
            self.waypoint_callback
        )
        
        self.bad_detection = rospy.Service(
            topics['bad_detections'],
            Empty,
            self.bad_detection_action
        )
        rospy.loginfo(f'{self.name}: Online')
        rospy.spin()
    
    def bad_detection_action(self, req):
        self.resume_trigger()
        return EmptyResponse()

    def depth_callback(self, req: DepthImgReq):
        self.amount = req.amount
        self.collect = True
        # Block until images collected 
        while len(self.imgs) != self.amount:
            pass
        else:
            self.collect = False
            msg = DepthImgReqResponse(
                depth_imgs=encode_rosnp_list(self.imgs)
            )
            self.amount = -1
            self.imgs.clear()
            return msg

    def depth_img_handler(self, msg: Image):
        if self.collect and len(self.imgs) < self.amount:
            shape = (msg.height, msg.width)
            data = msg.data  # Python 'bytes' object
            
            # Depth images are 16UC1 in this platform
            arr = np.ndarray(shape=shape, dtype=np.uint16, buffer=data)
            self.imgs.append(arr)
        else:
            pass

    def send_resume_signal(self):
        self.resume_trigger()

    def waypoint_callback(self, msg):
        print(f"ss00:\n{msg}")
        self.send_resume_signal()

if __name__ == '__main__':
    try:
        NodeLiaison()
    except rospy.ROSInterruptException:
        pass

