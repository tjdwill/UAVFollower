#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@date: 7 November 2023
@description:
    This program defines the node for coordinating communication
    for other subsystem nodes.
"""


import numpy as np
import rospy
from rosnp_msgs.rosnp_helpers import encode_rosnp_list
from rosnp_msgs.msg import ROSNumpyList_UInt16
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from uav_follower.srv import DepthImgReq, DepthImgReqResponse

class NodeLiaison:
    def __init__(self) -> None:
        ...
        rospy.init_node('ss03_Liaison', log_level=rospy.INFO)

        topics = rospy.get_param('topic')
        self.depth_req = rospy.Service(
            topics['depth_req'],
            DepthImgReq,
            self.depth_callback)
        self.resume_trigger = rospy.ServiceProxy(
            topics['resume_trigger'],
            Empty
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_raw',
            Image,
            self.depth_img_handler
        )

        self.collect = False
        self.imgs = []
        self.amount = -1
        rospy.spin()

    def depth_callback(self, req: DepthImgReq):
        ...
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
        self.resume_trigger(Empty())


if __name__ == '__main__':
    try:
        NodeLiaison()
    except rospy.ROSInterruptException:
        pass
