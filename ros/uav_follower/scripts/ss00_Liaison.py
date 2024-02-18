#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@title: ss00_Liaison
@date: 7 November 2023
@description:
    This program defines the node for coordinating communication
    for other subsystem nodes.
"""


import numpy as np
import rospy
from rosnp_msgs.rosnp_helpers import encode_rosnp_list
from move_base_msgs.msg import MoveBaseActionResult
from std_srvs.srv import Empty, EmptyResponse
from sensor_msgs.msg import Image
from uav_follower.srv import DepthImgReq, DepthImgReqResponse


class NodeLiaison:
    def __init__(self) -> None:
        rospy.init_node('ss00_Liaison', log_level=rospy.INFO)

        self.name = rospy.get_name()
        self.test_mode = rospy.get_param('test_mode')
        topics = rospy.get_param('topics')
        self.depth_req = rospy.Service(
            topics['depth_req'],
            DepthImgReq,
            self.depth_callback
        )
        self.bad_detection = rospy.Service(
            topics['bad_detections'],
            Empty,
            self.bad_detect_action
        )
        self.resume_trigger = rospy.ServiceProxy(
            topics['resume_trigger'],
            Empty
        )
        self.depth_sub = rospy.Subscriber(
            rospy.get_param("~depth_topic"),
            Image,
            self.depth_img_handler
        )
        
        if self.test_mode:
            # For tests in which we don't want to run the full navigation
            # but still want continuous detections.
            from geometry_msgs.msg import PointStamped
            self.test_sub = rospy.Subscriber(
                rospy.get_param("~waypoints"),
                PointStamped,
                self.test_version_resume
            )
        else:
            self.move_base_sub = rospy.Subscriber(
                rospy.get_param("~move_base_result"),
                MoveBaseActionResult,
                self.send_resume_signal
            )  
        # Machinery for depth image collection
        self.collect = False
        self.imgs = []
        self.amount = -1

        rospy.loginfo(f'{self.name}: Online.')
        rospy.spin()
    
    def bad_detect_action(self, req):
        """Send resume signal to ss02 and respond to ss03"""
        self.resume_trigger()
        return EmptyResponse()

    def depth_callback(self, req: DepthImgReq):
        """Handle ss03's request for depth images"""
        self.amount = req.amount
        self.collect = True
        # Block until images collected 
        while len(self.imgs) != self.amount:
            pass
        else:
            '''
            Toggling the boolean is the first thing to do to stop the
            collection. Do not send more depth images than requested.
            
            TODO: write an assertion in this node or handle it in ss03.
            '''
            self.collect = False
            msg = DepthImgReqResponse(
                depth_imgs=encode_rosnp_list(self.imgs)
            )
            self.amount = -1
            self.imgs.clear()
            return msg

    def depth_img_handler(self, msg: Image):
        """
        Collects depth images.

        Because it didn't seem feasible to resubscribe upon every request,
        this node subscribes to the depth camera topic for the entire duration
        of the system's runtime. As a result, there had to be a mechanism to
        only collect images when needed.

        This method hinges on the 'collect' bool and the length of the image
        list. Conditioning off of these two values prevents a race condition
        in which more images are collected than requested. In other words,
        this method won't begin collecting images until a request is received
        to do so. As a result, this method and `depth_callback` are
        tightly coupled.

        It's a way to bring synchronization to a decidely asynchronous system.
        Technically, the race condition is still possible, but it has yet to
        occur in a large variety of tests.
        """
        if self.collect and len(self.imgs) < self.amount:
            shape = (msg.height, msg.width)
            data = msg.data  # Python 'bytes' object
            
            # Depth images are 16UC1 in this platform
            # Must use np.ndarray initialization due to the data being
            # interpreted as bytes. Prevents a headache.
            # print(f"<{self.name}> Depth Encoding: {msg.encoding}\n")
            arr = np.ndarray(shape=shape, dtype=np.uint16, buffer=data)
            self.imgs.append(arr)
        else:
            pass

    def test_version_resume(self, msg):
        self.resume_trigger()
    
    def send_resume_signal(self, msg):
        """
        Resume data collection if last movement was successful.
        """
        # A visual test to see if we get multiple messages over one move
        # rospy.loginfo(f'{self.name} Received Move Base Status\n{msg}\n') 
        
        status = msg.status.status
        if status == msg.status.SUCCEEDED:
            self.resume_trigger()
        else:
            pass

if __name__ == '__main__':
    try:
        NodeLiaison()
    except rospy.ROSInterruptException:
        pass

