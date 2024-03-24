#! /usr/bin/env python3
# -*-coding: utf-8-*-

"""
@author: Terrance Williams
@title: ss01_Photographer
@creation_date: 6 November 2023
@last_edited: 22 March 2024
@description:
    This node captures images using the robot's on-board camera and publishes
    to a topic.
"""

import time
import cv2
import numpy as np
import rospy
from rosnp_msgs.msg import ROSNumpy
from rosnp_msgs.rosnp_helpers import encode_rosnp
from sensor_msgs.msg import Image


CAMERA = 0
BUFF_SZ = 2**24  # bytes (16 mebibytes)
SUB_QUEUE_SZ = 1

class Photographer:
    def __init__(self):
         # ROS Setup
        rospy.init_node('ss01_Photographer', anonymous=False, log_level=rospy.INFO)
        ## Parameters
        self.name = rospy.get_name()
        QUEUE_MAX = 1
        img_dims = rospy.get_param('frame_data')
        self.IMG_HEIGHT = img_dims['HEIGHT']
        self.IMG_WIDTH = img_dims['WIDTH']

        # OpenCV Setup
        self.cap = cv2.VideoCapture(CAMERA)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IMG_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IMG_WIDTH)
        if not self.cap.isOpened():
            rospy.logerr("Could not grab camera.")
            raise AttributeError
        
        ## Comms
        FPS = rospy.get_param('fps')
        self.rate = rospy.Rate(FPS)
        topics = rospy.get_param('topics')
        pub_topic = topics['img_topic']
        sub_topic = topics['depth_raw']
        self.pub = rospy.Publisher(pub_topic, ROSNumpy, queue_size=QUEUE_MAX)

        """Block until a node subscribes."""
        while self.pub.get_num_connections() < 1:
            rospy.sleep(1)

        self.depth_sub = rospy.Subscriber(
            sub_topic,
            Image,
            self.send_imgs,
            queue_size=SUB_QUEUE_SZ,
            buff_size=BUFF_SZ
        )

        rospy.loginfo(f"{self.name}: Online.")

    def send_imgs(self, msg: Image):
        # Capture and send images at given framerate
        try:
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn(f'Could not grab frame:\n{frame}')
                return
        except Exception:
            rospy.logerr(f"{self.name}: An error occurred.\n")
            self.release_resources()
            raise
        else:
            """
            Depth images are 16UC1 in this platform
            Must use np.ndarray initialization due to the data being
            interpreted as bytes. Prevents a headache.
            """
            shape = (msg.height, msg.width)
            assert shape[0] == self.IMG_HEIGHT
            assert shape[1] == self.IMG_WIDTH
            data = msg.data  # Python 'bytes' object
            # print(f"<{self.name}> Depth Encoding: {msg.encoding}\n")
            
            start = time.perf_counter()
            depth = np.ndarray(shape=shape, dtype=np.uint16, buffer=data)
            rospy.logdebug(f"{self.name}: Depth Instantiation (s): {time.perf_counter()-start}")
            
            rgb = frame[..., ::-1].astype(np.uint16)
            # Place data in (r,g,b,d) form
            start = time.perf_counter()
            rgbd = np.concatenate((rgb, depth[...,np.newaxis]), axis=2)
            rospy.logdebug(f"{self.name}: Array Concatenation (s): {time.perf_counter()-start}")
            assert rgbd.shape[-1] == 4
            
            start = time.perf_counter()
            output = encode_rosnp(rgbd)
            rospy.logdebug(f"{self.name}: Message Encoding (s): {time.perf_counter()-start}")
            
            start = time.perf_counter()
            self.pub.publish(output)
            rospy.logdebug(f"{self.name}: Publish Time (s): {time.perf_counter()-start}")
            # self.rate.sleep()

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

    # Define the context manager
    # https://realpython.com/python-with-statement/#coding-class-based-context-managers
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.release_resources()


if __name__ == '__main__':
    try:
        with Photographer() as ss01:
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
