#!/usr/bin/env python3
# -*-coding: utf-8-*-

import cv2
import numpy as np
import time
import rospy
from rosnp_msgs.msg import ROSNumpy_UInt8, ROSNumpy_UInt16
from rosnp_msgs.rosnp_helpers import encode_rosnp


def send_imgs():
    # ROS Setup
    rospy.init_node('ss01_Photographer', anonymous=False, log_level=rospy.INFO)
    
    name = rospy.get_name()
    QUEUE_MAX = 1
    FPS = 30
    img_data = rospy.get_param('frame_data')
    IMG_HEIGHT = img_data['HEIGHT']
    IMG_WIDTH = img_data['WIDTH']
    topics = rospy.get_param('topics')
    pub_topic = topics['img_topic']
    rate = rospy.Rate(FPS)
    
    rospy.wait_for_service(topics['resume_trigger'])
    pub = rospy.Publisher(pub_topic, ROSNumpy_UInt8, queue_size=QUEUE_MAX)
    rospy.loginfo(f"{name}: Online.")
    
    # OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    if not cap.isOpened():
        rospy.logerr("Could not grab camera.")
        exit()

    # Capture and send images at given framerate
    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn(f'Could not grab frame:\n{frame}')

            start =  rospy.get_time()
            msg = encode_rosnp(frame[..., ::-1])  # flip to RGB from BGR
            """msg = ROSNumpy_UInt16()
            dtype = frame.dtype.name
            shape = frame.shape
            rosnp = frame[..., ::-1].flatten().astype(np.uint16)
            msg.dtype, msg.shape, msg.rosnp = dtype, shape, rosnp
            """
            pub.publish(msg)
            # print(f'Time: {rospy.get_time() - start}')
            """cv2.imshow('Test', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            """
            rate.sleep()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        send_imgs()
    except rospy.ROSInterruptException:
        pass
