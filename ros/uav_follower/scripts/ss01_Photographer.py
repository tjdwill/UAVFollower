#!/usr/bin/env python3
# -*-coding: utf-8-*-

import cv2
import numpy as np
import rospy
from rosnp_msgs.msg import ROSNumpy_UInt16
from rosnp_msgs.helpers import encode_rosnp


def send_imgs():
    # ROS Setup
    rospy.init('ss01_Photographer', anonymous=False, log_level=rospy.INFO)

    QUEUE_MAX = 5
    FPS = 30
    img_data = rospy.get_param('frame_data')
    IMG_HEIGHT = img_data['HEIGHT']
    IMG_WIDTH = img_data['WIDTH']
    pub_topic = rospy.get_param('topics/img_topic', default='RGBHub')
    rate = rospy.Rate(FPS)
    
    pub = rospy.Publisher(pub_topic, ROSNumpy_UInt16, queue_size=QUEUE_MAX)
    rospy.loginfo("Online.")
    
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

            msg = encode_rosnp(frame[..., ::-1])  # flip to RGB from BGR
            pub.publish(msg)
            rate.sleep()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        send_imgs()
    except rospy.ROSInterruptException:
        pass