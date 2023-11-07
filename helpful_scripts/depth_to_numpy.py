#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
@author: Terrance Williams
@date: 2 November 2023
@description:
    This is a script to test if I can interact with depth images from the JetHexa
    and convert them to Numpy arrays.
"""

import numpy as np
import rospy
from sensor_msgs.msg import Image

count = 0
def callback(msg: Image):
    global count
    shape = (msg.height, msg.width)
    data = msg.data  # Python 'bytes' object
    
    # Depth images are 16UC1 in this platform
    arr = np.ndarray(shape=shape, dtype=np.uint16, buffer=data)
    
    # Only print one message to the screen
    if count == 0:
        print(f'Received image: {arr}')

        for row in arr:
            print(row)
        print(f'Received Depth Img of size {shape}')
        count += 1


def main():
    rospy.init_node('depth_catcher', anonymous=False)
    sub = rospy.Subscriber('/camera/depth/image_raw', Image, callback)
    rospy.spin()


if __name__ == '__main__':
    main()
