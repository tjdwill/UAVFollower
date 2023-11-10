#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@date: 6 November 2023
@description:
    Simulate ss02_Detector node in order to test ss03_DataProcessor.
@misc:
    Inspiration for the machine learning aspects of the code is found in the
    comment by Neel7317 at https://github.com/ultralytics/yolov5/issues/2045.
"""


import numpy as np
import rospy
from rosnp_msgs.msg import ROSNumpyList_Float32
from rosnp_msgs.rosnp_helpers import encode_rosnp_list
from std_srvs.srv import Empty, EmptyResponse


class UAVDetector:
    """A class for running a simulation of uav_follower/ss02_Detector node"""
    
    def __init__(self):
        rospy.init_node('ss02_Detector', log_level=rospy.INFO)
        
        # Get parameters
        self.name = rospy.get_name()
        self.topics: dict = rospy.get_param('topics')     
        self.DETECT_THRESH: int = rospy.get_param('detect_thresh', default=7)

        # Detections Infrastructure
        self.container = []
        self.detections = 0
        self.collecting = True  # whether to collect UAV detections or not

        # Define ROS Communications
        self.detections_pub = rospy.Publisher(
            self.topics['detections'],
            ROSNumpyList_Float32,
            queue_size=1
        )
        self.srv = rospy.Service(
            self.topics['resume_trigger'],
            Empty,
            self.resume
        )

        rospy.loginfo(f"{self.name}: Online.")
        rospy.sleep(7)
        self()
        rospy.spin()    
    
    def resume(self, req: Empty):
        self.collecting = True
        rospy.loginfo('Resumed Collections')
        # Is passing the same message back a valid action?
        return EmptyResponse()

    def simulate_ss02(self):
        """
        Generates fake inferences and sends them to the detections topic
        """
        if self.collecting:
            # Generate inference mssg
            for _ in range(self.DETECT_THRESH):
                rows = np.random.randint(1, 6)
                arr = np.random.random(size=(rows, 6)).astype(np.float32)
                self.container.append(arr)
                self.detections += 1
            else:
                rosnp_list_msg = encode_rosnp_list(self.container)
                print(rosnp_list_msg)
                self.detections_pub.publish(rosnp_list_msg)
                self.container.clear()
                self.detections = 0
                self.collecting = False
                rospy.loginfo('Stopped Collections')

    def __call__(self):
       while not rospy.is_shutdown():
           self.simulate_ss02()


if __name__ == '__main__':
    try:
        ss02 = UAVDetector()
    except rospy.ROSInterruptException:
        pass

