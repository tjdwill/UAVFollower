#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@title: ss00_Liaison
@creation_date: 7 November 2023
@last_edited: 22 March 2024
@description:
    This program defines the node for coordinating communication
    for other subsystem nodes.
"""


import rospy
from move_base_msgs.msg import MoveBaseActionResult
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import PoseStamped


class NodeLiaison:
    def __init__(self) -> None:
        rospy.init_node('ss00_Liaison', log_level=rospy.INFO)

        self.name = rospy.get_name()
        self.test_mode = rospy.get_param('test_mode')
        topics = rospy.get_param('topics')
        self.bad_detection = rospy.Service(
            topics['bad_detections'],
            Empty,
            self.bad_detect_action
        )
        self.resume_trigger = rospy.ServiceProxy(
            topics['resume_trigger'],
            Empty
        )
        
        if self.test_mode:
            # For tests in which we don't want to run the full navigation
            # but still want continuous detections.
            self.test_sub = rospy.Subscriber(
                rospy.get_param("~waypoints"),  # launch file
                PoseStamped,
                self.test_version_resume
            )
        else:
            self.move_base_sub = rospy.Subscriber(
                rospy.get_param("~move_base_result"),  # launch file
                MoveBaseActionResult,
                self.send_resume_signal
            )  

        rospy.loginfo(f'{self.name}: Online.')
        rospy.spin()
    
    def bad_detect_action(self, req: Empty):
        """Send resume signal to ss02 and respond to ss03"""
        self.resume_trigger()
        return EmptyResponse()


    def test_version_resume(self, msg: PoseStamped):
        self.resume_trigger()
    
    def send_resume_signal(self, msg: MoveBaseActionResult):
        """
        Resume data collection if last movement was successful.
        """
        # A visual test to see if we get multiple messages over one move
        # rospy.loginfo(f'{self.name} Received Move Base Status\n{msg}\n') 
        
        status = msg.status.status
        if status == msg.status.SUCCEEDED:
            rospy.sleep(0.5) # Wait a bit until robot stops moving
            self.resume_trigger()
        else:
            pass

if __name__ == '__main__':
    try:
        NodeLiaison()
    except rospy.ROSInterruptException:
        pass

