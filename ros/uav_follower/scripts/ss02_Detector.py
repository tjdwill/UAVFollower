#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@date: 6 November 2023
@description:
    The UAV detection system for the UAV Follower package.
    Runs inference on images, collects detections, and sends detection data
    to the relevant topic.
@misc:
    Inspiration for the machine learning aspects of the code is found in the
    comment by Neel7317 at https://github.com/ultralytics/yolov5/issues/2045.
"""
import cv2
import numpy as np
import torch
import rospy
from rosnp_msgs.msg import ROSNumpy_UInt8, ROSNumpy_UInt16, ROSNumpyList_Float32
from rosnp_msgs.rosnp_helpers import decode_rosnp, encode_rosnp_list
from std_srvs.srv import Empty


yolo_defaults = {
    'yolo': '/home/hiwonder/yolov5',
    'weights': '/home/hiwonder/yolov5/weights/20231101_drone_weights.pt',
    'conf': 0.35
}

class UAVDetector:
    """A class for running a YOLOv5 UAV Detection algorithm"""
    
    def __init__(self):
        rospy.init_node('ss02_Detector', log_level=rospy.INFO)
        
        # Get parameters
        self.topics: dict = rospy.get_param('topics')
        self.img_info: dict  = rospy.get_param('frame_data')
        self.yolo: dict  = rospy.get_param('~yolo', default=yolo_defaults)
        self.DETECT_THRESH: int = rospy.get_param('detect_thresh', default=7)
        self.SEEK_THRESH: float = rospy.get_param('~seek_thresh', default=7.)  
        self.IMG_HEIGHT: int = self.img_info['HEIGHT']
        self.IMG_WIDTH: int = self.img_info['WIDTH']
        self.window_name = 'JetHexa Live Feed'

        # Machine Learning Setup
        self.CONF: float = self.yolo['conf']
        self.model = torch.hub.load(
            self.yolo['yolo'],
            'custom',
            self.yolo['weights'],
            source='local'
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using Device: {self.device}")

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
        self.rgb_sub = rospy.Subscriber(
            self.topics['img_topic'],
            ROSNumpy_UInt8,
            self.img_callback
        )
        self.srv = rospy.Service(
            self.topics['resume_trigger'],
            Empty,
            self.resume
        )

        rospy.loginfo("Online.")
        rospy.spin()
    
    def box_display(self, frame: np.ndarray, xyxyn: torch.Tensor) -> bool:
        """
        Draw boundary boxes and confidence intervals on screen.
        Displays image whether BBox found or not.

        Parameters:
            - frame: np.ndarray
                original RGB image
            - xyxyn: torch.Tensor
                Tensor with normalized max and min box coordinates.
        Outputs:
            detected: bool
                If at least one detection was found, return True.
        """

        detected = False
        xyxyn = xyxyn.numpy()

        # Artist parameters
        box_color = (0, 255, 0)
        box_thickness = 3 
        font_color = (255, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.05

        # Loop through all inference matrix rows
        for detection in xyxyn:
            conf = detection[4].round(2)
            if conf >= self.CONF:
                detected = True   
                x1 = (detection[0] * self.IMG_WIDTH).astype(np.uint16)
                y1 = (detection[1] * self.IMG_HEIGHT).astype(np.uint16)
                x2 = (detection[2] * self.IMG_WIDTH).astype(np.uint16)
                y2 = (detection[3] * self.IMG_HEIGHT).astype(np.uint16)
                
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), box_color, box_thickness)
                
                if x2 > (5 * self.IMG_WIDTH) // 6:
                    cv2.putText(
                        frame, f'{conf}',(x1, y1 - 10),
                        font, font_scale, font_color
                    )
                else:
                    cv2.putText(
                        frame, f'{conf}', (x2, y1 - 10),
                        font, font_scale, font_color
                    )
        else:
            cv2.imshow(self.window_name, frame[..., ::-1])  # flip to BGR
            cv2.waitKey(1)
        return detected
    
    def resume(self, req: Empty):
        self.collecting = True
        # Is passing the same message back a valid action?
        return req
    
    def request_seeker(self):
        ...

    def img_callback(self, msg):
        """
        The main action of this node. Runs UAV inference on received images,
        collects them, and sends to the designated topic when the consecutive 
        detection threshold is reached.
        """
        rgb = decode_rosnp(msg)

        # Run inference
        self.model.to(self.device)  # Does this need to be called every loop iteration?
        inference = self.model(rgb, size=640)
        tensor = inference.xyxyn[0]
        tensor = tensor.cpu()

        # Detection logic
        if not self.box_display(rgb, tensor):
            self.container.clear()
            self.detections = 0
            """
            'SEEK' ACTION LOGIC HERE 
            """
        else:
            if self.collecting:
                self.container.append(tensor.numpy())
                self.detections += 1
                if self.detections >= self.DETECT_THRESH:
                    rosnp_list_msg = encode_rosnp_list(self.container)
                    self.detections_pub.publish(rosnp_list_msg)
                    self.container.clear()
                    self.detections = 0
                    # self.collecting = False

    def __del__(self):
        cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        ss02 = UAVDetector()
    except rospy.ROSInterruptException:
        pass
