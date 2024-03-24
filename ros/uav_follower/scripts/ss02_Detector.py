#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@title: ss02_Detector
@creation_date: 6 November 2023
@last_edited: 23 March 2024
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
from rosnp_msgs.msg import ROSNumpy, ROSNumpyList
from rosnp_msgs.rosnp_helpers import decode_rosnp, encode_rosnp_list, encode_rosnp
from std_srvs.srv import Empty, EmptyResponse


yolo_defaults = {
    'yolo': '/home/hiwonder/yolov5',
    'weights': '/home/hiwonder/yolov5/weights/20231101_drone_weights.pt',
    'conf': 0.50
}

# Define parameters for proper image subscription buffer.
# Significantly reduces lag
BUFF_SZ = 2**24  # bytes (16 mebibytes)
SUB_QUEUE_SZ = 1
BAD_DEPTH = np.inf

class UAVDetector:
    """A class for running a YOLOv5 UAV Detection algorithm"""
    
    def __init__(self):
        rospy.init_node('ss02_Detector', log_level=rospy.INFO)
        
        # Parameter Loading
        self.name = rospy.get_name()
        self.topics: dict = rospy.get_param('topics')
        self.img_info: dict  = rospy.get_param('frame_data')
        self.yolo: dict  = rospy.get_param('yolo', default=yolo_defaults)
        self.DETECT_THRESH: int = rospy.get_param('detect_thresh', default=7)
        self.SEEK_THRESH: float = rospy.get_param('~seek_thresh', default=7.)  
        self.IMG_HEIGHT: int = self.img_info['HEIGHT']
        self.IMG_WIDTH: int = self.img_info['WIDTH']
        self.test_mode = rospy.get_param('test_mode')
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
        self.model.to(self.device)

        # Detections Infrastructure
        self.container = []
        self.detections = 0
        self.collecting = True  # whether to collect UAV detections or not

        # Define ROS Communications
        self.detections_pub = rospy.Publisher(
            self.topics['detections'],
            ROSNumpyList,
            queue_size=1
        )

        if self.test_mode:
            self.last_frame_pub = rospy.Publisher(
                self.topics['last_frame'],
                ROSNumpy,
                queue_size=1
            )

        self.rgb_sub = rospy.Subscriber(
            self.topics['img_topic'],
            ROSNumpy,
            self.img_callback,
            queue_size=SUB_QUEUE_SZ,
            buff_size=BUFF_SZ
        )
        # Begin a service to allow this object to continue
        # collecting UAV detections
        self.srv = rospy.Service(
            self.topics['resume_trigger'],
            Empty,
            self.resume
        )
        
        rospy.loginfo(f"{self.name}: Online.")
    
    def box_display(
            self,
            frame: np.ndarray,
            depth_img: np.ndarray,
            xyxyn: np.ndarray,
    ) -> bool:
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
            depth_vals: np.ndarray
                The minimum non-zero depth values associated with each detection BBox.
                If there was a bad detection or the confidence interval is too low,
                the depth, Z_C = np.inf
        """

        detected = False
        depth_vals = []

        # Artist parameters
        box_color = (0, 255, 0)
        box_thickness = 3 
        font_color = (255, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.05
        font_thickness = 4

        # Loop through all inference matrix rows
        for detection in xyxyn:
            conf = detection[4].round(2)  # Why doesn't this round correctly?
            if conf < self.CONF:
                depth_vals.append(BAD_DEPTH)
            else:
                detected = True
                
                # Get BBox   
                x1 = (detection[0] * self.IMG_WIDTH).astype(np.uint16)
                y1 = (detection[1] * self.IMG_HEIGHT).astype(np.uint16)
                x2 = (detection[2] * self.IMG_WIDTH).astype(np.uint16)
                y2 = (detection[3] * self.IMG_HEIGHT).astype(np.uint16)
                
                # Get depth value
                region = depth_img[slice(y1, y2+1), slice(x1, x2+1)]
                nonzero_region = region[region != 0]
                
                try:
                    Z_c = np.min(nonzero_region)
                except ValueError as e:
                    rospy.logwarn(f"{self.name}:\n{e}")
                    Z_c = BAD_DEPTH
                finally:
                    depth_vals.append(Z_c)
                
                # Draw BBox
                conf_str = f'{conf:.3f}'
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), box_color, box_thickness
                )
                
                if x2 > (5 * self.IMG_WIDTH) // 6:
                    cv2.putText(
                            frame, conf_str, (x1, y1 - 10),
                        font, font_scale, font_color, thickness=font_thickness
                    )
                else:
                    cv2.putText(
                        frame, conf_str, (x2, y1 - 10),
                        font, font_scale, font_color, thickness=font_thickness
                    )
        else:
            # flip to BGR
            bgr = frame[..., ::-1].astype(np.uint8)
            cv2.imshow(self.window_name, bgr) 
            cv2.waitKey(1)
        return detected, np.array(depth_vals)
    
    def resume(self, req: Empty):
        self.collecting = True
        return EmptyResponse() 
    
    def request_seeker(self):
        """
        Declaring this for if/when a function must be written to
        request SEEKING action. The SEEKING action is one in which,
        after some pre-defined time, the robot makes movements
        to bring a UAV into view.

        The idea is to monitor how long it's been since a detection,
        call this function when the threshold is exceeded, and
        request the action (from a to-be-defined node) to move
        the robot as needed.
        """
        ...

    def img_callback(self, msg):
        """
        The main action of this node. Runs UAV inference on received images,
        collects them, and sends to the designated topic when the consecutive 
        detection threshold is reached.
        """
        start = rospy.get_time()
        rgbd = decode_rosnp(msg)
        rgb = rgbd[..., 0:-1].astype(np.uint8)
        dpth = rgbd[..., -1]

        # Run inference
        """
        How the data moves in this section:
            - Start with moving the model to CUDA
            - run inference: returns list of length 1
            - unpack the tensor into a variable: torch.Tensor
            - move tensor to CPU
            - torch.Tensor -> np.ndarray

        Then do whatever operations are needed.
        """
        #self.model.to(self.device)  # Does this need to be called every loop iteration?
        inference = self.model(rgb, size=640) 
        tensor = inference.xyxyn[0] 
        tensor = tensor.cpu()
        tensor = tensor.numpy()
        
        detected, depth_vals = self.box_display(rgb, dpth, tensor)
        # Detection logic
        if not detected:
            self.container.clear()
            self.detections = 0
            """
            'SEEK' ACTION LOGIC HERE
            - Keep track of time elapsed since the last detection
            - Send request to move bot and block until complete
            ...
            """
        else:
            if self.test_mode:
                print(f"Tensor:{tensor}\nDepth_vals:{depth_vals}\n")
                print(f"TensorShape:{tensor.shape}\nDepth_vals:{depth_vals.shape}\n")
            if self.collecting:
                # Tensor entry: [xmin, ymin, xmax, ymax, confidence, class, depth_val]
                tensor = np.concatenate((tensor, depth_vals[..., np.newaxis]), axis=1)
                self.container.append(tensor)
                self.detections += 1
                if self.detections >= self.DETECT_THRESH:
                    rosnp_list_msg = encode_rosnp_list(self.container)
                    self.detections_pub.publish(rosnp_list_msg)
                    self.container.clear()
                    self.detections = 0
                    if self.test_mode:
                        print(f'{self.name}: Time Elapsed: {rospy.get_time() - start}')
                    # Post last annotated image for saving.
                    if self.test_mode:
                        self.last_frame_pub.publish(encode_rosnp(rgb))

                    self.collecting = False

    def __del__(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        ss02 = UAVDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

