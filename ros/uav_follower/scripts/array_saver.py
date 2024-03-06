#!/usr/bin/env python3
# -*-coding: utf-8 -*-

"""
@author: Terrance Williams
@title: ArraySaver
@date: 18 December 2023
@description:
    This is a node that allows for the saving of Numpy arrays to file. It is to
    be used to collect depth images and an associated RGB image for analysis.
"""
from pathlib import Path
import time
import cv2
import numpy as np

import rospy
from rosnp_msgs.rosnp_helpers import decode_rosnp, decode_rosnp_list
from rosnp_msgs.msg import ROSNumpy_UInt8, ROSNumpy_UInt16
from uav_follower.srv import DepthImgReq


cat = "".join

# Define parameters for proper image subscription buffer.
# Significantly reduces lag
BUFF_SZ = 2**24  # mebibytes
SUB_QUEUE_SZ = 1

class ArraySaver:
    def __init__(self) -> None:
        self.rgb = []
        self.avg_depth = []
        self.get_frame = False
        
        # ROS setup
        rospy.init_node("array_saver", log_level=rospy.INFO)

        self.name = rospy.get_name()
        self.test_mode = rospy.get_param('test_mode')
        self.depth_count = rospy.get_param("depth_img_count")
        topics = rospy.get_param("topics")
        frame_data= rospy.get_param("frame_data")
        self.IMG_HEIGHT = frame_data['HEIGHT']
        self.IMG_WIDTH = frame_data['WIDTH']
        self.dir = Path(rospy.get_param("~log_dir"))
        self.dir = self.dir / f"depth_exp_{len(list(self.dir.iterdir())):02d}"
        if not self.dir.is_dir():
            self.dir.mkdir()
        
        # Set up comms
        ## Subscriber to images
        if not self.test_mode:
            self.rgb_sub = rospy.Subscriber(
                topics['img_topic'],
                ROSNumpy_UInt8,
                self.rgb_callback,
                queue_size = SUB_QUEUE_SZ,
                buff_size = BUFF_SZ
            )

            ## Service proxy to depth imgs
            rospy.wait_for_service(topics['depth_req'])
            self.depth_req = rospy.ServiceProxy(
                topics['depth_req'],
                DepthImgReq,
            )
        else:
            self.rgb_sub = rospy.Subscriber(
                topics['last_frame'],
                ROSNumpy_UInt8,
                self.get_last_frame,
                queue_size = SUB_QUEUE_SZ,
                buff_size = BUFF_SZ
            )

            self.avg_depth_sub = rospy.Subscriber(
                topics['avgd_depth_img'],
                ROSNumpy_UInt16,
                self.get_avg_depth,
                queue_size = SUB_QUEUE_SZ,
                buff_size = BUFF_SZ
            )

        rospy.loginfo(f"{self.name}: Online.")

    def rgb_callback(self, msg: ROSNumpy_UInt8) -> None:
        img = decode_rosnp(msg)
        cv2.imshow(self.name, img[...,::-1])
        cv2.waitKey(1)
        if self.get_frame:
            self.rgb = img
            self.get_frame = False
    
    def get_last_frame(self, msg:ROSNumpy_UInt8):
        self.rgb = decode_rosnp(msg)

    def get_avg_depth(self, msg:ROSNumpy_UInt16):
        self.avg_depth = decode_rosnp(msg)


    def uav_save(self):
        ...
        """
        Save Avg'd Depth image and last detected UAV frame to file.
        The last frame may not correlate 1:1 with the location of the depth
        values due to the UAV's movement. The depth images are requested later
        in time.

        This is just to get an idea of what readings I'm getting though, so I'm
        not looking for super precision.
        """
        exp_depth = -1  # signal that we don't know the actual depth value.
        while not rospy.is_shutdown():
            if (
                    isinstance(self.rgb, np.ndarray) and
                    isinstance(self.avg_depth, np.ndarray)
            ):
                data = np.array([exp_depth, self.avg_depth, self.rgb], dtype=np.ndarray)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                np_file = self.dir / cat([timestr, f"UAVDepth_{exp_depth}cm.npy"])
                with open(np_file, 'wb') as f:
                    np.save(f, data)
                
                self.avg_depth = []
                self.rgb = []


    def main(self):
        """
        * Get user input
        * Get and average depth images
        * Get frame
        * Package data
        * Save
        * Repeat
        """
        num_imgs = self.depth_count
        done = False
        quit = 'q'
        
        inf_signal = -1  # Insert this to say we are looking as far as possible.

        while not done:
            exp_depth = input(f"<{self.name}>: Experiment Depth (cm): ").lower()
            if exp_depth == quit:
                rospy.loginfo(f"<{self.name}>: Exiting...")
                done = True
                continue
            try:
                exp_depth = int(exp_depth)
                if not exp_depth >= inf_signal:
                    rospy.logwarn(f"Valid depth values: [-1, ...]\n") 
                    continue
            except (TypeError, ValueError) as e:
                print(f"Incorrect value. Please insert a number or enter '{quit}' to exit.\n")
                continue

            depth_msg = self.depth_req(num_imgs)
            depth_imgs = decode_rosnp_list(depth_msg.depth_imgs)
            assert num_imgs == len(depth_imgs)
            ## Convert type for math operations; prevent data overflow (ex. 65535 + 1 -> 0)
            depth_imgs = [arr.astype(np.float64) for arr in depth_imgs]

            # Average the depth images
            """
            avgd_depth_img = depth_imgs[0]
            for i in range(1, num_imgs):
                avgd_depth_img = avgd_depth_img + depth_imgs[i]
            else:
                avgd_depth_img = (avgd_depth_img / num_imgs)
            """

            avgd_depth_img = np.mean(depth_imgs, axis=0)

            self.get_frame = True
            # Wait for image update
            while self.get_frame:
                pass
            # Output data
            data = np.array([exp_depth, avgd_depth_img, self.rgb], dtype=np.ndarray)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            np_file = self.dir / cat([timestr, f"depth_{exp_depth}cm.npy"])
            with open(np_file, 'wb') as f:
                np.save(f, data)

    def __del__(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        arr_sav = ArraySaver()
        if arr_sav.test_mode:
            arr_sav.uav_save()
        else:
            arr_sav.main()
    except rospy.ROSInterruptException:
        pass

