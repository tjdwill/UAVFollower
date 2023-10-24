#! usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:53:37 2023

@author: Terrance Williams
@title: Data Set Automation
@description: A script to automatically generate the relevant TXT files
and JPG files for Machine Learning. Assumes the use of ArUCo tags to
detect the object.
"""

import argparse
import os
import cv2
import numpy as np
import time


# %% Globals
PAD_MAX = 5
FRAME_HEIGHT, FRAME_WIDTH = 480, 640
WIDTH_HEIGHT_RATIO = 138/35  # (mm) CoDrone width to height
K = WIDTH_HEIGHT_RATIO * 1.5

# Dictionary per PyImageSearch
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# %% Function Defs
def process_tags(tag_info: tuple, img: np.ndarray) -> tuple:
    '''
    - Processes the detected tag (assumes only one in frame).
    - Draws and labels tags onto given frame.
    - Calculates necessary data for YOLOv5 labelling.

    Parameters
    ----------
    tag_info : tuple
        Output data from cv2.aruco.detectMarkers function.
        (corners: (np.ndarray,) , ids, rejectedImgPoints)
    img: np.ndarray
        The frame from which tags were detected.

    Returns
    -------
    label_data: np.ndarray
        <Class index>
        <center-x>
        <center-y>
        <width of bounding box>
        <height of bounding box>

    wrk_img: np.ndarray
        The image that has the tag border(s) and center(s) drawn.
    '''

    corners, _, _ = tag_info
    # print(corners)
    wrk_img = np.copy(img)
    class_index = 0
    # Extract corners (top-left (tl), top-right (tr),
    # bottom-right (br), and bottom-left (bl) order)
    corners = corners[0].reshape((4, 2))
    tl, tr, br, bl = corners

    # Convert to numeric (x, y)
    tl = np.array([int(tl[0]), int(tl[1])])
    tr = np.array([int(tr[0]), int(tr[1])])
    br = np.array([int(br[0]), int(br[1])])
    bl = np.array([int(bl[0]), int(bl[1])])
    cntr_pt = (0.5 * (br + tl)).round(0)

    height = abs(tl[1] - bl[1])
    width = np.ceil(K * height)

    # Calculate bounding box coordinates (Simulate CoDrone Dimensions)
    cntr_x, cntr_y = cntr_pt

    bb_tl = np.int16(np.ceil(np.array([max(0, cntr_x - (width/2)),
                              max(0, cntr_y - (width/4))])))
    bb_tr = np.int16(np.ceil(np.array([min(FRAME_WIDTH - 1, cntr_x + (width/2)),
                              max(0, cntr_y - (width/4))])))
    bb_br = np.int16(np.ceil(np.array([min(FRAME_WIDTH - 1, cntr_x + (width/2)),
                              min(FRAME_HEIGHT, cntr_y + (width/4))])))
    bb_bl = np.int16(np.ceil(np.array([max(0, cntr_x - (width/2)),
                              min(FRAME_HEIGHT, cntr_y + (width/4))])))

    """for pt in [bb_tl, bb_tr, bb_br, bb_bl]:
        print(pt)"""
    # Draw bounding box
    line_color = (0, 255, 0)
    circ_color = (0, 0, 255)
    line_thickness = 4
    rad_val = 3
    cv2.line(wrk_img, np.int16(bb_tl), bb_tr, line_color, thickness=line_thickness)
    cv2.line(wrk_img, bb_tr, bb_br, line_color, thickness=line_thickness)
    cv2.line(wrk_img, bb_br, bb_bl, line_color, thickness=line_thickness)
    cv2.line(wrk_img, bb_bl, bb_tl, line_color, thickness=line_thickness)

    cv2.circle(wrk_img, (int(cntr_x), int(cntr_y)),
               radius=rad_val, color=circ_color, thickness=-1)
    # Normalize height and width
    width_normed = width / FRAME_WIDTH
    height_normed = height / FRAME_HEIGHT

    return ((class_index,
            (cntr_x) / FRAME_WIDTH),
            (cntr_y / FRAME_HEIGHT),
            width_normed,
            height_normed),
            wrk_img)


# %% MAIN Program
if __name__ == '__main__':
    # %% Argument Parsing

    parser = argparse.ArgumentParser()

    parser.add_argument("fileName",
                      metavar="givenName",
                      help="This is the name used to prefix each image and text file. Result Example:\ngivenName_x.jpg")
    parser.add_argument('-o', '--output',
                      default=os.path.join(os.getcwd(), 'JPEGImages'),
                      help="Provide the absolute path to the desired output directory. Creates directory if it does not exist.\nOtherwise, creates an output folder in current working directory.")
    parser.add_argument('-p', '--padding',
                      default='2',
                      choices=[i for i in range(PAD_MAX)],
                      type=int,
                      help='Determines how much the image/txt file number is padded.\nEx. "-p 3" --> givenName_00x.jpg')
    parser.add_argument('-d', '--dict',
                      default="DICT_7X7_50",
                      help="The ArUCo dictionary to use for detection. Defaults to 'DICT_7X7_50'")

    # I could add arguments for the image dimensions, but I know what they are already. Maybe in the future.

    try:
        args = parser.parse_args()
    except:
        raise

    # %% Initial Checks and setups

    file_prefix = args.fileName
    output_path = args.output
    padding = args.padding
    try:
        tag_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[args.dict])
        tag_params = cv2.aruco.DetectorParameters_create()
    except KeyError:
        print("Provided dictionary not in ArUCo dictionary list. Input one of the following:")
        for key in ARUCO_DICT:
            print(key)
        raise

    # Create directory if need be
    box_dir = os.path.join(output_path, 'boxes')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(box_dir):
        os.mkdir(box_dir)
    # %% Computer Vision Setup

    cap = cv2.VideoCapture(0)  # default camera
    try:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        if not cap.isOpened():
            print("Could not open camera. Exiting.")
            exit()

        print("\n Press 'q' to exit.\n")
        detect_count = 0
        while True:
            '''
            Capture images and analyze them for ArUCo tags.
            If found, process them and create the relevant files
            from generated data.
            '''
            ret, frame = cap.read()
            if not ret:
                print("Error in image capture.")
                break

            tag_info = cv2.aruco.detectMarkers(frame,
                                               tag_dict,
                                               parameters=tag_params)
            if not tag_info[0]:
                cv2.imshow("Detection Window", frame)
                key = cv2.waitKey(1)
                if key == ord('q') & 0xFF:
                    break
                continue
            else:
                label_data, img = process_tags(tag_info, frame)
                # print(label_data)
                # Display image
                cv2.imshow("Detection Window", img)
                key = cv2.waitKey(1)
                if key == ord('q') & 0xFF:
                    break

                # Create TXT file
                txt_name = "".join([file_prefix, '_',
                                    f'{detect_count:0{padding}d}',
                                    '.txt'])
                txt_name = os.path.join(output_path, txt_name)
                data = []
                for entry in label_data:
                    data.append(str(entry))
                    data.append(" ")
                data = data[:-1]
                with open(txt_name, mode='w') as txt:
                    txt.write("".join(data))

                # Save image
                img_name = "".join([file_prefix, '_',
                                    f'{detect_count:0{padding}d}',
                                    '.jpg'])
                img_name_path = os.path.join(output_path, img_name)
                cv2.imwrite(img_name_path, frame)
                cv2.imwrite(os.path.join(box_dir, img_name), img)
                detect_count += 1
                print(f'Wrote: {txt_name}')
                time.sleep(0.1)
    except:
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
