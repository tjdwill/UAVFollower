# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:00:28 2023

@author: Terrance Williams
@description:
    A script to save images from a camera source. Needed for saving background
    images for use in my ML dataset.
"""

from pathlib import Path
import argparse
import time
import cv2


SLEEP_TIME = 0.01

#%% Set up CL-args
parser = argparse.ArgumentParser()
parser.add_argument(
    'prefix',
    help="The desired name prefix for a set of images. Example: 'uav_set'"
)
parser.add_argument(
    '-o', '--outpath',
    default='.',
    help=("The desired output path for the images. "
          "Creates directory if it doesn't exist.")
)


#%% MAIN Program
if __name__ == '__main__':
    args = parser.parse_args()
    prefix = args.prefix
    outpath = Path(args.outpath).resolve()
    if not outpath.is_dir():
        outpath.mkdir()

    #%% Camera things
    cap = cv2.VideoCapture(0)
    i = 0
    if not cap.isOpened():
        cap.release()
        print('Could not open camera.')
        exit()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Could not read image.')
                break

            # Save files
            filename = outpath / "".join([prefix, f'_background_{i:02d}.jpg'])
            print(filename)
            cv2.imwrite(str(filename), frame)
            cv2.imshow("Captured Frames", frame)
            key = cv2.waitKey(1)
            if key == ord('q') & 0xFF:
                break
            i += 1
            time.sleep(SLEEP_TIME)
    finally:
        cap.release()
        cv2.destroyAllWindows()
