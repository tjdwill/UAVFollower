# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:17:14 2023

@author: Terrance Williams
@description:
    This script is meant to be an interactive exploratory tool to conduct
    characterization experiments on the JetHexa's depth camera. It's to be
    used in an environment such as Spyder that allows command-line interaction
    and variable/object exploration.
"""


import cv2
import numpy as np
from scipy.stats import mode
from pathlib import Path


# %% Numpy data loading

pth = Path(
    ('C:/users/tj/documents/gradschool/thesis/programming/'
     'ros/logs/depth_arrays')
)

assert pth.is_dir()

dirs = [file for file in pth.iterdir()]
dirs = tuple(filter(lambda x: x.is_dir() == True, dirs))

#%% Load the arrays
array_files = [file for file in dirs[-1].iterdir()]
# print(files)

arrays = []

for file in array_files:
    with open(file, 'rb') as f:
        arrays.append(np.load(f, allow_pickle=True))

#%% Analysis Function (Plot image)

def analyze_arr(t_depth: np.ndarray) -> tuple:
    """Extract information for a given depth collection"""
    target, depth, rgb = t_depth
    cv2.imshow("Corresponding RGB", rgb[...,::-1])
    cv2.waitKey(1)
    return target, depth, rgb

def get_stats(t_depth: np.ndarray) -> tuple:
    if t_depth.dtype.name == 'object':
        _, depth, _ = t_depth
    else:
        depth = t_depth
    nonzero = depth[depth!=0]
    mean = np.mean(nonzero)
    median = np.median(nonzero)
    val_mode = mode(nonzero.astype(np.uint16))

    return mean, median, val_mode


dw = cv2.destroyAllWindows  # quick alias
def wk():
    cv2.waitKey(0)
#%%
# TODO: Figure out a way to filter the background from a drone detection.
# How do I get the middle values?

"""
* The depth camera does not handle slopes and non-uniform surfaces all that well. At least, not where the variation is non-linear.
  * the values will vary quite drastically w/in a few pixels
* There were a lot of 0s in one particular area that was black in color, even
  though the object was far enough in distance to be a valid value.
* Depth values aren't necessarily accurate. Some values got up above 5000 (max
  was ~7300mm).  
* As expected, objects that are too close result in 0s.
* I think the raw image may also be a factor in the values I get along the edge of the image. Would rectified be better?
* Result: the depth camera is accurate, especially along the middle region. Therefore, I need a way to filter out the background
info in my drone detections.

### PROPOSED FILTER METHOD
   
0. Rather than finding the mean of the non-zero depth values, use the median or mode. For the latter, convert the array to np.uint16 to have a higher chance of correct values being selected.
1. Filter for values less than the mean and then take the max
"""
