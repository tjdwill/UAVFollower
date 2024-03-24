# -*- coding: utf-8 -*-
"""
@author: Terrance Williams
@date: 23 October 2023
@last_edited: 22 March 2024
@description: Revised helper functions for the construction and deconstruction
of ROSNumpy-type messages.
"""

from typing import List
import numpy as np
from rosnp_msgs.msg import ROSNumpy, ROSNumpyList


def encode_rosnp(array: np.ndarray):
    """
    Construct a ROSNumpy-typed message from a provided ndarray.
    Because Numpy arrays are contiguous in memory, we can flatten the array
    and reconstruct it if we know both the shape and dtype.

    Parameter(s):
    array: np.ndarray

    Output(s):
    msg
        The corresponding ROSNumpy message.
    """
    func_name = "rosnp_helpers.encode_rosnp"

    if not isinstance(array, np.ndarray):
        raise ValueError(f"<{func_name}> Input is not a Numpy array.")
    
    shape = array.shape
    dtype = array.dtype.name
    # ROS' uint8[] -> bytes serialization 
    rosnp = array.tobytes()
    
    # Select the message class and instantiate an object.
    msg = ROSNumpy()
    msg.shape, msg.dtype, msg.rosnp = shape, dtype, rosnp
    return msg    


def encode_rosnp_list(array_list: List[np.ndarray]):
    """
    Create a ROSNumpyList message of the necessary type.
    Infers the type based on the dtype of the first array.
    Assumes all arrays in list have uniform dtype.

    Parameter(s):
    array_list: List[np.ndarray]
        The list of Numpy arrays to send as a message.
        NOT a list of ROSNumpy messages.

    Output(s):
    msg:
        The ROSNumpyList message of requisite data type.
    """
    func_name = "rosnp_helpers.encode_rosnp_list"
    # Would silent failure be better than Exceptions?
    # should I return a msg with a Numpy message of NaN?
    if not array_list:
        raise ValueError(
            f"<{func_name}>"
            " Cannot make message from empty list."
        )
    
    # Create list of msgs
    msg_arr = [encode_rosnp(arr) for arr in array_list]
    msg = ROSNumpyList()
    msg.rosnp_list = msg_arr
    return msg
    
    
def decode_rosnp(msg):
    """
    Reconstructs Numpy array from ROSNumpy-typed message.

    Parameter(s):
    msg: 
        An instance of a ROSNumpy msg
    
    Output(s):
    result_array: np.ndarray
        The corresponding array from the decoded message.
    """
    func_name = "rosnp_helpers.decode_rosnp"
    shape, dtype, data = msg.shape, msg.dtype, msg.rosnp
    result_array = np.ndarray(shape, dtype=dtype, buffer=data)
    return result_array


def decode_rosnp_list(msg):
    """
    Decodes all messages in a ROSNumpyList-typed message.

    Parameter(s):
    msg: 
        An instance of a ROSNumpyList msg
    
    Output(s):
    result_array: List[np.ndarray]
        The corresponding list of Numpy arrays from the decoded message.
    """
    func_name = "rosnp_helpers.decode_rosnp_list"
    result_list = [decode_rosnp(message) for message in msg.rosnp_list]
    return result_list

