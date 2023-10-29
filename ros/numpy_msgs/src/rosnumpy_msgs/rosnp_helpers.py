# -*- coding: utf-8 -*-
"""
@author: Terrance Williams
@date: 23 October 2023
@description: Revised helper functions for the construction and deconstruction
of ROSNumpy-type messages.
"""

from typing import List
import numpy as np
from rosnumpy_msgs.msg import *

rosnp_dict = {
    'int8': ROSNumpy_Int8,
    'int16': ROSNumpy_Int16,
    'int32': ROSNumpy_Int32,
    'int64': ROSNumpy_Int64,
    'uint8': ROSNumpy_UInt8,
    'uint16': ROSNumpy_UInt16,
    'uint32': ROSNumpy_UInt32,
    'uint64': ROSNumpy_UInt64,
    'float32': ROSNumpy_Float32,
    'float64': ROSNumpy_Float64
}

rosnp_list_dict = {
    ROSNumpy_Int8: ROSNumpyList_Int8,
    ROSNumpy_Int16: ROSNumpyList_Int16,
    ROSNumpy_Int32: ROSNumpyList_Int32,
    ROSNumpy_Int64: ROSNumpyList_Int64,
    ROSNumpy_UInt8: ROSNumpyList_UInt8,
    ROSNumpy_UInt16: ROSNumpyList_UInt16,
    ROSNumpy_UInt32: ROSNumpyList_UInt32,
    ROSNumpy_UInt64: ROSNumpyList_UInt64,
    ROSNumpy_Float32: ROSNumpyList_Float32,
    ROSNumpy_Float64: ROSNumpyList_Float64
}


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

    if not isinstance(array, np.ndarray):
        raise ValueError("[rosnp_helpers.encode_rosnp] Input is not a Numpy array.")
    
    shape = array.shape
    dtype = array.dtype.name
    # correct ROS' uint8[] -> bytes serialization 
    # All other numeric-typed arrays
    # are serialized as tuples anyway, so just cast it.
    rosnp = tuple(array.flatten())
    
    # Select the message class and instantiate an object.
    try:
        msg = rosnp_dict[dtype]()
    except KeyError:
        print((f"<rosnp_helpers.encode_rosnp> Input dtype {dtype}"
              " is not among accepted formats. Use one of the following:"))
        for key in rosnp_dict:
            print(key)
        raise
    else:
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

    # Would silent failure be better than Exceptions?
    # should I return a msg with a Numpy message of NaN?
    if not array_list:
        raise ValueError(("<rosnp_helpers.encode_rosnp_list>",
                          " Cannot make message from empty list."))
    
    # Create list of msgs
    msg_arr = [encode_rosnp(arr) for arr in array_list]

    # Determine message to use
    msg_type = type(msg_arr[0])
    try:
        # Select the correct message and instantiate it.
        msg = rosnp_list_dict[msg_type]()
        """# Alternate Method
        dtype = array_list[0].dtype.name
        msg = rosnp_list_dict[rosnp_dict[dtype]]()"""
    except KeyError:
        print(("<rosnp_helpers.encode_rosnp_list>",
                          f" Message type {msg_type} not among supported types."
                            "Supported Types:"))
        for key in rosnp_list_dict:
            print(key)
        raise
    else:
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
    # print(type(msg.rosnp), msg.rosnp)
    msg_types = list(rosnp_dict.values())
    if type(msg) not in msg_types:
        print((f"<rosnp_helpers.decode_rosnp> Message type {type(msg)} not"
               "among supported types.\nSupported types:"))
        for msg_type in msg_types:
            print(msg_type)
        raise TypeError   
    
    shape, dtype, data = msg.shape, msg.dtype, tuple(msg.rosnp)
    result_array = np.array(data, dtype=dtype).reshape(shape)
    
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
    msg_types = list(rosnp_list_dict.values())

    if type(msg) not in msg_types:
        print((f"<rosnp_helpers.decode_rosnp_list> Message type {type(msg)} not"
                "among supported types.\nSupported types:"))
        for msg_type in msg_types:
            print(msg_type)
        raise TypeError

    result_list = [decode_rosnp(message) for message in msg.rosnp_list]
    return result_list
