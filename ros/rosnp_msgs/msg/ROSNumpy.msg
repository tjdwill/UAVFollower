# Send a Numpy ndarray across the ROS network

uint16[] shape      # The array dimension (ex. 3x3 -> (3, 3)
string dtype        # The data type of the array elements
uint8[] rosnp       # Array elements as `bytes` object.