def process_corners(corner_container: list, ndim:int=2):
    '''
    Gets the list of boundary box corners and reorganizes them to desired format.
    Also determines which subcontainer has the most entries. This is the
    container that will be used as the initial means for k-Means segmentation.

    The "length" of this subcontainer is the number of segments, k.

    Parameters
    ----------
    corner_container : list
        A list containing all appended boundary box arrays from Machine Learning detection.
        Each array is in format [tl, tr, br, bl] each of which is (x, y, ...) coordinates:
            - tl = top-left corner point
            - tr = top-right
            - br = bottom-right
            - bl = bottom-left

            - x = horizontal index (along image columns)
            - y = vertical index (along image rows)
     ndim: int (optional)
        Number of dimensions a given point occupies. Defaults to two (ex. x-y coordinates).

    Returns
    -------
    centers: list
        The flat container of calculated box centers.

    means: np.ndarray
        The array with the most entries to be used as initial means in clustering.
        The idea is to mitigate effects of outliers in the data because said outliers would
        (hopefully) serve as cluster centroids, meaning they are less likely to be included
        in the clusters that correspond to the "real", non-noisy data.

    k_val: int
        The number of segments to use for k-Means clustering.
        Determined by the array with the most entries.
    '''

    corners = [arr.reshape(-1, 4, ndim) for arr in corner_container]

    centers, means, k_val = [], [], 0

    for arr in corners:
        # Determine k-Means parameters
        num_boxes = arr.shape[0]
        if num_boxes > k_val:
            k_val = num_boxes
            means.clear()
            # Store center points
            for box in arr:
                tl, tr, br, bl = box
                center = ((tl + br) / 2).astype(np.int16)
                # print(center)
                centers.append(center)
                means.append(center)
        else:
        # DON'T append to the means list.
            for box in arr:
                tl, tr, br, bl = box
                center = ((tl + br) / 2).astype(np.int16)
                # print(center)
                centers.append(center)

    assert len(means) == k_val

    return centers, means, k_val