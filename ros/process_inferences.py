import numpy as np


def process_inferences(inference_container: list, ndim:int=4):
    """
    Gets the list of boundary box data and reorganizes them to desired format.
    Also determines which subcontainer has the most entries. This is the
    container that will be used as the initial means for k-Means segmentation.

    The "length" of this subcontainer is the number of segments, k.

    Parameters
    ----------
    xyxy_container: list
        A list containing all appended boundary box xyxy values from
        Machine Learning inferences.
        Each array is in format [xmin, ymin, xmax, ymax]. The four corner
        points can be constructed from these values, and/or the center
        point can be calculated via midpoint of corresponding values
            - x = horizontal index (along image columns)
            - y = vertical index (along image rows)
            
     ndim: int (optional)
        number of elements in each given array.
        Defaults to four (xmin, ymin, xmax, ymax)
        However, confidence intervals may be transferred as well...
        I'll figure it out later.

    Returns
    -------
    - processed_data: tuple
        Using a tuple in order to facilitate ease of refactoring.
        If I need to return more data
        (ex. calculated area, confidence intervals, etc.),
        I will be able to do so after modifying the inner-processing.
        A dictionary is chosen for the second element because it is
        query-able by key.
        I need not know the order in which the data is stored.

        0. kmeans_data: dict
            A dictionary of data for kmeans clustering.
            Includes (in order)

            - centers: list
                The key is 'centers'.

                The flat container of calculated box centers.


            - k_val: int
                The key is 'k'.

                The number of segments to use for k-Means clustering.
                Determined by the array with the most entries.

            - means: np.ndarray
                The key is 'means'.

                The array with the most entries to be used as initial
                means in clustering. The idea is to mitigate effects
                of outliers in the data because said outliers would
                hopefully) serve as cluster centroids, meaning they are
                less likely to be included in the clusters that
                correspond to the "real", non-noisy data.

        1. other_data: dict
            A dictionary storing extra data needed for processing (if any).
            Its keys should be strings that semantically convey the information
            stored within.
    """

    # Squeeze data representation.
    xyxys = [arr.reshape(-1, ndim) for arr in inference_container]

    centers, means, k_val = [], [], 0

    center_count = 0
    for arr in xyxys:
        num_boxes = arr.shape[0]

        # Store center points
        for box in arr:
            xmin, ymin, xmax, ymax = box[:4]
            center = np.array([
                (xmin + xmax) / 2,
                (ymin + ymax) / 2]
            ).astype(np.int16)
            # print(center)
            centers.append(center)

        # k-means parameter updates
        if num_boxes > k_val:
            k_val = num_boxes
            # Set means to be the recently-calculated set of centers
            means = centers[center_count:center_count + num_boxes]
            center_count += num_boxes

    assert len(means) == k_val

    # Package the output
    kmeans_data = {
        'centers': centers,
        'k': k_val,
        'means': means
        }
    other_data = {}
    processed_data = (kmeans_data, other_data)

    return processed_data


if __name__ == '__main__':
    static_test_data = np.array([0, 0, 2, 2])

    dynamic_test_data = [
        np.random.randint(0, 300, size=(np.random.randint(1, 10), 4))
        for _ in range(5)]

    kmeans_data, other_data = process_inferences([static_test_data])

    kmeans_data2, other_data2 = process_inferences(dynamic_test_data)

    center_test = np.array([1, 1])

    # Testing accesses
    assert np.all(np.equal(kmeans_data['centers'][0], center_test))  # Pass!
    assert kmeans_data2['k'] == len(kmeans_data2['means'])  # Pass!
    
