#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Terrance Williams
@original_creation: June 2023
@version: 1.2.1
@revision_date: 23 March 2024
@description: A class for k-means clustering
"""
# from __future__ import annotations

from collections.abc import MutableMapping
from typing import ClassVar
import copy
import numpy as np


class KMeans:
    """
        A class for implementing k-Means clustering, KMeans
        segments data into clusters and has the ability
        to perform image segmentation.

        Initializing (Data Format)
        --------------------------
        Data must be in a 2D array. Meaning, if you have some data
        such as data = np.array([0, 1, 2, 3, 4]), do the following:

        `data = data.reshape(data.shape[-1], -1)`
        or `data = data[..., np.newaxis]`
        It should make each point a row entry:
            [[0], [1], [2], [3], [4]]
        Pass this version into the KMeans constructor.

        Data of higher dimensions (ex. a multi-channeled image)
        should be flattened using the number of indices
        for the deepest dimension. So, for an image with shape
        (480, 640, 3), run
            `data = data.reshape(-1, data.shape[-1])`
        and pass this data into the constructor.

        Features
        --------
        - Clustering (of course!):
            Cluster data into a specified number of
            clusters using user-defined thresholding and
            iteration limit. All three parameters are adjustable via
            attribute assignment.
    """

    # =================
    # Class Variables
    # =================
    _THRESH_MAX: ClassVar[int] = 1
   
    # =================
    # Instance Variables
    # =================
    data: list
    segments: int
    threshold: float
    maxIterations: int
    initial_means: np.ndarray
    ndim: int

    # =================
    # Initialization
    # =================
    def __init__(
            self,
            data: np.ndarray,
            *,
            ndim=0,
            segments=2,
            initial_means=None,
            threshold=0.5,
            maxIterations=100
    ):

        self._data = data
        self._initial_means = initial_means
        self._segments = segments
        self._threshold = threshold
        self._maxIterations = maxIterations

        if ndim == 0:
            self._ndim = min(len(x) for x in data)
        else:
            self._ndim = ndim
        self._validateParams()


    # ============
    # Properties
    # ============
    '''Use these instead of explicit getters and setters'''
    @property
    def data(self):
        """Returns a copy of the object's data"""
        return copy.deepcopy(self._data)

    @property
    def segments(self):
        """How many segments into which the data is clustered."""
        return self._segments

    @segments.setter
    def segments(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        if self._initial_means is not None:
            raise ValueError(
                'Cannot change \'k\' value when initial means are given.'
            )
        old_val = self._segments
        try:
            self._segments = value
            self._validateParams()
        except ValueError:
            self._segments = old_val
            raise

    @property
    def threshold(self):
        """Threshold for k-means clustering."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if 0 <= value <= KMeans._THRESH_MAX:
            self._threshold = value
        else:
            raise ValueError("Threshold must be between 0 and"
                             f' {KMeans._THRESH_MAX}')

    @property
    def maxIterations(self):
        """Max number of iterations for k-Means clustering"""
        return self._maxIterations

    @maxIterations.setter
    def maxIterations(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        elif value < 1:
            raise ValueError("Value must be at least 1.")
        else:
            self._maxIterations = value


    @property
    def ndim(self):
        """Number of dimensions of the data to be clustered."""
        return self._ndim

    @ndim.setter
    def ndim(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        elif value < 1:
            raise ValueError("Value must be at least 1.")
        else:
            old_ndim = self.ndim
            self._ndim = value
            try:
                self._validateParams()
            except ValueError:
                self._ndim = old_ndim
                raise


    # ===============
    # Class Methods
    # ===============

    # ::Public methods::
    def cluster(self) -> tuple:
        """
        The main event; performs the data clustering operation.

        Parameters
        ----------
        display : bool, optional
            Whether to live-plot the data or not. The default is False.

        Returns
        -------
        tuple
            [Clusters, Centroids, IterationCount]

        """
        # Initialize Variables
        this_func = 'KMeans.cluster'
        data = self._data
        K_NUM = self._segments
        THRESH = self._threshold
        ndim = self.ndim

        # Declare variables
        clusters, centroids, means = None, None, None

        # ===================
        # Set initial means
        # ===================

        # Perform checks on user-passed means.
        if self._initial_means is not None:
            means = self._initial_means
        else:
            # get k random points to serve as initial means
            print(f"<{this_func}>: Generating initial means...")
            means_found = False

            # Loop until the mean points are found. It shouldn't loop at
            # all since repeat points are unlikely, but with randomization,
            # there's always a chance, so I placed a check for uniqueness.
            while not means_found:
                # index = np.random.randint(0, len(data))
                means = np.array([
                    data[np.random.randint(0, len(data))][:ndim]
                        for _ in range(K_NUM)
                ])

                # Sanity Check
                length = len(means)
                assert length == K_NUM
                # Ensure no duplicate point; don't care about row order in
                # means array, so no need to pass `return_index=True` and
                # recover original means sort order.
                check = np.unique(means, axis=0)
                '''print(f'{this_func}: Means Generated:\n{means}')
                print(check)
                print('')'''
                if len(check) == length:
                    means_found = True
                    # print(means)
        # print(f'Means Check:\n{means}')
        # Begin loop; Currently, the program will loop until each calculated
        # cluster centroid is within THRESH distance from the mean found in the
        # previous iteration, or until the iteration limit is reached,
        # whichever happens first.
        # I may change this calculation to use statistical variance in the future
        # if that's more "legitimate".
        thresh_reached = False
        iterations = 0
        print(f"<{this_func}>: Cluster Iteration Count:")

        while not thresh_reached:
            # Assign clusters
            iterations += 1
            if iterations >= self._maxIterations:
                print(f"<{this_func}>: Max iterations reached. Returning output.")
                thresh_reached = True
            print(iterations)

            clusters = self._assignLabels(means, data)
            # print(f'Clusters: {clusters}')
            centroids = self._findCentroids(clusters)
            # print(f'Centroids: {centroids}')

            # Compare centroids to previous means.
            for i in range(len(centroids)):
                distance = self._calcDistance(centroids[i], means[i])
                if distance > THRESH:
                    # Assign new means
                    means = centroids
                    break
            else:
                thresh_reached = True
        else:
            print(f"{this_func}: Successful cluster operation.\n")
        return [clusters, centroids, {"iterations": iterations, "ndim": ndim}]

    # ::Private methods::
    def _validateParams(self):
        """
        Validate configuration for KMeans object.
        """
        # access and validate the data
        data = self._data
        K_NUM = self._segments
        THRESH = self._threshold
        ndim = self.ndim
        MAX_ITERATIONS = self._maxIterations
        initial_means = self._initial_means
        accepted_types = [list, tuple, np.ndarray]

        # Check if data is stored properly (flat and in a container)

        # Ensure data has suitable dimensionality
        if ndim <= 0:
            raise ValueError("Data must have at least one dimension.")
        if any([len(arr) < ndim for arr in data]):
            raise ValueError(
                f"Each data point must have at least {ndim} components."
            )

        if K_NUM < 1:
            raise ValueError("Number of segments must be at least one.")
        elif K_NUM > len(data):
            raise ValueError(
                "Number of segments cannot exceed "
                "number of data points.\n"
                "Length Data: {}, Segments: {}".format(len(data), K_NUM)
            )

        if not 0 < THRESH < KMeans._THRESH_MAX:
            raise ValueError("Threshold Value must be between 0 and 1.")
        if MAX_ITERATIONS <= 0:
            raise ValueError("Must have at least one iteration.")

        if initial_means is not None:
            if type(initial_means) not in accepted_types:
                raise TypeError('Means container must be one of the following:'
                                f'\n{accepted_types}')
            else:
                temp_means = [arr[0:self.ndim] for arr in initial_means]
                temp_data = [arr for arr in data]
                initial_means = np.array(temp_means, dtype=np.float64)
                data = np.array(temp_data)
                # print(initial_means)
            # Check element types and values.
            if any([type(arr) not in accepted_types for arr in initial_means]):
                raise TypeError('All means points must be one of the following input types:'
                                f'\n{accepted_types}')
            """
            Nested logic comparison:
                - check = np.equal(data[:, 0:ndim], entry)
                    - checks for element-wise equality between a given entry
                    in `initial_means` and each row in `data` up to the `ndim`th element.
                    Output array is of shape (len(data), ndim).

                - check2 = check.all(axis=1)
                    - Tests if all elements in a row of `check` are True for
                    each row in `check`
                    - "For each row in `check`, test if all column entries
                    are True."
                    - Passing in axis=0 would check if all elements among
                    columns were True for each column in `check`
                    - Left with a 1-D T/F array (i.e., flat) of length len(data).

                - check3 = (any(np.equal(data, entry).all(1) for entry in initial_means[:, 0:ndim)
                    - Returns a generator where each element is a single T/F value.
                    - Each element is communicating if there are any entries in `data` that are completely equivalent
                    with an entry in initial_means.
                    - Outputs generator (length = K_NUM)

                - all(check3)
                    - Is every entry in initial_means represented in `data`?
                    - Outputs a single T/F.
            """
            if not all(
                    any(
                        np.equal(data[:, 0:ndim], entry).all(1)
                    )
                    for entry in initial_means[:, 0:ndim]
            ):
                raise ValueError("Provided means must be among the data.")

            # Remove duplicates and check remaining length
            # Turns out np.unique changes the order of the data.
            # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved

            # print(f'Before: {initial_means}')

            # Check for unique rows in means array; no duplicate means
            # Use slices of initial_means because we are only concerned about uniqueness among the dimensions we are
            # clustering
            filtered_initial_means, ndx = np.unique(
                np.copy(initial_means[:, :ndim]),  # use copy so original array order remains the same.
                axis=0,
                return_index=True
            )
            # Check length
            try:
                length = len(filtered_initial_means)
                assert length == K_NUM
            except AssertionError:
                print("\nNumber of unique mean points must == number of segments.\n")
                raise
            # print(ndx)

            # If the function gets here, there were no duplicate means among the slices.
        # Retain conversion to Numpy array if made
        self._data = data
        self._initial_means = initial_means
        return True

    def _assignLabels(self, means: np.ndarray, data: np.ndarray):
        """
        Separate the data into clusters.

        Parameters
        ----------
        means : np.ndarray
            The current list of cluster means.
            Randomly chosen for first iteration.
        data : np.ndarray
            The data to be organized.

        Returns
        -------
        clusters : dictionary

        """
        # Organizes the data into clusters based on which mean
        # is closest to a given point.
        K_NUM = self._segments
        clusters = {k: [] for k in range(K_NUM)}
        index = None
        ndim = self.ndim

        for point in data:
            # Initialize ridiculously high number to begin comparisons.
            curr_dist = 1E1000
            for i in range(K_NUM):
                new_dist = self._calcDistance(point[0:ndim], means[i][0:ndim])

                if new_dist < curr_dist:
                    # Track index of the closest mean point
                    (curr_dist, index) = (new_dist, i)
            else:
                # Add entire point to label bin (all dimensions; not just ndim)
                clusters[index].append(point)
        return clusters

    @staticmethod
    def _calcDistance(point1: np.ndarray, point2: np.ndarray = None):
        """
        Calculate Euclidean distance between two points.

        Parameters
        ----------
        point1 : np.ndarray
        point2 : np.ndarray, optional
            Defaults to the zero vector.

        Returns
        -------
        distance : np.float64

        """
        # check input
        this_func = 'KMeans._calcDistance'
        if point2 is None:
            point2 = np.zeros(len(point1))

        try:
            assert len(point1) == len(point2)
        except AssertionError:
            print(
                (f'\n<{this_func}>: Both points must have same dimension.\n'
                 f'Point 1: {point1}\nPoint 2: {point2}')
            )
            raise

        # Cast as numpy arrays to prevent overflow
        point1 = point1.astype(np.float64)
        point2 = point2.astype(np.float64)

        # Perform Calculation
        return np.linalg.norm(point1 - point2)

    def _findCentroids(self, clusters: dict):
        """
        Calculate the centroid for each cluster in the bin.
        Parameters
        ----------
        clusters: dict
            The clustered data.

        Returns
        -------
        centroids : list
            - List of the centroids of each cluster

        """
        # Calculate the centroid for each cluster in the bin.
        # Takes in any iterable, but seeing as the data is labeled,
        # it should be a dictionary whose keys range from 0 to some n.
        # However, I'm leaving it to work for more iterables in case I need to
        # change the design in the future.

        centroids = []
        for cluster in clusters.values():
            temp = [arr[:self.ndim] for arr in cluster]
            cluster = np.array(temp, dtype=np.float64)
            rows = cluster.shape[0]
            # Calc centroid
            centroid = cluster[:, 0:self.ndim].sum(axis=0)/rows
            centroids.append(centroid)
        return centroids
# ---

