#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Terrance Williams
@date: 26 October 2023
@description: A class for k-means clustering
"""

from collections.abc import Iterable
from typing import ClassVar
import copy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class KMeans:
    """
        A class for implementing k-Means clustering, KMeans
        segments data into clusters and has the ability
        to perform image segmentation.

        Initializing (Data Format)
        --------------------------
        Data must be in a 2D array. Meaning, if you have some data
        such as data = np.array([0, 1, 2, 3, 4]), do the following:

        * data.shape = (1, data.shape[0])
        It should make each point a column entry:
            [[0], [1], [2], [3], [4]]
        Pass this version into the KMeans constructor.

        Data of higher dimensions (ex. a multi-channeled image)
        should be flattened using the number of indices
        for the deepest dimension. So, for an image with shape
        (480, 640, 3), run
            * data = data.reshape(-1, 3)
        and pass this data into the constructor.

        Features
        --------
        - Clustering (of course!):
            Cluster data into a specified number of
            clusters using user-defined thresholding and
            iteration limit. All three parameters are adjustable via
            attribute assignment.

        - Segmenting Images:
            Once you've clustered an image's colorspace
            (if you're also using the Image class, there is a method
            for this), pass in an RGB-ordered version of
            the image (again, Image class can provide this, or just flip
            the array about its color columns 'img_array[..., ::-1]'),
            pass in the RGB image, the clusters, and the centroids.
            The method can segment images using random colors or use the
            centroids as the cluster colors.

            *NOTE*:
            Because the method has to iterate through every pixel of
            every cluster, it can take a lot of time to run
            (~0.056 s / pixel). At the time of writing, the author is
            unaware of alternative methods.

        - (BONUS) Re-opening the figure(s):
            Accidentally closing a Matplotlib figure and not being able to
            open it again can be bothersome, so there is a method that can
            "re-open" a figure.
    """

    # =================
    # Class Variables
    # =================
    _THRESH_MAX: ClassVar[int] = 1
    colors: ClassVar[list] = [color for color in
                              list(mcolors.TABLEAU_COLORS.values())]
    WRAP_FACTOR: ClassVar[int] = len(colors)

    # =================
    # Instance Variables
    # =================
    data: list
    segments: int
    threshold: float
    maxIterations: int
    initial_means: Iterable

    # =================
    # Initialization
    # =================
    def __init__(
            self,
            data,
            *,
            segments=2,
            initial_means=None,
            threshold=0.5,
            maxIterations=100):
            
        self._data = data
        self._segments = segments
        self._threshold = threshold
        self._maxIterations = maxIterations
        self._initial_means = initial_means

        self._validateParams()

        # Plotting (2D and 3D cases)
        self.figure2D = plt.figure()
        self.axes2D = self.figure2D.add_subplot()
        self.figure3D = plt.figure()
        self.axes3D = self.figure3D.add_subplot(projection='3d')
        # Close unused figure(s)
        self.closefig()
        # plt.ion()

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

    # ===============
    # Class Methods
    # ===============

    # ::Public methods::
    def cluster(self,
                display: bool = True) -> list:
        """
        The main event; performs the data clustering operation.

        Parameters
        ----------
        display : bool, optional
            Whether to live-plot the data or not. The default is True.

        Returns
        -------
        list
            - Clusters: dict
            - Centroids: list
            - IterationCount: int

        """
        # Initialize Variables
        data = self.data
        K_NUM = self.segments
        THRESH = self.threshold

        if not display:
            plt.close(self.figure2D)
            plt.close(self.figure3D)
        # Check (preclude inf. loop)
        # If the length of the data is less than the target segment number,
        # getting the initial means will result in an infinite loop.
        # The program would never be able to get
        # the target number of unique points.
        if len(data) < K_NUM:
            raise ValueError("Number of segments exceeds data points."
                             " Ensure data is in a 1-D iterable.\n"
                             "Length Data: {}, Segments: {}".format(len(data),
                                                                    K_NUM)
                             )
        # print(f'Data: {data}\nSegments:{K_NUM}')

        # Declare variables
        clusters, centroids = None, None
        means = None

        # ===================
        # Set initial means
        # ===================

        # Perform checks on user-passed means.
        if self._initial_means is not None:
            means = self._initial_means
        else:
            # get k random points to serve as initial means
            print("Generating initial means...")
            means_found = False

            # Loop until the mean points are found. It shouldn't loop at
            # all since repeat points are unlikely, but with randomization,
            # there's always a chance, so I placed the check for uniqueness.
            while not means_found:
                means = np.array([data[np.random.randint(0, len(data))]
                                  for _ in range(K_NUM)])
                # Sanity Check
                length = len(means)
                assert length == K_NUM
                # Ensure no duplicate point
                check = np.unique(means, axis=0)
                if len(check) == length:
                    means_found = True
                    print(means)
        # print(f'Means Check:\n{means}')
        # Begin loop; Currently, the program will loop until each calculated
        # cluster centroid is within THRESH distance from the mean found in the
        # previous iteration, or until the iteration limit is reached,
        # whichever happens first.
        # I may change this calculation to use data variance in the future
        # if that's more "official."
        thresh_reached = False
        iterations = 0
        print("Cluster Iteration Count:")

        while not thresh_reached:
            # Assign clusters
            iterations += 1
            if iterations >= self._maxIterations:
                print("Max iterations reached. Returning output.")
                thresh_reached = True
            print(iterations)

            clusters = self._assignLabels(means, data)
            # print(f'Clusters: {clusters}')
            centroids = self._findCentroid(clusters)
            # print(f'Centroids: {centroids}')
            # Live plot the data
            if display:
                self._display([clusters, centroids, iterations])

            # Compare centroids to previous means.
            for i in range(len(centroids)):
                distance = self._calcDistance(centroids[i], means[i])
                if distance > THRESH:
                    # Assign new means
                    means = centroids
                    break
            else:
                thresh_reached = True
        if iterations < self._maxIterations:
            print("Successful cluster operation.\n")
        return [clusters, centroids, iterations]

    @staticmethod
    def segment_img(image: np.ndarray, clusters: dict, centroids: list,
                    random_colors: bool = False) -> np.ndarray:
        """
        Perform image segmentation from k-Means clustering

        Parameters
        ----------
        image : np.ndarray (RGB)
        clusters : dict
        centroids : list
        random_colors : bool, optional
            Whether to use the centroids as colors, or generate random ones.
            The default is False.
        Returns
        -------
        seg_img : np.ndarray
            The segmented image. (RGB)

        """
        # Setup: Copy the image and get the colors;
        # Do I want to check for repeat colors?
        # The chance of that happening is so miniscule.
        print(f'Beginning Image Segmentation: {len(clusters)} segments.')
        seg_img = np.copy(image)
        if random_colors:
            colors = [(np.random.randint(0, 256),
                      np.random.randint(0, 256),
                      np.random.randint(0, 256)) for _ in range(len(clusters))]
        else:
            # Use centroid color
            colors = np.round(centroids, 0)

        # Inspiration (and much gratitude):
        # https://stackoverflow.com/questions/16094563/numpy-get-index-where-value-is-true
        # Use Numpy nonzero function to find the indices of all elements that
        # match a given pixel in a cluster.
        # Use those indices to replace the pixel values
        # therein with the segmentation color.

        '''
        Iterating through a given cluster takes the most time.
        Can I make this faster?
        '''
        seg_img = seg_img.reshape(-1, 3)
        # Prepare percentage calculation variables
        total_cnt = 0
        for key in clusters:
            total_cnt += len(clusters[key])

        print(total_cnt)

        count = 0
        print('Segmentation Completion (%)')
        old_percent = -1
        for cluster in clusters:
            # Remember that the keys in the dictionary range from
            # 0 to k-1, so they also
            # double as indices.
            seg_color = colors[cluster]
            for pixel in clusters[cluster]:
                # Get indices where image is equal to pixel
                # print(f'Pixel: {type(pixel)}')
                indices = np.nonzero(np.all(np.equal(pixel, seg_img), axis=1))
                # print(indices[0])
                seg_img[indices[0]] = seg_color
                count += 1

                # Print progress
                percentage = round((count / total_cnt) * 100, ndigits=2)
                if percentage % 5 == 0 and percentage != old_percent:
                    print(percentage, end=' ')
                    old_percent = percentage
        else:
            # Please include the 'else' statement to prevent a tremendous
            # debugging headache. Watch the indentation.
            seg_img = seg_img.reshape(image.shape)
        return seg_img

    def closefig(self, close_all=False):
        """
        A method to close the generated figures.

        Parameter(s):
        close_all: bool
            Close all (both) plots. Defaults to False.
        """
        # Close unused plot
        # Intention: Get the number of dimensions of the data
        # (Ex. [(0,0,0)]) has dimension 3 for the data.
        data_dim = np.array(self._data).shape[-1]
        # print(f"VALIDATE DATA: {data_dim}")
        if close_all:
            plt.close(fig=self.figure2D)
            plt.close(fig=self.figure3D)
        elif data_dim == 2:
            plt.close(fig=self.figure3D)
        elif data_dim == 3:
            plt.close(fig=self.figure2D)
        else:
            # Close both
            plt.close(fig=self.figure2D)
            plt.close(fig=self.figure3D)

    def openfig(self, which_dimension: str):
        """
        A way to re-open a closed figure.

        Useful for remedying accidental exits.

        Parameter(s):
        which_dimension: str
            Which plot to open (2d or 3d)
        """
        # Credit: https://stackoverflow.com/questions/31729948/matplotlib-how-to-show-a-figure-that-has-been-closed  #noqa
        dim_string = which_dimension.lower()

        if dim_string == '2d':
            blank = plt.figure()
            fm = blank.canvas.manager
            fig = self.figure2D
            fm.canvas.figure = fig
            plt.show()
        elif dim_string == '3d':
            blank = plt.figure()
            fm = blank.canvas.manager
            fig = self.figure3D
            fm.canvas.figure = fig
            fig.set_canvas(blank.canvas)
            plt.show()
        else:
            print('Invalid input. Pass "2d" or "3d".')
            return

    # ::Private methods::
    def _validateParams(self):
        """Helps consolidate edge-case checks."""
        # access and validate the data
        data = self._data
        K_NUM = self._segments
        THRESH = self._threshold
        MAX_ITERATIONS = self._maxIterations
        initial_means = self._initial_means
        accepted_types = [list, tuple, np.ndarray]

        if np.array(data).ndim != 2:
            raise ValueError("Data *must* be w/in a 1-D container.\nEx. "
                             "[(0, 0), (2,3)]")
        if K_NUM < 1:
            raise ValueError("Number of segments must be at least one.")
        elif K_NUM > len(data):
            raise ValueError("Number of segments cannot exceed "
                             "number of data points.\n"
                             "Length Data: {}, Segments: {}".format(len(data),
                                                                    K_NUM))
        if THRESH < 0 or THRESH > KMeans._THRESH_MAX:
            raise ValueError("Cannot have a negative threshold value.")
        if MAX_ITERATIONS <= 0:
            raise ValueError("Must have at least one iteration.")

        if initial_means is not None:
            if type(initial_means) not in accepted_types:
                raise TypeError('Means container must be one of the following:'
                                f'\n{accepted_types}')
            # Check element types and values.
            if not all([type(arr) in accepted_types for arr in initial_means]):
                raise TypeError('Elements must be one of the following types:'
                                f'\n{accepted_types}')
            if not all(any(np.equal(data, element).all(1))
                       for element in initial_means):
                raise ValueError("Provided means must be among the data.")
            # Remove duplicates and check remaining length
            # Turns out np.unique changes the order of the data.
            # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
            # print(f'Before: {initial_means}')
            initial_means, ndx = np.unique(initial_means, axis=1,
                                           return_index=True)
            # print(ndx)
            initial_means = initial_means[:, ndx]
            # print(f'After {initial_means}')
            # Check length
            try:
                length = len(initial_means)
                assert length == K_NUM
            except AssertionError:
                print("\nNumber of mean points must == number of segments.\n")
                raise
            self._initial_means = initial_means
        # print("KMeans: All parameters valid.")

        return True


    def _assignLabels(self, means: list, data: list):
        """
        Separate the data into clusters.

        Parameters
        ----------
        means : list
            The current list of cluster means.
            Randomly chosen for first iteration.
        data : list
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

        for point in data:
            # Initialize ridiculously high number to begin comparisons.
            old_dist = 1E1000
            for i in range(len(means)):
                new_dist = self._calcDistance(point, means[i])

                if new_dist < old_dist:
                    # Track index of the closest mean point
                    (old_dist, index) = (new_dist, i)
            else:
                # Add point to label bin
                clusters[index].append(point)

        return clusters

    @staticmethod
    def _calcDistance(point1: Iterable, point2: Iterable = None):
        """
        Calculate Euclidean distance between two points.

        Parameters
        ----------
        point1 : Iterable
        point2 : Iterable, optional
            Defaults to the zero vector.

        Returns
        -------
        distance : np.float64

        """
        # check input
        if not isinstance(point1, Iterable):
            point1 = [point1]
        if point2 is None:
            point2 = np.zeros(len(point1))
        else:
            if not isinstance(point2, Iterable):
                point2 = [point2]

        try:
            assert len(point1) == len(point2)
        except AssertionError:
            print('\nBoth points must have same dimension.')
            raise

        # Cast as numpy arrays to prevent overflow
        point1 = np.array(point1, dtype=np.float64)
        point2 = np.array(point2, dtype=np.float64)

        # Perform Calculation
        sqr_dist = (point1-point2)**2
        sqr_dist = sqr_dist.sum()
        distance = np.sqrt(sqr_dist)

        return distance

    @staticmethod
    def _findCentroid(clusters: Iterable):
        """
        Calculate the centroid for each cluster in the bin.
        Parameters
        ----------
        clusters : Iterable (dictionary)
            The clustered data.

        Returns
        -------
        centroids : list
            List of the centroids of each cluster

        """
        # Calculate the centroid for each cluster in the bin.
        # Takes in any iterable, but seeing as the data is labeled,
        # it should be a dictionary whose keys range from 0 to some n.
        # However, I'm leaving it to work for more iterables in case I need to
        # change the design in the future.

        centroids = []
        for i in range(len(clusters)):
            label_bin = np.array(clusters[i], dtype=np.float64)
            rows = label_bin.shape[0]
            # Calc centroid
            centroid = label_bin.sum(axis=0)/rows
            centroids.append(centroid)
        return centroids

    def _display(self, data: Iterable):
        ...
        # Assume we are passed the clusters, centroids, and iterations count
        # TO-DO: Figure out how to get the centroid to show in a 3D cluster.

        # Color list
        # Matplotlib Colors:
        # https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_color_data.py
        """
        The WRAP_FACTOR causes the colors to be reused after exhaustion
        For example, if len(colors) == 10, but K_NUM == 11,
        The 11th cluster will use the first value in colors because
        10 % 10 == 0 (Remember 0-based indexing).
        """

        colors = KMeans.colors
        WRAP_FACTOR = KMeans.WRAP_FACTOR

        # Get data
        clusters, centroids, iterations = data
        # print(clusters[0][0])
        dimensions = len(clusters[0][0])
        # print(f'Data Dimensions: {dimensions}')
        labels = ['Cluster {}'.format(i) for i in range(len(clusters))]

        # 2D case
        if dimensions == 2:
            # plt.close(fig=self.figure3D)
            ax = self.axes2D
            ax.clear()
            ax.set(xlabel='x', ylabel='y',
                   title='k-Means Iteration {}\nk = {}'.format(iterations,
                                                               self._segments))
            for i in range(len(clusters)):
                x, y = zip(*clusters[i])
                cenX, cenY = centroids[i]
                ax.scatter(x, y, s=10, color=colors[i % WRAP_FACTOR],
                           label=labels[i])
                ax.scatter(cenX, cenY, color=colors[i % WRAP_FACTOR],
                           marker='o', s=50, zorder=3, edgecolor='k')
            # ax.legend()
            # ax.grid(visible=True, axis='both')
            plt.pause(0.05)

        # 3D case
        elif dimensions == 3:
            # plt.close(fig=self.figure2D)
            ax = self.axes3D
            ax.clear()

            # Plot setup
            ax.set(xlabel='R', ylabel='G', zlabel='B',
                   title='k-Means Iteration {}\nk = {}'.format(iterations,
                                                               self._segments))
            # Get R, G, B points and plot them for each cluster
            # Matplotlib automatically switches color for each call to scatter.
            for i in range(len(clusters)):
                R, G, B = zip(*clusters[i])
                # cenR, cenG, cenB = centroids[i]
                ax.scatter(R, G, B, s=10, color=colors[i % WRAP_FACTOR])
                # ax.scatter(cenR, cenG, cenB ,marker='*', s=(72*2), zorder=5)
            plt.pause(0.05)
        # plt.show()
        else:
            # print("Data is neither 2D nor 3D. Returning.")
            return
# ---
