#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams @date: 7 November 2023 @description:
    This program defines the node for the Data Processor that processes
    bounding boxes, calculates centers, performs k-means clustering and more.
"""


from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import rospy
from rosnp_msgs.msg import ROSNumpyList_Float32, ROSNumpyList_UInt16
from rosnp_msgs.rosnp_helpers import decode_rosnp_list
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped
from uav_follower.kmeans import KMeans
from uav_follower.srv import DepthImgReq
from std_srvs.srv import Empty

class DataProcessor:
    def __init__(self):
        rospy.init_node('ss03_DataProcessor', log_level=rospy.INFO)

        # Define necessary parameters
        self.name = rospy.get_name()
        self.debug = rospy.get_param('~debug')
        self.COUNT_THRESH = int(0.8 * rospy.get_param('detect_thresh'))
        self.DENSITY_THRESH = rospy.get_param('~density_thresh', default=1.5)
        self.MAX_ACCEL = rospy.get_param('~max_accel', default=5)
        topics = rospy.get_param('topics')
        waypoints_topic = rospy.get_param('~waypoints')
        self.base_frame_topic = rospy.get_param('~base_frame_topic')

        img_data = rospy.get_param('frame_data')
        self.IMG_HEIGHT, self.IMG_WIDTH = img_data['HEIGHT'], img_data['WIDTH']
        self.f_px = rospy.get_param('~focal_length', default=359.0439147949219)
        principal_point = rospy.get_param('~principal_point')
        self.cx, self.cy = principal_point

        # Define communication points
        self.detections_sub = rospy.Subscriber(
            topics['detections'],
            ROSNumpyList_Float32,
            self.detections_callback
        )
        
        rospy.wait_for_service(topics['depth_req'])
        self.depth_req = rospy.ServiceProxy(
            topics['depth_req'],
            DepthImgReq
        )

        self.waypoint_pub = rospy.Publisher(
            waypoints_topic,
            PointStamped,
            queue_size=1
        )
        
        self.bad_detect_req = rospy.ServiceProxy(
            topics['bad_detections'],
            Empty
        )

        rospy.loginfo(f"{self.name} Online.")
        rospy.spin()
        
    def process_detections(
            self,
            xyxyn_container: ROSNumpyList_Float32,
            ndim:int=6) -> Tuple[dict]:
        """
    Gets the list of boundary box data and reorganizes them to desired format.
    Also determines which subcontainer has the most entries. This is the
    container that will be used as the initial means for k-Means segmentation.

    The "length" of this subcontainer is the number of segments, k.

    Parameters
    ----------
    xyxyn_container: list
        A list containing all appended boundary box xyxyn values from Machine
        Learning inferences. Each array is in format [xmin, ymin, xmax, ymax].
        The four corner points can be constructed from these values, and/or the
        center point can be calculated via midpoint of corresponding values
            - x = horizontal index (along image columns)
            - y = vertical index (along image rows)
            
     ndim: int (optional)
        number of elements in each given array. Defaults to six (xmin, ymin,
        xmax, ymax, conf, label_id)

    Returns
    -------
    - <processed_data>: tuple
        Using a tuple in order to facilitate ease of refactoring. If I need to
        return more data (ex. calculated area, confidence intervals, etc.), I
        will be able to do so after modifying the inner-processing. A
        dictionary is chosen for both elements because they are query-able by
        key. I need not know the order in which the data is stored.

        0. kmeans_data: dict
            A dictionary of data for kmeans clustering. Includes (in order)

            - xyxyns: list
                The key is 'xyxyns'.

                The flat container of bbox coordinates.


            - k_val: int
                The key is 'k'.

                The number of segments to use for k-Means clustering.
                Determined by the array with the most entries.

            - means: np.ndarray
                The key is 'means'.

                The array with the most entries to be used as initial means in
                clustering. The idea is to mitigate effects of outliers in the
                data because said outliers would hopefully) serve as cluster
                centroids, meaning they are less likely to be included in the
                clusters that correspond to the "real", non-noisy data.

        1. other_data: dict
            A dictionary storing extra data needed for processing (if any). Its
            keys should be strings that semantically convey the information
            stored within.
    """
        detections = [
            arr.reshape(-1, ndim)
            for arr in decode_rosnp_list(xyxyn_container)
        ]
        
        xyxyns = [arr[..., :4] for arr in detections]  # normalized bbox coords
        
        # index = 0
        flattened, k_val, means = [], 0, None
        for arr in xyxyns:
            rows = arr.shape[0]
            for row in arr:
                flattened.append(row)
            if rows > k_val:
                k_val = rows
                # means = flattened[index:index+rows]
                means = arr
            # index += rows
        else:
            assert k_val == len(means)

        kmeans_params = {
            'xyxyns': flattened,
            'k': k_val,
            'means': means
        }

        other_data = {}

        return kmeans_params, other_data
    
    def cluster_data(self, kmeans_params: dict) -> Tuple[dict, list, int]:
        """Performs kmeans clustering and returns data"""
        k = kmeans_params['k']
        xyxyns = kmeans_params['xyxyns']
        init_means = kmeans_params['means']

        self.kmeans = KMeans(
            data=xyxyns,
            initial_means=init_means,
            segments=k,
            threshold=0.05
        )
        clusters, centroids, _ = self.kmeans.cluster()
        return clusters, centroids
    
    def filter_clusters(
            self,
            clusters: dict,
            centroids: np.ndarray) -> dict:
        """
        Remove clusters that are likely not valid detections.

        Parameters
        ----------
        clusters : dict
            Clusters from KMeans.cluster function. Comes in [x,y] order.

        centroids : np.ndarray
            Centroids from KMeans.cluster function. Comes in [x,y] order.

        COUNT_THRESH: int
            Minimum points in a cluster to be a qualifying candidate.

        Returns
        -------
        comparison_dict
            A Dictionary with the following information:
                key: cluster number value: tuple Value Contents:
                    - cluster_centroid ([y, x] order): np.ndarray
                        Center of the cluster
                    - cluster_point_count: int
                        Number of points in the cluster
                    - cluster_point_density: float
                        Point concentration (pts/sqr. pixel)
        """
        # Data setup
        comparison_data = {}
        ACCEL_MAX = self.MAX_ACCEL
        MIN_POINT_COUNT = self.COUNT_THRESH

        # ================ Begin Filtering
        # ================
        for key in clusters:

            # Get given cluster and centroid
            cluster = clusters[key]
            centroid = centroids[key]

            # Stage 1: Minimum point count
            clust_point_count = len(cluster)
            if clust_point_count >= MIN_POINT_COUNT:
                ...
                # Stage 2: Density Calculation
                distances = np.array([np.linalg.norm(cluster[i] - centroid)
                                    for i in range(clust_point_count)])
                distances = np.sort(distances)

                vel = distances[1:]-distances[0:-1]
                accel = vel[1:] - vel[:-1]
                if max(accel) > ACCEL_MAX:
                    # Find the index of max acceleration. Grab the distance two
                    # sorted points away from the point with the max
                    # acceleration. Use this value as the radius.
                    index = np.nonzero(np.equal(accel, max(accel)))[0]
                    index_num = index[0]
                    r_max = distances[index_num]
                else:
                    r_max = max(distances)
                area = np.pi * (r_max**2)
                # Data is now np.float64
                clust_point_density = clust_point_count / area

                clust_centroid = centroid[::-1]  # [col, row] -> [row, col]

                # Gather data
                comparison_data.update(
                    {
                        key: (
                            clust_centroid,
                            clust_point_count,
                            clust_point_density
                        )
                    }
                )
            else:
                pass
        else:
            # Wrap-up
            rospy.loginfo("<filter_clusters>: Data filtering complete.\n")
            return comparison_data

    def vote(self, cluster_candidates: dict) -> np.ndarray:
        """
        Produces best guess on the true tennis ball cluster

        Parameters
        ----------
        cluster_candidates : dict
            Output from the filter_clusters function.

        Returns
        -------
        winning_centroid: np.ndarray
            The centroid belonging to the winning cluster (in [x, y] form).
        """
        # NOTE: DEBUG
        if self.debug:
            print('<vote>: Vote DEBUGGING')
            for key in cluster_candidates:
                print(cluster_candidates[key])

        # Recall the dict values have order (clust_centroid, clust_point_count,
        # clust_point_density) Matching Index: 0, 1, 2

        num_candidates = len(cluster_candidates)

        # Simple Case Checks
        if num_candidates == 0:
            rospy.loginfo("<vote>: No winner.")
            return np.array([])
        elif num_candidates == 1:
            key = [*cluster_candidates][0]
            winner = cluster_candidates[key][0]
            rospy.loginfo(f"<vote>: Winner is obvious; Cluster {key}")
            return winner

        # Begin winner analysis
        first_place_density, fpindex = 0, None
        second_place_density, spindex = 0, None
        count = 0

        for key in cluster_candidates:
            # Stage 1
            rospy.loginfo(f'<vote>: Cluster {key}')
            candidate = cluster_candidates[key]
            density = candidate[2]
            if count == 0:
                first_place_density, fpindex = density, key
                second_place_density, spindex = density, key
            else:
                if density > first_place_density:
                    # update metric trackers
                    second_place_density, spindex = first_place_density, fpindex
                    first_place_density, fpindex = density, key
                elif density > second_place_density:
                    second_place_density, spindex = density, key
                elif count == 1 and (fpindex == spindex):
                    # Get the second place contender; Doing this will result in
                    # the Function voting properly in the case of only two
                    # candidates with identical densities. Otherwise, both 1st
                    # and 2nd place would be the same candidate.
                    second_place_density, spindex = density, key
            count += 1
        else:
            # Compare densities
            if first_place_density/second_place_density >= self.DENSITY_THRESH:
                # We have a winner
                winner = cluster_candidates[fpindex][0]
                rospy.loginfo(f"<vote>: Winner Decided by Density; Cluster {fpindex}")
            else:
                # Stage 2: Compare point counts
                count1 = cluster_candidates[fpindex][2]
                count2 = cluster_candidates[spindex][2]

                if count1 >= count2:
                    winner = cluster_candidates[fpindex][0]
                    rospy.loginfo(f"<vote>: Winner decided by point count; Cluster {fpindex}")

                else:
                    winner = cluster_candidates[spindex][0]
                    rospy.loginfo(f"VOTE: Upset! Winner decided by point count' Cluster {spindex}")
            # Return point in (x, y) form
            return winner[::-1]
    
    def pointstamped_from_imgcoord(
            self,
            normalized_bbox_coordinates: np.ndarray
            ):
        """
        Generate a PointStamped message from [x,y] coordinates
        
        Steps:
            1. Request depth images and average them
            2. Back project depth region and (x, y) into proper coord frame. 
            3. Create and return PointStamped message of the waypoint
        """
        # Have x_px and y_px be uint16 for slicing. Convert later
        x_min = (normalized_bbox_coordinates[0] * self.IMG_WIDTH).astype(np.uint16)
        y_min = (normalized_bbox_coordinates[1] * self.IMG_HEIGHT).astype(np.uint16)
        x_max = (normalized_bbox_coordinates[2] * self.IMG_WIDTH).astype(np.uint16)
        y_max = (normalized_bbox_coordinates[3] * self.IMG_HEIGHT).astype(np.uint16)
        
        cx, cy = self.cx, self.cy
        f_px = self.f_px
        
        # Retrieve and convert depth images
        num_imgs: int = 3
        depth_imgs_msg = self.depth_req(num_imgs)
        depth_imgs = decode_rosnp_list(depth_imgs_msg.depth_imgs)
        assert num_imgs == len(depth_imgs)
        ## Convert type for math operations; prevent values "rolling over" (ex. 65535 + 1 -> 0)
        depth_imgs = [arr.astype(np.float64) for arr in depth_imgs]

        # Average the depth images
        avgd_depth_img = depth_imgs[0]
        for i in range(1, num_imgs):
            avgd_depth_img = avgd_depth_img + depth_imgs[i]
        else:
            avgd_depth_img = (avgd_depth_img / num_imgs)

        # Select bbox region and average values to get the average Z val
        slice_y = slice((y_min + 1), y_max)
        slice_x = slice((x_min + 1), x_max)
        try:
            region = avgd_depth_img[
                slice_y,
                slice_x
            ]
            Z_c = np.average(region)
            print(avgd_depth_img[slice_y, slice_x])
        except IndexError:
            rospy.logwarn(
                (f'Index Error; Img_size: {avgd_depth_img.shape}\n'
                'Slice index: '
                f'[{slice_y}, '
                f'{slice_x}]')
            )
            Z_c = avgd_depth_img[y_px, x_px]
        if Z_c == 0:
            return None
        else:
            Z_c = Z_c / 1000  # convert to meters
        rospy.loginfo(f'Averaged Z_val: {Z_c}')

        # Craft output message
        header = Header(
            frame_id=self.base_frame_topic,
            stamp=rospy.Time.now()
        )

        # Calculate body-frame vector
        """
        The conversion from body-frame to image frame is:
            - one +90 deg rotation about the z-axis
            - one subsequent -90 deg rotation about the resulting x-axis.
        To recover the body-frame coordinate, we invert the rotation and apply
        the correct scaling.

        The formula is body_frame = [x, y, z]^T = (Z_c / f_px)*[f_px, -x_px,
        -y_px]^T
        """
        x_px = np.average(np.float64[x_min, x_max])
        y_px = np.average(np.float64[y_min, y_max])
        body_frame = (Z_c / f_px) * np.array([f_px, cx-x_px, cy-y_px]).astype(np.float64)
        print(body_frame) 
        # body_frame = np.float64([0.2, .1, 0])
        point_msg = Point()
        point_msg.x, point_msg.y, point_msg.z = body_frame[:2], 0.
        if self.debug:
            print(f'UAV Detected: {body_frame}')
            
        return PointStamped(
            header=header,
            point=point_msg
        ) 
    
    def detections_callback(self, detections_msg: ROSNumpyList_Float32):
        kmeans_data, _ = self.process_detections(detections_msg)
        clusters, centroids = self.cluster_data(kmeans_data)
        uav_candidates = self.filter_clusters(
            clusters=clusters,
            centroids=centroids
        )
        # End callback if no candidates
        if not uav_candidates:
            self.bad_detect_req()
            return
        uav_xyxyn = self.vote(uav_candidates)

        """
        Run depth image processing here Output PointStamped message
        """
        waypoint = self.pointstamped_from_imgcoord(uav_xyxyn)
        if waypoint is None:
            self.bad_detect_req()
            return
        self.waypoint_pub.publish(waypoint)
    
if __name__ == '__main__':
    try:
        DataProcessor()
    except rospy.ROSInterruptException:
        pass

