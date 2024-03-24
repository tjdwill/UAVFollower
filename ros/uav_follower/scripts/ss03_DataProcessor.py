#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
@author: Terrance Williams
@title: ss03_DataProcessor
@creation_date: 7 November 2023
@last_edited: 23 March 2024
@description:
    This program defines the node for the Data Processor that processes
    bounding boxes, calculates centers, performs k-means clustering and more.
"""

from typing import Tuple
import numpy as np
import rospy
from rosnp_msgs.msg import ROSNumpyList, ROSNumpy
from rosnp_msgs.rosnp_helpers import decode_rosnp_list, encode_rosnp
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, Pose
from uav_follower.kmeans import KMeans
from uav_follower.srv import TF2Poll
from std_srvs.srv import Empty


MAX_DEPTH = 1000  # mm (Depends on operating environment)
class DataProcessor:
    """
    Performs multiple tasks to process the detections and ultimately produce
    a navigation goal point.

    Process:
    - 
    """
    def __init__(self):
        rospy.init_node('ss03_DataProcessor', log_level=rospy.INFO)

        # Parameter Definitions
        self.name = rospy.get_name()
        self.test_mode = rospy.get_param('test_mode')
        self.COUNT_THRESH = np.ceil(0.8 * rospy.get_param('detect_thresh')).astype(np.uint16)
        self.DENSITY_THRESH = rospy.get_param('~density_thresh', default=1.5)
        self.MAX_ACCEL = rospy.get_param('~max_accel', default=5)
        self.MAX_DEPTH_VAL = rospy.get_param("~max_depth", default=MAX_DEPTH)
        self.FOLLOW_DIST = rospy.get_param("follow_distance")
        self.CONF_THRESH = rospy.get_param("yolo/conf")
        topics = rospy.get_param('topics')
        waypoints_topic = rospy.get_param('~waypoints')  # launch file
        self.frame_id = rospy.get_param('~frame_id')  # launch file

        img_dims = rospy.get_param('frame_data')
        self.IMG_HEIGHT, self.IMG_WIDTH = img_dims['HEIGHT'], img_dims['WIDTH']
        self.f_px = rospy.get_param('~focal_length', default=359.0439147949219)
        principal_point = rospy.get_param('~principal_point')
        self.cx, self.cy = principal_point

        # Define communication points
        ## Gets the UAV detection data
        self.detections_sub = rospy.Subscriber(
            topics['detections'],
            ROSNumpyList,
            self.detections_callback
        )
        
        ## Publishes the goal point
        self.waypoint_pub = rospy.Publisher(
            waypoints_topic,
            PoseStamped,
            queue_size=1
        )
        ## Publishes the depth value used in calculations
        self.depth_pub = rospy.Publisher(
            topics['depth_val'],
            Float32,
            queue_size=2
        )
        ## Calculated Drone Position
        self.drone_pos_pub = rospy.Publisher(
            topics['calcd_drone_pos'],
            PointStamped,
            queue_size=2
        )

        ## Requests new data (send resume signal)
        self.bad_detect_req = rospy.ServiceProxy(
            topics['bad_detections'],
            Empty
        )
        if self.test_mode:
            ...
        else:
            '''
            Only use this service during production since
            it requires tf2 transforms to be available
            '''
            self.tf2_req = rospy.ServiceProxy(
                    topics['tf2'],
                    TF2Poll
            )

        rospy.loginfo(f"{self.name} Online.")
        rospy.spin()

    def process_detections(
            self,
            detection_container: ROSNumpyList,
            ndim:int=7
    ) -> Tuple[dict]:
        """
        Sanitizes data to be compatible for KMeans Clustering.

        Parameters
        ----------
        detection_container: list
            A list containing all appended boundary box xyxyn (normalized xyxy)
            values from Machine Learning inferences. Each array is in format
            [xmin, ymin, xmax, ymax, confidence_interval, label_id, depth_val]
                    
        ndim: int (optional)
            number of elements in each given array. Defaults to num of entries
            in a given detection.

        Returns
        -------
        - <processed_data>: tuple
            Using a tuple in order to facilitate ease of refactoring. If I need
            to return more data (ex. calculated area, confidence intervals,
            etc.), I will be able to do so after modifying the
            inner-processing. A dictionary is chosen for both elements because
            they are queryable by key. I need not know the order in which the
            data is stored.

            0. kmeans_data: dict
                A dictionary of data for kmeans clustering. Includes (in order)

                - detections: list
                    key: 'detections'.
                    
                    This holds the flat container of bbox coordinates.

                - k_val: int
                    key: 'k'

                    The number of segments to use for k-Means clustering.
                    Determined by the array with the most entries.

                - means: np.ndarray
                    key: 'means'
                    
                    The array with the most entries to be used as initial means
                    in clustering. The idea is to mitigate effects of outliers
                    in the data because said outliers would hopefully) serve as
                    cluster centroids, meaning they are less likely to be
                    included in the clusters that correspond to the "real",
                    non-noisy data.

            1. other_data: dict
                A dictionary storing extra data needed for processing (if any).
                Its keys should be strings that semantically convey the
                information stored within.
        """
        detections = [
            arr.reshape(-1, ndim)
            for arr in decode_rosnp_list(detection_container)
        ]
        
        # detections = [arr for arr in detections]
        
        flattened, k_val, means = [], 0, None
        for arr in detections:
            num_rows = arr.shape[0]
            for row in arr:
                flattened.append(row)
            if num_rows > k_val:
                k_val = num_rows
                means = arr
        else:
            assert k_val == len(means)

        kmeans_params = {
            'detections': flattened,
            'k': k_val,
            'means': means
        }
        # print(f'\n<process_detections>: flattened arrays: {flattened}')
        # print(f'\n<process_detections>: Means: \n{means}\n')
        other_data = {}
        return kmeans_params, other_data
    
    def cluster_data(self, kmeans_params: dict) -> Tuple[dict, list, dict]:
        """
        Performs kmeans clustering and returns relevant data
        Only clustering off of the bounding box data (the first four columns),
        so ndim=4.
        """
        k = kmeans_params['k']
        detections = kmeans_params['detections']
        init_means = kmeans_params['means']

        self.kmeans = KMeans(
            data=detections,
            ndim=4,
            initial_means=init_means,
            segments=k,
            threshold=0.05
        )
        return self.kmeans.cluster()
    
    def filter_clusters(
            self,
            clusters: dict,
            centroids: np.ndarray,
            extra_data: dict
    ) -> dict:
        """
        Remove clusters that are likely not valid detections.

        Parameters
        ----------
        clusters : dict
            Clusters from KMeans.cluster function.

        centroids : np.ndarray
            Centroids from KMeans.cluster function.

        extra_data: dict
            Any extra data deemed relevant from the cluster operation.

        Returns
        -------
        comparison_dict
            A Dictionary with the following information:
                key: cluster number
                value: tuple
                    Value Contents:
                    - cluster_centroid ([y, x] order): np.ndarray
                        Center of the cluster
                    - cluster_point_count: int
                        Number of points in the cluster
                    - cluster_point_density: float
                        Point concentration (pts/sqr. pixel)
                    - conf: float
                        Average Confidence Interval
                    - Z_c: np.uint16
                        The smallest depth value within the cluster
        """
        comparison_data = {}
        ACCEL_MAX = self.MAX_ACCEL
        # Minimum points in a cluster to be a qualifying candidate.
        MIN_POINT_COUNT = self.COUNT_THRESH
        ndim = extra_data["ndim"]

        # ================
        # Begin Filtering
        # ================
        for key in clusters:

            cluster = clusters[key]
            centroid = centroids[key]
            conf = np.array([x[4] for x in cluster]).mean()
            depth_vals = np.array([x[-1] for x in cluster])
            # Use min instead of mean because np.inf.
            # Could potentially use median as an alternative.
            Z_c = np.min(depth_vals)
            if Z_c == np.inf:
                """All Z values were invalid"""
                continue
            # Stage 1: Minimum point count
            clust_point_count = len(cluster)
            if clust_point_count >= MIN_POINT_COUNT and conf >= self.CONF_THRESH:
                # Perform Density Calculation
                '''
                Calculate cluster density by finding the relevant distance
                from the centroid and using it as the radius of a circle.
                Use the area of said circle to calculate density as 
                    points/area
                '''
                distances = np.array([np.linalg.norm(entry[:ndim] - centroid)
                                    for entry in cluster])
                distances = np.sort(distances)

                vel = distances[1:]-distances[0:-1]
                accel = vel[1:] - vel[:-1]
                if max(accel) > ACCEL_MAX:
                    '''
                    Find the index of max acceleration. Grab the distance two
                    sorted points away from the point with the max
                    acceleration. Use this value as the radius.
                    This is done to mitigate effects of outliers on the 
                    density calculation.
                    '''
                    index = np.nonzero(np.equal(accel, max(accel)))[0]
                    index_num = index[0]
                    r_max = distances[index_num]
                else:
                    # just use the max distance otherwise
                    r_max = max(distances)
                area = np.pi * (r_max**2)
                # NOTE: Data is now np.float64
                clust_point_density = clust_point_count / area
                clust_centroid = centroid

                # Gather data
                comparison_data.update(
                    {
                        key: (
                            clust_centroid,
                            clust_point_count,
                            clust_point_density,
                            conf,
                            Z_c
                        )
                    }
                )
            else:
                pass
        else:
            rospy.loginfo("<filter_clusters>: Data filtering complete.\n")
            return comparison_data

    def vote(self, cluster_candidates: dict) -> np.ndarray:
        """
        Produces best guess on the detected UAV.
        Uses multiple metrics to decide which
        candidate is best to choose as the UAV to follow.

        Parameters
        ----------
        cluster_candidates : dict
            Output from the filter_clusters function.
            Format:
            {
                key: (
                    clust_centroid,
                    clust_point_count,
                    clust_point_density,
                    conf,
                    Z_c
                )
            }

        Returns
        -------
        winner: Tuple[np.ndarray, np.ndarray]
            0: The centroid belonging to the winning cluster
            1: The depth value, Z_c
        """
        
        IDX_CENTROID, IDX_PT_CNT, IDX_DENSITY, IDX_CONF, IDX_Z = tuple(range(5))
        def get_winner(winning_key, clusters=cluster_candidates):
            cluster = clusters[winning_key]
            centroid, Z_c = cluster[IDX_CENTROID], cluster[IDX_Z]
            return centroid, Z_c

        

        # NOTE: Print out for debugging
        if self.test_mode:
            print(f'{self.name}.vote: Vote Candidates')
            for key in cluster_candidates:
                print(cluster_candidates[key])

        # Recall the dict values have order (clust_centroid, clust_point_count,
        # clust_point_density) Matching Index: 0, 1, 2

        num_candidates = len(cluster_candidates)

        # Simple Case Checks
        if num_candidates == 0:
            rospy.loginfo(f"{self.name}.vote: No winner.")
            return ()
        elif num_candidates == 1:
            key = [*cluster_candidates][0]
            winner = get_winner(key)
            rospy.loginfo(f"{self.name}.vote: Winner is obvious; Cluster {key}")
            return winner

        # Begin winner analysis
        first_place_density, fpindex = 0, None
        second_place_density, spindex = 0, None
        count = 0

        for key in cluster_candidates:
            rospy.loginfo(f'{self.name}.vote: Cluster {key}')
            candidate = cluster_candidates[key]
            density = candidate[IDX_DENSITY]
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
                winner = get_winner(fpindex)
                rospy.loginfo(f"{self.name}.vote: Winner Decided by Density; Cluster {fpindex}")
            else:
                # Stage 2: Compare point counts
                """
                You can change the index here to compare depth (distance)
                and/or confidence interval values instead.
                """
                count1 = cluster_candidates[fpindex][IDX_PT_CNT]
                count2 = cluster_candidates[spindex][IDX_PT_CNT]

                if count1 >= count2:
                    winner = get_winner(fpindex)
                    rospy.loginfo(f"{self.name}.vote: Winner decided by point count; Cluster {fpindex}")
                else:
                    winner = get_winner(spindex)
                    rospy.loginfo(f"VOTE: Upset! Winner decided by point count' Cluster {spindex}")
            return winner
    
    def mapcoord_from_imgcoord(
            self,
            uav_data: Tuple[np.ndarray, np.ndarray]
    ) -> PoseStamped:
        """
        Generate a PointStamped message from normalized
        bounding box coordinates
        
        Steps:
            1. Request depth images and average them
            2. Back project depth region and (x, y) into proper coord frame. 
            3. Create and return PointStamped message of the waypoint
        """
        
        BAD_RETURN = None
        normalized_bbox_coordinates, Z_c = uav_data
        # check z value
        if np.isnan(Z_c) or Z_c >= self.MAX_DEPTH_VAL or Z_c == np.inf:
            rospy.logwarn(f'{self.name}: Invalid Z value {Z_c}. Max Depth is {self.MAX_DEPTH_VAL/1000}m')
            return BAD_RETURN
        else:
            # convert to meters
            Z_c = Z_c.astype(np.float32) / 1000
            rospy.loginfo(f'{self.name}: Z_val (m): {Z_c}')
            self.depth_pub.publish(Float32(Z_c))
 
        x_min = (normalized_bbox_coordinates[0] * self.IMG_WIDTH).astype(np.uint16)
        y_min = (normalized_bbox_coordinates[1] * self.IMG_HEIGHT).astype(np.uint16)
        x_max = (normalized_bbox_coordinates[2] * self.IMG_WIDTH).astype(np.uint16)
        y_max = (normalized_bbox_coordinates[3] * self.IMG_HEIGHT).astype(np.uint16)
        x_px = np.average(np.array([x_min, x_max]).astype(np.float64))
        y_px = np.average(np.array([y_min, y_max]).astype(np.float64))

        body_frame_translation, q1 = self._get_transform(
            np.array([x_px, y_px, Z_c])
        )

        rospy.loginfo(
            (f"{self.name}: Displacement transform (from img frame):\nTranslation:\t{body_frame_translation}"
            f"\nQuaternion:\t{q1}")
        )
        # =====================
        # Craft output message
        # =====================
        
        if self.test_mode:
            from geometry_msgs.msg import Vector3
            curr_pos = Vector3(0., 0., 0.)
            q0 = Quaternion(0., 0., 0., 1.)
        else:
            # Request tf2 Data (TF2PollResponse msg)
            tf2_resp = self.tf2_req()
            if not tf2_resp.successful:
                rospy.logwarn(f'{self.name}: Could not get transform.')
                return BAD_RETURN
            else:
                curr_pos = tf2_resp.transform.translation
                q0 = tf2_resp.transform.rotation
                rospy.loginfo(
                    f'{self.name} Received Transform (map->base_link):\n'
                    f'{curr_pos}\n{q0}\n'
                )

        # Apply the rotations
        R_UAV = np.linalg.inv(self._get_quaternion_matrix(q1))
        # R_map = np.linalg.inv(self._get_quaternion_matrix(q0))
        R_map = self._get_quaternion_matrix(q0)
        """
        Imagine the hexapod rotates to align its body with the UAV. What is the
        UAV position in the new body frame?
        """
        aligned_pos = R_UAV @ body_frame_translation
        print(f"\n<{self.name}.goalpoint_func> Aligned Position:\n{aligned_pos}")  # y should always be near 0
    
        # Convert to map coordinates
        goal_pos_bf = np.copy(aligned_pos)
        goal_pos_bf[0] -= self.FOLLOW_DIST
        goal_pos_map_aligned = R_map @ goal_pos_bf # align with map frame
        print(f"\nCalculated Translation: {goal_pos_map_aligned}")
        goalpoint_msg = Point()
        goalpoint_msg.x =  goal_pos_map_aligned[0] + curr_pos.x
        goalpoint_msg.y =  goal_pos_map_aligned[1] + curr_pos.y
        goalpoint_msg.z =  goal_pos_map_aligned[2] + curr_pos.z

        # Orientation
        q = self._quat_product(q0, q1)
        quat = Quaternion(*q)
        # quat = Quaternion(0., 0., 0., 1.)
        pose_msg = Pose(position=goalpoint_msg, orientation=quat)
        rospy.loginfo(f'{self.name}: Pose msg:\n{pose_msg}')
            
        header = Header(
            frame_id=self.frame_id,
            stamp=rospy.Time.now()
        )

        # Publish Drone data
        drone_pos = R_map @ np.copy(aligned_pos)
        drone_msg = Point()
        drone_msg.x = drone_pos[0] + curr_pos.x  
        drone_msg.y = drone_pos[1] + curr_pos.y  
        drone_msg.z = drone_pos[2] + curr_pos.z 

        self.drone_pos_pub.publish(PointStamped(header=header, point=drone_msg))

        return PoseStamped(
            header=header,
            pose=pose_msg
        ) 
    
    def _get_transform(self, img_coords: np.ndarray) -> tuple:
        """
            Returns the calculated transform to the detected UAV
        
            Returns:
                drone_in_bf: np.ndarray
                    Usual [x,y,z] format
                rotation: np.ndarray 
                    A quaternion
        """
        x_px, y_px, Z_c = img_coords
        drone_in_bf = np.array([0, 0, 0])
        rotation = np.array([0, 0, 0, 1])

        # Calculate body-frame vector
        """
        The conversion from body-frame to image frame is:
            - one +90 deg rotation about the z-axis
            - one subsequent -90 deg rotation about the resulting x-axis.
        To recover the body-frame coordinate, we invert the rotation and apply
        the correct scaling.

        The formula is body_frame = [x, y, z]^T 
        = (Z_c / f_px)*[f_px, -x_px, -y_px]^T
        """
        # Intrinsic Camera Parameters for back projection
        cx, cy = self.cx, self.cy
        f_px = self.f_px
        drone_in_bf = (Z_c / f_px) * np.array([f_px, cx-x_px, cy-y_px])

        # Quaternion calculation
        """
            The axis of rotation is the z-axis, which is the same directionally
            in the map and body frames.

            I use the dot product between the projection of displacement vector
            onto the xy-plane, v, and the
            body-frame x-axis, u, to find the rotation angle

            v = <x, y, 0>
            u = <1, 0, 0>
            cos{theta} = (u . v)/(||u|| ||v||)
                = (x)/||v||  
            |theta| = arccos(x/sqrt(x^{2} + y^{2}))

            From there, I calculate the quaternion knowing the rotation is
            about the z-axis.
        """
        disp_xproj = np.copy(drone_in_bf)
        disp_xproj[-1] = 0.

        v_x, v_y, _ = disp_xproj
        # the denom is never 0 because x will always be non-zero when used in
        # this program; no check necessary
        theta = np.arccos(v_x/np.linalg.norm(disp_xproj)) 
        if v_y < 0:
            theta *= -1
        print(f"<{self.name}._get_tranform> Calculated theta: {np.degrees(theta)}") 
        # Convert quaternion message to ndarray; Doing it this way ensures the
        # array is ordered correctly.
        rotation = Quaternion()
        rotation.x = 0
        rotation.y = 0
        rotation.z = np.sin(theta/2)
        rotation.w = np.cos(theta/2)

        return drone_in_bf, self._np_quat(rotation)
    

    def _get_quaternion_matrix(self, quaternion=np.array([0,0,0,1])):
        
        # Quaternion Matrix from Szeliski `Computer Vision - Algoirthms and
        # Applications (2nd ed.)` pg.39
        
        if isinstance(quaternion, Quaternion):
            quaternion = self._np_quat(quaternion)
        qx, qy, qz, qw = quaternion
        R_mat = np.array([
            [1-2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1-2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx**2 + qy**2)],
        ])
        return R_mat
    
    def _quat_product(self, q1, q0) -> np.ndarray:
        """A quaternion that represents the rotation about q0 followed by q1"""
        if isinstance(q1, Quaternion):
            q1 = self._np_quat(q1)
        if isinstance(q0, Quaternion):
            q0 = self._np_quat(q0)

        x1, y1, z1, w1 = q1
        x0, y0, z0, w0 = q0

        w = w1*w0 - x1*x0 - y1*y0 - z1*z0
        x = w1*x0 + x1*w0 + y1*z0 - z1*y0
        y = w1*y0 - x1*z0 + y1*w0 + z1*x0
        z = w1*z0 + x1*y0 - y1*x0 + z1*w0

        return np.array([x, y, z, w])
    
    @staticmethod
    def _np_quat(q: Quaternion) -> np.ndarray:
        return np.array([q.x, q.y, q.z, q.w])

    def detections_callback(self, detections_msg: ROSNumpyList) -> None:
        """
        This is the main function of the node.
        It coordinates the other methods.
        """
        kmeans_data: dict
        clusters: dict
        centroids: list
        extra_data: dict
        uav_data: Tuple[np.ndarray, np.ndarray]
        waypoint: PoseStamped


        kmeans_data, _ = self.process_detections(detections_msg)
        clusters, centroids, extra_data = self.cluster_data(kmeans_data)
        uav_candidates = self.filter_clusters(
            clusters=clusters,
            centroids=centroids,
            extra_data=extra_data
        )
        # End callback if no candidates
        if not uav_candidates:
            self.bad_detect_req()
            return
        uav_data = self.vote(uav_candidates)

        """
        Run depth image processing here output PointStamped message
        """
        waypoint = self.mapcoord_from_imgcoord(uav_data)
        if waypoint is None:
            self.bad_detect_req()
            return
        self.waypoint_pub.publish(waypoint)


if __name__ == '__main__':
    try:
        DataProcessor()
    except rospy.ROSInterruptException:
        pass

