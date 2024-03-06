#! /usr/bin/env python3 -*-coding: utf8-*-

"""
Writing a script to extract message data to a more manipulable form.
"""

import csv
import argparse
from pathlib import Path

import rosbag
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped, TransformStamped

cat = "".join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")

    args = parser.parse_args()
    file_path = Path(args.file_path)
    if not file_path.is_file() or file_path.suffix != '.bag':
        raise ValueError("Input file must be a bag file.\n")
    
    csv_dir = Path(file_path.parent, "csv")
    if not csv_dir.is_dir():
        csv_dir.mkdir()

    # Define file names
    csv_rootname = csv_dir / f"{file_path.stem}_"
    
    amcl_csv = Path(cat([str(csv_rootname), "amcl_pose.csv"]))
    goalpt_csv = Path(cat([str(csv_rootname), "goal_point.csv"]))
    depth_csv = Path(cat([str(csv_rootname), "depth_val.csv"]))
    drone_est_csv = Path(cat([str(csv_rootname), "drone_est.csv"]))
    tf2_csv = Path(cat([str(csv_rootname), "tf2_record.csv"]))
    print(f"Using root name: {str(csv_rootname)}\n")

    # Load the bag file
    bag = rosbag.Bag(file_path.resolve())
    topics_list = list(bag.get_type_and_topic_info()[1].keys())

    tf_topic = "/tf"

    if tf_topic in topics_list:
        topics_list.remove('/tf')
    print(topics_list)

    # I have to manually code the process for each file. /jethexa/amcl_pose:
    # Want time, position, orientation, and covariant x)
    # geometry_msgs/PoseWithCovarianceStamped
    with open(amcl_csv, 'w', newline="") as csv_file:
        csv_scribe = csv.writer(csv_file)
        csv_scribe.writerow(
            ["Time (ns)", "X (m)", "Y (m)", "Z (m)", "x", "y", "z", "w", "X_covar(m)"]
        )
        for topic, msg, t in bag.read_messages(topics=["/jethexa/amcl_pose"]):
            X = msg.pose.pose.position.x
            Y = msg.pose.pose.position.y
            Z = msg.pose.pose.position.z
            x = msg.pose.pose.orientation.x
            y = msg.pose.pose.orientation.y
            z = msg.pose.pose.orientation.z
            w = msg.pose.pose.orientation.w
            x_covar = msg.pose.covariance[0]
            csv_scribe.writerow([t, X, Y, Z, x, y, z, w, x_covar])


    # /jethexa/move_base_simple_goal: time, position, orientation
    # geometry_msgs/PoseStamped
    with open(goalpt_csv, 'w', newline="") as csv_file:
        csv_scribe = csv.writer(csv_file)
        csv_scribe.writerow(
            ["Time (ns)", "X (m)", "Y (m)" , "Z (m)", "x", "y", "z", "w"]
        )
        for topic, msg, t in bag.read_messages(
            topics=["/jethexa/move_base_simple/goal"]
        ):
            X = msg.pose.position.x
            Y = msg.pose.position.y
            Z = msg.pose.position.z
            x = msg.pose.orientation.x
            y = msg.pose.orientation.y
            z = msg.pose.orientation.z
            w = msg.pose.orientation.w
            csv_scribe.writerow([t, X, Y, Z, x, y, z, w])
    
    # /uav_follower/calcd_depth_val: time, depth_val 
    # std_msgs/Float32
    with open(depth_csv, 'w', newline="") as csv_file:
        csv_scribe = csv.writer(csv_file)
        csv_scribe.writerow(["Time (ns)", "Depth (m)"])
        for topic, msg, t in bag.read_messages(
            topics=["/uav_follower/calcd_depth_val"]
        ):
            depth = msg.data
            csv_scribe.writerow([t, depth])
    
    # /uav_follower/calcd_drone_pos: time, pos
    # geometry_msgs/PointStamped
    with open(drone_est_csv, 'w', newline="") as csv_file:
        csv_scribe = csv.writer(csv_file)
        csv_scribe.writerow(["Time (ns)", "X (m)", "Y (m)", "Z (m)"])
        for topic, msg, t, in bag.read_messages(
            topics=["/uav_follower/calcd_drone_pos"]
        ):
            X = msg.point.x
            Y = msg.point.y
            Z = msg.point.z
            csv_scribe.writerow([t, X, Y, Z])
    
    # /uav_follower/tf2_record: time, hexapod pos
    # geometry_msgs/TransformStamped
    tf2_topic = "/uav_follower/tf2_record"
    if tf2_topic in topics_list:
        with open(tf2_csv, 'w', newline="") as csv_file:
            csv_scribe = csv.writer(csv_file)
            csv_scribe.writerow(
                ["Time (ns)", "X (m)", "Y (m)" , "Z (m)", "x", "y", "z", "w"]
            )
            for topic, msg, t in bag.read_messages(
                topics=["/uav_follower/tf2_record"]
            ):
                X = msg.transform.translation.x
                Y = msg.transform.translation.y
                Z = msg.transform.translation.z
                x = msg.transform.rotation.x
                y = msg.transform.rotation.y
                z = msg.transform.rotation.z
                w = msg.transform.rotation.w
                csv_scribe.writerow([t, X, Y, Z, x, y, z, w])

    print("All CSV scribing completed.\n")
