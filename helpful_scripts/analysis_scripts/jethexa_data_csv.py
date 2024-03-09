#! /usr/bin/env python3 
#-*-coding: utf8-*-

"""
Writing a script to extract message data to a more manipulable form. Once you've generated the ROS Bag from recording the
topic that listens to the /jethexa/map <-> /jethexa/base_link transform, use this script to save the data to a CSV. 
This is done for ease of plotting in Matplotlib.
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

    jethexa_csv = Path(cat([str(csv_rootname), "jethexa_pos.csv"]))
    print(f"Using root name: {str(csv_rootname)}\n")

    # Load the bag file
    bag = rosbag.Bag(file_path.resolve())
    topics_list = list(bag.get_type_and_topic_info()[1].keys())

    # /uav_follower/tf2_record: time, hexapod pos
    # geometry_msgs/TransformStamped
    tf2_topic = "/tf2Pub"
    if tf2_topic in topics_list:
        with open(jethexa_csv, 'w', newline="") as csv_file:
            csv_scribe = csv.writer(csv_file)
            csv_scribe.writerow(
                ["Time", "X (m)", "Y (m)" , "Z (m)", "x", "y", "z", "w"]
            )
            for topic, msg, t in bag.read_messages(
                topics=[tf2_topic]
            ):
                X = msg.transform.translation.x
                Y = msg.transform.translation.y
                Z = msg.transform.translation.z
                x = msg.transform.rotation.x
                y = msg.transform.rotation.y
                z = msg.transform.rotation.z
                w = msg.transform.rotation.w
                csv_scribe.writerow([t, X, Y, Z, x, y, z, w])

    print(f"Wrote file: {jethexa_csv.resolve()}")
