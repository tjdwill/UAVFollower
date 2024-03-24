#!/usr/bin/env zsh
cd /home/hiwonder/test_ws/src/uav_follower/bags
rosbag record /jethexa/amcl_pose /jethexa/initialpose /uav_follower/calcd_depth_val /uav_follower/calcd_drone_pos /jethexa/move_base_simple/goal /tf /uav_follower/tf2_record --output-prefix uav_follow_exp --bz2
