"""
@author: Terrance Williams
@date: 4 March 2024
@title: CoDrone EDU Position Saver
@description:
    This is a short script used to save the drone's positional data to a CSV for later parsing analysis.
    The drone can be piloted once the program begins by switching modes. This will allow me to collect data
    on the drone's position.
"""

import csv
import time
from pathlib import Path
from datetime import datetime

from tjdrone import TDrone

# Initial setup

dir_name = "positions"
pos_dir = Path(dir_name)
if not pos_dir.exists():
    pos_dir.mkdir()
posfile = pos_dir / f"uav_pos_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"

# Drone interfacing
BATTERY_LIM = 40
PERIOD = 0.25
TRIGGER_ALT = 0.15  # meters

data_format = ["Time Elapsed (s)", "X (m)", "Y (m)", "Z (m)"]
with TDrone() as drone:
    drone.land_reset()
    with open(posfile, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data_format)
        while drone.get_pos_z('m') < TRIGGER_ALT:
            time.sleep(PERIOD)
            pass
        print("Beginning Data Collection.\n")
        time.sleep(PERIOD)
        START = time.time()
        while drone.get_battery() > BATTERY_LIM and drone.get_pos_z(unit="m") > TRIGGER_ALT:
            # Get data
            pos = drone.get_position_data()
            time_elapsed = time.time() - START
            writer.writerow([time_elapsed, *pos[1:]])

            time.sleep(PERIOD)
        else:
            print("End Data Collection.\n")
