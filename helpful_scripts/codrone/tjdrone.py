"""
@author: Terrance Williams
@date: 12 September 2023
@last_modified: 4 March 2024
@title: TDrone: A Context Manager for the CoDrone EDU
"""


from codrone_edu.drone import Drone
import time


class TDrone(Drone):
    """
    A class to implement a context manager for Robolink's CoDrone EDU platform.
    Now automatically pairs and disconnects when entering and exiting the `with` context,
    increasing runtime safety and ease of development and testing.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.connected = self.isOpen()
        self.previous_yaw = 0.

    def __enter__(self):
        # Pair the drone
        self.pair()
        time.sleep(0.2)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.land()
        # self.set_drone_LED(255, 0, 0, 100)
        # Shutdown Connection
        self.close()
        # print(f"Connected?: {self.connected}")

        # Print Errors
        if exc_value is not None:
            print(exc_type, exc_value, exc_tb, sep='\n')

    """
        # Testing Context Manager via Overloads

        def open(self, portname=None):
            print("OPEN: Opened!")
            self.connected = True

        def close(self):
            print("CLOSE: Closing CoDroneEDU Connection.")
            self.connected = False
    """

    def __del__(self):
        # Overload to prevent close from being called twice.
        pass

    def land_reset(self):
        """Treat current landing point as origin"""
        self.previous_yaw = self.get_z_angle()
        self.land()
        time.sleep(0.5)
        # print(f'Previous land: {self.previous_land}')
        self.previous_land = [0, 0]
        # print(f'New Land: {self.previous_land}')

    def fire_start(self):
        """
        Convenience method to block until key-input is entered.
        """
        start_key = 's'
        self.takeoff()
        self.hover()
        ready = False
        # self.set_drone_LED(0, 0, 255, 100)
        while not ready:
            # print(f'Drone Pos: {self.get_position_data()}')
            key = input(f"Press {start_key} to begin: ")
            if key.lower() == start_key:
                # print("Beginning Flight.")
                ready = True
                # self.set_drone_LED(0, 255, 0, 100)
                time.sleep(0.1)

    def relative_takeoff(self):
        """Overload the takeoff to always return to the origin"""
        self.turn_degree(-self.previous_yaw)
        self.takeoff()
        # self.hover(1)
        for _ in range(2):
            self.goto_waypoint([0.00, 0.00, 0.], 0.5)
            # print(self.get_position_data())
        self.hover(2.5)
        # self.goto_waypoint([0, 0, 0], 0.75)
