import numpy as np
import tensorflow as tf
from OpenNTN.utils import compute_satellite_speed

class MovingSatellite:
    """
    Class to simulate a moving satellite in a 3D space.
    """
    def __init__(self, h_sat: float, inclination_deg: float = 0.0, initial_phase_shift_deg: float = 90.0):
        """
        Initializes the MovingSatellite with its initial position and height.
        
        Parameters:
            initial_position: Initial 3D position of the satellite [x, y, z] in meters.
            h_sat: Height of the satellite above the ground in meters.
        """
        self.h_sat = h_sat
        self.inclination_deg = inclination_deg
        self.theta_0 = initial_phase_shift_deg
        self.r_earth = 6371000.0  # Earth radius in meters
        self.r_orbit = self.r_earth + self.h_sat  # Earth radius + satellite height
        self.v_sat_mag =  compute_satellite_speed(tf.constant(self.h_sat, dtype=tf.float32)).numpy()

    def get_satellite_position_at_time(self, t_sec: float) -> np.ndarray:
        """
        Calculates the satellite's position at a given time t_sec.
        
        Parameters:
            t_sec: Time in seconds since the start of the simulation.
        
        Returns:
            3D position of the satellite at time t_sec [x, y, z] in meters.
        """
        # Orbital speed from OpenNTN utility function 
        # in ECEF frame, assuming circular orbit, 
        # consider earth is not rotating for simplicity
        omega = self.v_sat_mag / self.r_orbit  # Angular velocity (rad/s)
        theta_sat = omega * t_sec  # Angular position of satellite in orbit
        inclination_rad = np.radians(self.inclination_deg)  # Convert inclination to radians
        sat_x = self.r_orbit * np.cos(theta_sat)
        sat_y = self.r_orbit * np.sin(theta_sat) * np.cos(inclination_rad)
        sat_z = self.r_orbit * np.sin(theta_sat) * np.sin(inclination_rad)

        # 3D position vector in Cartesian ENU-like frame centered at UT origin ground level
        r_sat = np.array([sat_x, sat_y, sat_z], dtype=np.float32)

        # 3D velocity vector derivative
        v_sat_x = -self.v_sat_mag * np.sin(theta_sat)
        v_sat_y = self.v_sat_mag * np.cos(theta_sat) * np.cos(inclination_rad)
        v_sat_z = self.v_sat_mag * np.cos(theta_sat) * np.sin(inclination_rad)

        # 3D velocity vector in Cartesian ENU-like frame
        v_sat = np.array([v_sat_x, v_sat_y, v_sat_z], dtype=np.float32)
        return {"r_sat": r_sat, "v_sat": v_sat}
    
    
    def get_n_sat_position_at_time(self, n_sat: int, t_sec: float) -> np.ndarray:
        """
        Calculates the positions of multiple satellites at a given time t_sec.
        
        Parameters:
            n_sat: Number of satellites to calculate positions for.
            t_sec: Time in seconds since the start of the simulation.
        
        Returns:
            Array of shape (n_sat, 3) containing the 3D positions of the satellites at time t_sec [x, y, z] in meters.
        """
        # here we will not rotate the orbit unlike the previous function
        n_sat_vec = np.arange(n_sat)
        omega = self.v_sat_mag / self.r_orbit  # Angular velocity (rad/s)
        theta_sat = omega * t_sec  # Angular position of satellite in orbit
        theta_sat_array= theta_sat*n_sat_vec[:, np.newaxis]  # shape (n_sat, 1)
        inclination_rad = np.radians(self.inclination_deg)  # Convert inclination to radians
        sat_x = self.r_orbit * np.cos(theta_sat_array)
        sat_y = self.r_orbit * np.sin(theta_sat_array) * np.cos(inclination_rad)
        sat_z = self.r_orbit * np.sin(theta_sat_array) * np.sin(inclination_rad)

        # 3D position vector in Cartesian ENU-like frame centered at UT origin ground level
        r_sat = np.stack((sat_x, sat_y, sat_z), axis=-1)  # shape (n_sat, 3)

        # 3D velocity vector derivative
        v_sat_x = -self.v_sat_mag * np.sin(theta_sat_array)
        v_sat_y = self.v_sat_mag * np.cos(theta_sat_array) * np.cos(inclination_rad)
        v_sat_z = self.v_sat_mag * np.cos(theta_sat_array) * np.sin(inclination_rad)

        # 3D velocity vector in Cartesian ENU-like frame
        v_sat = np.stack((v_sat_x, v_sat_y, v_sat_z), axis=-1)  # shape (n_sat, 3)
        
        return {"r_sat": r_sat, "v_sat": v_sat}
    
    def ecef_to_enu(self, r_sat_ecef: np.ndarray) -> np.ndarray:
        """
        Transforms the satellite position from ECEF to ENU coordinates.
        
        Parameters:
            r_sat_ecef: Satellite position in ECEF coordinates [x, y, z] in meters.

        Returns:
            3D position of the satellite in ENU coordinates [east, north, up] in meters.
        """
        # ref point on the earth surface in ecef
        r_ref_ecef = np.array([self.r_earth, 0.0, 0.0], dtype=np.float32)  # Reference point at the North Pole
        r_sat_rel_ecef = r_sat_ecef - r_ref_ecef
        east = r_sat_rel_ecef[:,:,1]
        north = r_sat_rel_ecef[:,:,2]
        up = r_sat_rel_ecef[:,:,0]  # Up is the same as x in this simplified model
        r_sat_enu = np.stack((east, north, up), axis=-1)  # Swap x and y for ENU

        return r_sat_enu

if __name__ == "__main__":
    # Example usage of the MovingSatellite class
    h_sat = 600000.0  # Height of the satellite in meters
    moving_satellite = MovingSatellite(h_sat)

    # Get positions of 5 satellites at time t = 1 seconds
    n_sat = 5
    t_sec = 1.0
    positions = moving_satellite.get_n_sat_position_at_time(n_sat, t_sec)
    positions_enu = moving_satellite.ecef_to_enu(positions["r_sat"])

    print("Positions of satellites at time t =", t_sec, "seconds:")
    print(positions["r_sat"])
    print(type(positions["r_sat"]))
    print(positions["r_sat"].shape)
    print("Positions of satellites in ENU coordinates at time t =", t_sec, "seconds:")
    print(positions_enu)
    print(type(positions_enu))
    print(positions_enu.shape)