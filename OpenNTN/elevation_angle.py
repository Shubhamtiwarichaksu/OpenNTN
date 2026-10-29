import numpy as np
from OpenNTN.moving_satellite import MovingSatellite
from OpenNTN.moving_ut import MovingUT
import matplotlib.pyplot as plt
import tensorflow as tf

class ComputeElevationAngleAndDopplerShift:

    # Class to compute the elevation angle between a satellite and a user terminal (UT).

    def __init__(self, carrier_frequency: float):
        self.carrier_frequency = carrier_frequency
    
    #@staticmethod
    def compute_elevation_angle(self, sat_position: tf.Tensor, ut_position: tf.Tensor) -> float:
        """
        Computes the elevation angle in degrees between the satellite and the UT.

        Parameters:
            sat_position: 3D position of the satellite [x, y, z] in meters.
            ut_position: 3D position of the UT [x, y, z] in meters.
        """
        # Vector from UT to Satellite
        los_vec = sat_position - ut_position
        #z coordinate
        z_coordinate = los_vec[...,2]

        # Distance from UT to Satellite
        distance_ut_to_sat = tf.linalg.norm(los_vec, axis=-1)

        # Elevation angle calculation
        elevation_angle_rad = tf.asin(z_coordinate / distance_ut_to_sat)

        # Convert to degrees
        elevation_angle_deg = tf.multiply(elevation_angle_rad, 180.0 / np.pi)

        return elevation_angle_deg

    def compute_elevation_angle_and_doppler_shift(self,sat_position: tf.Tensor, ut_position: tf.Tensor, sat_velocity: tf.Tensor, ut_velocity: tf.Tensor) -> float:
        """
        Computes the elevation angle in degrees between the satellite and the UT.

        Parameters:
            sat_position: 3D position of the satellite [x, y, z] in meters.
            ut_position: 3D position of the UT [x, y, z] in meters.

        Returns:
            Elevation angle in degrees.
        """
        # Vector from UT to Satellite
        los_vec = tf.squeeze(sat_position - ut_position)

        # Distance from UT to Satellite
        distance_ut_to_sat = np.linalg.norm(los_vec)

        # unit vector along the line of sight
        unit_los = los_vec / distance_ut_to_sat

        # Relative velocity along the line of sight
        relative_velocity = np.dot(tf.squeeze(sat_velocity - ut_velocity), unit_los)
        v_light = 3e8  # Speed of light in m/s

        # Elevation angle calculation
        elevation_angle_rad = np.arcsin(los_vec[2] / distance_ut_to_sat)

        # Convert to degrees
        elevation_angle_deg = (elevation_angle_rad)*(180.0/np.pi)

        # Doppler shift calculation
        doppler_shift_hz = (self.carrier_frequency / v_light) * relative_velocity

        # delay calculation
        delay_sec = distance_ut_to_sat / v_light

        return {"elevation_angle_deg": elevation_angle_deg,
                "doppler_shift_hz": doppler_shift_hz,
                "delay_sec": delay_sec}
'''
    def compute_doppler_shift(self, sat_pos: np.ndarray, ut_pos: np.ndarray, sat_velocity: np.ndarray, ut_velocity: np.ndarray, carrier_frequency: float) -> float:
        """
        Computes the Doppler shift based on the relative velocity along the line of sight.

        Parameters:
            sat_velocity: 3D velocity of the satellite [vx, vy, vz] in m/s.
            ut_velocity: 3D velocity of the UT [vx, vy, vz] in m/s.         
            los_vec: Line-of-sight vector from UT to Satellite.
            carrier_frequency: Carrier frequency in Hz.

        Returns:
            Doppler shift in Hz.
        """
        los_vec = sat_pos - ut_pos
        # Unit vector along the line of sight
        unit_los = los_vec / np.linalg.norm(los_vec)

        # Relative velocity along the line of sight
        relative_velocity = np.dot(sat_velocity - ut_velocity, unit_los)

        # Speed of light
        c = 3e8  # m/s

        # Doppler shift calculation
        doppler_shift = (carrier_frequency / c) * relative_velocity
        
        return doppler_shift

'''     
if __name__ == "__main__":
    # Example usage of the ComputeElevationAngle class
    sat_position_start = np.array([0.0, 0.0, 600000.0])  # Example satellite position at 600 km altitude
    ut_position_start = np.array([0.0, 0.0, 1.5]).reshape(1, 1, 3)
    ut_velocity_vec = np.array([30.0, 0.0, 0.0])  # Example UT velocity vector (moving along x-axis)
    time_steps = np.linspace(-350, 350, 15)  # Simulate for 10 seconds with 1-second intervals
    
    Satellite_kinematics=MovingSatellite(600000.0)
    ut_kinematics= MovingUT(ut_position_start, ut_velocity_vec)  # Stationary UT
    elevation_kinematics= ComputeElevationAngleAndDopplerShift(carrier_frequency=3.5e9)

    
    doppler_history = []
    elevation_history = []
    delay_history = []

    for t in time_steps:
        # Update satellite and UT positions based on time
        satellite_state = Satellite_kinematics.get_satellite_position_at_time(t)
        sat_pos, sat_vel = satellite_state["r_sat"], satellite_state["v_sat"]

        print(f"sat_pos type", type(sat_pos))
        print(f"Time {t:.1f}s: Satellite Position = {sat_pos}, Satellite Velocity = {sat_vel}")

        ut_state = ut_kinematics.get_ut_position_at_time(t)
        ut_pos, ut_vel = ut_state["r_ut"], ut_state["v_ut"]
        print(f"Time {t:.1f}s: UT Position = {ut_pos}, UT Velocity = {ut_vel}")

        elevation_state = elevation_kinematics.compute_elevation_angle_and_doppler_shift(sat_pos, ut_pos, sat_vel, ut_vel)
        print(f"Time {t:.1f}s: Elevation Angle = {elevation_state['elevation_angle_deg']:.2f} degrees")

        #doppler_shift = ComputeElevationAngle().compute_doppler_shift(sat_pos, ut_pos, sat_vel, ut_vel, 2.4e9)
        print(f"Time {t:.1f}s: Doppler Shift = {elevation_state['doppler_shift_hz']:.2f} Hz")
        doppler_history.append(elevation_state['doppler_shift_hz']/1e3)  # Convert to kHz
        elevation_history.append(elevation_state['elevation_angle_deg'])
        delay_history.append(elevation_state['delay_sec']*1e3)  # Convert to milliseconds

    # Plotting Kinematics Progression
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    axs[0].plot(time_steps, doppler_history, 'r-o')
    axs[0].set_ylabel('Doppler Shift (kHz)')
    axs[0].grid(True)
    axs[0].set_title('Dynamic LEO Pass Trajectory Metrics')

    axs[1].plot(time_steps, elevation_history, 'b-o')
    axs[1].set_ylabel('Elevation Angle (deg)')
    axs[1].grid(True)

    axs[2].plot(time_steps, delay_history, 'g-o')
    axs[2].set_ylabel('Delay (msec)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
