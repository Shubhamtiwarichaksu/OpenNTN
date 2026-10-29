import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# OpenNTN Imports
from OpenNTN.urban import Urban
from OpenNTN.antenna import PanelArray 
from OpenNTN.utils import (
    compute_satellite_speed,
    time_lag_discrete_time_channel,
    cir_to_time_channel
)

tf.random.set_seed(42)

class NTNKinematicsEngine:
    """
    Computes dynamic 3D positions and relative velocity vectors for 
    a LEO satellite pass and a moving ground User Terminal (UT).
    """
    def __init__(self, h_sat_m=600000.0, carrier_frequency=3.5e9, orbit_inclination_deg=45.0):
        self.h_sat = float(h_sat_m)
        self.fc = float(carrier_frequency)
        self.r_earth = 6371000.0  # Earth radius in meters
        self.c = 299792458.0      # Speed of light in m/s
        self.inclination_rad = np.radians(orbit_inclination_deg)
        
        # Orbital speed from OpenNTN utility function
        self.v_sat_mag = compute_satellite_speed(tf.constant(self.h_sat, dtype=tf.float32)).numpy()
        self.r_orbit = self.r_earth + self.h_sat
        self.omega = self.v_sat_mag / self.r_orbit  # Angular velocity (rad/s)

    def get_kinematics_at_time(self, t_sec: float, ut_start_pos: np.ndarray, ut_vel_vec: np.ndarray):
        """
        Calculates instantaneous spatial positions and relative kinematics at time t_sec.
        
        Parameters:
            t_sec: Simulation timestamp in seconds.
            ut_start_pos: Initial 3D UT position [x, y, z] in meters.
            ut_vel_vec: Ground UT velocity vector [vx, vy, vz] in m/s.
        """
        # 1. Update Ground UT Position over time
        r_ut = np.array(ut_start_pos, dtype=np.float32) + np.array(ut_vel_vec, dtype=np.float32) * t_sec
        v_ut = np.array(ut_vel_vec, dtype=np.float32)

        # 2. Orbital Arc Motion for Satellite
        # Satellite passes overhead along an inclined orbital arc
        angle = self.omega * t_sec
        
        # 3D position vector in Cartesian ENU-like frame centered at UT origin ground level
        sat_x = self.r_orbit * np.sin(angle) * np.sin(self.inclination_rad)
        sat_y = self.r_orbit * np.sin(angle) * np.cos(self.inclination_rad)
        sat_z = self.r_orbit * np.cos(angle) - self.r_earth
        
        r_sat = np.array([sat_x, sat_y, sat_z], dtype=np.float32)

        # 3D velocity vector derivative
        v_sat_x = self.v_sat_mag * np.cos(angle) * np.sin(self.inclination_rad)
        v_sat_y = self.v_sat_mag * np.cos(angle) * np.cos(self.inclination_rad)
        v_sat_z = -self.v_sat_mag * np.sin(angle)
        
        v_sat = np.array([v_sat_x, v_sat_y, v_sat_z], dtype=np.float32)

        # 3. Line-of-Sight (LoS) Vector & Distance
        los_vec = r_sat - r_ut
        dist = np.linalg.norm(los_vec)
        unit_los = los_vec / dist

        # 4. Instantaneous Elevation Angle (Degrees)
        elevation_rad = np.arcsin(np.clip(los_vec[2] / dist, -1.0, 1.0))
        elevation_deg = np.degrees(elevation_rad)

        # 5. Relative Velocity (UT relative to Satellite)
        # OpenNTN expects ut_velocities = v_UT - v_Sat
        v_relative_ut_frame = v_ut - v_sat

        # 6. Physical LoS Doppler Shift and Delay
        v_radial = np.dot(v_sat - v_ut, unit_los)
        doppler_hz = (self.fc / self.c) * v_radial
        delay_sec = dist / self.c

        return {
            "r_sat": r_sat,
            "v_sat": v_sat,
            "r_ut": r_ut,
            "v_ut": v_ut,
            "v_relative_ut_frame": v_relative_ut_frame,
            "elevation_deg": elevation_deg,
            "doppler_hz": doppler_hz,
            "delay_sec": delay_sec,
            "distance_m": dist
        }


###### Simulation Example: Dynamic LEO Pass with Moving Ground UT ######

if __name__ == "__main__":
    carrier_freq = 3.5e9
    
    # Initialize Antenna Arrays
    bs_array = PanelArray(num_rows_per_panel=4, num_cols_per_panel=4, 
                          polarization="dual", polarization_type="VH", 
                          antenna_pattern="38.901", carrier_frequency=carrier_freq)

    ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, 
                          polarization="single", polarization_type="V", 
                          antenna_pattern="omni", carrier_frequency=carrier_freq)

    # Initialize Kinematics Engine and OpenNTN Urban Channel
    kinematics = NTNKinematicsEngine(h_sat_m=600000.0, carrier_frequency=carrier_freq, orbit_inclination_deg=90.0)
    channel_model = Urban(carrier_frequency=carrier_freq, ut_array=ut_array, bs_array=bs_array, direction="downlink", elevation_angle=50.0)

    # UT Initial State: Ground vehicle driving East at 30 m/s (108 km/h)
    ut_start_pos = np.array([0.0, 0.0, 1.5])
    ut_vel_vec = np.array([30.0, 0.0, 0.0])

    # Simulate a 40-second pass window in 4-second time steps
    time_steps = np.linspace(-350, 350, 15)
    
    doppler_history = []
    elevation_history = []
    delay_history = []

    print(f"{'Time (s)':<10} | {'Elev (deg)':<12} | {'Dist (km)':<12} | {'Doppler (kHz)':<15} | {'Delay (ms)':<12}")
    print("-" * 72)

    for t in time_steps:
        # A. Compute spatial positions at time t
        state = kinematics.get_kinematics_at_time(t, ut_start_pos, ut_vel_vec)
        
        # B. Prepare OpenNTN Tensors
        ut_loc = tf.constant(state["r_ut"].reshape(1, 1, 3), dtype=tf.float32)
        bs_loc = tf.constant(state["r_sat"].reshape(1, 1, 3), dtype=tf.float32)
        ut_vel = tf.constant(state["v_relative_ut_frame"].reshape(1, 1, 3), dtype=tf.float32)
        
        ut_orientations = tf.zeros((1, 1, 3), dtype=tf.float32)
        bs_orientations = tf.zeros((1, 1, 3), dtype=tf.float32)
        in_state = tf.constant([[False]])

        channel_model.elevation_angle = tf.constant(state["elevation_deg"], dtype=tf.float32)

        # C. Update OpenNTN Topology with moving geometry
        channel_model.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations, ut_vel, in_state)


        # D. Sample Channel Impulse Response at updated topology
        a, tau = channel_model(num_time_samples=100, sampling_frequency=1e6)
        print("a:", tf.shape(a))
        print("tau:", tf.shape(tau))

        # E. Record metrics for trajectory plotting
        doppler_history.append(state["doppler_hz"] / 1e3)
        elevation_history.append(state["elevation_deg"])
        delay_history.append(state["delay_sec"] * 1e3)
    
        print(f"{t:<10.1f} | {state['elevation_deg']:<12.2f} | {state['distance_m']/1000:<12.2f} | {state['doppler_hz']/1e3:<15.2f} | {state['delay_sec']*1e3:<12.3f}")

    # Plotting Kinematics Progression
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(time_steps, doppler_history, 'r-o')
    axs[0].set_ylabel('Doppler Shift (kHz)')
    axs[0].grid(True)
    axs[0].set_title('Dynamic LEO Pass Trajectory Metrics')

    axs[1].plot(time_steps, elevation_history, 'b-o')
    axs[1].set_ylabel('Elevation Angle (deg)')
    axs[1].grid(True)

    axs[2].plot(time_steps, delay_history, 'g-o')
    axs[2].set_ylabel('Propagation Delay (ms)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
    #plt.savefig('leo_pass_kinematics.png')
    print("\nSimulation complete.")
 