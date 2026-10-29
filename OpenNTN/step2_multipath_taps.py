import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from OpenNTN.urban import Urban
from OpenNTN.antenna import PanelArray 
from OpenNTN.utils import cir_to_time_channel,sample_bernoulli
from OpenNTN.step1_moving_satellite import NTNKinematicsEngine

tf.random.set_seed(42)

def generate_dynamic_multipath_taps(sim_duration_sec=10.0, dt_sec=1.0, bandwidth_hz=20e6):
    """
    Step 2: Generates phase-continuous discrete-time channel taps (h_time)
    across a LEO satellite trajectory.
    """
    carrier_freq = 3.5e9
    
    # Initialize Antenna Arrays
    bs_array = PanelArray(num_rows_per_panel=2, num_cols_per_panel=2, 
                          polarization="single", polarization_type="V",antenna_pattern="38.901", carrier_frequency=carrier_freq)
    ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, 
                          polarization="single", polarization_type="V", antenna_pattern="38.901", carrier_frequency=carrier_freq)

    # Initialize Engine & OpenNTN Channel Model
    kinematics = NTNKinematicsEngine(h_sat_m=600000.0, carrier_frequency=carrier_freq)
    channel_model = Urban(carrier_frequency=carrier_freq, ut_array=ut_array, bs_array=bs_array, direction="downlink", elevation_angle=50.0)

    ut_start_pos = np.array([0.0, 0.0, 1.5])
    ut_vel_vec = np.array([0.0, 0.0, 0.0]) # Stationary UT on ground

    time_steps = np.arange(0.0, sim_duration_sec, dt_sec)
    h_time_list = []
    
    accumulated_phase = 0.0

    print(f"Generating multipath taps for Bandwidth = {bandwidth_hz/1e6:.1f} MHz...")
    
    for t in time_steps:
        # 1. Update Spatial Trajectory
        state = kinematics.get_kinematics_at_time(t, ut_start_pos, ut_vel_vec)
        
        # 2. Update Channel Elevation Angle & Topology
        channel_model.elevation_angle = tf.constant(state["elevation_deg"], dtype=tf.float32)
        
        ut_loc = tf.constant(state["r_ut"].reshape(1, 1, 3), dtype=tf.float32)
        bs_loc = tf.constant(state["r_sat"].reshape(1, 1, 3), dtype=tf.float32)
        ut_vel = tf.constant(state["v_relative_ut_frame"].reshape(1, 1, 3), dtype=tf.float32)
        
        channel_model.set_topology(ut_loc, bs_loc, 
                                  tf.zeros((1, 1, 3)), tf.zeros((1, 1, 3)), 
                                  ut_vel, tf.constant([[False]]))

        # 3. Generate Raw Channel Impulse Response (Clusters: a, tau)
        a, tau = channel_model(num_time_samples=100, sampling_frequency=bandwidth_hz)

        # 4. Apply Phase Continuity (Doppler Phase Rotation over time delta dt)
        if len(h_time_list) > 0:
            accumulated_phase += 2.0 * np.pi * state["doppler_hz"] * dt_sec
            phase_rotation = tf.complex(tf.cos(accumulated_phase), tf.sin(accumulated_phase))
            phase_rotation = tf.cast(phase_rotation, dtype=a.dtype)
            a = a * phase_rotation

        # 5. Convert Continuous Clusters to Discrete Time Domain Taps (h_time)
        h_time = cir_to_time_channel(bandwidth_hz, a, tau, l_min=-5, l_max=25)
        print(f"h_time_shape at t={t:.1f}s: {h_time.shape}")
        
        # Store discrete tap vector for timestamp t (Shape: num_taps)
        # Squeeze batch/antenna dimensions for single SISO pair
        tap_vector = np.squeeze(h_time.numpy())
        h_time_list.append(tap_vector)

    h_time_matrix = np.array(h_time_list) # Shape: (time_steps, num_taps)
    return time_steps, h_time_matrix

if __name__ == "__main__":
    time_steps, h_matrix = generate_dynamic_multipath_taps(sim_duration_sec=10.0, dt_sec=1.0)
    #print("Generated h_matrix shape:", h_matrix.shape)
"""
    # Plot Tap Magnitude over Time (Delay-Doppler Heatmap)
    plt.figure(figsize=(9, 5))
    plt.imshow(np.abs(h_matrix).T, aspect='auto', origin='lower',
               extent=[time_steps[0], time_steps[-1], -5, 25], cmap='viridis')
    plt.colorbar(label='Tap Magnitude |h(t, l)|')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Discrete Tap Index (Delay Bin)')
    plt.title('Time-Varying Discrete Channel Taps (OpenNTN Multipath)')
   # plt.savefig('multipath_taps_over_time.png')
    print("Step 2 Complete! Heatmap saved as 'multipath_taps_over_time.png'.")
"""