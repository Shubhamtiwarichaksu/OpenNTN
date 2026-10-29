import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# OpenNTN Imports
from OpenNTN.urban import Urban
from OpenNTN.antenna import PanelArray 

from OpenNTN.moving_ut import MovingUT
from OpenNTN.moving_satellite import MovingSatellite
from OpenNTN.elevation_angle import ComputeElevationAngleAndDopplerShift

from OpenNTN.utils import (
    compute_satellite_speed,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    gen_single_sector_topology,
    time_to_ofdm_channel,
    cir_to_ofdm_channel,
    subcarrier_frequencies
)
from sionna.phy.ofdm import ResourceGrid 
#import OpenNTN.utils as utils
from OpenNTN.step1_moving_satellite import NTNKinematicsEngine

tf.random.set_seed(42)

# Initialize Antenna Arrays
bs_array = PanelArray(num_rows_per_panel=4, num_cols_per_panel=4, 
                          polarization="dual", polarization_type="VH", 
                          antenna_pattern="38.901", carrier_frequency=2.4e9)

ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, 
                          polarization="single", polarization_type="V", 
                          antenna_pattern="omni", carrier_frequency=2.4e9)
#uts_topology= utils.gen_uts_topology(batch_size=1,num_ut=1, num_interferers=0, scenario="urb")
topology = gen_single_sector_topology(batch_size=1,num_ut=1, scenario="urb")
#print("ut_loc_matrix", topology["ut_loc"])
#print("ut_velocities_matrix", topology["ut_velocities"])
#print("bs_loc_matrix", topology["bs_loc"])
#print("ut_orientations_matrix", topology["ut_orientations"])
#print("in_state_matrix", topology["in_state"])  

#ut_loc_start, bs_loc_start, ut_orientations, bs_orientations, ut_velocities_start, in_state = topology
ut_loc_start = np.array([0.0, 0.0, 1.5])#tf.constant(topology["ut_loc"], dtype=tf.float32)
bs_loc_start = np.array([0.0, 0.0, 600000.0])#tf.constant(topology["bs_loc"], dtype=tf.float32)
ut_orientations = tf.zeros((1, 1, 3), dtype=tf.float32)
bs_orientations = tf.zeros((1, 1, 3), dtype=tf.float32)
#ut_velocities_start = tf.constant(topology["ut_velocities"], dtype=tf.float32)
ut_velocity_vec = np.array([30.0, 0.0, 0.0])
in_state = tf.constant([[False]])

#print("ut_loc:", ut_loc_start)
#print("bs_loc:", bs_loc_start)

channel_model = Urban(carrier_frequency=2.4e9, ut_array=ut_array, bs_array=bs_array, direction="downlink", elevation_angle=50.0)

#kinematics=NTNKinematicsEngine(h_sat_m=600000.0, carrier_frequency=2.4e9, orbit_inclination_deg=30.0)

ut_kinematics= MovingUT(ut_loc_start, ut_velocity_vec)
sat_kinematics= MovingSatellite(h_sat=600000.0)
elevation_kinematics= ComputeElevationAngleAndDopplerShift(carrier_frequency=2.4e9)


time_slot=np.arange(0, 1, 1)

for t in time_slot:
    #state = kinematics.get_kinematics_at_time(t, ut_loc_start, ut_velocity_vec)
    state_ut= ut_kinematics.get_ut_position_at_time(t)
    state_sat= sat_kinematics.get_satellite_position_at_time(t)
    ut_loc= tf.constant(state_ut["r_ut"].reshape(1, 1, 3), dtype=tf.float32)
    print(f"ut_loc at t={t:.1f}s:", ut_loc)
    ut_velocity= tf.constant(ut_velocity_vec.reshape(1, 1, 3), dtype=tf.float32)
    bs_loc= tf.constant(state_sat["r_sat"].reshape(1, 1, 3), dtype=tf.float32)
    print(f"bs_loc at t={t:.1f}s:", bs_loc)
    bs_velocity= tf.constant(state_sat["v_sat"].reshape(1, 1, 3), dtype=tf.float32)
    #print(f"satellite pos type at t={t:.1f}s:", type(state_sat["r_sat"]))
    #ut_velocities= tf.constant(state["v_relative_ut_frame"].reshape(1, 1, 3), dtype=tf.float32)

    elevation_deg= elevation_kinematics.compute_elevation_angle_and_doppler_shift(sat_position=bs_loc, ut_position=ut_loc,sat_velocity=bs_velocity, ut_velocity=ut_velocity)

    #elevation_deg= tf.constant(state["elevation_deg"], dtype=tf.float32)

    channel_model.elevation_angle = elevation_deg
    #print(f"ut_loc at t={t:.1f}s:", ut_loc.shape)
    #print(f"bs_loc at t={t:.1f}s:", bs_loc.shape)

    channel_model.set_topology(
        ut_loc=ut_loc,
        bs_loc=bs_loc,
        ut_orientations=ut_orientations,
        bs_orientations=bs_orientations,
        ut_velocities=ut_velocity,
        in_state=in_state)
    show_3d_topology=channel_model.show_topology(bs_index=0, batch_index=0)
    plt.show()
    a, tau = channel_model(num_time_samples=100, sampling_frequency=1e6)
    #channel= OFDMChannel(bandwidth_hz=1e6, num_taps=31)
    #print("a:", tf.shape(a))
    #print("tau:", tf.shape(tau))
    l_min,l_max= time_lag_discrete_time_channel(1e6)
    #h_t= tf.constant([100, l_max-l_min+1], dtype=tf.complex32)
    #h_f= cir_to_ofdm_channel(h_t, rg, l_min )
    #print("l_min:", l_min)
    #print("l_max:", l_max)
    h_time = cir_to_time_channel(1e6, a, tau, l_min=l_min, l_max=l_max)
    #print("h_time:", tf.shape(h_time))

    rg=ResourceGrid(num_ofdm_symbols=1, fft_size=1024, subcarrier_spacing=15e3)

    #h_f= cir_to_ofdm_channel(2.4e6, a, tau)
    #print("h_f:", tf.shape(h_f))
    frequencies= subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    print("frequencies:", tf.shape(frequencies))
    print("frequencies:", frequencies)
    #tdl= TDL("A", frequencies, 1e6, a, tau)
    #cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)

    #h_freq= tf.squeeze(cir_to_ofdm_channel(1e6, a, tau, rg, l_min=l_min))
    #print("h_freq:", tf.shape(h_freq))
"""
    h_freq_hat= tf.squeeze(time_to_ofdm_channel(h_time, rg, l_min))
    print("h_freq_hat:", tf.shape(h_freq_hat))
    print("h_freq_hat:", h_freq_hat)
 
    plt.figure(figsize=(10, 6))
    plt.plot(np.real(h_freq_hat), label='Real Part')
    plt.plot(np.imag(h_freq_hat), label='Imaginary Part')
    plt.title(f'Frequency Response of the Channel at t={t:.1f}s')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Channel Frequency Response')
    plt.legend()
    plt.grid()
    plt.show()
"""