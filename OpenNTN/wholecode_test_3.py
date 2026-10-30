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
    subcarrier_frequencies,
    generate_uts_topology
)
from sionna.phy.ofdm import ResourceGrid 
#import OpenNTN.utils as utils
from OpenNTN.step1_moving_satellite import NTNKinematicsEngine
#from OpenNTN.system_level_scenario import SystemLevelScenario

tf.random.set_seed(42)

carrier_frequency = tf.constant(3.5e9)

# Initialize Antenna Arrays
bs_array = PanelArray(num_rows_per_panel=4, 
                      num_cols_per_panel=4, 
                      polarization="dual", 
                      polarization_type="cross", 
                      antenna_pattern="38.901", 
                      carrier_frequency=carrier_frequency)

ut_array = PanelArray(num_rows_per_panel=1, 
                      num_cols_per_panel=1, 
                      polarization="single", 
                      polarization_type="V", 
                      antenna_pattern="omni", 
                      carrier_frequency=carrier_frequency)

initial_ut_pos = np.array([0.0, 0.0, 1.5])# w.r.t the origin of the ENU frame (ground level)
initial_ut_velocity = np.array([30.0, 0.0, 0.0])  # Moving at 30 m/s along the x-axis
stationary_ut_velocity = np.array([0.0, 0.0, 0.0])  # Stationary UT for comparison
moving_ut = MovingUT(initial_ut_pos, initial_ut_velocity)
ut_tensor =moving_ut.get_n_ut_position_at_time(5, 1)  # Get position at t=1s for 5 UTs
ut_expand_axes= np.expand_dims(ut_tensor["r_ut"], axis=1)  # Expand to shape (5, 1, 3)
ut_loc_tensor= tf.constant(ut_expand_axes, dtype=tf.float32)
moving_satellite = MovingSatellite(h_sat=600000.0)
bs_tensor = moving_satellite.get_n_sat_position_at_time(5, 1)  # Get position at t=1s for 5 satellites
bs_loc_ecef= bs_tensor["r_sat"]
bs_loc_enu= moving_satellite.ecef_to_enu(bs_loc_ecef)
bs_loc_tensor= tf.constant(bs_loc_enu, dtype=tf.float32)
ut_orientations= tf.zeros((1, 5, 3), dtype=tf.float32)
bs_orientations= tf.zeros((1, 5, 3), dtype=tf.float32)
compute_elevation= ComputeElevationAngleAndDopplerShift(carrier_frequency=carrier_frequency)    
#print("UT Locations at t=1s:", ut_loc_tensor)
#print("Satellite Locations at t=1s:", bs_loc_tensor)
ut_loc_tensor= tf.reshape(ut_loc_tensor, (1, 5, 3))
bs_loc_tensor= tf.reshape(bs_loc_tensor, (1, 5, 3))
elevation_angle= compute_elevation.compute_elevation_angle(bs_loc_tensor, ut_loc_tensor)
elevation_angle=tf.tile(tf.expand_dims(elevation_angle,axis=1),[1,5,1])
print("UT Locations at t=1s:", ut_loc_tensor)
print("Satellite Locations at t=1s:", bs_loc_tensor)
print("Elevation Angle at t=1s:", elevation_angle)
#bs_loc_tensor= bs_loc_tensor[0:1, 0:1, :]    # Ensure shape is 1,1,3

#elevation_angle= tf.constant(50.0, dtype=tf.float32)  # Example elevation angle in degrees


channel_model = Urban(carrier_frequency=carrier_frequency, ut_array=ut_array, bs_array=bs_array, direction="downlink", elevation_angle=elevation_angle)



#print("ut_loc from system_level_scenario:", channel_model._scenario.ut_loc)
#print("bs_loc from system_level_scenario:", channel_model._scenario.bs_loc)
#print("ut_orientations from system_level_scenario:", channel_model._scenario.ut_orientations)
#print("Scenario type:", type(channel_model._scenario))
#print("Has _lsp_log_mean:", hasattr(channel_model._scenario, "_lsp_log_mean"))
#print("Has _lsp_log_std:", hasattr(channel_model._scenario, "_lsp_log_std"))
# print("LSP Log Mean:", channel_model._scenario.lsp_log_mean())


channel_model.set_topology(ut_loc=ut_loc_tensor,
                          bs_loc=bs_loc_tensor,
                          ut_orientations=ut_orientations,
                          bs_orientations=bs_orientations,
                          ut_velocities=tf.zeros((1, 5, 3), dtype=tf.float32),
                          in_state=tf.fill([1,5],False))
print("Channel Model LOS Clusters Shape:", channel_model._scenario.num_clusters_los.shape)
print("Channel Model NLOS Clusters Shape:", channel_model._scenario.num_clusters_nlos.shape)

# ---------------------------------------------------------------------------
# 1. GEOMETRY & PROPAGATION DISTANCES
# ---------------------------------------------------------------------------
print("=== GEOMETRY & PROPAGATION ===")
print("2D Distances Shape:", channel_model._scenario.distance_2d.shape)
print("2D Distances (m):\n", channel_model._scenario.distance_2d.numpy())

print("\n3D Distances Shape:", channel_model._scenario.distance_3d.shape)
print("3D Distances (m):\n", channel_model._scenario.distance_3d.numpy())

# Line-of-Sight State (True = LOS, False = NLOS)
print("\nLOS State Shape:", channel_model._scenario.los.shape)
print("LOS State:\n", channel_model._scenario.los.numpy())

# Arrival and Departure Angles (in Radians/Degrees)
#print("\nAzimuth Angle of Departure (AoD) Shape:", channel_model._scenario.aod.shape)
#print("Zenith Angle of Departure (ZoD) Shape:", channel_model._scenario.zod.shape)

# ---------------------------------------------------------------------------
# 2. PATH LOSS & ATTENUATION
# ---------------------------------------------------------------------------
print("\n=== PATH LOSS & ATTENUATION ===")
# Basic Free-Space / 3GPP Path Loss
if hasattr(channel_model._scenario, "path_loss"):
    print("Path Loss Shape:", channel_model._scenario.path_loss.shape)
    print("Path Loss (dB):\n", channel_model._scenario.path_loss.numpy())

# Line-of-Sight Probability
if hasattr(channel_model._scenario, "p_los"):
    print("\nP_LOS Probability:\n", channel_model._scenario.p_los.numpy())

# ---------------------------------------------------------------------------
# 3. LARGE-SCALE PARAMETERS (LSPs)
# ---------------------------------------------------------------------------
print("\n=== LARGE-SCALE PARAMETERS (LSPs) ===")
lsp_obj = channel_model._lsp

# Each LSP parameter has shape [batch_size, num_bs, num_ut]
print("Shadow Fading (SF) [dB] Shape:", lsp_obj.sf.shape)
print("Shadow Fading (SF) Values:\n", lsp_obj.sf.numpy())

print("\nDelay Spread (DS) [s] Shape:", lsp_obj.ds.shape)
print("Delay Spread (DS) Values:\n", lsp_obj.ds.numpy())

print("\nRician K-Factor (K) [dB] Values:\n", lsp_obj.k_factor.numpy())
print("\nAzimuth Spread of Arrival (ASA) [deg]:\n", lsp_obj.asa.numpy())
print("Azimuth Spread of Departure (ASD) [deg]:\n", lsp_obj.asd.numpy())
print("Zenith Spread of Arrival (ZSA) [deg]:\n", lsp_obj.zsa.numpy())
print("Zenith Spread of Departure (ZSD) [deg]:\n", lsp_obj.zsd.numpy())


# ---------------------------------------------------------------------------
# 4. CORRELATION MATRICES
# ---------------------------------------------------------------------------
print("\n=== CORRELATION MATRICES ===")

# Cross-LSP 7x7 Cholesky matrix
cross_sqrt = channel_model._lsp_sampler._cross_lsp_correlation_matrix_sqrt
print("Cross-LSP Cholesky Matrix (7x7) Shape:", cross_sqrt.shape)

# Spatial correlation Cholesky matrix across UT trajectory steps
spatial_sqrt = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt
print("Spatial Correlation Cholesky Matrix Shape:", spatial_sqrt.shape)
# Shape: [batch, num_bs, 7, num_ut, num_ut] or [batch, num_bs, num_ut, num_ut]

#a,tau=channel_model(num_time_samples=100, sampling_frequency=1e6)
#print("a:", tf.shape(a))
#print("tau:", tf.shape(tau))
#show_3d_topology=channel_model.show_topology(bs_index=0, batch_index=0)
#plt.show()

'''
channel_model = Urban(carrier_frequency=2.4e9, ut_array=ut_array, bs_array=bs_array, direction="downlink", elevation_angle=50.0)

#kinematics=NTNKinematicsEngine(h_sat_m=600000.0, carrier_frequency=2.4e9, orbit_inclination_deg=30.0)

ut_kinematics= MovingUT(ut_loc, ut_velocities)
sat_kinematics= MovingSatellite(h_sat=600000.0)
elevation_kinematics= ComputeElevationAngleAndDopplerShift(carrier_frequency=2.4e9)

time_slot=np.arange(0, 10, 1)

for t in time_slot:
    #state = kinematics.get_kinematics_at_time(t, ut_loc_start, ut_velocity_vec)
    state_ut= ut_kinematics.get_ut_position_at_time(t)
    state_sat= sat_kinematics.get_satellite_position_at_time(t)
    ut_loc= tf.constant(state_ut["r_ut"].reshape(1, 1, 3), dtype=tf.float32)
    ut_velocity= tf.constant(ut_velocity_vec.reshape(1, 1, 3), dtype=tf.float32)
    bs_loc= tf.constant(state_sat["r_sat"].reshape(1, 1, 3), dtype=tf.float32)
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

'''
