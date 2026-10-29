import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from OpenNTN.urban import Urban
from OpenNTN.antenna import PanelArray
from OpenNTN.utils import cir_to_time_channel, sample_bernoulli
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
    kinematics = NTNKinematicsEngine(h_sat_m=600000.0, carrier_frequency=3.5e9)
    channel_model = Urban(carrier_frequency=3.5e9, ut_array=ut_array, bs_array=bs_array, direction="downlink", elevation_angle=50.0)
    """
    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, in_state=None, los=None):
    """

    ut_initial_pos= np.array([0.0, 0.0, 1.5])
    ut_vel_vec= np.array([0.0, 0.0, 0.0]) # Stationary UT on ground
    
    bs_initial_pos= np.array([0.0, 0.0, 600000.0])
    
    in_state= tf.constant([[False]])

    a,tau=channel_model.set_topology(ut_loc=tf.constant(ut_initial_pos.reshape(1, 1, 3), dtype=tf.float32),
                              bs_loc=tf.constant(bs_initial_pos.reshape(1, 1, 3), dtype=tf.float32),
                              ut_orientations=tf.zeros((1, 1, 3)),
                              bs_orientations=tf.zeros((1, 1, 3)),
                              ut_velocities=tf.constant(ut_vel_vec.reshape(1, 1, 3), dtype=tf.float32),
                              in_state=tf.constant([[False]]))
    
    print('test message')
    print("a:", tf.shape(a))
    print("a:", a)
    print("tau:", tf.shape(tau))
    print("tau:", tau)

#generate_dynamic_multipath_taps()