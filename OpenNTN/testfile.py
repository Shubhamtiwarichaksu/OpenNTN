import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
print("NumPy version:", np.__version__)
import sionna as sn
print("Sionna version:", sn.__version__)
import mitsuba
print("Mitsuba version:", mitsuba.__version__)
import drjit
print("DrJIT version:", drjit.__version__)

from sionna.phy.channel import tr38811

print("OpenNTN integration:", tr38811.__file__)
print("OpenNTN successfully initialized!")

#from sionna import OpenNTN
#print("OpenNTN version:", OpenNTN.__version__)

