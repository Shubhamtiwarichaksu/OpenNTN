import numpy as np

class MovingUT:
    """
    Class to simulate a moving User Terminal (UT) in a 3D space.
    """
    def __init__(self, initial_position: np.ndarray, velocity_vector: np.ndarray):
        """
        Initializes the MovingUT with its initial position and velocity vector.
        
        Parameters:
            initial_position: Initial 3D position of the UT [x, y, z] in meters.
            velocity_vector: Ground UT velocity vector [vx, vy, vz] in m/s.
        """
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.velocity_vector = np.array(velocity_vector, dtype=np.float32)
    
    def get_ut_position_at_time(self, t_sec: float) -> np.ndarray:
        """
        Calculates the position of the UT at a given time t_sec.

        Parameters:
            t_sec: Time in seconds since the start of the simulation.

        Returns:
            3D position of the UT at time t_sec [x, y, z] in meters.
        """
        r_ut = self.initial_position + self.velocity_vector * t_sec
        v_ut = self.velocity_vector

        return {"r_ut": r_ut, "v_ut": v_ut}

    def get_n_ut_position_at_time(self, n_ut: int, t_sec: float) -> np.ndarray:
        """
        Calculates the positions of multiple UTs at a given time t_sec.

        Parameters:
            n_ut: Number of UTs to calculate positions for.
            t_sec: Time in seconds since the start of the simulation.

        Returns:
            Array of shape (n_ut, 3) containing the 3D positions of the UTs at time t_sec [x, y, z] in meters.
        """
        n_ut_vec = np.arange(n_ut)
        r_ut = self.initial_position + self.velocity_vector * t_sec*n_ut_vec[:, np.newaxis] #shape (n_ut, 3)
        v_ut = self.velocity_vector


        return {"r_ut": r_ut, "v_ut": v_ut}

if __name__ == "__main__":
    # Example usage of the MovingUT class
    initial_position = [0.0, 0.0, 1.5]  # Initial position of the UT in meters
    velocity_vector = [30.0, 0.0, 0.0]   # Velocity vector of the UT in m/s

    moving_ut = MovingUT(initial_position, velocity_vector)

    # Get positions of 5 UTs at time t = 1 seconds
    n_ut = 5
    t_sec = 1.0
    positions = moving_ut.get_n_ut_position_at_time(n_ut, t_sec)

    print("Positions of UTs at time t =", t_sec, "seconds:")
    print(positions["r_ut"])
    print(type(positions["r_ut"]))
    print(positions["r_ut"].shape)