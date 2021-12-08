from filterpy.kalman import KalmanFilter
import numpy as np

class IP_KF():
    def __init__(self, init_x, init_y):
        
        # System and Measurement model
        # x = [x_postition, x_velocity, y_position, y_velocity] : system states
        # x = [x_postition, y_position] : Measurement states
        
        self.dt = 0.001
        self.F = np.array([                     # system matrix
        [1, self.dt, 0,  0],
        [0,  1, 0,  0],
        [0,  0, 1, self.dt],
        [0,  0, 0,  1]], dtype=np.float)
        
        self.H = np.array([                     # measurement matrix
        [1, 0, 0, 0],
        [0, 0, 1, 0]])

        # self.H = np.array([                     # measurement matrix
        # [1, 0],
        # [0, 1]])
        
        self.Q = 0.9*np.eye(4, dtype=np.float)  # system error matrix
        self.R = np.array([                     # measurement error matrix
        [100, 0],
        [0, 100]], dtype=np.float)
    
        # Kalman filter using filterpy
        
        self.kf = KalmanFilter (dim_x=4, dim_z=2)
        self.kf.x = np.array([[init_x], [0], [init_y], [0]])                         
        self.kf.F = self.F
        self.kf.H = self.H
        self.kf.Q = self.Q
        self.kf.R = self.R
        self.kf.P *= 1000.
        
    def update(self, x, y):
        # self.dt = 0.001
        # self.F = np.array([                     # system matrix
        # [1, self.dt, 0,  0],
        # [0,  1, 0,  0],
        # [0,  0, 1, self.dt],
        # [0,  0, 0,  1]], dtype=np.float)

        
        z = np.array([[x], [y]])
        self.kf.predict()
        self.kf.update(z)
        return self.kf.x