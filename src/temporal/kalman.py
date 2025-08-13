import time
import torch
import numpy as np


class QuaternionKalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial estimate covariance

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.normalize_quaternion()

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        self.normalize_quaternion()

    def normalize_quaternion(self):
        q = self.x[3:7]
        self.x[3:7] = q / np.linalg.norm(q)

    def get_state(self):
        return self.x


class KalmanFilterPosSimple:
    def __init__(self, dt, position):
        self.dt = dt
        self.e, self.a, self.h, self.q, self.r, self.p = self.reset(dt, position)

    def reset(self, dt, position):

        self.dt = dt

        # Create matrices and not arrays to use matrix operators: *, .T, .I

        # Initial state vector (vertical) = [position, speed]
        e = np.matrix([position[0], position[1], position[2], 0, 0, 0]).T

        # Transition matrix
        a = np.matrix([[1, 0, 0, dt, 0, 0],
                       [0, 1, 0, 0, dt, 0],
                       [0, 0, 1, 0, 0, dt],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])

        # Observation matrix (we only observe x, y and z, not speed)
        h = np.matrix([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0]])

        # Noise relative to system evolution
        q = np.matrix(np.eye(6) * 1)

        # Measurement Noise Covariance Matrix.
        # It's a measure of how much trust the filter should place in the incoming data
        # High values means noisy data
        r = np.matrix(np.eye(3) * 100)

        p = np.matrix(np.eye(6) * 1)

        return e, a, h, q, r, p

    def predict(self):
        self.e = self.a * self.e
        # Compute error covariance
        self.p = self.a * self.p * self.a.T + self.q
        return self.e

    def update(self, position):
        # position must be a vertical numpy vector
        # Compute kalman gain
        k = self.p * self.h.T * (self.h * self.p * self.h.T + self.r).I
        # Correction / innovation
        self.e = self.e + k * (position - self.h * self.e)
        self.p = (np.matrix(np.eye(6)) - k * self.h) * self.p
        return self.e
