import copy
import math
import numpy as np

class SensorNoiseModel:
    def __init__(self, obs_dist_noise_std=0.012, obs_angle_noise_std_deg=0.5, theta_ref_noise_std_rad=math.radians(0.3)):
        self.obs_dist_noise_std = obs_dist_noise_std
        self.obs_angle_noise_std_deg = obs_angle_noise_std_deg
        self.theta_ref_noise_std_rad = theta_ref_noise_std_rad

    def apply(self, state_raw, enable_sensor_noise=True):
        if not enable_sensor_noise:
            return state_raw
        s = copy.deepcopy(state_raw).astype(np.float64)
        if s[0] != -1 and s[1] != -1:
            s[0] = s[0] + np.random.normal(0.0, self.obs_dist_noise_std)
            s[0] = max(s[0], 0.0)
            s[1] = s[1] + np.random.normal(0.0, self.obs_angle_noise_std_deg)
            s[1] = float(np.clip(s[1], -180.0, 180.0))
        s[2] = s[2] + np.random.normal(0.0, self.theta_ref_noise_std_rad)
        s[2] = float(np.clip(s[2], -math.pi, math.pi))
        return s
