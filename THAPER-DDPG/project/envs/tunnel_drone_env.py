import airsim
import time
import copy
import numpy as np
import math
from collections import deque

from algorithms.modules.delay_compensator import DelayCompensator
from algorithms.modules.sensor_noise import SensorNoiseModel

goal_threshold = 1
np.set_printoptions(precision=3, suppress=True)

TARGET_X = 160
TARGET_Y = 0.0
TARGET_Z = -5

START_X = 0.0
START_Y = 0.0
START_Z = TARGET_Z

DEFAULT_START = [START_X, START_Y, START_Z]
DEFAULT_AIM = [TARGET_X, TARGET_Y, TARGET_Z]

class TunnelDroneEnv:
    client = None

    def __init__(self, name, start=None, aim=None, scaling_factor=1):
        if start is None:
            start = DEFAULT_START
        if aim is None:
            aim = DEFAULT_AIM

        self.threshold = goal_threshold
        self.left = True
        self.scaling_factor = scaling_factor
        self.start = np.array(start, dtype=np.float64)
        self.aim = np.array(aim, dtype=np.float64)
        self.name = name
        self.height_limit = 20
        self.t_bizhang = 0
        self.path_bizhang = 0
        self.t_current = 0
        self.isbizhang = False
        self.info_before = None
        self.t_start = 0
        self.forward_atten_min = 0.3
        self.forward_atten_max = 1.0
        self.forward_atten_dscale = 4.5
        self.enable_sensor_noise = True
        self.obs_dist_noise_std = 0.012
        self.obs_angle_noise_std_deg = 0.5
        self.theta_ref_noise_std_rad = math.radians(0.3)
        self.dropout_prob = 0.002
        self.enable_delay_model = True
        self.delay_steps = 1
        self.delay_comp_alpha = 0.1
        self.state_buffer = deque(maxlen=20)
        self.last_output_state = None
        self.sensor_noise_model = SensorNoiseModel(
            obs_dist_noise_std=self.obs_dist_noise_std,
            obs_angle_noise_std_deg=self.obs_angle_noise_std_deg,
            theta_ref_noise_std_rad=self.theta_ref_noise_std_rad,
        )
        self.delay_compensator = DelayCompensator(
            delay_steps=self.delay_steps,
            delay_comp_alpha=self.delay_comp_alpha
        )

        if aim is None:
            self.rand = True
            self.start = np.array(DEFAULT_START, dtype=np.float64)
        else:
            self.rand = False
            self.aim_height = self.aim[2]

    def reset_aim(self):
        self.isbizhang = False
        self.t_bizhang = 0
        self.path_bizhang = 0
        self.t_current = time.perf_counter()
        self.t_start = self.t_current
        self.info_before = None
        self.state_buffer.clear()
        self.last_output_state = None

    def reset(self):
        self.reset_aim()
        self.client.enableApiControl(True, self.name)
        self.client.armDisarm(True, self.name)

        if self.left:
            if self.name == "Drone2":
                self.client.moveToPositionAsync(
                    self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2],
                    2, 10, vehicle_name=self.name
                )
            elif self.name == "Drone3":
                self.client.moveToPositionAsync(
                    self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2],
                    2, 10, vehicle_name=self.name
                )
            else:
                self.client.moveToPositionAsync(
                    self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2],
                    2, 10, vehicle_name=self.name
                )
            self.left = False
        else:
            if self.name == "Drone2":
                self.client.moveToPositionAsync(
                    self.start.tolist()[0], self.start.tolist()[1] - 1.5, self.start.tolist()[2],
                    2, 10, vehicle_name=self.name
                )
            elif self.name == "Drone3":
                self.client.moveToPositionAsync(
                    self.start.tolist()[0], self.start.tolist()[1] - 1.5, self.start.tolist()[2],
                    2, 10, vehicle_name=self.name
                )
            else:
                self.client.moveToPositionAsync(
                    self.start.tolist()[0], self.start.tolist()[1] - 1.5, self.start.tolist()[2],
                    2, 10, vehicle_name=self.name
                )
            self.left = True

        time.sleep(2)
        self.state = self.getState()
        return self.state

    def _calc_pos_ref(self, t_now):
        pos_start = np.array([0.0, 0.0, float(self.start[2])], dtype=np.float64)
        dpos0 = self.aim - pos_start
        temp_ref = np.sqrt(dpos0[0] ** 2 + dpos0[1] ** 2 + dpos0[2] ** 2)
        if temp_ref <= 1e-6:
            temp_ref = 1e-6

        dx_ref = dpos0[0] / temp_ref * 1
        dy_ref = dpos0[1] / temp_ref * 1
        dz_ref = dpos0[2] / temp_ref * 1

        pos_x_ref = dx_ref * (t_now - self.t_start)
        pos_y_ref = dy_ref * (t_now - self.t_start)
        pos_z_ref = dz_ref * (t_now - self.t_start) + float(self.start[2])

        return np.array([pos_x_ref, pos_y_ref, pos_z_ref], dtype=np.float64)

    def _theta_ref_and_d_ref(self, pos_ref, pos, yaw):
        v = pos_ref[:2] - pos[:2]
        nv = float(np.linalg.norm(v))
        if nv <= 1e-6:
            theta_ref = 0.0
        else:
            fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
            cross = float(fwd[0] * v[1] - fwd[1] * v[0])
            dot = float(fwd[0] * v[0] + fwd[1] * v[1])
            theta_ref = float(math.atan2(cross, dot))
        d_ref = float(np.linalg.norm(pos_ref - pos))
        return theta_ref, d_ref

    def _calc_forward_atten(self, d_obs):
        if d_obs is None or d_obs < 0:
            return 1.0
        k = self.forward_atten_min + (self.forward_atten_max - self.forward_atten_min) * (
            d_obs / self.forward_atten_dscale
        )
        return float(np.clip(k, self.forward_atten_min, self.forward_atten_max))

    def _apply_sensor_noise_to_state(self, state_raw):
        return self.sensor_noise_model.apply(
            state_raw,
            enable_sensor_noise=self.enable_sensor_noise
        )

    def _apply_delay_and_compensation(self, state_raw):
        return self.delay_compensator.apply(
            self.state_buffer,
            state_raw,
            enable_delay_model=self.enable_delay_model
        )

    def getState(self):
        min_distance, angle_min_distance = self.getlarder_data()
        t_now = time.perf_counter()
        pos_ref = self._calc_pos_ref(t_now)
        st = self.client.getMultirotorState(vehicle_name=self.name)
        p = st.kinematics_estimated.position
        pos = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float64)
        (_, _, yaw) = airsim.to_eularian_angles(st.kinematics_estimated.orientation)
        theta_ref, d_ref = self._theta_ref_and_d_ref(pos_ref, pos, yaw)
        state = np.array([min_distance, angle_min_distance, theta_ref, d_ref], dtype=np.float64)

        if self.enable_sensor_noise and self.last_output_state is not None:
            if np.random.rand() < self.dropout_prob:
                return copy.deepcopy(self.last_output_state)

        state = self._apply_sensor_noise_to_state(state)
        state = self._apply_delay_and_compensation(state)
        self.last_output_state = copy.deepcopy(state)
        return state

    def moveByDist(self, diff, forward):
        temp = airsim.YawMode()
        temp.is_rate = not forward
        self.client.moveByVelocityAsync(
            diff[0], diff[1], diff[2], 2,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=temp,
            vehicle_name=self.name
        )
        time.sleep(0.2)
        return 0

    def moveByBodyDist(self, diff, forward):
        state = self.client.getMultirotorState(vehicle_name=self.name)
        (_, _, yaw) = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        temp = airsim.YawMode()
        temp.is_rate = not forward
        temp.yaw_or_rate = yaw / 3.14 * 180
        self.client.moveByVelocityBodyFrameAsync(
            diff[0], diff[1], diff[2], 2,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=temp,
            vehicle_name=self.name
        )
        time.sleep(0.2)
        return 0

    def _normalize_state_and_reward(self, state_, reward):
        s = copy.deepcopy(state_).astype(np.float64)
        if s[0] != -1 and s[1] != -1:
            s[0] = s[0] * np.sign(s[1]) / 4.0
            s[1] = s[1] / 180.0
        s[2] = s[2] / math.pi
        s[3] = s[3] / 20.0
        r = reward / 50.0
        self.state = s
        return s, r

    def step(self, action):
        t_jiange = time.perf_counter() - self.t_current
        self.t_current = time.perf_counter()

        pos_ref = self._calc_pos_ref(self.t_current)
        pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)
        dpos = pos_ref - pos
        temp = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2 + dpos[2] ** 2)

        if temp >= 4:
            self.scaling_factor = 2
        else:
            self.scaling_factor = 1

        if temp <= 0.001:
            temp = 0.001

        state_ = self.getState()
        reward = 0
        done = False
        info = None

        if state_[0] == -1:
            dx = dpos[0] / temp * self.scaling_factor
            dy = dpos[1] / temp * self.scaling_factor
            dz = dpos[2] / temp * self.scaling_factor
            self.moveByDist([dx, dy, dz], True)

            if self.isbizhang:
                self.isbizhang = False
                info = "success bizhang"
                if self.t_bizhang != 0 and self.path_bizhang != 0:
                    reward = (8 * 1.14 / self.t_bizhang + 8 * 1.14 / self.path_bizhang) * 40
                self.t_bizhang = 0
                self.path_bizhang = 0

            state_ = self.getState()

        else:
            if abs(state_[1]) >= 90:
                dx = dpos[0] / temp * self.scaling_factor
                dy = dpos[1] / temp * self.scaling_factor
                dz = dpos[2] / temp * self.scaling_factor

                if dpos[0] != 0 and dpos[1] != 0 and dpos[2] != 0:
                    self.moveByDist([dx, dy, dz], True)

                if self.isbizhang:
                    self.isbizhang = False
                    info = "success bizhang"
                    reward = (8 * 1.14 / self.t_bizhang + 8 * 1.14 / self.path_bizhang) * 20
                    self.t_bizhang = 0
                    self.path_bizhang = 0

                state_ = self.getState()

            else:
                self.isbizhang = True
                self.t_bizhang = self.t_bizhang + t_jiange
                vel = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.linear_velocity
                vel_linear = np.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
                self.path_bizhang = vel_linear * t_jiange + self.path_bizhang

                d_obs = float(state_[0])
                k = self._calc_forward_atten(d_obs)
                dx_raw = float(action[0] + 2.0)
                dx = dx_raw * k
                dy = float(action[1])
                dz = float(self.scaling_factor * dpos[2] / temp)
                self.moveByBodyDist([dx, dy, dz], True)

                state_ = self.getState()

                if (-pos[2] + self.aim_height) > self.height_limit and info is None:
                    info = "too high"
                    reward = -30
                    done = True
                    norm_state, reward = self._normalize_state_and_reward(state_, reward)
                    return norm_state, reward, done, info

                if pos[2] > -0.15 and info is None:
                    info = "too low"
                    reward = -30
                    done = True
                    norm_state, reward = self._normalize_state_and_reward(state_, reward)
                    return norm_state, reward, done, info

                if state_[0] <= 0.65 and state_[0] > 0:
                    reward = -100
                    info = "collision"
                    done = True
                    print(self.name, "+", info)
                    norm_state, reward = self._normalize_state_and_reward(state_, reward)
                    return norm_state, reward, done, info

                if abs(state_[1]) < 90 and info is None:
                    if state_[0] <= 2.4 and state_[0] > 0.65:
                        info = "too close"
                        reward = -50
                    if state_[0] > 2.4:
                        info = "safe distance"
                        xishu = 2.2
                        if (state_[1] > 0 and action[1] > 0) or (state_[1] < 0 and action[1] < 0):
                            xishu = -2.2
                            info = "safe distance but wrong direction"
                        reward = 30 * xishu

        if self.isDone() and info is None:
            print("success", "+", self.name)
            info = "success"
            done = True
            norm_state, reward = self._normalize_state_and_reward(state_, reward)
            return norm_state, reward, done, info

        if self.isChaoshi() and info is None:
            print("Timeout", "+", self.name)
            info = "Timeout"
            reward = -50
            done = True
            norm_state, reward = self._normalize_state_and_reward(state_, reward)
            return norm_state, reward, done, info

        norm_state, reward = self._normalize_state_and_reward(state_, reward)
        return norm_state, reward, done, info

    def isDone(self):
        pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        return self.distance(self.aim, pos) < self.threshold

    def isChaoshi(self):
        pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        if pos[0] - self.aim[0] >= 5:
            print("Exceeded target point")
            return True
        return False

    def distance(self, pos1, pos2):
        return np.sqrt(
            abs(pos1[0] - pos2[0]) ** 2 +
            abs(pos1[1] - pos2[1]) ** 2 +
            abs(pos1[2] - pos2[2]) ** 2
        )

    def getlarder_data(self):
        lidar_name = "LidarSensor1"
        if self.name == "Drone2":
            lidar_name = "LidarSensor2"
        if self.name == "Drone3":
            lidar_name = "LidarSensor3"

        lidarData = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=self.name)
        if len(lidarData.point_cloud) < 3:
            return -1, -1

        points = self.parse_lidarData(lidarData)
        min_distance = 10
        angle_min_distance = 180

        for j in range(0, len(points) - 1):
            distance = np.sqrt(points[j][0] ** 2 + points[j][1] ** 2 + points[j][2] ** 2)
            if distance < min_distance:
                min_distance = distance
                min_y = points[j][1]
                if min_distance > 1e-8:
                    ratio = np.clip(min_y / min_distance, -1.0, 1.0)
                    angle_min_distance = math.asin(ratio) / math.pi * 180
                    if points[j][0] < 0:
                        angle_min_distance = np.sign(angle_min_distance) * 180 - angle_min_distance
                else:
                    angle_min_distance = 0.0

        return abs(min_distance), angle_min_distance

    def parse_lidarData(self, data):
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        return np.reshape(points, (int(points.shape[0] / 3), 3))

drone_env_collisionabvoidance = TunnelDroneEnv
