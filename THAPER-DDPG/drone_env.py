import AirSimClient
import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import math
import time

goal_threshold = 1
np.set_printoptions(precision=3, suppress=True)

#-------------------------------------------------------
# Obstacle avoidance algorithm based on LiDAR
# Action output is desired velocity
# State includes distance to obstacle, angle between obstacle direction and motion direction

class drone_env_collisionabvoidance:
	def __init__(self, name, start=[0, 0, -5], aim=[190, 0, -5], scaling_factor=1):
		self.threshold = goal_threshold
		self.left = True
		self.scaling_factor = scaling_factor
		self.start = np.array(start)
		self.aim = np.array(aim)
		self.name = name
		self.height_limit = 20
		self.t_bizhang = 0
		self.path_bizhang = 0
		self.t_current = 0
		self.isbizhang = False
		self.info_before = None
		self.t_start = 0;
		# self.rand = False
		if aim == None:
			self.rand = True
			self.start = np.array([0, 0, -5])
		else:
			self.aim_height = self.aim[2]

	def reset_aim(self):
		#self.aim = (np.random.rand(3) * 300).astype("int") - 150
		#self.aim[2] = -np.random.randint(10) - 5
		#print("Our aim is: {}".format(self.aim).ljust(80, " "), end='\r')
		#self.aim_height = self.aim[2]
		self.isbizhang = False 
		self.t_bizhang = 0
		self.path_bizhang = 0
		self.t_current = time.clock()
		self.t_start = self.t_current
		self.info_before = None 
		
	def reset(self):
		# if self.rand:
		self.reset_aim()
		self.client.enableApiControl(True, self.name)
		self.client.armDisarm(True, self.name)
		if self.left:
			if self.name == "Drone2":
				self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2],
												2, 10, vehicle_name=self.name)
			if self.name == "Drone3":
				self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2],
												2, 10, vehicle_name=self.name)
			else:
				self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2],
												2, 10, vehicle_name=self.name)
			# self.aim[1] = 0
			self.left = False
		else:
			if self.name == "Drone2":
				self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1] - 1.5,
												self.start.tolist()[2], 2, 10, vehicle_name=self.name)
			if self.name == "Drone3":
				self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1] - 1.5, self.start.tolist()[2],
												2, 10, vehicle_name=self.name)
			else:
				self.client.moveToPositionAsync(self.start.tolist()[0], self.start.tolist()[1] - 1.5,
												self.start.tolist()[2], 2, 10, vehicle_name=self.name)
			# self.aim[1] = -1.5
			self.left = True
		time.sleep(2)
		self.state = self.getState()
		return self.state

	def getState(self):
		# pos = v2t(self.client.getPosition())
		# pos = self.client.getMultirotorState(vehicle_name='').kinematics_estimated.position
		# pos = np.array([pos.x_val, pos.y_val, pos.z_val])
		# vel = self.client.getMultirotorState(vehicle_name='').kinematics_estimated.linear_velocity
		# vel_linear = np.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
		min_distance, angle_min_distance = self.getlarder_data()
		# img = self.getImg()
		# state = [img, np.array([min_distance, angle_min_distance, vel_linear])]
		state = np.array([min_distance, angle_min_distance])

		return state

	def moveByDist(self,diff, forward):
		temp = airsim.YawMode()
		temp.is_rate = not forward
		self.client.moveByVelocityAsync(diff[0], diff[1], diff[2], 2, drivetrain=airsim.DrivetrainType.ForwardOnly,
										yaw_mode=temp, vehicle_name=self.name)
		time.sleep(0.2)
		return 0

	def moveByBodyDist(self, diff, forward):
		state = self.client.getMultirotorState(vehicle_name=self.name)
		(pitch, roll, yaw) = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
		temp = airsim.YawMode()
		temp.is_rate = not forward
		temp.yaw_or_rate = yaw / 3.14 * 180
		self.client.moveByVelocityBodyFrameAsync(diff[0], diff[1], diff[2], 2, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=temp, vehicle_name=self.name)
		time.sleep(0.2)
		return 0

	def render(self, extra1="", extra2=""):
		pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
		pos = np.array([pos.x_val, pos.y_val, pos.z_val])
		goal = self.distance(self.aim, pos)
		print(extra1, "distance:", int(goal), "position:", pos.astype("int"), extra2)

	def help(self):
		print("drone simulation environment")

	def step(self, action):
		t_jiange = time.clock() - self.t_current
		self.t_current = time.clock()
		pos_start = np.array([0, 0, -5])
		dpos = self.aim - pos_start
		temp_ref = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2 + dpos[2] ** 2)
		dx_ref = dpos[0] / temp_ref * 1
		dy_ref = dpos[1] / temp_ref * 1
		dz_ref = dpos[2] / temp_ref * 1
		pos_x_ref = dx_ref * (self.t_current - self.t_start)
		pos_y_ref = dy_ref * (self.t_current - self.t_start)
		pos_z_ref = dz_ref * (self.t_current - self.t_start)-5
		pos_ref = np.array([pos_x_ref,pos_y_ref,pos_z_ref])
		pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
		pos = np.array([pos.x_val, pos.y_val, pos.z_val])
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
		if state_[0] == -1:  # no obstacle nearby, fly straight to target
			dx = dpos[0] / temp * self.scaling_factor
			dy = dpos[1] / temp * self.scaling_factor
			dz = dpos[2] / temp * self.scaling_factor
			self.moveByDist([dx, dy, dz], True)
			if self.isbizhang:
				self.isbizhang = False
				info = "success bizhang"
				if self.t_bizhang != 0 and self.path_bizhang != 0:
					reward = (8 * 1.14 / self.t_bizhang + 8 * 1.14 / self.path_bizhang) * 40
				# print(reward)
				self.t_bizhang = 0
				self.path_bizhang = 0
			# print("No obstacle")
			# print(dpos[0], dpos[1], dpos[2])
			state_ = self.getState()  # get next state
		else:  # obstacle nearby
			if abs(state_[1]) >= 90:  # motion direction away from obstacle, fly straight to target
				dx = dpos[0] / temp * self.scaling_factor
				dy = dpos[1] / temp * self.scaling_factor
				dz = dpos[2] / temp * self.scaling_factor
				if dpos[0] != 0 and dpos[1] != 0 and dpos[2] != 0:
					self.moveByDist([dx, dy, dz], True)
				# print("Obstacle present but will not collide")
				if self.isbizhang:
					self.isbizhang = False
					info = "success bizhang"
					reward = (8 * 1.14 / self.t_bizhang + 8 * 1.14 / self.path_bizhang) * 20
					# print(reward)
					self.t_bizhang = 0
					self.path_bizhang = 0
				state_ = self.getState()  # get next state
			else:  # motion direction towards obstacle, enter avoidance mode
				self.isbizhang = True
				self.t_bizhang = self.t_bizhang + t_jiange
				vel = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.linear_velocity
				vel_linear = np.sqrt(vel.x_val ** 2 + vel.y_val ** 2 + vel.z_val ** 2)
				self.path_bizhang = vel_linear * t_jiange + self.path_bizhang
				dx = action[0] + 2
				dy = action[1]
				dy = dy.astype(np.float64)
				dz = self.scaling_factor * dpos[2] / temp
				self.moveByBodyDist([dx, dy, dz], True)
				# print("Obstacle ahead")
				state_ = self.getState()  # get next state
				if (- pos[2] + self.aim_height) > self.height_limit and info == None:
					info = "too high"
					reward = -30
					done = True
					state_[0] = state_[0] * np.sign(state_[1]) / 4
					state_[1] = state_[1] / 180
					reward = reward / 50
					self.state = state_
					norm_state = copy.deepcopy(state_)
					return norm_state, reward, done, info
				if pos[2] > -0.15 and info == None:
					info = "too low"
					reward = -30
					done = True
					state_[0] = state_[0] * np.sign(state_[1]) / 4
					state_[1] = state_[1] / 180
					reward = reward / 50
					self.state = state_
					norm_state = copy.deepcopy(state_)
					return norm_state, reward, done, info
				if state_[0] <= 0.65 and state_[0] > 0:
					reward = -100
					info = "collision"
					done = True
					print(self.name, "+", info)
					state_[0] = state_[0] * np.sign(state_[1]) / 4
					state_[1] = state_[1] / 180
					reward = reward / 50
					self.state = state_
					norm_state = copy.deepcopy(state_)
					return norm_state, reward, done, info
				if abs(state_[1]) < 90 and info == None:
					if state_[0] <= 2.4 and state_[0] > 0.65:
						info = "too close"
						reward = -50
					if state_[0] <= 5 and state_[0] > 4:
						info = "too far"
						reward = -50
					if state_[0] > 2.4 and state_[0] <= 4:
						info = "safe distance"
						xishu = 2.2
						if (state_[1] > 0 and action[1] > 0) or (state_[1] < 0 and action[1] < 0):
							xishu = -2.2
							info = "safe distance but wrong direction"
						reward = 30 * xishu
		#print(self.name,"Start point:  ",self.start)
		#print(self.name,"Target point:  ",self.aim)
		#print(self.name," X distance: ",pos[0] - self.aim[0]," Y distance: ",pos[1] - self.aim[1])
		if (self.isDone() and info == None) :
			print("success", "+", self.name)
			#reward = 100
			info = "success"
			done = True
			state_[0] = state_[0] * np.sign(state_[1]) / 4
			state_[1] = state_[1] / 180
			reward = reward / 50
			self.state = state_
			norm_state = copy.deepcopy(state_)
			return norm_state, reward, done, info
		
		if(self.isChaoshi() and info == None):
			print("Timeout", "+", self.name)
			info = "Timeout"
			reward = -50
			done = True
			state_[0] = state_[0] * np.sign(state_[1]) / 4
			state_[1] = state_[1] / 180
			reward = reward / 50
			self.state = state_
			norm_state = copy.deepcopy(state_)
			return norm_state, reward, done, info
			
		state_[0] = state_[0] * np.sign(state_[1]) / 4
		state_[1] = state_[1] / 180
		reward = reward / 50
		self.state = state_
		norm_state = copy.deepcopy(state_)

		return norm_state, reward, done, info

	def isDone(self):
		pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
		pos = np.array([pos.x_val, pos.y_val, pos.z_val])
		if self.distance(self.aim, pos) < self.threshold:
			return True
		return False
	
	def isChaoshi(self):
		pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
		pos = np.array([pos.x_val, pos.y_val, pos.z_val])
		if (pos[0] - self.aim[0] >=5) :
			info = "Timeout"
			print("Exceeded target point")
			return True
		return False

	def distance(self, pos1, pos2):
		# pos1 = v2t(pos1)
		# pos2 = v2t(pos2)
		dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
		# dist = np.linalg.norm(pos1 - pos2)

		return dist

	def rewardf(self, state, state_):
		pos = state[1][0]
		pos_ = state_[1][0]
		reward = - abs(pos_) + 5

		return reward

	def getlarder_data(self):
		lidar_name = "LidarSensor1"
		if self.name == "Drone2":
			lidar_name = "LidarSensor2"
		if self.name == "Drone3":
			lidar_name = "LidarSensor3"
		lidarData = self.client.getLidarData(lidar_name = lidar_name, vehicle_name=self.name)
		if (len(lidarData.point_cloud) < 3):
			return -1, -1
		else:
			points = self.parse_lidarData(lidarData)
			min_distance = 10
			angle_min_distance = 180
			for j in range(0, len(points) - 1):
				distance = np.sqrt(points[j][0] ** 2 + points[j][1] ** 2 + points[j][2] ** 2)
				if distance < min_distance:
					min_distance = distance
					min_y = points[j][1]
					angle_min_distance = math.asin(min_y / min_distance) / math.pi * 180
					if points[j][0] < 0:
						angle_min_distance = np.sign(angle_min_distance) * 180 - angle_min_distance
			return abs(min_distance), angle_min_distance
			# print("min distance is %d" % (min_distance))
			# print("intersection angle is %f°" % (angle_min_distance))

	def parse_lidarData(self, data):

		# reshape array of floats to array of [X,Y,Z]
		points = np.array(data.point_cloud, dtype=np.dtype('f4'))
		points = np.reshape(points, (int(points.shape[0] / 3), 3))

		return points