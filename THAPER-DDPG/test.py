import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import airsim
import time
import os
from DDPG_T import DDPG_agent
from drone_env import drone_env_collisionabvoidance
from priority_memory import Memory

PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "test")
tf.set_random_seed(22)

def main():
	with tf.device("/gpu:0"):

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			drone_env_collisionabvoidance.client = airsim.MultirotorClient()
			drone_env_collisionabvoidance.client.confirmConnection()
			env1 = drone_env_collisionabvoidance(name="Drone1")
			env2 = drone_env_collisionabvoidance(name="Drone2")
			env3 = drone_env_collisionabvoidance(name="Drone3")
			drone_env_collisionabvoidance.client.reset()
			state1 = env1.reset()
			state2 = env2.reset()
			state3 = env3.reset()
			state_shape = 2
			action_bound = 1.7
			action_dim = 2
			memory_size = 10000
			DDPG_agent.replay_memory = Memory(memory_size)
			agent1 = DDPG_agent(sess, state_shape, action_bound, action_dim, name="Drone1")
			agent2 = DDPG_agent(sess, state_shape, action_bound, action_dim, name="Drone2")
			agent3 = DDPG_agent(sess, state_shape, action_bound, action_dim, name="Drone3")
			saver = tf.train.Saver()

			DIR1 = os.path.join(DIR, "Drone1")
			DIR2 = os.path.join(DIR, "Drone2")
			DIR3 = os.path.join(DIR, "Drone3")

			if not agent1.load(saver, DIR1):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR1):
					os.mkdir(DIR1)
			else:
				print ("model loaded-------------------------")
			if not agent2.load(saver, DIR2):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR2):
					os.mkdir(DIR2)
			else:
				print("model loaded-------------------------")
			if not agent3.load(saver, DIR3):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR3):
					os.mkdir(DIR3)
			else:
				print("model loaded-------------------------")

			e = 0
			info1, info2, info3 = None, None, None
			qianjin1, qianjin2, qianjin3 = False, False, False
			pos1, pos2, pos3 = [0, 0, 0], [0, 0, 0], [0, 0, 0]
			pos1_target, pos2_target, pos3_target = [0, 0, 0], [0, 0, 0], [0, 0, 0]
			total_success = 0
			t = time.clock()
			env1.t_start = t
			env2.t_start = t
			env3.t_start = t
			yaw1, yaw2, yaw3 = 0, 0, 0
			pos1_reserve, pos2_reserve, pos3_reserve = np.array([[0.], [0.], [-5.]]), np.array([[0.], [-12.], [-5.]]), np.array([[0.], [12.], [-5.]])
			while True:
				point1_reverse = [airsim.Vector3r(pos1_reserve[0,0], pos1_reserve[1,0], pos1_reserve[2,0])]
				pos1 = env1.client.simGetGroundTruthKinematics(vehicle_name="Drone1")
				pos1 = np.array([[pos1.position.x_val], [pos1.position.y_val], [pos1.position.z_val]])
				point1 = [airsim.Vector3r(pos1[0,0], pos1[1,0], pos1[2,0])]
				env1.client.simPlotLineList(point1+point1_reverse, color_rgba=[0.0, 0.0, 1.0, 1.0], thickness=10, is_persistent=True)
				pos1_reserve = pos1
				point2_reverse = [airsim.Vector3r(pos2_reserve[0,0], pos2_reserve[1,0], pos2_reserve[2,0])]
				pos2 = env2.client.simGetGroundTruthKinematics(vehicle_name="Drone2")
				pos2 = np.array([[pos2.position.x_val], [pos2.position.y_val-12], [pos2.position.z_val]])
				point2 = [airsim.Vector3r(pos2[0,0], pos2[1,0], pos2[2,0])]
				env2.client.simPlotLineList(point2+point2_reverse, color_rgba=[0.0, 1.0, 0.0, 1.0], thickness=10, is_persistent=True)
				pos2_reserve = pos2
				point3_reverse = [airsim.Vector3r(pos3_reserve[0,0], pos3_reserve[1,0], pos3_reserve[2,0])]
				pos3 = env3.client.simGetGroundTruthKinematics(vehicle_name="Drone3")
				pos3 = np.array([[pos3.position.x_val], [pos3.position.y_val+12], [pos3.position.z_val]])
				point3 = [airsim.Vector3r(pos3[0,0], pos3[1,0], pos3[2,0])]
				env3.client.simPlotLineList(point3+point3_reverse, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness=10, is_persistent=True)
				pos3_reserve = pos3
				if info1 != "success":
					action1 = agent1.act(state1, info1)
					if qianjin1 == True:
						temp = airsim.YawMode()
						temp.is_rate = not True
						temp.yaw_or_rate = yaw1 / 3.14 * 180
						env1.client.moveToPositionAsync(pos1_target[0], pos1_target[1], pos1_target[2], 0.8,drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=temp, vehicle_name="Drone1")
						pos = env1.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
						pos = np.array([pos.x_val, pos.y_val, pos.z_val])
						if env1.distance(pos, pos1_target) <= 1:
							qianjin1 = False
					else:
						state = env1.client.getMultirotorState(vehicle_name="Drone1")
						(pitch1, roll1, yaw1) = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
						next_state1, reward1, terminal1, info1 = env1.step(action1)
					if info1 == "success bizhang":
						qianjin1 = True
						info1 = None
						pos = env1.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
						pos1_target = np.array([pos.x_val, pos.y_val, pos.z_val])
						pos1_target[0] = pos1_target[0] + 10
				if info2 != "success":
					action2 = agent2.act(state2, info2)
					if qianjin2 == True:
						temp = airsim.YawMode()
						temp.is_rate = not True
						temp.yaw_or_rate = yaw2 / 3.14 * 180
						env2.client.moveToPositionAsync(pos2_target[0], pos2_target[1], pos2_target[2], 0.8,drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=temp, vehicle_name="Drone2")
						pos = env2.client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
						pos = np.array([pos.x_val, pos.y_val, pos.z_val])
						if env2.distance(pos, pos2_target) <= 1:
							qianjin2 = False
					else:
						state = env1.client.getMultirotorState(vehicle_name="Drone2")
						(pitch2, roll2, yaw2) = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
						next_state2, reward2, terminal2, info2 = env2.step(action2)
					if info2 == "success bizhang":
						qianjin2 = True
						info2 = None
						pos = env2.client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
						pos2_target = np.array([pos.x_val, pos.y_val, pos.z_val])
						pos2_target[0] = pos2_target[0] + 10
				if info3 != "success":
					action3 = agent3.act(state3, info3)
					if qianjin3 == True:
						temp = airsim.YawMode()
						temp.is_rate = not True
						temp.yaw_or_rate = yaw2 / 3.14 * 180
						env3.client.moveToPositionAsync(pos3_target[0], pos3_target[1], pos3_target[2], 0.8,drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=temp, vehicle_name="Drone3")
						pos = env3.client.getMultirotorState(vehicle_name="Drone3").kinematics_estimated.position
						pos = np.array([pos.x_val, pos.y_val, pos.z_val])
						if env3.distance(pos, pos3_target) <= 1:
							qianjin3 = False
					else:
						state = env1.client.getMultirotorState(vehicle_name="Drone3")
						(pitch2, roll2, yaw2) = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
						next_state3, reward3, terminal3, info3 = env3.step(action3)
					if info3 == "success bizhang":
						qianjin3 = True
						info3 = None
						pos = env3.client.getMultirotorState(vehicle_name="Drone3").kinematics_estimated.position
						pos3_target = np.array([pos.x_val, pos.y_val, pos.z_val])
						pos3_target[0] = pos3_target[0] + 10
				state1 = next_state1
				state2 = next_state2
				state3 = next_state3

				if terminal1 and terminal2 and terminal3:
					if info1 == "success" and info2 == "success" and info3 == "success":
						total_success += 1
						time.sleep(20)
						print("episode ", e, "finish, success", "total success:", total_success)
					else:
						print("episode ", e, "finish, failed", "total success:", total_success)
					e += 1
					drone_env_collisionabvoidance.client.reset()
					state1 = env1.reset()
					state2 = env2.reset()
					state3 = env3.reset()
					info1, info2, info3 = None, None, None


if __name__ == "__main__":
	main()