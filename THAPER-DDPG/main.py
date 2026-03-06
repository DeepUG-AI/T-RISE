import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import time
import numpy as np
import airsim
import os
from DDPG_T import DDPG_agent
from drone_env import drone_env_collisionabvoidance
import xlwt
#from ReplayMemory import ReplayMemory
from priority_memory import Memory

workbook=xlwt.Workbook(encoding="utf-8")    # create xls object
worksheet=workbook.add_sheet("PM_DDPG")  # create a sheet
PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "data")
tf.set_random_seed(22)
#tf.random.set_seed(22)
PREMODEL = True
np.set_printoptions(precision=3, suppress=True)

def main():

	with tf.device("/gpu:0"):

		config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
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
				print ("continue1------------------")

			if not agent2.load(saver, DIR2):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR2):
					os.mkdir(DIR2)
			else:
				print ("continue2------------------")
			if not agent3.load(saver, DIR3):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR3):
					os.mkdir(DIR3)
			else:
				print ("continue3------------------")

			e = 0
			episode_reward1, episode_reward2, episode_reward3 = 0, 0, 0
			step_count1, step_count2, step_count3 = 0, 0, 0
			success1, success2, success3 = 0, 0, 0
			hang = 0
			info1, info2, info3 = None, None, None
			while True:
				if info1 != "success":
					action1 = agent1.act(state1, info1, noise=True)
					next_state1, reward1, terminal1, info1 = env1.step(action1)
				if info2 != "success":
					action2 = agent2.act(state2, info2, noise=True)
					next_state2, reward2, terminal2, info2 = env2.step(action2)
				if info3 != "success":
					action3 = agent3.act(state3, info3, noise=True)
					next_state3, reward3, terminal3, info3 = env3.step(action3)

				if info1 != None and info1 != "success":
					episode_reward1 += reward1
					# DDPG_agent.replay_memory.append(state1, action1, reward1, next_state1, terminal1)
					transition = np.hstack((state1, action1, reward1, next_state1, terminal1))
					if (not np.isnan(state1[0])) and (not np.isnan(state1[1])) and (not np.isnan(action1[0])) and (not np.isnan(action1[1])) and (not np.isnan(next_state1[0])) and (not np.isnan(next_state1[1])):
						DDPG_agent.replay_memory.store(transition)
					agent1.train()
					step_count1 += 1
				state1 = next_state1
				if info2 != None and info2 != "success":
					episode_reward2 += reward2
					# DDPG_agent.replay_memory.append(state2, action2, reward2, next_state2, terminal2)
					transition = np.hstack((state2, action2, reward2, next_state2, terminal2))
					if (not np.isnan(state2[0])) and (not np.isnan(state2[1])) and (not np.isnan(action2[0])) and (not np.isnan(action2[1])) and (not np.isnan(next_state2[0])) and (not np.isnan(next_state2[1])):
						DDPG_agent.replay_memory.store(transition)
					agent2.train()
					step_count2 += 1
				state2 = next_state2
				if info3 != None and info3 != "success":
					episode_reward3 += reward3
					# DDPG_agent.replay_memory.append(state2, action2, reward2, next_state2, terminal2)
					transition = np.hstack((state3, action3, reward3, next_state3, terminal3))
					if (not np.isnan(state3[0])) and (not np.isnan(state3[1])) and (not np.isnan(action3[0])) and (not np.isnan(action3[1])) and (not np.isnan(next_state3[0])) and (not np.isnan(next_state3[1])):
						DDPG_agent.replay_memory.store(transition)
					agent3.train()
					step_count3 += 1
				state3 = next_state3

				if info1 == "collision" or info2 == "collision" or info3 == "collision" or (info1 == "success" and info2 == "success" and info3 == "success")  or info1 == "Timeout" or info2 == "Timeout" or info3 == "Timeout" :
					worksheet.write(hang, 0, episode_reward1)
					if info1 == "success":
						success1 += 1
						worksheet.write(hang, 1, 1)
					else:
						worksheet.write(hang, 1, 0)
					worksheet.write(hang, 2, episode_reward2)
					if info2 == "success":
						success2 += 1
						worksheet.write(hang, 3, 1)
					else:
						worksheet.write(hang, 3, 0)
					worksheet.write(hang, 4, episode_reward3)
					if info3 == "success":
						success3 += 1
						worksheet.write(hang, 5, 1)
					else:
						worksheet.write(hang, 5, 0)
					hang += 1

					workbook.save("data_report.xls")
					print(" " * 80, end="\r")
					print("episode {} finish".format(e))
					print("name: {}, reward1: {:.5f}, total success1: {}, result1: {}, step1: {}".format(env1.name, episode_reward1, success1, info1, step_count1).ljust(80, " "))
					print("name: {}, reward2: {:.5f}, total success2: {}, result2: {}, step2: {}".format(env2.name, episode_reward2, success2, info2, step_count2).ljust(80, " "))
					print("name: {}, reward3: {:.5f}, total success3: {}, result3: {}, step3: {}".format(env3.name, episode_reward3, success3, info3, step_count3).ljust(80, " "))
					episode_reward1 = 0
					episode_reward2 = 0
					episode_reward3 = 0
					step_count1 = 0
					step_count2 = 0
					step_count3 = 0
					e += 1
					if e % 10 == 0:
						nDir = os.path.join(PATH, "data/" + str(int(e // 10)))
						if not os.path.exists(nDir):
							os.mkdir(nDir)
						nDir1 = os.path.join(nDir, "Drone1")
						nDir2 = os.path.join(nDir, "Drone2")
						nDir3 = os.path.join(nDir, "Drone3")
						if not os.path.exists(nDir1):
							os.mkdir(nDir1)
						agent1.save(saver, nDir1)
						if not os.path.exists(nDir2):
							os.mkdir(nDir2)
						agent2.save(saver, nDir2)
						if not os.path.exists(nDir3):
							os.mkdir(nDir3)
						agent3.save(saver, nDir3)
					info1, info2, info3 = None, None, None
					drone_env_collisionabvoidance.client.reset()
					state1 = env1.reset()
					state2 = env2.reset()
					state3 = env3.reset()

if __name__ == "__main__":
	main()