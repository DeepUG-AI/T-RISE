import os
import shutil
import airsim
import xlwt
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from algorithms.tunnel_ddpg import TunnelAwareDDPGAgent
from algorithms.modules.state_adapter import (
    get_shared_position,
    build_coord_state,
    FORMATION_OFFSETS_LOCAL,
    REFERENCE_PATH_DIRECTION_LOCAL,
    USE_SHARED_REFERENCE_COORD,
)
from algorithms.modules.human_prior_demonstration import (
    compute_warm_start_guidance_rate,
    apply_human_prior_action_guidance
)
from algorithms.modules.reward_shaper import coordination_reward
from envs.tunnel_drone_env import TunnelDroneEnv

workbook = xlwt.Workbook(encoding="utf-8")
worksheet = workbook.add_sheet("PM_DDPG_COORD")

PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "data_coord")

tf.set_random_seed(22)
np.set_printoptions(precision=3, suppress=True)
PREMODEL = False

def main():
    with tf.device("/gpu:0"):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            TunnelDroneEnv.client = airsim.MultirotorClient()
            TunnelDroneEnv.client.confirmConnection()

            env1 = TunnelDroneEnv(name="Drone1")
            env2 = TunnelDroneEnv(name="Drone2")
            env3 = TunnelDroneEnv(name="Drone3")

            TunnelDroneEnv.client.reset()

            base_state1 = env1.reset()
            base_state2 = env2.reset()
            base_state3 = env3.reset()

            pos1_shared = get_shared_position(
                env1,
                formation_offsets=FORMATION_OFFSETS_LOCAL,
                use_shared_reference=USE_SHARED_REFERENCE_COORD
            )
            pos2_shared = get_shared_position(
                env2,
                formation_offsets=FORMATION_OFFSETS_LOCAL,
                use_shared_reference=USE_SHARED_REFERENCE_COORD
            )
            pos3_shared = get_shared_position(
                env3,
                formation_offsets=FORMATION_OFFSETS_LOCAL,
                use_shared_reference=USE_SHARED_REFERENCE_COORD
            )

            state1 = build_coord_state(
                base_state1, pos1_shared, pos2_shared, pos3_shared,
                path_direction=REFERENCE_PATH_DIRECTION_LOCAL
            )
            state2 = build_coord_state(
                base_state2, pos2_shared, pos1_shared, pos3_shared,
                path_direction=REFERENCE_PATH_DIRECTION_LOCAL
            )
            state3 = build_coord_state(
                base_state3, pos3_shared, pos1_shared, pos2_shared,
                path_direction=REFERENCE_PATH_DIRECTION_LOCAL
            )

            state_shape = 6
            action_bound = 1.7
            action_dim = 2
            memory_size = 10000
            transition_len = 2 * state_shape + action_dim + 2

            TunnelAwareDDPGAgent.initialize_replay_memory(memory_size, transition_len)

            agent1 = TunnelAwareDDPGAgent(sess, state_shape, action_bound, action_dim, name="Drone1")
            agent2 = TunnelAwareDDPGAgent(sess, state_shape, action_bound, action_dim, name="Drone2")
            agent3 = TunnelAwareDDPGAgent(sess, state_shape, action_bound, action_dim, name="Drone3")

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            os.makedirs(os.path.join(DIR, "Drone1"), exist_ok=True)
            os.makedirs(os.path.join(DIR, "Drone2"), exist_ok=True)
            os.makedirs(os.path.join(DIR, "Drone3"), exist_ok=True)

            e = 0
            global_step = 0

            episode_reward1 = episode_reward2 = episode_reward3 = 0
            step_count1 = step_count2 = step_count3 = 0
            success1 = success2 = success3 = 0
            hang = 0
            info1 = info2 = info3 = None
            consecutive_success = 0
            best_saved_after_50 = False
            last_periodic_save_root = None

            while True:
                global_step += 1

                min_action_taken = min(agent1.num_action_taken, agent2.num_action_taken, agent3.num_action_taken)
                force_coord_rate = compute_warm_start_guidance_rate(min_action_taken, agent1.train_after)

                if info1 != "success":
                    action1 = agent1.act(state1, info1, noise=True, consecutive_success=consecutive_success)
                else:
                    action1 = np.zeros((action_dim,), dtype=np.float32)

                if info2 != "success":
                    action2 = agent2.act(state2, info2, noise=True, consecutive_success=consecutive_success)
                else:
                    action2 = np.zeros((action_dim,), dtype=np.float32)

                if info3 != "success":
                    action3 = agent3.act(state3, info3, noise=True, consecutive_success=consecutive_success)
                else:
                    action3 = np.zeros((action_dim,), dtype=np.float32)

                pos1_shared = get_shared_position(
                    env1,
                    formation_offsets=FORMATION_OFFSETS_LOCAL,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD
                )
                pos2_shared = get_shared_position(
                    env2,
                    formation_offsets=FORMATION_OFFSETS_LOCAL,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD
                )
                pos3_shared = get_shared_position(
                    env3,
                    formation_offsets=FORMATION_OFFSETS_LOCAL,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD
                )

                action1, action2, action3, _ = apply_human_prior_action_guidance(
                    action1, action2, action3,
                    state1, state2, state3,
                    pos1_shared, pos2_shared, pos3_shared,
                    path_direction=REFERENCE_PATH_DIRECTION_LOCAL,
                    k_form=0.5,
                    delta_s=1.5,
                    action_bound=action_bound,
                    safe_dist_m=4.5,
                    guidance_rate=force_coord_rate
                )

                if info1 != "success":
                    next_base_state1, reward1, terminal1, info1 = env1.step(action1)
                else:
                    next_base_state1, reward1, terminal1 = base_state1, 0.0, True

                if info2 != "success":
                    next_base_state2, reward2, terminal2, info2 = env2.step(action2)
                else:
                    next_base_state2, reward2, terminal2 = base_state2, 0.0, True

                if info3 != "success":
                    next_base_state3, reward3, terminal3, info3 = env3.step(action3)
                else:
                    next_base_state3, reward3, terminal3 = base_state3, 0.0, True

                npos1_shared = get_shared_position(
                    env1,
                    formation_offsets=FORMATION_OFFSETS_LOCAL,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD
                )
                npos2_shared = get_shared_position(
                    env2,
                    formation_offsets=FORMATION_OFFSETS_LOCAL,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD
                )
                npos3_shared = get_shared_position(
                    env3,
                    formation_offsets=FORMATION_OFFSETS_LOCAL,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD
                )

                next_state1 = build_coord_state(
                    next_base_state1, npos1_shared, npos2_shared, npos3_shared,
                    path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                )
                next_state2 = build_coord_state(
                    next_base_state2, npos2_shared, npos1_shared, npos3_shared,
                    path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                )
                next_state3 = build_coord_state(
                    next_base_state3, npos3_shared, npos1_shared, npos2_shared,
                    path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                )

                if info1 is not None and info1 != "success":
                    reward1 += coordination_reward(
                        npos1_shared, npos1_shared, npos2_shared, npos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL,
                        lambda_form=0.08,
                        delta_s=1.5
                    )

                if info2 is not None and info2 != "success":
                    reward2 += coordination_reward(
                        npos2_shared, npos1_shared, npos2_shared, npos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL,
                        lambda_form=0.08,
                        delta_s=1.5
                    )

                if info3 is not None and info3 != "success":
                    reward3 += coordination_reward(
                        npos3_shared, npos1_shared, npos2_shared, npos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL,
                        lambda_form=0.08,
                        delta_s=1.5
                    )

                if info1 is not None and info1 != "success":
                    episode_reward1 += reward1
                    transition = np.hstack((state1, action1, reward1, next_state1, terminal1))
                    if (not np.any(np.isnan(state1))) and (not np.any(np.isnan(action1))) and (not np.any(np.isnan(next_state1))):
                        TunnelAwareDDPGAgent.replay_memory.store(transition)
                    agent1.train()
                    step_count1 += 1

                state1 = next_state1
                base_state1 = next_base_state1

                if info2 is not None and info2 != "success":
                    episode_reward2 += reward2
                    transition = np.hstack((state2, action2, reward2, next_state2, terminal2))
                    if (not np.any(np.isnan(state2))) and (not np.any(np.isnan(action2))) and (not np.any(np.isnan(next_state2))):
                        TunnelAwareDDPGAgent.replay_memory.store(transition)
                    agent2.train()
                    step_count2 += 1

                state2 = next_state2
                base_state2 = next_base_state2

                if info3 is not None and info3 != "success":
                    episode_reward3 += reward3
                    transition = np.hstack((state3, action3, reward3, next_state3, terminal3))
                    if (not np.any(np.isnan(state3))) and (not np.any(np.isnan(action3))) and (not np.any(np.isnan(next_state3))):
                        TunnelAwareDDPGAgent.replay_memory.store(transition)
                    agent3.train()
                    step_count3 += 1

                state3 = next_state3
                base_state3 = next_base_state3

                if (
                    info1 == "collision" or info2 == "collision" or info3 == "collision"
                    or (info1 == "success" and info2 == "success" and info3 == "success")
                    or info1 == "Timeout" or info2 == "Timeout" or info3 == "Timeout"
                ):
                    worksheet.write(hang, 0, episode_reward1)
                    worksheet.write(hang, 1, 1 if info1 == "success" else 0)
                    worksheet.write(hang, 2, episode_reward2)
                    worksheet.write(hang, 3, 1 if info2 == "success" else 0)
                    worksheet.write(hang, 4, episode_reward3)
                    worksheet.write(hang, 5, 1 if info3 == "success" else 0)
                    hang += 1
                    workbook.save("data_report_coord.xls")

                    if info1 == "success":
                        success1 += 1
                    if info2 == "success":
                        success2 += 1
                    if info3 == "success":
                        success3 += 1

                    all_success = (info1 == "success" and info2 == "success" and info3 == "success")
                    if all_success:
                        consecutive_success += 1
                    else:
                        consecutive_success = 0
                        best_saved_after_50 = False

                    print(" " * 100, end="\r")
                    print(f"episode {e} finish")
                    print("name: {}, reward1: {:.5f}, total success1: {}, result1: {}, step1: {}".format(
                        env1.name, episode_reward1, success1, info1, step_count1).ljust(100, " "))
                    print("name: {}, reward2: {:.5f}, total success2: {}, result2: {}, step2: {}".format(
                        env2.name, episode_reward2, success2, info2, step_count2).ljust(100, " "))
                    print("name: {}, reward3: {:.5f}, total success3: {}, result3: {}, step3: {}".format(
                        env3.name, episode_reward3, success3, info3, step_count3).ljust(100, " "))
                    print(f"consecutive_all_success: {consecutive_success}".ljust(100, " "))

                    episode_reward1 = episode_reward2 = episode_reward3 = 0
                    step_count1 = step_count2 = step_count3 = 0
                    e += 1

                    if e > 0 and e % 10 == 0:
                        nDir = os.path.join(PATH, "data_coord", str(int(e // 10)))
                        nDir1 = os.path.join(nDir, "Drone1")
                        nDir2 = os.path.join(nDir, "Drone2")
                        nDir3 = os.path.join(nDir, "Drone3")
                        os.makedirs(nDir1, exist_ok=True)
                        os.makedirs(nDir2, exist_ok=True)
                        os.makedirs(nDir3, exist_ok=True)
                        agent1.save(saver, nDir1)
                        agent2.save(saver, nDir2)
                        agent3.save(saver, nDir3)
                        last_periodic_save_root = nDir
                        print(f">>> Periodic coordinated model saved at: {last_periodic_save_root}")

                    if consecutive_success >= 50 and not best_saved_after_50:
                        best_root = os.path.join(PATH, "best_models_after_50success_coord")
                        if last_periodic_save_root is None:
                            print(">>> WARNING: 50 consecutive successes reached, but no periodic saved model folder is available yet.")
                        else:
                            latest_tag = os.path.basename(last_periodic_save_root)
                            best_target = os.path.join(best_root, latest_tag)
                            if os.path.exists(best_target):
                                shutil.rmtree(best_target)
                            os.makedirs(best_root, exist_ok=True)
                            shutil.copytree(last_periodic_save_root, best_target)
                            print(">>> Best coordinated model copied from latest periodic save folder after 50 consecutive all-success episodes.")
                            print(f">>> Source: {last_periodic_save_root}")
                            print(f">>> Target: {best_target}")
                            best_saved_after_50 = True

                    info1 = info2 = info3 = None
                    TunnelDroneEnv.client.reset()

                    base_state1 = env1.reset()
                    base_state2 = env2.reset()
                    base_state3 = env3.reset()

                    pos1_shared = get_shared_position(
                        env1,
                        formation_offsets=FORMATION_OFFSETS_LOCAL,
                        use_shared_reference=USE_SHARED_REFERENCE_COORD
                    )
                    pos2_shared = get_shared_position(
                        env2,
                        formation_offsets=FORMATION_OFFSETS_LOCAL,
                        use_shared_reference=USE_SHARED_REFERENCE_COORD
                    )
                    pos3_shared = get_shared_position(
                        env3,
                        formation_offsets=FORMATION_OFFSETS_LOCAL,
                        use_shared_reference=USE_SHARED_REFERENCE_COORD
                    )

                    state1 = build_coord_state(
                        base_state1, pos1_shared, pos2_shared, pos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                    )
                    state2 = build_coord_state(
                        base_state2, pos2_shared, pos1_shared, pos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                    )
                    state3 = build_coord_state(
                        base_state3, pos3_shared, pos1_shared, pos2_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                    )

if __name__ == "__main__":
    main()
