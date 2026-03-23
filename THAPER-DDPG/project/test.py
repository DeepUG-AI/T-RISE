import os
import time
import airsim
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
from algorithms.modules.human_prior_demonstration import apply_human_prior_action_guidance
from envs.tunnel_drone_env import TunnelDroneEnv

tf.set_random_seed(22)
np.set_printoptions(precision=3, suppress=True)

PATH = os.path.dirname(os.path.abspath(__file__))
BEST_ROOT = os.path.join(PATH, "best_models_after_50success_coord")

if not os.path.exists(BEST_ROOT):
    raise FileNotFoundError(f"Best model root directory not found: {BEST_ROOT}")

subdirs = [d for d in os.listdir(BEST_ROOT) if os.path.isdir(os.path.join(BEST_ROOT, d)) and d.isdigit()]
if len(subdirs) == 0:
    raise FileNotFoundError(f"No numbered subfolder was found under {BEST_ROOT}, such as 7/ or 8/")

latest_tag = str(max(int(d) for d in subdirs))
DIR = os.path.join(BEST_ROOT, latest_tag)
DIR1 = os.path.join(DIR, "Drone1")
DIR2 = os.path.join(DIR, "Drone2")
DIR3 = os.path.join(DIR, "Drone3")

TEST_EPISODES = 100
RENDER_SLEEP = 0.0
FINAL_RESULTS = {"success", "collision", "Timeout"}


def result_to_print_str(info):
    return "None" if info is None else str(info)


def update_single_drone_stats(info, success_total, collision_total, timeout_total):
    if info == "success":
        success_total += 1
    elif info == "collision":
        collision_total += 1
    elif info == "Timeout":
        timeout_total += 1
    return success_total, collision_total, timeout_total


def is_finished(info):
    return info in FINAL_RESULTS


def zero_action(action_dim):
    return np.zeros((action_dim,), dtype=np.float32)


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

            state_shape = 6
            action_bound = 1.7
            action_dim = 2

            agent1 = TunnelAwareDDPGAgent(sess, state_shape, action_bound, action_dim, name="Drone1")
            agent2 = TunnelAwareDDPGAgent(sess, state_shape, action_bound, action_dim, name="Drone2")
            agent3 = TunnelAwareDDPGAgent(sess, state_shape, action_bound, action_dim, name="Drone3")

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if not agent1.load(saver, DIR1):
                raise FileNotFoundError(f"Failed to load Drone1 model: {DIR1}")
            if not agent2.load(saver, DIR2):
                raise FileNotFoundError(f"Failed to load Drone2 model: {DIR2}")
            if not agent3.load(saver, DIR3):
                raise FileNotFoundError(f"Failed to load Drone3 model: {DIR3}")

            success1_total = collision1_total = timeout1_total = 0
            success2_total = collision2_total = timeout2_total = 0
            success3_total = collision3_total = timeout3_total = 0
            all_success_total = 0
            formation_fail_total = 0

            for ep in range(TEST_EPISODES):
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

                info1 = info2 = info3 = None

                while True:
                    finished1 = is_finished(info1)
                    finished2 = is_finished(info2)
                    finished3 = is_finished(info3)

                    if not finished1:
                        action1 = agent1.act(state1, info1, noise=False)
                    else:
                        action1 = zero_action(action_dim)

                    if not finished2:
                        action2 = agent2.act(state2, info2, noise=False)
                    else:
                        action2 = zero_action(action_dim)

                    if not finished3:
                        action3 = agent3.act(state3, info3, noise=False)
                    else:
                        action3 = zero_action(action_dim)

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
                        guidance_rate=0.0
                    )

                    if finished1:
                        action1 = zero_action(action_dim)
                    if finished2:
                        action2 = zero_action(action_dim)
                    if finished3:
                        action3 = zero_action(action_dim)

                    if not finished1:
                        next_base_state1, _, _, info1 = env1.step(action1)
                    else:
                        next_base_state1 = base_state1

                    if not finished2:
                        next_base_state2, _, _, info2 = env2.step(action2)
                    else:
                        next_base_state2 = base_state2

                    if not finished3:
                        next_base_state3, _, _, info3 = env3.step(action3)
                    else:
                        next_base_state3 = base_state3

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

                    state1 = build_coord_state(
                        next_base_state1, npos1_shared, npos2_shared, npos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                    )
                    base_state1 = next_base_state1

                    state2 = build_coord_state(
                        next_base_state2, npos2_shared, npos1_shared, npos3_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                    )
                    base_state2 = next_base_state2

                    state3 = build_coord_state(
                        next_base_state3, npos3_shared, npos1_shared, npos2_shared,
                        path_direction=REFERENCE_PATH_DIRECTION_LOCAL
                    )
                    base_state3 = next_base_state3

                    if RENDER_SLEEP > 0:
                        time.sleep(RENDER_SLEEP)

                    if is_finished(info1) and is_finished(info2) and is_finished(info3):
                        success1_total, collision1_total, timeout1_total = update_single_drone_stats(
                            info1, success1_total, collision1_total, timeout1_total
                        )
                        success2_total, collision2_total, timeout2_total = update_single_drone_stats(
                            info2, success2_total, collision2_total, timeout2_total
                        )
                        success3_total, collision3_total, timeout3_total = update_single_drone_stats(
                            info3, success3_total, collision3_total, timeout3_total
                        )

                        fail1_total = collision1_total + timeout1_total
                        fail2_total = collision2_total + timeout2_total
                        fail3_total = collision3_total + timeout3_total

                        if info1 == "success" and info2 == "success" and info3 == "success":
                            all_success_total += 1
                        else:
                            formation_fail_total += 1

                        print("=" * 80)
                        print(f"Test Episode {ep + 1}/{TEST_EPISODES}")
                        print(f"Drone1 | result={result_to_print_str(info1):>9}")
                        print(f"Drone2 | result={result_to_print_str(info2):>9}")
                        print(f"Drone3 | result={result_to_print_str(info3):>9}")
                        print("-" * 80)
                        print("Current cumulative statistics:")
                        print(f"Drone1 | total successes={success1_total} | total failures={fail1_total}")
                        print(f"Drone2 | total successes={success2_total} | total failures={fail2_total}")
                        print(f"Drone3 | total successes={success3_total} | total failures={fail3_total}")
                        print(f"Formation | total simultaneous successes={all_success_total} | total formation failures={formation_fail_total}")
                        break

            fail1_total = collision1_total + timeout1_total
            fail2_total = collision2_total + timeout2_total
            fail3_total = collision3_total + timeout3_total

            print("\n" + "#" * 100)
            print("Testing completed. Summary of results:")
            print(f"Number of test episodes TEST_EPISODES = {TEST_EPISODES}")
            print(f"Loaded best model folder ID = {latest_tag}")
            print("-" * 100)
            print(f"Drone1 success count   : {success1_total}")
            print(f"Drone1 collision count : {collision1_total}")
            print(f"Drone1 timeout count   : {timeout1_total}")
            print(f"Drone1 failure count   : {fail1_total}")
            print(f"Drone1 success rate    : {success1_total / TEST_EPISODES:.4f}")
            print("-" * 100)
            print(f"Drone2 success count   : {success2_total}")
            print(f"Drone2 collision count : {collision2_total}")
            print(f"Drone2 timeout count   : {timeout2_total}")
            print(f"Drone2 failure count   : {fail2_total}")
            print(f"Drone2 success rate    : {success2_total / TEST_EPISODES:.4f}")
            print("-" * 100)
            print(f"Drone3 success count   : {success3_total}")
            print(f"Drone3 collision count : {collision3_total}")
            print(f"Drone3 timeout count   : {timeout3_total}")
            print(f"Drone3 failure count   : {fail3_total}")
            print(f"Drone3 success rate    : {success3_total / TEST_EPISODES:.4f}")
            print("-" * 100)
            print(f"Three-drone simultaneous success count : {all_success_total}")
            print(f"Formation failure count                : {formation_fail_total}")
            print(f"Three-drone simultaneous success rate  : {all_success_total / TEST_EPISODES:.4f}")
            print("#" * 100)


if __name__ == "__main__":
    main()
