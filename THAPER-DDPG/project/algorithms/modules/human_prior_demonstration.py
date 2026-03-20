import numpy as np
from algorithms.modules.state_adapter import get_longitudinal_progress

def compute_warm_start_guidance_rate(num_action_taken, train_after):
    return 1.0 if num_action_taken < train_after else 0.0

def apply_human_prior_action_guidance(action1, action2, action3,
                                      state1, state2, state3,
                                      pos1_shared, pos2_shared, pos3_shared,
                                      path_direction=None,
                                      k_form=0.5,
                                      delta_s=1.5,
                                      action_bound=1.7,
                                      safe_dist_m=4.5,
                                      guidance_rate=1.0):

    prog1 = get_longitudinal_progress(pos1_shared, path_direction)
    prog2 = get_longitudinal_progress(pos2_shared, path_direction)
    prog3 = get_longitudinal_progress(pos3_shared, path_direction)

    prog_avg = (prog1 + prog2 + prog3) / 3.0

    progs = [prog1, prog2, prog3]
    states = [state1, state2, state3]
    actions = [action1.copy(), action2.copy(), action3.copy()]

    for i in range(3):
        err = progs[i] - prog_avg

        d_obs_signed_norm = states[i][0]

        if d_obs_signed_norm == -1:
            allow_guidance = True
        else:
            d_obs_m = abs(d_obs_signed_norm) * 4.0
            allow_guidance = (d_obs_m > safe_dist_m)

        if abs(err) > delta_s and allow_guidance and guidance_rate > 0:
            delta_a = -k_form * err * guidance_rate
            actions[i][0] = np.clip(actions[i][0] + delta_a, -action_bound, action_bound)

    return actions[0], actions[1], actions[2], prog_avg
