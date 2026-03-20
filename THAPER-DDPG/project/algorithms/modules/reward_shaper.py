from algorithms.modules.state_adapter import get_longitudinal_progress

def coordination_reward(pos_self_shared,
                        pos1_shared, pos2_shared, pos3_shared,
                        path_direction=None,
                        lambda_form=0.08,
                        delta_s=1.5):

    prog_self = get_longitudinal_progress(pos_self_shared, path_direction)
    prog1 = get_longitudinal_progress(pos1_shared, path_direction)
    prog2 = get_longitudinal_progress(pos2_shared, path_direction)
    prog3 = get_longitudinal_progress(pos3_shared, path_direction)

    prog_avg = (prog1 + prog2 + prog3) / 3.0
    err = abs(prog_self - prog_avg)

    if err <= delta_s:
        return 0.0

    return -lambda_form * (err - delta_s)
