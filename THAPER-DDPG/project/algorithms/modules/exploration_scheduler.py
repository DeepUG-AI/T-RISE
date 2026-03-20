def get_noise_scale(consecutive_success):
    if consecutive_success >= 30:
        return 0.0
    elif consecutive_success >= 10:
        return 1.0 - (consecutive_success - 10) / 20.0
    else:
        return 1.0
