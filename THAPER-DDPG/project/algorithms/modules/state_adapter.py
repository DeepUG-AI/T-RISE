import numpy as np

USE_SHARED_REFERENCE_COORD = True

FORMATION_OFFSETS_LOCAL = {
    "Drone1": [0.0, 0.0, 0.0],
    "Drone2": [0.0, -12.0, 0.0],
    "Drone3": [0.0, 12.0, 0.0],
}

REFERENCE_PATH_DIRECTION_LOCAL = [1.0, 0.0, 0.0]

COORD_FEATURE_SCALE = 20.0

def _to_np3(v):
    arr = np.asarray(v, dtype=np.float32).reshape(3,)
    return arr

def _normalize(v):
    v = _to_np3(v)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return v / norm

def get_formation_offsets(custom_offsets=None):
    if custom_offsets is None:
        custom_offsets = FORMATION_OFFSETS_LOCAL

    offsets = {}
    for k, v in custom_offsets.items():
        offsets[k] = _to_np3(v)
    return offsets

def get_reference_path_unit_vector(path_direction=None):
    if path_direction is None:
        path_direction = REFERENCE_PATH_DIRECTION_LOCAL
    return _normalize(path_direction)

def get_local_position(env):
    pos = env.client.getMultirotorState(vehicle_name=env.name).kinematics_estimated.position
    return np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)], dtype=np.float32)

def local_to_shared(pos_local,
                    drone_name,
                    formation_offsets=None,
                    use_shared_reference=USE_SHARED_REFERENCE_COORD):

    pos_local = _to_np3(pos_local)

    if not use_shared_reference:
        return pos_local

    offsets = get_formation_offsets(formation_offsets)
    if drone_name not in offsets:
        raise ValueError(f"Unknown drone_name: {drone_name}. Available keys: {list(offsets.keys())}")

    return pos_local + offsets[drone_name]

def get_shared_position(env,
                        formation_offsets=None,
                        use_shared_reference=USE_SHARED_REFERENCE_COORD):
    pos_local = get_local_position(env)
    return local_to_shared(
        pos_local=pos_local,
        drone_name=env.name,
        formation_offsets=formation_offsets,
        use_shared_reference=use_shared_reference
    )

def get_longitudinal_progress(pos_shared, path_direction=None):

    pos_shared = _to_np3(pos_shared)
    dir_unit = get_reference_path_unit_vector(path_direction)
    return float(np.dot(pos_shared, dir_unit))

def build_coord_state(base_state,
                      pos_self_shared,
                      pos_other1_shared,
                      pos_other2_shared,
                      path_direction=None,
                      proj_scale=COORD_FEATURE_SCALE):

    prog_self = get_longitudinal_progress(pos_self_shared, path_direction)
    prog_other1 = get_longitudinal_progress(pos_other1_shared, path_direction)
    prog_other2 = get_longitudinal_progress(pos_other2_shared, path_direction)

    ds1 = np.clip((prog_other1 - prog_self) / proj_scale, -1.0, 1.0)
    ds2 = np.clip((prog_other2 - prog_self) / proj_scale, -1.0, 1.0)

    return np.array(
        [base_state[0], base_state[1], base_state[2], base_state[3], ds1, ds2],
        dtype=np.float32
    )
