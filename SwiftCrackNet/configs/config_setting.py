import os
from datetime import datetime
from utils import *


class setting_config:
    """
    training config
    """

    # =========================
    # 1. Basic experimental information
    # =========================
    project_name = 'crack_seg'

    # Dataset name
    dataset_name = 'newdata2'

    # Model name: it is recommended to keep it consistent with the model registration name / class name
    # In the training script, it is best to construct the model dynamically based on this name
    model_name = 'SwiftCrackNet'

    # Model version: used to distinguish different iteration versions
    model_version = 'v5'

    # Experiment tag: used to distinguish baseline / ablation / debug / fine-tuning, etc.
    exp_tag = 'WBceTverskyLoss'

    # Whether to automatically append a timestamp
    use_timestamp = True

    # Root directory for all experimental results
    work_dir_root = r'H:\SwiftCrackNet\SwiftCrackNet-main\results'


    # =========================
    # 2. Model configuration
    # =========================

    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        # 'c_list': [32, 64, 72, 96, 128],
        'c_list': [24, 40, 64, 80, 96],
        'pretrained_path': '',
        'drop_path_rate': 0.03,
        'use_sigmoid': True
    }

    # Test weights, can be left empty
    test_weights = ''


    # =========================
    # 3. Dataset path configuration
    # =========================
    if dataset_name == 'newdata2':
        data_path = r'H:\SwiftCrackNet\SwiftCrackNet-main\newdata2'
    else:
        raise Exception(f'dataset_name "{dataset_name}" is not supported!')


    # =========================
    # 4. Basic training parameters
    # =========================
    # criterion = BceDiceLoss()
    criterion = WBceTverskyLoss()

    num_classes = 1
    input_size_h = 512
    input_size_w = 512
    input_channels = 3

    distributed = False
    local_rank = -1
    world_size = None
    rank = None

    num_workers = 8
    seed = 42
    amp = False

    batch_size = 8
    epochs = 200

    # Early stopping parameters
    early_stop_patience = 50

    # Threshold
    threshold = 0.5


    # =========================
    # 5. Logging / validation / checkpoint saving frequency
    # =========================
    print_interval = 200
    val_interval = 10
    save_interval = 100


    # =========================
    # 6. Optimizer configuration
    # =========================
    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False


    # =========================
    # 7. Learning rate scheduler configuration
    # =========================
    sch = 'CosineAnnealingLR'
    T_max = 50
    eta_min = 1e-5
    last_epoch = -1


    # =========================
    # 8. Training resumption settings
    # =========================
    resume_from = ''   # If left empty, training will resume from latest.pth in the current experiment directory by default.
    auto_resume = True


    # =========================
    # 9. Automatically generate the experiment name and working directory
    # =========================
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    exp_name_parts = [
        dataset_name,
        model_name,
        model_version,
        exp_tag
    ]

    if use_timestamp:
        exp_name_parts.append(timestamp)

    exp_name = '_'.join(exp_name_parts)

    work_dir = os.path.join(work_dir_root, exp_name)
