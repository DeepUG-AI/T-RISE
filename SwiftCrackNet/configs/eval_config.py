import os
from datetime import datetime
from configs.config_setting import setting_config


class eval_config:

    # =====================================================
    # 1. Global output directory
    # =====================================================
    project_name = "crack_seg_eval"
    summary_name = "SwiftCrackNetV5_eval"
    use_timestamp = True

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    global_save_root_base = r"H:\SwiftCrackNet\SwiftCrackNet-main\eval_outputs"
    global_save_root = os.path.join(
        global_save_root_base,
        f"{summary_name}_{timestamp}" if use_timestamp else summary_name
    )

    # =====================================================
    # 2. Default parameters (used automatically when not specified in each experiment)
    # =====================================================
    default_data_root = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2"
    default_split = "test"
    default_eval_size = 512
    default_batch_size = 4
    default_num_workers = 4
    default_seed = 42

    default_num_thresholds = 101
    default_num_bootstrap = 200
    default_deltas = (1, 2, 3, 5)

    # =====================================================
    # 3. Checkpoint saving settings
    # 1 = save all, 10 = save 1 out of every 10 images, 0 = save none
    # =====================================================
    default_save_score_interval = 30
    default_save_binary_interval = 30

    default_save_score_png = True
    default_save_score_npy = True

    # =====================================================
    # 4. Error map configuration
    # =====================================================
    default_save_error_map = True
    default_save_error_map_interval = 30

    # "ods" -> use the ODS threshold corresponding to the current delta
    # "fixed" -> use the fixed threshold default_fixed_error_map_threshold
    default_error_map_threshold_mode = "ods"
    default_fixed_error_map_threshold = 0.5

    # Whether to save the overlay of the original image and the error map
    default_save_error_overlay = True

    # Overlay transparency
    default_error_overlay_alpha = 0.5

    # =====================================================
    # 5. Automatically load the training configuration snapshot
    # =====================================================
    default_auto_load_train_config = True

    # =====================================================
    # 6. Multi-experiment configuration
    # Each experiment corresponds to one dictionary.
    # =====================================================
    experiments = [
        {
            "method_name": "SwiftCrackNet_v5_bestdice",
            "model_name": "SwiftCrackNet",
            "model_config": setting_config.model_config,
            "checkpoint_path": r"H:\SwiftCrackNet\SwiftCrackNet-main\results\newdata2_SwiftCrackNet_reversion_v5_WBceTverskyLoss_2026_03_19_12_34_55\checkpoints\best_dice_epoch18_dice0.7834.pth",
            "save_subdir": "SwiftCrackNet_v5_bestdice",
        },
        {
            "method_name": "SwiftCrackNet_v5_bestloss",
            "model_name": "SwiftCrackNet",
            "model_config": setting_config.model_config,
            "checkpoint_path": r"H:\SwiftCrackNet\SwiftCrackNet-main\results\newdata2_SwiftCrackNet_reversion_v5_WBceTverskyLoss_2026_03_19_12_34_55\checkpoints\best_loss_epoch16_loss0.3575.pth",
            "save_subdir": "SwiftCrackNet_v5_bestloss",
        },
    ]
