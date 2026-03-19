import os
import csv
import copy
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from configs.eval_config import eval_config

from models.SwiftCrackNet_v5 import SwiftCrackNetV5
# from models.SwiftCrackNet_v4 import SwiftCrackNetV4


# from models.unet import UNet
# from models.deeplabv3plus import DeepLabV3Plus


# =========================================================
# 0. Model registry
# =========================================================
MODEL_REGISTRY = {
    "SwiftCrackNet": SwiftCrackNetV5,
}


# =========================================================
# 1. Basic utilities
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def save_csv(file_path, header, rows):
    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_ci(values, alpha=0.95):
    lower = np.percentile(values, (1 - alpha) / 2 * 100)
    upper = np.percentile(values, (1 + alpha) / 2 * 100)
    mean_val = np.mean(values)
    return mean_val, lower, upper


def find_mask_path(mask_dir, base_name):
    candidates = [
        base_name + "_mask.png",
        base_name + ".png",
        base_name + ".jpg",
        base_name + ".jpeg",
        base_name + ".bmp",
        base_name + ".tif",
        base_name + ".tiff",
        base_name + "_label.png",
        base_name + "_label.jpg",
    ]
    for name in candidates:
        path = os.path.join(mask_dir, name)
        if os.path.exists(path):
            return path
    return None


def should_save_by_interval(index, interval):
    if interval is None or interval <= 0:
        return False
    return (index % interval) == 0


def get_model_state_keys(model):
    return list(model.state_dict().keys())


def get_experiment_root_from_checkpoint(checkpoint_path):
    ckpt_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(ckpt_dir) == "checkpoints":
        return os.path.dirname(ckpt_dir)
    return os.path.dirname(checkpoint_path)


def try_load_train_config_from_checkpoint(checkpoint_path):
    exp_root = get_experiment_root_from_checkpoint(checkpoint_path)
    candidate_files = [
        os.path.join(exp_root, "config_snapshot.json"),
        os.path.join(exp_root, "config.json"),
    ]

    for file_path in candidate_files:
        if os.path.exists(file_path):
            try:
                cfg = load_json(file_path)
                print(f"[Auto Load Train Config] loaded from: {file_path}")
                return cfg, file_path
            except Exception as e:
                print(f"[Auto Load Train Config Failed] {file_path}, error: {e}")

    print("[Auto Load Train Config] no train config json found.")
    return None, None


def merge_single_experiment_config(exp_cfg, global_cfg):

    merged = {}

    mapping = {
        "data_root": "default_data_root",
        "split": "default_split",
        "eval_size": "default_eval_size",
        "batch_size": "default_batch_size",
        "num_workers": "default_num_workers",
        "seed": "default_seed",
        "num_thresholds": "default_num_thresholds",
        "num_bootstrap": "default_num_bootstrap",
        "deltas": "default_deltas",
        "save_score_interval": "default_save_score_interval",
        "save_binary_interval": "default_save_binary_interval",
        "save_score_png": "default_save_score_png",
        "save_score_npy": "default_save_score_npy",
        "save_error_map": "default_save_error_map",
        "save_error_map_interval": "default_save_error_map_interval",
        "error_map_threshold_mode": "default_error_map_threshold_mode",
        "fixed_error_map_threshold": "default_fixed_error_map_threshold",
        "save_error_overlay": "default_save_error_overlay",
        "error_overlay_alpha": "default_error_overlay_alpha",
        "auto_load_train_config": "default_auto_load_train_config",
    }

    for k, default_k in mapping.items():
        merged[k] = getattr(global_cfg, default_k, None)

    for k, v in exp_cfg.items():
        merged[k] = v

    checkpoint_path = merged.get("checkpoint_path", "")
    train_cfg_dict = None
    train_cfg_path = None

    if merged.get("auto_load_train_config", True) and checkpoint_path:
        train_cfg_dict, train_cfg_path = try_load_train_config_from_checkpoint(checkpoint_path)

    if train_cfg_dict is not None:
        if merged.get("model_name", None) in [None, ""]:
            merged["model_name"] = train_cfg_dict.get("model_name", train_cfg_dict.get("network", None))

        if merged.get("model_config", None) in [None, {}]:
            merged["model_config"] = train_cfg_dict.get("model_config", None)

        if merged.get("data_root", None) in [None, ""]:
            merged["data_root"] = train_cfg_dict.get("data_root", train_cfg_dict.get("data_path", None))

        if merged.get("eval_size", None) in [None, 0]:
            merged["eval_size"] = train_cfg_dict.get("input_size_h", 512)

    merged["loaded_train_config_path"] = train_cfg_path
    return merged


# =========================================================
# 2. Dataset
# =========================================================
class DeepCrackEvalDataset(Dataset):
    def __init__(self, root_dir, split="test", eval_size=512):
        self.root_dir = root_dir
        self.split = split
        self.eval_size = eval_size

        self.img_dir = os.path.join(root_dir, split, "imgs")
        self.mask_dir = os.path.join(root_dir, split, "mask")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.img_names = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])

        if len(self.img_names) == 0:
            raise ValueError(f"No image files found in: {self.img_dir}")

        self.samples = []
        for img_name in self.img_names:
            base_name = os.path.splitext(img_name)[0]
            mask_path = find_mask_path(self.mask_dir, base_name)
            if mask_path is None:
                raise FileNotFoundError(f"Missing mask file for image: {img_name}")
            self.samples.append((img_name, mask_path))

        print(f"[EvalDataset] root={self.root_dir}")
        print(f"[EvalDataset] split={self.split}, image_count={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, mask_path = self.samples[index]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        raw_h, raw_w = mask.shape[:2]

        if self.eval_size is not None and self.eval_size > 0:
            image_resized = cv2.resize(image, (self.eval_size, self.eval_size), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image

        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        return image_tensor, img_name, raw_h, raw_w


# =========================================================
# 3. Model construction and loading
# =========================================================
def build_model(model_name, model_config, device):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name: {model_name}, available: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**model_config)
    model = model.to(device)
    model.eval()
    return model


def load_checkpoint_to_model(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model_keys = get_model_state_keys(model)
    model_has_module = len(model_keys) > 0 and model_keys[0].startswith("module.")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if model_has_module and (not new_k.startswith("module.")):
            new_k = "module." + new_k
        elif (not model_has_module) and new_k.startswith("module."):
            new_k = new_k[7:]
        new_state_dict[new_k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"[load_checkpoint] missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}")
    if len(missing_keys) > 0:
        print(f"[load_checkpoint] example missing keys: {missing_keys[:10]}")
    if len(unexpected_keys) > 0:
        print(f"[load_checkpoint] example unexpected keys: {unexpected_keys[:10]}")

    model.to(device)
    model.eval()
    return model


# =========================================================
# 4. Complexity statistics
# =========================================================
def count_parameters(model):
    base_model = model.module if hasattr(model, "module") else model
    return sum(p.numel() for p in base_model.parameters() if p.requires_grad)


class DeepCrackInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        pred = parse_model_output(outputs)
        return pred


def try_compute_flops(model, input_size=(1, 3, 512, 512), device="cpu"):
    try:
        from thop import profile

        base_model = model.module if hasattr(model, "module") else model
        model_copy = copy.deepcopy(base_model).to(device)
        model_copy.eval()

        wrapper = DeepCrackInferenceWrapper(model_copy).to(device)
        wrapper.eval()

        dummy = torch.randn(*input_size).to(device)

        with torch.no_grad():
            flops, params = profile(wrapper, inputs=(dummy,), verbose=False)

        del wrapper
        del model_copy
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return flops, params

    except Exception as e:
        print(f"[Warning] FLOPs computation skipped: {e}")
        return None, None


# =========================================================
# 5. Model output parsing
# =========================================================
def parse_model_output(outputs):
    if isinstance(outputs, dict):
        for key in ["fused", "output", "pred", "prediction", "out"]:
            if key in outputs:
                pred = outputs[key]
                break
        else:
            pred = list(outputs.values())[-1]
    elif isinstance(outputs, (tuple, list)):
        pred = outputs[-1]
    else:
        pred = outputs

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)

    if pred.shape[1] > 1:
        pred = pred[:, 0:1, :, :]

    if pred.min().item() < 0.0 or pred.max().item() > 1.0:
        pred = torch.sigmoid(pred)

    return pred


# =========================================================
# 6. Run inference and save score maps
# =========================================================
@torch.no_grad()
def infer_and_save_score_maps(
    model,
    dataloader,
    device,
    score_map_dir,
    data_root,
    split="test",
    save_score_interval=1,
    save_score_png=True,
    save_score_npy=True
):
    create_dir(score_map_dir)
    model.eval()

    image_records = []
    use_amp = device.type == "cuda"
    mask_dir = os.path.join(data_root, split, "mask")

    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    first_batch_logged = False
    global_index = 0
    saved_score_count = 0

    for images, img_names, raw_hs, raw_ws in tqdm(dataloader, desc="infer_score_maps"):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            probs = parse_model_output(outputs)

        if not first_batch_logged:
            print(f"[Debug] prediction range after parsing: min={probs.min().item():.6f}, max={probs.max().item():.6f}")
            print(f"[Debug] prediction shape: {tuple(probs.shape)}")
            first_batch_logged = True

        probs = probs.detach().float().cpu().numpy()

        batch_size = probs.shape[0]
        for i in range(batch_size):
            img_name = img_names[i]
            base_name = os.path.splitext(img_name)[0]
            raw_h = int(raw_hs[i])
            raw_w = int(raw_ws[i])

            prob_map = np.ascontiguousarray(probs[i, 0].astype(np.float32))
            prob_map_resized = cv2.resize(prob_map, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
            prob_map_resized = np.ascontiguousarray(prob_map_resized.astype(np.float32))

            mask_path = find_mask_path(mask_dir, base_name)
            if mask_path is None:
                raise FileNotFoundError(f"Cannot find mask for image: {img_name}")

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")
            gt_mask = (gt_mask > 127).astype(np.uint8)

            record = {
                "img_name": img_name,
                "base_name": base_name,
                "score_map_path": None,
                "score_map": prob_map_resized,
                "gt_mask": gt_mask,
                "global_index": global_index,
                "score_saved": False,
            }

            if should_save_by_interval(global_index, save_score_interval):
                npy_path = os.path.join(score_map_dir, base_name + "_score.npy")
                png_path = os.path.join(score_map_dir, base_name + "_score.png")

                if save_score_npy:
                    np.save(npy_path, prob_map_resized)
                    record["score_map_path"] = npy_path

                if save_score_png:
                    vis_map = np.clip(prob_map_resized * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(png_path, vis_map)

                record["score_saved"] = True
                saved_score_count += 1

            image_records.append(record)
            global_index += 1

    print(f"[Score Map Saving] total_images={len(image_records)}, saved={saved_score_count}, save_score_interval={save_score_interval}")
    return image_records


# =========================================================
# 7. Metric computation
# =========================================================
def compute_confusion_with_tolerance(pred_binary, gt_binary, delta):

    pred_binary = pred_binary.astype(np.uint8)
    gt_binary = gt_binary.astype(np.uint8)

    pred_sum = int(pred_binary.sum())
    gt_sum = int(gt_binary.sum())

    # Edge case: both are empty
    if pred_sum == 0 and gt_sum == 0:
        return 0, 0, 0, 1.0, 1.0, 1.0

    # No prediction, but the GT contains targets
    # To make the left end of the PR curve more well-defined, set precision to 1.0 and recall to 0.0.
    if pred_sum == 0 and gt_sum > 0:
        return 0, 0, gt_sum, 1.0, 0.0, 0.0

    # Predictions exist, but the GT is empty.
    if pred_sum > 0 and gt_sum == 0:
        return 0, pred_sum, 0, 0.0, 0.0, 0.0

    kernel_size = delta * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    gt_dilated = cv2.dilate(gt_binary, kernel, iterations=1)
    pred_dilated = cv2.dilate(pred_binary, kernel, iterations=1)

    matched_pred = int((pred_binary & gt_dilated).sum())
    matched_gt = int((gt_binary & pred_dilated).sum())

    fp = pred_sum - matched_pred
    fn = gt_sum - matched_gt

    precision = matched_pred / (pred_sum + 1e-8)
    recall = matched_gt / (gt_sum + 1e-8)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return matched_pred, fp, fn, precision, recall, f1


def precompute_threshold_stats(image_records, delta, thresholds):

    n_images = len(image_records)
    n_thr = len(thresholds)

    tp_p_mat = np.zeros((n_images, n_thr), dtype=np.int64)   # matched_pred
    fp_mat = np.zeros((n_images, n_thr), dtype=np.int64)     # pred_sum - matched_pred
    tp_r_mat = np.zeros((n_images, n_thr), dtype=np.int64)   # matched_gt
    fn_mat = np.zeros((n_images, n_thr), dtype=np.int64)     # gt_sum - matched_gt
    f1_mat = np.zeros((n_images, n_thr), dtype=np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (delta * 2 + 1, delta * 2 + 1))

    for i, record in enumerate(tqdm(image_records, desc=f"precompute_delta{delta}", leave=False)):
        score_map = record["score_map"]
        gt_mask = record["gt_mask"].astype(np.uint8)
        gt_sum = int(gt_mask.sum())

        # gt_dilated depends only on gt and delta, so it can be computed once in advance.
        gt_dilated = cv2.dilate(gt_mask, kernel, iterations=1) if gt_sum > 0 else gt_mask

        for t, threshold in enumerate(thresholds):
            pred_binary = (score_map >= threshold).astype(np.uint8)
            pred_sum = int(pred_binary.sum())

            if pred_sum == 0 and gt_sum == 0:
                tp_p = 0
                fp = 0
                tp_r = 0
                fn = 0
                precision = 1.0
                recall = 1.0
                f1 = 1.0

            elif pred_sum == 0 and gt_sum > 0:
                tp_p = 0
                fp = 0
                tp_r = 0
                fn = gt_sum
                precision = 1.0
                recall = 0.0
                f1 = 0.0

            elif pred_sum > 0 and gt_sum == 0:
                tp_p = 0
                fp = pred_sum
                tp_r = 0
                fn = 0
                precision = 0.0
                recall = 0.0
                f1 = 0.0

            else:
                pred_dilated = cv2.dilate(pred_binary, kernel, iterations=1)

                # prediction-side matched pixels
                tp_p = int((pred_binary & gt_dilated).sum())
                fp = pred_sum - tp_p

                # GT-side recalled pixels
                tp_r = int((gt_mask & pred_dilated).sum())
                fn = gt_sum - tp_r

                # More standardized: directly defined by the total number of predictions / total number of GT pixels
                precision = tp_p / (pred_sum + 1e-8)
                recall = tp_r / (gt_sum + 1e-8)

                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                else:
                    f1 = 0.0

            tp_p_mat[i, t] = tp_p
            fp_mat[i, t] = fp
            tp_r_mat[i, t] = tp_r
            fn_mat[i, t] = fn
            f1_mat[i, t] = f1

    return {
        "tp_p_mat": tp_p_mat,
        "fp_mat": fp_mat,
        "tp_r_mat": tp_r_mat,
        "fn_mat": fn_mat,
        "f1_mat": f1_mat,
        "thresholds": np.asarray(thresholds, dtype=np.float32)
    }



def compute_metrics_from_precomputed(precomp, indices=None, need_pr_rows=True):

    tp_p_mat = precomp["tp_p_mat"]   # matched_pred
    fp_mat = precomp["fp_mat"]       # pred_sum - matched_pred
    tp_r_mat = precomp["tp_r_mat"]   # matched_gt
    fn_mat = precomp["fn_mat"]       # gt_sum - matched_gt
    f1_mat = precomp["f1_mat"]
    thresholds = precomp["thresholds"]

    if indices is None:
        tp_p_sum = tp_p_mat.sum(axis=0)
        fp_sum = fp_mat.sum(axis=0)
        tp_r_sum = tp_r_mat.sum(axis=0)
        fn_sum = fn_mat.sum(axis=0)
        f1_sel = f1_mat
    else:
        tp_p_sum = tp_p_mat[indices].sum(axis=0)
        fp_sum = fp_mat[indices].sum(axis=0)
        tp_r_sum = tp_r_mat[indices].sum(axis=0)
        fn_sum = fn_mat[indices].sum(axis=0)
        f1_sel = f1_mat[indices]

    # The “implicit form” of tp / (tp + fp) is no longer used directly here.
    # Instead, pred_sum and gt_sum are explicitly restored to make the definition clearer.
    pred_sum = tp_p_sum + fp_sum
    gt_sum = tp_r_sum + fn_sum

    # Key modification:
    # When pred_sum == 0, set precision to 1.0
    # When gt_sum == 0, set recall to 0.0 (this usually does not affect the main results for the whole dataset)
    precisions = np.where(pred_sum > 0, tp_p_sum / (pred_sum + 1e-8), 1.0)
    recalls = np.where(gt_sum > 0, tp_r_sum / (gt_sum + 1e-8), 0.0)

    mean_f1s = f1_sel.mean(axis=0)

    best_idx = int(np.argmax(mean_f1s))
    ods_f1 = float(mean_f1s[best_idx])
    ods_threshold = float(thresholds[best_idx])
    ois_f1 = float(f1_sel.max(axis=1).mean())

    # More standardized AP calculation
    order = np.argsort(recalls)
    recalls_sorted = recalls[order]
    precisions_sorted = precisions[order]

    unique_recalls = np.unique(recalls_sorted)
    max_precisions = np.array(
        [np.max(precisions_sorted[recalls_sorted == r]) for r in unique_recalls],
        dtype=np.float64
    )

    # monotonic precision envelope
    for i in range(len(max_precisions) - 2, -1, -1):
        max_precisions[i] = max(max_precisions[i], max_precisions[i + 1])

    ap = float(np.trapezoid(max_precisions, unique_recalls))

    pr_rows = None
    if need_pr_rows:
        pr_rows = [
            [float(thresholds[i]), float(precisions[i]), float(recalls[i]), float(mean_f1s[i])]
            for i in range(len(thresholds))
        ]

    return {
        "ods_f1": ods_f1,
        "ois_f1": ois_f1,
        "ap": ap,
        "ods_threshold": ods_threshold,
        "pr_rows": pr_rows
    }



def bootstrap_ci_fast(precomp, delta, num_bootstrap=200, seed=42):
    rng = np.random.default_rng(seed)
    n = precomp["tp_p_mat"].shape[0]

    ods_f1_samples = []
    ap_samples = []

    for _ in tqdm(range(num_bootstrap), desc=f"bootstrap_delta{delta}", leave=False):
        indices = rng.integers(0, n, n)
        metrics = compute_metrics_from_precomputed(precomp, indices=indices, need_pr_rows=False)
        ods_f1_samples.append(metrics["ods_f1"])
        ap_samples.append(metrics["ap"])

    ods_mean, ods_lower, ods_upper = format_ci(ods_f1_samples, alpha=0.95)
    ap_mean, ap_lower, ap_upper = format_ci(ap_samples, alpha=0.95)

    return {
        "ods_f1_mean": ods_mean,
        "ods_f1_ci_lower": ods_lower,
        "ods_f1_ci_upper": ods_upper,
        "ap_mean": ap_mean,
        "ap_ci_lower": ap_lower,
        "ap_ci_upper": ap_upper,
    }


# =========================================================
# 8. Visualization tools: binary maps / PR curves / error maps
# =========================================================
def save_ods_binary_maps(image_records, ods_threshold, delta, save_dir, save_binary_interval=1):
    create_dir(save_dir)
    saved_count = 0

    for record in image_records:
        global_index = record.get("global_index", 0)
        if not should_save_by_interval(global_index, save_binary_interval):
            continue

        base_name = record["base_name"]
        score_map = record["score_map"]

        pred_binary = (score_map >= ods_threshold).astype(np.uint8) * 255
        save_path = os.path.join(save_dir, base_name + f"_ods_delta{delta}.png")
        cv2.imwrite(save_path, pred_binary)
        saved_count += 1

    print(f"[Binary Map Saving][delta={delta}] total_images={len(image_records)}, saved={saved_count}, save_binary_interval={save_binary_interval}")


def plot_pr_curve(pr_rows, save_path, method_name, delta):
    if pr_rows is None or len(pr_rows) == 0:
        print(f"[plot_pr_curve] Empty pr_rows, skip: {save_path}")
        return

    precisions = np.array([r[1] for r in pr_rows], dtype=np.float64)
    recalls = np.array([r[2] for r in pr_rows], dtype=np.float64)

    valid = np.isfinite(recalls) & np.isfinite(precisions)
    recalls = np.clip(recalls[valid], 0.0, 1.0)
    precisions = np.clip(precisions[valid], 0.0, 1.0)

    if len(recalls) == 0:
        print(f"[plot_pr_curve] No valid PR points, skip: {save_path}")
        return

    order = np.argsort(recalls)
    recalls_sorted = recalls[order]
    precisions_sorted = precisions[order]

    unique_recalls = np.unique(recalls_sorted)
    max_precisions = np.array(
        [np.max(precisions_sorted[recalls_sorted == r]) for r in unique_recalls],
        dtype=np.float64
    )

    for i in range(len(max_precisions) - 2, -1, -1):
        max_precisions[i] = max(max_precisions[i], max_precisions[i + 1])

    plt.figure(figsize=(6, 5))
    plt.plot(unique_recalls, max_precisions, linewidth=2.2, label=method_name, color="#D62728")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve under δ = {delta} px")
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()




def load_original_image(data_root, split, img_name):
    img_dir = os.path.join(data_root, split, "imgs")
    img_path = os.path.join(img_dir, img_name)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Cannot find image path for: {img_name}")

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def build_error_map(pred_binary, gt_binary):
    """
    RGB:
    TP -> green
    FP -> red
    FN -> blue
    TN -> black
    """
    pred_binary = pred_binary.astype(np.uint8)
    gt_binary = gt_binary.astype(np.uint8)

    tp = (pred_binary == 1) & (gt_binary == 1)
    fp = (pred_binary == 1) & (gt_binary == 0)
    fn = (pred_binary == 0) & (gt_binary == 1)

    h, w = pred_binary.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    error_map[tp] = [0, 255, 0]    # green
    error_map[fp] = [255, 0, 0]    # red
    error_map[fn] = [0, 0, 255]    # blue

    return error_map


def overlay_error_map_on_image(image_rgb, error_map_rgb, alpha=0.5):
    image_rgb = image_rgb.astype(np.uint8)
    error_map_rgb = error_map_rgb.astype(np.uint8)

    overlay = image_rgb.copy()
    mask = np.any(error_map_rgb > 0, axis=-1)

    overlay[mask] = (
        (1 - alpha) * image_rgb[mask] + alpha * error_map_rgb[mask]
    ).astype(np.uint8)

    return overlay


def save_error_maps(
    image_records,
    data_root,
    split,
    delta,
    threshold,
    save_dir,
    save_error_map_interval=1,
    save_overlay=True,
    overlay_alpha=0.5
):
    create_dir(save_dir)
    raw_dir = os.path.join(save_dir, "raw")
    overlay_dir = os.path.join(save_dir, "overlay")
    create_dir(raw_dir)
    if save_overlay:
        create_dir(overlay_dir)

    saved_count = 0

    for record in image_records:
        global_index = record.get("global_index", 0)
        if not should_save_by_interval(global_index, save_error_map_interval):
            continue

        img_name = record["img_name"]
        base_name = record["base_name"]
        score_map = record["score_map"]
        gt_mask = record["gt_mask"].astype(np.uint8)

        pred_binary = (score_map >= threshold).astype(np.uint8)
        error_map = build_error_map(pred_binary, gt_mask)

        raw_save_path = os.path.join(raw_dir, base_name + f"_error_delta{delta}.png")
        cv2.imwrite(raw_save_path, cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR))

        if save_overlay:
            image_rgb = load_original_image(data_root, split, img_name)
            if image_rgb.shape[:2] != error_map.shape[:2]:
                image_rgb = cv2.resize(
                    image_rgb,
                    (error_map.shape[1], error_map.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            overlay = overlay_error_map_on_image(image_rgb, error_map, alpha=overlay_alpha)
            overlay_save_path = os.path.join(overlay_dir, base_name + f"_error_overlay_delta{delta}.png")
            cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        saved_count += 1

    print(
        f"[Error Map Saving][delta={delta}] total_images={len(image_records)}, "
        f"saved={saved_count}, interval={save_error_map_interval}"
    )


# =========================================================
# 9. Single-experiment evaluation
# =========================================================
def evaluate_single_experiment(exp_cfg):
    data_root = exp_cfg["data_root"]
    checkpoint_path = exp_cfg["checkpoint_path"]
    save_root = exp_cfg["save_root"]
    split = exp_cfg.get("split", "test")
    eval_size = exp_cfg.get("eval_size", 512)
    batch_size = exp_cfg.get("batch_size", 4)
    num_workers = exp_cfg.get("num_workers", 4)
    method_name = exp_cfg.get("method_name", "UnknownMethod")
    model_name = exp_cfg.get("model_name", None)
    model_config = exp_cfg.get("model_config", None)
    num_thresholds = exp_cfg.get("num_thresholds", 101)
    num_bootstrap = exp_cfg.get("num_bootstrap", 200)
    seed = exp_cfg.get("seed", 42)
    deltas = exp_cfg.get("deltas", (1, 2, 3, 5))
    save_score_interval = exp_cfg.get("save_score_interval", 1)
    save_binary_interval = exp_cfg.get("save_binary_interval", 1)
    save_score_png = exp_cfg.get("save_score_png", True)
    save_score_npy = exp_cfg.get("save_score_npy", True)
    save_error_map = exp_cfg.get("save_error_map", True)
    save_error_map_interval = exp_cfg.get("save_error_map_interval", 10)
    error_map_threshold_mode = exp_cfg.get("error_map_threshold_mode", "ods")
    fixed_error_map_threshold = exp_cfg.get("fixed_error_map_threshold", 0.5)
    save_error_overlay = exp_cfg.get("save_error_overlay", True)
    error_overlay_alpha = exp_cfg.get("error_overlay_alpha", 0.5)
    loaded_train_config_path = exp_cfg.get("loaded_train_config_path", None)

    if model_name is None or model_config is None:
        raise ValueError(f"model_name/model_config missing for experiment: {method_name}")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*100}")
    print(f"[Evaluate] method={method_name}")
    print(f"[Evaluate] model_name={model_name}")
    print(f"[Evaluate] checkpoint={checkpoint_path}")
    print(f"[Evaluate] save_root={save_root}")
    print(f"{'='*100}\n")

    create_dir(save_root)

    score_map_dir = os.path.join(save_root, "score_maps")
    binary_root = os.path.join(save_root, "binary_maps")
    error_root = os.path.join(save_root, "error_maps")
    pr_curve_dir = os.path.join(save_root, "pr_curves")
    table_dir = os.path.join(save_root, "tables")

    create_dir(score_map_dir)
    create_dir(binary_root)
    create_dir(error_root)
    create_dir(pr_curve_dir)
    create_dir(table_dir)

    eval_dataset = DeepCrackEvalDataset(root_dir=data_root, split=split, eval_size=eval_size)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = build_model(model_name=model_name, model_config=model_config, device=device)
    model = load_checkpoint_to_model(model, checkpoint_path, device)
    model.eval()

    params = count_parameters(model)
    flops, _ = try_compute_flops(
        model=model,
        input_size=(1, 3, eval_size, eval_size),
        device=device
    )

    image_records = infer_and_save_score_maps(
        model=model,
        dataloader=eval_loader,
        device=device,
        score_map_dir=score_map_dir,
        data_root=data_root,
        split=split,
        save_score_interval=save_score_interval,
        save_score_png=save_score_png,
        save_score_npy=save_score_npy
    )

    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)

    summary_lines = []
    summary_lines.append(f"Method: {method_name}")
    summary_lines.append(f"Model name: {model_name}")
    summary_lines.append(f"Checkpoint: {checkpoint_path}")
    summary_lines.append(f"Dataset root: {data_root}")
    summary_lines.append(f"Split: {split}")
    summary_lines.append(f"Eval image count: {len(image_records)}")
    summary_lines.append(f"Params: {int(params)}")
    summary_lines.append(f"FLOPs: {float(flops) if flops is not None else 'N/A'}")
    summary_lines.append(f"Save score interval: {save_score_interval}")
    summary_lines.append(f"Save binary interval: {save_binary_interval}")
    summary_lines.append(f"Save error map: {save_error_map}")
    summary_lines.append(f"Save error map interval: {save_error_map_interval}")
    summary_lines.append(f"Error map threshold mode: {error_map_threshold_mode}")
    summary_lines.append(f"Fixed error map threshold: {fixed_error_map_threshold}")
    summary_lines.append(f"Save error overlay: {save_error_overlay}")
    summary_lines.append(f"Error overlay alpha: {error_overlay_alpha}")
    summary_lines.append(f"Loaded train config: {loaded_train_config_path}")
    summary_lines.append("")

    save_csv(
        os.path.join(table_dir, "model_complexity.csv"),
        header=["Method", "ModelName", "Params", "FLOPs"],
        rows=[[method_name, model_name, int(params), float(flops) if flops is not None else "N/A"]]
    )

    all_results_rows = []
    all_ci_rows = []
    summary_metrics = []

    for delta in deltas:
        print(f"\n========== Evaluating {method_name} | delta = {delta} px ==========")

        precomp = precompute_threshold_stats(image_records, delta, thresholds)
        metrics = compute_metrics_from_precomputed(precomp, indices=None, need_pr_rows=True)

        ods_f1 = metrics["ods_f1"]
        ois_f1 = metrics["ois_f1"]
        ap = metrics["ap"]
        ods_threshold = metrics["ods_threshold"]
        pr_rows = metrics["pr_rows"]

        save_csv(
            os.path.join(pr_curve_dir, f"PR_delta{delta}.csv"),
            header=["Threshold", "Precision", "Recall", "F1"],
            rows=pr_rows
        )

        plot_pr_curve(
            pr_rows,
            os.path.join(pr_curve_dir, f"PR_delta{delta}.png"),
            method_name=method_name,
            delta=delta
        )

        save_csv(
            os.path.join(table_dir, f"results_delta{delta}.csv"),
            header=["Method", "ModelName", "Delta", "ODS_F1", "OIS_F1", "AP", "ODS_Threshold", "Params", "FLOPs"],
            rows=[[
                method_name, model_name, delta,
                f"{ods_f1:.6f}", f"{ois_f1:.6f}", f"{ap:.6f}", f"{ods_threshold:.6f}",
                int(params), float(flops) if flops is not None else "N/A"
            ]]
        )

        save_ods_binary_maps(
            image_records=image_records,
            ods_threshold=ods_threshold,
            delta=delta,
            save_dir=os.path.join(binary_root, f"ods_delta{delta}"),
            save_binary_interval=save_binary_interval
        )

        if save_error_map:
            if error_map_threshold_mode == "fixed":
                error_threshold = fixed_error_map_threshold
            else:
                error_threshold = ods_threshold

            save_error_maps(
                image_records=image_records,
                data_root=data_root,
                split=split,
                delta=delta,
                threshold=error_threshold,
                save_dir=os.path.join(error_root, f"delta_{delta}"),
                save_error_map_interval=save_error_map_interval,
                save_overlay=save_error_overlay,
                overlay_alpha=error_overlay_alpha
            )

        ci_metrics = bootstrap_ci_fast(
            precomp=precomp,
            delta=delta,
            num_bootstrap=num_bootstrap,
            seed=seed + delta
        )

        save_csv(
            os.path.join(table_dir, f"ci_delta{delta}.csv"),
            header=["Method", "ModelName", "Delta", "ODS_F1_Mean", "ODS_F1_CI_Lower", "ODS_F1_CI_Upper", "AP_Mean", "AP_CI_Lower", "AP_CI_Upper"],
            rows=[[
                method_name, model_name, delta,
                f"{ci_metrics['ods_f1_mean']:.6f}",
                f"{ci_metrics['ods_f1_ci_lower']:.6f}",
                f"{ci_metrics['ods_f1_ci_upper']:.6f}",
                f"{ci_metrics['ap_mean']:.6f}",
                f"{ci_metrics['ap_ci_lower']:.6f}",
                f"{ci_metrics['ap_ci_upper']:.6f}"
            ]]
        )

        all_results_rows.append([
            method_name, model_name, delta,
            f"{ods_f1:.6f}", f"{ois_f1:.6f}", f"{ap:.6f}", f"{ods_threshold:.6f}",
            int(params), float(flops) if flops is not None else "N/A"
        ])

        all_ci_rows.append([
            method_name, model_name, delta,
            f"{ci_metrics['ods_f1_mean']:.6f}",
            f"{ci_metrics['ods_f1_ci_lower']:.6f}",
            f"{ci_metrics['ods_f1_ci_upper']:.6f}",
            f"{ci_metrics['ap_mean']:.6f}",
            f"{ci_metrics['ap_ci_lower']:.6f}",
            f"{ci_metrics['ap_ci_upper']:.6f}"
        ])

        summary_metrics.append({
            "delta": delta,
            "ods_f1": ods_f1,
            "ois_f1": ois_f1,
            "ap": ap,
            "ods_threshold": ods_threshold,
            "error_map_threshold": float(error_threshold) if save_error_map else None,
            "ods_f1_mean": ci_metrics["ods_f1_mean"],
            "ods_f1_ci_lower": ci_metrics["ods_f1_ci_lower"],
            "ods_f1_ci_upper": ci_metrics["ods_f1_ci_upper"],
            "ap_mean": ci_metrics["ap_mean"],
            "ap_ci_lower": ci_metrics["ap_ci_lower"],
            "ap_ci_upper": ci_metrics["ap_ci_upper"],
        })

        line = (
            f"delta={delta}px | "
            f"ODS F1={ods_f1:.6f} | "
            f"OIS F1={ois_f1:.6f} | "
            f"AP={ap:.6f} | "
            f"ODS threshold={ods_threshold:.6f}"
        )
        print(line)
        summary_lines.append(line)

    save_csv(
        os.path.join(table_dir, "all_results.csv"),
        header=["Method", "ModelName", "Delta", "ODS_F1", "OIS_F1", "AP", "ODS_Threshold", "Params", "FLOPs"],
        rows=all_results_rows
    )

    save_csv(
        os.path.join(table_dir, "all_ci_results.csv"),
        header=["Method", "ModelName", "Delta", "ODS_F1_Mean", "ODS_F1_CI_Lower", "ODS_F1_CI_Upper", "AP_Mean", "AP_CI_Lower", "AP_CI_Upper"],
        rows=all_ci_rows
    )

    with open(os.path.join(save_root, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    save_json(
        os.path.join(save_root, "eval_config_used.json"),
        exp_cfg
    )

    print(f"\n[Finished] {method_name} -> {save_root}")

    return {
        "method_name": method_name,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "save_root": save_root,
        "params": int(params),
        "flops": float(flops) if flops is not None else None,
        "metrics": summary_metrics
    }


# =========================================================
# 10. Overall summary table for multiple models
# =========================================================
def save_multi_model_summary(all_results, global_save_root):
    create_dir(global_save_root)

    summary_rows = []
    summary_ci_like_rows = []
    delta_to_rows = {}

    for result in all_results:
        method_name = result["method_name"]
        model_name = result["model_name"]
        checkpoint_path = result["checkpoint_path"]
        save_root = result["save_root"]
        params = result["params"]
        flops = result["flops"]

        for m in result["metrics"]:
            delta = m["delta"]

            row = [
                method_name,
                model_name,
                delta,
                f"{m['ods_f1']:.6f}",
                f"{m['ois_f1']:.6f}",
                f"{m['ap']:.6f}",
                f"{m['ods_threshold']:.6f}",
                params,
                flops if flops is not None else "N/A",
                checkpoint_path,
                save_root
            ]
            summary_rows.append(row)

            ci_row = [
                method_name,
                model_name,
                delta,
                f"{m['ods_f1_mean']:.6f}",
                f"{m['ods_f1_ci_lower']:.6f}",
                f"{m['ods_f1_ci_upper']:.6f}",
                f"{m['ap_mean']:.6f}",
                f"{m['ap_ci_lower']:.6f}",
                f"{m['ap_ci_upper']:.6f}",
                params,
                flops if flops is not None else "N/A",
                checkpoint_path,
                save_root
            ]
            summary_ci_like_rows.append(ci_row)

            if delta not in delta_to_rows:
                delta_to_rows[delta] = []
            delta_to_rows[delta].append(row)

    save_csv(
        os.path.join(global_save_root, "multi_models_summary.csv"),
        header=[
            "Method", "ModelName", "Delta",
            "ODS_F1", "OIS_F1", "AP", "ODS_Threshold",
            "Params", "FLOPs", "CheckpointPath", "SaveRoot"
        ],
        rows=summary_rows
    )

    save_csv(
        os.path.join(global_save_root, "multi_models_summary_with_ci.csv"),
        header=[
            "Method", "ModelName", "Delta",
            "ODS_F1_Mean", "ODS_F1_CI_Lower", "ODS_F1_CI_Upper",
            "AP_Mean", "AP_CI_Lower", "AP_CI_Upper",
            "Params", "FLOPs", "CheckpointPath", "SaveRoot"
        ],
        rows=summary_ci_like_rows
    )

    for delta, rows in delta_to_rows.items():
        rows_sorted = sorted(rows, key=lambda x: float(x[3]), reverse=True)
        save_csv(
            os.path.join(global_save_root, f"multi_models_summary_delta{delta}.csv"),
            header=[
                "Method", "ModelName", "Delta",
                "ODS_F1", "OIS_F1", "AP", "ODS_Threshold",
                "Params", "FLOPs", "CheckpointPath", "SaveRoot"
            ],
            rows=rows_sorted
        )

    save_json(
        os.path.join(global_save_root, "multi_models_summary.json"),
        all_results
    )

    print(f"\n[Summary Saved] global_save_root: {global_save_root}")
    print("  - multi_models_summary.csv")
    print("  - multi_models_summary_with_ci.csv")
    for delta in sorted(delta_to_rows.keys()):
        print(f"  - multi_models_summary_delta{delta}.csv")


# =========================================================
# 11. Batch evaluation from eval_config
# =========================================================
def evaluate_multiple_models_from_config(cfg):
    experiments = getattr(cfg, "experiments", None)
    if experiments is None or len(experiments) == 0:
        raise ValueError("No experiments found in configs/eval_config.py")

    global_save_root = getattr(cfg, "global_save_root", "./multi_model_eval_outputs")
    create_dir(global_save_root)

    all_results = []

    for idx, exp in enumerate(experiments):
        print(f"\n{'#'*120}")
        print(f"[Multi Model Eval] {idx + 1}/{len(experiments)}")
        print(f"{'#'*120}")

        merged_exp = merge_single_experiment_config(exp, cfg)

        save_subdir = merged_exp.get("save_subdir", merged_exp.get("method_name", f"exp_{idx+1}"))
        merged_exp["save_root"] = os.path.join(global_save_root, save_subdir)

        for required_key in ["data_root", "checkpoint_path", "save_root", "model_name", "model_config"]:
            if merged_exp.get(required_key, None) in [None, ""]:
                raise ValueError(f"Missing required key: {required_key} in experiment: {merged_exp}")

        result = evaluate_single_experiment(merged_exp)
        all_results.append(result)

    save_multi_model_summary(all_results, global_save_root)
    return all_results


# =========================================================
# 12. main
# =========================================================
def main():
    evaluate_multiple_models_from_config(eval_config)


if __name__ == "__main__":
    main()
