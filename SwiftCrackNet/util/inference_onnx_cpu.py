import torch
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config_setting import setting_config
from utils import *

# Ensure no GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def measure_onnx_inference_speed(onnx_model_path, input_size, num_iterations=1000):
    # Create an ONNX Runtime session with CPU provider
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    # Generate a fake input tensor in NumPy format
    fake_input = np.random.randn(*input_size).astype(np.float32)

    # Measure inference time with tqdm for progress visualization
    times = []
    for _ in tqdm(range(num_iterations), desc="Measuring inference speed"):
        start_time = time.time()
        _ = session.run(None, {"input": fake_input})
        end_time = time.time()

        elapsed_time_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
        times.append(elapsed_time_ms)

    min_time = min(times)
    max_fps = 1000 / min_time  # Convert ms to seconds and calculate FPS

    print(f"Lowest inference time: {min_time:.4f} ms")
    print(f"Highest FPS: {max_fps:.4f}")

    # Return average time and FPS for consistency
    avg_time_ms = sum(times) / num_iterations
    avg_fps = 1000 / avg_time_ms
    return avg_time_ms, avg_fps


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)
    log_config_info(config, logger)

    print('#----------Measuring Inference Speed of ONNX Model----------#')
    model_cfg = config.model_config
    input_size = (1, model_cfg['input_channels'], 256, 256)  # Example input size
    print(input_size)

    # Path to the ONNX model
    onnx_model_path = "model_weights/model_cpu_optimized.onnx"  # Ensure this is the correct path to your ONNX model

    # Measure inference speed
    inference_time, fps = measure_onnx_inference_speed(onnx_model_path, input_size)
    print(f"Average inference time per forward pass: {inference_time:.4f} ms")
    print(f"Frames per second (FPS): {fps:.4f}")
    logger.info(f"Average inference time per forward pass: {inference_time:.4f} ms")
    logger.info(f"Frames per second (FPS): {fps:.4f}")


if __name__ == '__main__':
    config = setting_config
    main(config)
