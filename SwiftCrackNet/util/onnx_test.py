from torch.utils.data import DataLoader
from loader import *
from tqdm import tqdm
import os
import sys
import onnxruntime as ort
import numpy as np
from sklearn.metrics import confusion_matrix

# Append project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from configs.config_setting import setting_config
import warnings
warnings.filterwarnings("ignore")


def run_onnx_inference_with_loader(model_path, data_loader, criterion, logger, config, test_data_name=None):
    # Create an ONNX Runtime session with CPU support only
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


    # Get the input name for the ONNX model
    input_name = session.get_inputs()[0].name

    # Prepare for evaluation
    preds = []
    gts = []
    loss_list = []

    # Iterate through the test loader
    for i, data in enumerate(tqdm(data_loader)):
        img, msk = data

        # Convert the PyTorch tensor to NumPy array
        img_numpy = img.numpy().astype(np.float32)
        msk = msk.squeeze(1).cpu().detach().numpy()
        gts.append(msk)

        # Prepare the input dictionary
        onnx_inputs = {input_name: img_numpy}

        # Run inference
        onnx_output = session.run(None, onnx_inputs)[0]

        # Append predictions
        preds.append(onnx_output)

        # # Save images at intervals
        # if i % config.save_interval == 0:
        #     save_imgs(img, msk, onnx_output, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

    # Convert predictions and ground truths to NumPy arrays for evaluation
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    # Apply thresholding
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    # Calculate confusion matrix and metrics
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
    recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    # Log evaluation results
    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    log_info = f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                precision: {precision}, recall: {recall}, confusion_matrix: {confusion}'
    print(log_info)
    logger.info(log_info)

    return


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')

    global logger
    logger = get_logger('test', log_dir)
    log_config_info(config, logger)



    print('#----------Preparing dataset----------#')
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion


    # Specify the path to the ONNX model and input size
    model_path = "model_weights/model_cpu_int8.onnx"  # Use your actual ONNX model path

    # Run inference with actual data from test_loader and evaluate
    loss = run_onnx_inference_with_loader(model_path, test_loader, criterion, logger, config)



if __name__ == '__main__':
    config = setting_config
    main(config)
