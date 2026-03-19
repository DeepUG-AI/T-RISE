import os
import sys
import glob
import warnings

import torch
from torch.utils.data import DataLoader

from util.loader import *

from models.SwiftCrackNet_v5 import SwiftCrackNetV5
from util.engine import *
from utils import *
from configs.config_setting import setting_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def remove_old_best_model(checkpoint_dir, pattern, logger=None):

    old_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    for f in old_files:
        try:
            os.remove(f)
            msg = f'[Remove Old Best] deleted: {f}'
            print(msg)
            if logger is not None:
                logger.info(msg)
        except Exception as e:
            msg = f'[Remove Old Best Failed] file: {f}, error: {e}'
            print(msg)
            if logger is not None:
                logger.info(msg)


def save_best_model(model, checkpoint_dir, best_type, epoch, metric_value, val_loss, val_dice, logger=None):

    if best_type == 'loss':
        remove_old_best_model(checkpoint_dir, 'best_loss_epoch*_loss*.pth', logger)
        save_path = os.path.join(
            checkpoint_dir,
            f'best_loss_epoch{epoch}_loss{metric_value:.4f}.pth'
        )
        msg = (
            f'[Best Loss Updated] epoch={epoch}, '
            f'val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, '
            f'saved to {save_path}'
        )
    elif best_type == 'dice':
        remove_old_best_model(checkpoint_dir, 'best_dice_epoch*_dice*.pth', logger)
        save_path = os.path.join(
            checkpoint_dir,
            f'best_dice_epoch{epoch}_dice{metric_value:.4f}.pth'
        )
        msg = (
            f'[Best Dice Updated] epoch={epoch}, '
            f'val_dice={val_dice:.4f}, val_loss={val_loss:.4f}, '
            f'saved to {save_path}'
        )
    else:
        raise ValueError("best_type must be 'loss' or 'dice'")

    torch.save(model.module.state_dict(), save_path)
    print(msg)
    if logger is not None:
        logger.info(msg)

    return save_path


def find_existing_best_model(checkpoint_dir, best_type):
    """
    Search for the existing best model file
    """
    if best_type == 'loss':
        files = glob.glob(os.path.join(checkpoint_dir, 'best_loss_epoch*_loss*.pth'))
    elif best_type == 'dice':
        files = glob.glob(os.path.join(checkpoint_dir, 'best_dice_epoch*_dice*.pth'))
    else:
        return None

    if len(files) == 0:
        return None

    files.sort()
    return files[-1]


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)

    print('Load one training sample for testing...')
    img, mask = train_dataset[0]
    print('train image shape:', img.shape)
    print('train mask shape:', mask.shape)
    print('train image min/max:', img.min().item(), img.max().item())
    print('train mask unique:', torch.unique(mask))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False,
    )

    print('#----------Prepareing Models----------#')
    # model = SwiftCrackNet(**config.model_config)
    model = SwiftCrackNetV5(**config.model_config)
    # print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params:", total_params)
    print("trainable params:", trainable_params)

    model = model.cuda()
    cal_params_flops(model, config.input_size_h, logger)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    logger.info(model)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = float('inf')
    max_dice = 0.0
    start_epoch = 1
    min_loss_epoch = 1
    max_dice_epoch = 1

    best_loss_model_path = None
    best_dice_model_path = None

    # Early Stopping parameters
    early_stop_patience = 30
    early_stop_counter = 0

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'), weights_only=False)
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch = saved_epoch + 1

        min_loss = checkpoint.get('min_loss', float('inf'))
        min_loss_epoch = checkpoint.get('min_loss_epoch', 1)
        max_dice = checkpoint.get('max_dice', 0.0)
        max_dice_epoch = checkpoint.get('max_dice_epoch', 1)

        best_loss_model_path = checkpoint.get('best_loss_model_path', find_existing_best_model(checkpoint_dir, 'loss'))
        best_dice_model_path = checkpoint.get('best_dice_model_path', find_existing_best_model(checkpoint_dir, 'dice'))

        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        early_stop_patience = checkpoint.get('early_stop_patience', 30)

        last_val_loss = checkpoint.get('loss', None)
        last_val_dice = checkpoint.get('val_dice', None)

        log_info = (
            f'resuming model from {resume_model}. '
            f'resume_epoch: {saved_epoch}, '
            f'min_loss: {min_loss:.4f}, min_loss_epoch: {min_loss_epoch}, '
            f'max_dice: {max_dice:.4f}, max_dice_epoch: {max_dice_epoch}, '
            f'early_stop_counter: {early_stop_counter}/{early_stop_patience}, '
            f'last_val_loss: {last_val_loss}, last_val_dice: {last_val_dice}'
        )
        logger.info(log_info)
        print(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config
        )

        val_loss, val_dice = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        # 1. Save the best model according to val_loss
        if val_loss < min_loss:
            min_loss = val_loss
            min_loss_epoch = epoch
            best_loss_model_path = save_best_model(
                model=model,
                checkpoint_dir=checkpoint_dir,
                best_type='loss',
                epoch=epoch,
                metric_value=min_loss,
                val_loss=val_loss,
                val_dice=val_dice,
                logger=logger
            )

        # 2. Save the best model according to val_dice + use it as the criterion for Early Stopping
        if val_dice > max_dice:
            max_dice = val_dice
            max_dice_epoch = epoch
            early_stop_counter = 0

            best_dice_model_path = save_best_model(
                model=model,
                checkpoint_dir=checkpoint_dir,
                best_type='dice',
                epoch=epoch,
                metric_value=max_dice,
                val_loss=val_loss,
                val_dice=val_dice,
                logger=logger
            )

            msg = f'[Early Stopping] val_dice improved, counter reset to 0/{early_stop_patience}'
            print(msg)
            logger.info(msg)
        else:
            early_stop_counter += 1
            msg = (
                f'[Early Stopping] no improvement in val_dice for '
                f'{early_stop_counter}/{early_stop_patience} epochs '
                f'(current: {val_dice:.4f}, best: {max_dice:.4f})'
            )
            print(msg)
            logger.info(msg)

        # 3. Save latest
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_loss_epoch': min_loss_epoch,
                'max_dice': max_dice,
                'max_dice_epoch': max_dice_epoch,
                'loss': val_loss,
                'val_dice': val_dice,
                'best_loss_model_path': best_loss_model_path,
                'best_dice_model_path': best_dice_model_path,
                'early_stop_counter': early_stop_counter,
                'early_stop_patience': early_stop_patience,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            resume_model
        )

        # 4. Trigger early stopping
        if early_stop_counter >= early_stop_patience:
            stop_msg = (
                f'[Early Stopping Triggered] '
                f'val_dice has not improved for {early_stop_patience} consecutive epochs. '
                f'Training stopped at epoch {epoch}. '
                f'Best dice: {max_dice:.4f} at epoch {max_dice_epoch}.'
            )
            print(stop_msg)
            logger.info(stop_msg)
            break

    print('#----------Training Finished----------#')
    final_msg_1 = f'Best loss model: epoch={min_loss_epoch}, val_loss={min_loss:.4f}, path={best_loss_model_path}'
    final_msg_2 = f'Best dice model: epoch={max_dice_epoch}, val_dice={max_dice:.4f}, path={best_dice_model_path}'
    print(final_msg_1)
    print(final_msg_2)
    logger.info(final_msg_1)
    logger.info(final_msg_2)

    # By default, test best_dice.
    if best_dice_model_path is not None and os.path.exists(best_dice_model_path):
        print('#----------Testing best dice model----------#')
        logger.info('#----------Testing best dice model----------#')
        best_weight = torch.load(best_dice_model_path, map_location=torch.device('cpu'), weights_only=False)
        model.module.load_state_dict(best_weight, strict=False)
        test_one_epoch(test_loader, model, criterion, logger, config)

    if best_loss_model_path is not None and os.path.exists(best_loss_model_path):
        print('#----------Testing best loss model----------#')
        logger.info('#----------Testing best loss model----------#')
        best_weight = torch.load(best_loss_model_path, map_location=torch.device('cpu'), weights_only=False)
        model.module.load_state_dict(best_weight, strict=False)
        test_one_epoch(test_loader, model, criterion, logger, config)


if __name__ == '__main__':
    config = setting_config
    main(config)
