import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    model.train()
    loss_list = []

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Train Epoch [{epoch}/{config.epochs}]",
        leave=True,
        ncols=120
    )

    for iter, data in pbar:
        optimizer.zero_grad()

        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        avg_loss = np.mean(loss_list)

        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{now_lr:.6f}")

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {avg_loss:.4f}, lr: {now_lr}'
            logger.info(log_info)

    scheduler.step()

    final_log = f'train: epoch {epoch}, mean_loss: {np.mean(loss_list):.4f}, lr: {optimizer.state_dict()["param_groups"][0]["lr"]}'
    print(final_log)
    logger.info(final_log)


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    '''
    validate model for one epoch
    return:
        avg_loss, val_dice
    '''
    model.eval()
    preds = []
    gts = []
    loss_list = []

    pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"Val   Epoch [{epoch}/{config.epochs}]",
        leave=True,
        ncols=120
    )

    with torch.no_grad():
        for iter, data in pbar:
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            avg_loss = np.mean(loss_list)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

            gts.append(msk.squeeze(1).cpu().detach().numpy())

            if isinstance(out, tuple):
                out = out[0]

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
    recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    avg_loss = np.mean(loss_list)
    val_dice = f1_or_dsc

    if epoch % config.val_interval == 0:
        log_info = (
            f'val epoch: {epoch}, loss: {avg_loss:.4f}, '
            f'miou: {miou:.6f}, dice: {val_dice:.6f}, accuracy: {accuracy:.6f}, '
            f'precision: {precision:.6f}, recall: {recall:.6f}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)
    else:
        log_info = f'val epoch: {epoch}, loss: {avg_loss:.4f}, dice: {val_dice:.6f}'
        print(log_info)
        logger.info(log_info)

    return avg_loss, val_dice


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    '''
    test model
    '''
    model.eval()
    preds = []
    gts = []
    loss_list = []

    pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Test",
        leave=True,
        ncols=120
    )

    with torch.no_grad():
        for i, data in pbar:
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            avg_loss = np.mean(loss_list)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            if isinstance(out, tuple):
                out = out[0]

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            # if i % config.save_interval == 0:
            #     save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)

        log_info = (
            f'test of best model, loss: {np.mean(loss_list):.4f}, '
            f'miou: {miou:.6f}, f1_or_dsc: {f1_or_dsc:.6f}, accuracy: {accuracy:.6f}, '
            f'precision: {precision:.6f}, recall: {recall:.6f}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
