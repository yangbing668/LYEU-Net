from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

from tqdm import tqdm
import gc
import argparse
from glob import glob
import pandas as pd
import os
import torch.optim as optim
from dataset import CrackData
from model.LYEU import UNet
from utils.callbacks import CallBacks
from utils.utils import *
from config import trainLoaderConfig, valLoaderConfig
import warnings

warnings.filterwarnings("ignore")


def train_step(model, optim, criteria, loader, epoch, max_epochs, scaler, accumulation_steps):
    model.train()
    train_logs = init_log()
    bar = tqdm(loader, dynamic_ncols=True)
    torch.cuda.empty_cache()
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.enable_grad():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                output = model(imgs)
                output = output.squeeze(1)
                op_preds = torch.sigmoid(output)
                masks = masks.squeeze(1)
                # loss = criteria(op_preds, masks)
                loss = criteria(op_preds, masks) / accumulation_steps

            batch_size = imgs.size(0)

            scaler.scale(loss).backward()

            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(loader)):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            train_logs['loss'].update(round(loss.item(), 4), batch_size)
            train_logs['time'].update(round(time.time() - start, 4), batch_size)
            train_logs['dice'].update(round(compute_dice2(op_preds, masks).item(), 4), batch_size)
            train_logs['iou'].update(round(get_IoU(op_preds, masks), 4), batch_size)
            train_logs['acc'].update(round(accuracy(op_preds, masks).item(), 4), batch_size)
            p, r, f, ba, gm = precision_recall_f1(op_preds, masks)
            train_logs['precision'].update(round(p.item(), 4), batch_size)
            train_logs['recall'].update(round(r.item(), 4), batch_size)
            train_logs['f1'].update(round(f.item(), 4), batch_size)

            bar.set_description(f"Training Epoch: [{epoch}/{max_epochs}] Loss: {train_logs['loss'].avg:.4f}"
                                f" Dice: {train_logs['dice'].avg:.4f} IoU: {train_logs['iou'].avg:.4f}"
                                f" Accuracy: {train_logs['acc'].avg:.4f} Precision: {train_logs['precision'].avg:.4f}"
                                f" Recall: {train_logs['recall'].avg:.4f} F1: {train_logs['f1'].avg:.4f}")
            del imgs
            del masks
            gc.collect()

    return train_logs


def val(model, criteria, loader, epoch, epochs, split='Validation'):
    model.eval()
    val_logs = init_log()
    bar = tqdm(loader, dynamic_ncols=True)
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            output = model(imgs)
            output = output.squeeze(1)
            op_preds = torch.sigmoid(output)
            masks = masks.squeeze(1)
            loss = criteria(op_preds, masks)

            batch_size = imgs.size(0)
            val_logs['loss'].update(round(loss.item(), 4), batch_size)
            val_logs['time'].update(round(time.time() - start, 4), batch_size)
            val_logs['dice'].update(round(compute_dice2(op_preds, masks).item(), 4), batch_size)
            val_logs['iou'].update(round(get_IoU(op_preds, masks), 4), batch_size)
            val_logs['acc'].update(round(accuracy(op_preds, masks).item(), 4), batch_size)
            p, r, f, ba, gm = precision_recall_f1(op_preds, masks)
            val_logs['precision'].update(round(p.item(), 4), batch_size)
            val_logs['recall'].update(round(r.item(), 4), batch_size)
            val_logs['f1'].update(round(f.item(), 4), batch_size)

            bar.set_description(f"{split} Epoch: [{epoch}/{epochs}] Loss: {val_logs['loss'].avg:.4f}"
                                f" Dice: {val_logs['dice'].avg:.4f} IoU: {val_logs['iou'].avg:.4f}"
                                f" Accuracy: {val_logs['acc'].avg:.4f} Precision: {val_logs['precision'].avg:.4f}"
                                f" Recall: {val_logs['recall'].avg:.4f} F1: {val_logs['f1'].avg:.4f}")

    return val_logs


def getDataLoaders(dfTrain, dfVal, **kwargs):
    dataTrain = CrackData(dfTrain,
                          img_transforms=kwargs['training_data']['transforms'],
                          mask_transform=kwargs['training_data']['transforms'],
                          aux_transforms=None)

    trainLoader = DataLoader(dataTrain,
                             batch_size=kwargs['training_data']['batch_size'],
                             shuffle=kwargs['training_data']['shuffle'],
                             pin_memory=torch.cuda.is_available(),
                             num_workers=kwargs['training_data']['num_workers'])

    dataVal = CrackData(dfVal,
                        img_transforms=kwargs['val_data']['transforms'],
                        mask_transform=kwargs['val_data']['transforms'],
                        aux_transforms=None)
    valLoader = DataLoader(dataVal,
                           batch_size=kwargs['val_data']['batch_size'],
                           shuffle=kwargs['val_data']['shuffle'],
                           pin_memory=torch.cuda.is_available(),
                           num_workers=kwargs['val_data']['num_workers'])

    return trainLoader, valLoader


def buildModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model = model.to(device)
    return model


def buildDataset(train_image_path, train_masks_path, val_image_path, val_masks_path):
    train_data = {
        'images': sorted(glob(train_image_path + "/*.jpg")),
        'masks': sorted(glob(train_masks_path + "/*.jpg"))
    }
    val_data = {
        'images': sorted(glob(val_image_path + "/*.jpg")),
        'masks': sorted(glob(val_masks_path + "/*.jpg"))
    }

    # test to see if there are images coresponding to masks
    for img_path, mask_path in zip(train_data['images'], train_data['masks']):
        assert img_path[:-4].split('\\')[-1] == mask_path[:-4].split('\\')[-1]

    for img_path, mask_path in zip(val_data['images'], val_data['masks']):
        assert img_path[:-4].split('\\')[-1] == mask_path[:-4].split('\\')[-1]

    dfTrain, dfVal = pd.DataFrame(train_data), pd.DataFrame(val_data)
    trainLoader, valLoader = getDataLoaders(dfTrain,
                                            dfVal,
                                            training_data=trainLoaderConfig,
                                            val_data=valLoaderConfig)

    return trainLoader, valLoader


if __name__ == "__main__":
    datasetPaths = ['ShadowCrack'] # 'AigleRN-TRIMM', 'GAPs384', 'Crack500', 'DeepCrack',
    for path in datasetPaths:
        print(f"Training on {path}")
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_path_images", type=str, default=f'Datasets/{path}/train',
                            help="Enter path to images folder.")
        parser.add_argument("--train_path_masks", type=str, default=f'Datasets/{path}/train_lab',
                            help="Enter path to masks folder.")
        parser.add_argument("--val_path_images", type=str, default=f'Datasets/{path}/valid',
                            help="Enter path to images folder.")
        parser.add_argument("--val_path_masks", type=str, default=f'Datasets/{path}/valid_lab',
                            help="Enter path to masks folder.")
        parser.add_argument("--out_path", type=str, default='output1', help="Output path, model saving path.")
        args = parser.parse_args()
        # 0.7173 0.7873 0.7307 Crack500
        # 0.6859 0.7235 0.6933 AigleRN-TRIMM
        # 0.8439 0.8840 0.8451 DeepCrack
        # 0.5793 0.5724 0.5447 GAPs384
        train_image_path = args.train_path_images
        train_masks_path = args.train_path_masks
        val_image_path = args.val_path_images
        val_masks_path = args.val_path_masks
        out_path = args.out_path
        name = train_image_path.split('/')[1]

        trainLoader, valLoader = buildDataset(train_image_path, train_masks_path, val_image_path, val_masks_path)

        model = buildModel()

        lr = 0.01
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        criteria = myLoss()  # DiceLoss() TverskyLoss() BceDiceLoss()
        epochs = 120
        best_iou = 0.
        accumulation_steps = 4
        scaler = GradScaler()
        iteration = 0

        cb = CallBacks(best_iou, out_path + "/models")

        results = {"train_loss": [], "train_dice": [], "train_iou": [], 'train_acc': [],
                   "train_pre": [], "train_rec": [], "train_f1": [],
                   "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [],
                   "val_pre": [], "val_rec": [], "val_f1": []}

        save_path = out_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            model_dir = out_path + "/models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = out_path + "/models/base.pth"
            if os.path.exists(model_path):
                print("loading base model")
                model.load_state_dict(torch.load(model_path, map_location=device))

        earlyStopEpoch = 20

        try:
            for epoch in range(1, epochs + 1):
                iteration = epoch
                train_logs = train_step(model, optimizer, criteria, trainLoader, epoch, epochs, scaler,
                                        accumulation_steps)
                # print("\n")
                val_logs = val(model, criteria, valLoader, epoch, epochs)

                scheduler.step(val_logs['iou'].avg)

                results['train_loss'].append(train_logs['loss'].avg)
                results['train_dice'].append(train_logs['dice'].avg)
                results['train_iou'].append(train_logs['iou'].avg)
                results['train_acc'].append(train_logs['acc'].avg)
                results['train_pre'].append(train_logs['precision'].avg)
                results['train_rec'].append(train_logs['recall'].avg)
                results['train_f1'].append(train_logs['f1'].avg)
                results['val_loss'].append(val_logs['loss'].avg)
                results['val_dice'].append(val_logs['dice'].avg)
                results['val_iou'].append(val_logs['iou'].avg)
                results['val_acc'].append(val_logs['acc'].avg)
                results['val_pre'].append(val_logs['precision'].avg)
                results['val_rec'].append(val_logs['recall'].avg)
                results['val_f1'].append(val_logs['f1'].avg)

                data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
                data_frame.to_csv(f'{save_path}/logs_{name}.csv', index_label='epoch')

                # print("\n")

                cb.saveBestModel(val_logs['iou'].avg, model, name)
                cb.earlyStoping(val_logs['iou'].avg, earlyStopEpoch)

        except KeyboardInterrupt:
            data_frame = pd.DataFrame(data=results, index=range(1, iteration + 1))
            data_frame.to_csv(f'{save_path}/logs_{name}.csv', index_label='epoch')
            val_logs = val(model, criteria, valLoader, 1, 1)
            cb.saveBestModel(val_logs['iou'].avg, model)
