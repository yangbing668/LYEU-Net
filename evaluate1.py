import argparse
import os
from glob import glob
from time import time

import pandas as pd
from PIL import ImageOps, Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import valLoaderConfig
from dataset import CrackDataTest
from model.LYEU import UNet
from utils.utils import *


def score_per_sample(model, dataset, criteria, save_path, name=None, split="test"):
    results_test = {"img": [], "test_loss": [], "test_dice": [], "test_iou": [], "test_acc": [],
                    "test_pre": [], "test_rec": [], "test_f1": []}
    model = model.cpu()
    idx = 0
    with torch.inference_mode():
        bar = tqdm(range(len(dataset)))
        for i in bar:
            images, masks, path = dataset[i]
            images, masks = images.cpu(), masks.cpu()
            images = images.unsqueeze(dim=0)

            mask_pred = model(images)
            masks_2 = torch.sigmoid(mask_pred.cpu())

            loss = criteria(masks_2, masks)
            results_test['img'].append(path)
            results_test['test_loss'].append(loss.item())
            results_test['test_dice'].append(compute_dice2(masks_2, masks).item())
            results_test['test_iou'].append(get_IoU(masks_2, masks).item())
            results_test['test_acc'].append(accuracy(masks_2, masks).item())
            p, r, f, ba, gm = precision_recall_f1(masks_2, masks)
            results_test['test_pre'].append(p.item())
            results_test['test_rec'].append(r.item())
            results_test['test_f1'].append(f.item())
            bar.set_description(f"Saving Test Results")
            idx += 1

    data_frame = pd.DataFrame(data=results_test)
    data_frame.to_csv(f'{save_path}/results_per_image_{name}_{split}.csv')
    return


def plot_test(model, dataset, criteria, save_path, save_plots, name=None, split="test"):
    idx = 0
    if not os.path.exists(save_plots):
        os.makedirs(save_plots)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results_test = {"img": [], "test_loss": [], "test_dice": [], "test_iou": [], "test_acc": [],
                    "test_pre": [], "test_rec": [], "test_f1": [], "test_ba": [], "test_gm": []}
    model = model  # .cpu()
    model.eval()
    with torch.no_grad():
        bar = tqdm(range(len(dataset)))
        for i in bar:
            images, masks, path = dataset[i]
            images, masks = images.to('cuda'), masks.to('cuda')
            images = images.unsqueeze(dim=0)

            mask_pred = model(images)
            masks_2 = torch.sigmoid(mask_pred.cpu())
            masks_2, masks = masks_2.cpu(), masks.cpu()

            loss = criteria(masks_2, masks)
            results_test['img'].append(path)
            results_test['test_loss'].append(loss.item())
            results_test['test_dice'].append(compute_dice2(masks_2, masks).item())
            results_test['test_iou'].append(get_IoU(masks_2, masks))
            results_test['test_acc'].append(accuracy(masks_2, masks).item())
            p, r, f, ba, gm = precision_recall_f1(masks_2, masks)
            results_test['test_pre'].append(p.item())
            results_test['test_rec'].append(r.item())
            results_test['test_f1'].append(f.item())
            results_test['test_ba'].append(ba.item())
            results_test['test_gm'].append(gm.item())

            # print(images.shape, masks.shape, masks_2.shape)
            masks *= 255.
            masks_2 = masks_2.squeeze(dim=0)
            masks_2 = masks_2.to(torch.float)
            masks_2 *= 255.
            image = transforms.ToPILImage()(images[0])
            gt = transforms.ToPILImage()(masks.byte().cpu())
            pred = transforms.ToPILImage()(masks_2.byte().cpu())

            image = ImageOps.expand(image, border=5, fill='white')
            gt = ImageOps.expand(gt, border=5, fill='white')
            pred = ImageOps.expand(pred, border=5, fill='white')

            (img_width, img_height) = image.size
            (gt_width, gt_height) = gt.size
            (pred_width, pred_height) = pred.size

            imgName = path.split('\\')[-1][:-4]
            final_width, final_height = (img_width + gt_width + pred_width), max(img_height,
                                                                                 max(gt_height, pred_height))
            result = Image.new('RGB', (final_width, final_height))
            result.paste(im=image, box=(0, 0))
            result.paste(im=gt, box=(img_width, 0))
            result.paste(im=pred, box=(img_width + gt_width, 0))

            result.save(f"{save_plots}/{imgName}_res.png")
            bar.set_description(f"Saving Test Results")
            idx += 1
    print("Saving Test Results")
    data_frame = pd.DataFrame(data=results_test)
    data_frame.to_csv(f'{save_path}/logs_{name}_{split}.csv')
    return


def score(model, criteria, loader):
    model.eval()
    val_logs = init_log()
    # Batch size should be 1
    bar = tqdm(loader, dynamic_ncols=True)
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for idx, data in enumerate(bar):
            imgs, masks = data[0], data[1]
            # imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            output = model(imgs)
            output = output.squeeze(1)
            op_preds = torch.sigmoid(output)
            masks = masks.squeeze(1)
            loss = criteria(op_preds, masks)

            batch_size = imgs.size(0)
            val_logs['loss'].update(loss.item(), batch_size)
            val_logs['time'].update(time() - start)
            val_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
            val_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
            val_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
            p, r, f, ba, gm = precision_recall_f1(op_preds, masks)
            val_logs['precision'].update(p.item(), batch_size)
            val_logs['recall'].update(r.item(), batch_size)
            val_logs['f1'].update(f.item(), batch_size)
            print(f"Loss: {val_logs['loss'].avg:.4f}"
                  f" Dice: {val_logs['dice'].avg:.4f} IoU: {val_logs['iou'].avg:.4f}"
                  f" Accuracy: {val_logs['acc'].avg:.4f} Precision: {val_logs['precision'].avg:.4f}"
                  f" Recall: {val_logs['recall'].avg:.4f} F1: {val_logs['f1'].avg:.4f}")
    return val_logs


def getTestDataLoader(dfTest, **kwargs):
    dataTest = CrackDataTest(dfTest,
                             img_transforms=kwargs['val_data']['transforms'],
                             mask_transform=kwargs['val_data']['transforms'],
                             aux_transforms=None)
    testLoader = DataLoader(dataTest,
                            batch_size=kwargs['val_data']['batch_size'],
                            shuffle=kwargs['val_data']['shuffle'],
                            pin_memory=torch.cuda.is_available(),
                            num_workers=kwargs['val_data']['num_workers'])

    return testLoader, dataTest


def buildModel(modelPath=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model = model.to(device)
    print("loading best model...", modelPath)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    return model


def buildDataset(imgs_path, masks_path):
    data = {
        'images': sorted(glob(imgs_path + "/*.jpg")),
        'masks': sorted(glob(masks_path + "/*.jpg"))
    }

    # test to see if there are images coresponding to masks
    for img_path, mask_path in zip(data['images'], data['masks']):
        assert img_path[:-4].split('\\')[-1] == mask_path[:-4].split('\\')[-1]

    df = pd.DataFrame(data)
    testLoader, dataTest = getTestDataLoader(df, val_data=valLoaderConfig)

    return testLoader, dataTest


if __name__ == '__main__':
    datasetPaths = ['AigleRN-TRIMM', 'GAPs384', 'Crack500', 'ShadowCrack', 'DeepCrack']  #
    datasetPaths = ['ShadowCrack']
    for path in datasetPaths:
        parser = argparse.ArgumentParser()
        parser.add_argument("--path_images", type=str, default=f'Datasets/{path}/test',
                            help="Enter path to images folder.")
        parser.add_argument("--path_masks", type=str, default=f'Datasets/{path}/test_lab',
                            help="Enter path to masks folder.")
        parser.add_argument("--result_path", type=str, default='output1/B', help="Path to results")
        parser.add_argument("--plot_path", type=str, default=f'output1/B/{path}Plot',
                            help="Path where plot or predictions would be saved")

        args = parser.parse_args()

        image_path = args.path_images
        masks_path = args.path_masks
        plot_path = args.plot_path
        out_path = args.result_path
        name = image_path.split('/')[1]
        model_path = f'output1/B/models/model_best_{name}.pth'

        testLoader, testDataset = buildDataset(image_path, masks_path)
        model = buildModel(model_path)
        criteria = BceDiceLoss()

        # testLog = score(model, criteria, testLoader)
        plot_test(model, testDataset, criteria, out_path, plot_path, name=name)
        # score_per_sample(model, testDataset, criteria, out_path, name=name)
