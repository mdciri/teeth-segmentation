import os
import argparse
import json
import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

from loader import TuftsDataset
from augmentation import get_transorms
from metric import MeanDiceScore
from loss import MeanDiceLoss

def evaluate(device, model, data_loader, criterion, metric):

    len_dl = len(data_loader)
    Loss, Dice = [], []
    
    with torch.no_grad():
        model.eval()

        for batch_data in tqdm.tqdm(data_loader, total=len_dl):
        
            inputs = batch_data["img"].to(device)
            targets = batch_data["seg"].to(device)
            
            outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            loss = criterion(outputs, targets)
            dice = metric(outputs, targets)
        
            Loss.append(loss.cpu().numpy())
            Dice.append(dice.cpu().numpy())

            return Loss, Dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script", add_help=False)
    parser.add_argument("-md", "--model_dir", type=str, help="model directory")
    parser.add_argument("-d", "--device", default="mps", type=str, help="device type")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="GPU-ID position")
    parser.add_argument("-bs", "--batch_size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.pt")
    assert os.path.exists(model_path) == True

    jfile = json.load(open("data.json"))
    class_names = jfile["class_names"]
    num_classes = len(class_names)
    class_weights = torch.tensor(list(jfile["class_weights"].values()), dtype=torch.float32)

    # set device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = args.device + ":" + str(args.gpu_id)
    else:
        device = "cpu"
    print(f"Using {device} device.")

    # create datasets
    new_shape = (256, 512)

    valid_transform = get_transorms(
        new_shape, 
        num_classes=num_classes
    )

    train_ds = TuftsDataset(jfile["train"], masking=True, transform=valid_transform)
    valid_ds = TuftsDataset(jfile["valid"], masking=True, transform=valid_transform)
    test_ds = TuftsDataset(jfile["test"], masking=True, transform=valid_transform)

    # create dataloaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # build model
    model = UNet(
        spatial_dims = 2,
        in_channels = 1,
        out_channels = num_classes,
        channels = (32, 64, 128, 256, 512),
        strides = (2, 2, 2, 2),
        num_res_units = 2,
        norm = Norm.BATCH,
        act = Act.LEAKYRELU
    ).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded.")

    # loss function and dice metric
    metric = MeanDiceScore(softmax=False, weights=None, epsilon=0.)
    criterion = MeanDiceLoss(softmax=False, weights=class_weights)

    # evaluate model
    print(f"Evaluating a model over the training, validation, and test dataset:\n")

    train_loss, train_dice = evaluate(device, model, train_ds, criterion, metric)
    valid_loss, valid_dice = evaluate(device, model, valid_ds, criterion, metric)
    test_loss, test_dice = evaluate(device, model, test_ds, criterion, metric)

    print(len(train_loss), len(train_dice))

    print(f"Training: {np.mean(train_loss, 0):.4f} loss, {np.nanmean(train_dice, 0):.4f} dice.")
    print(f"Validation: {np.mean(valid_loss, 0):.4f} loss, {np.nanmean(valid_dice, 0):.4f} dice.")
    print(f"Test: {np.mean(test_loss, 0):.4f} loss, {np.nanmean(test_dice, 0):.4f} dice.")

    out = {
        "file_name": [],
        "set_name": ["train"]*len(jfile["train"]) + ["valid"]*len(jfile["valid"]) + ["test"]*len(jfile["test"]),
        "loss": train_loss + valid_loss + test_loss,
        "dice": train_dice + valid_dice + test_dice
    }

    for set_name in ["train", "valid", "test"]:
        for i, data in enumerate(jfile[set_name]):
            out["file_name"].append(data["img"])

    df = pd.DataFrame(out)
    df.sort_values(by=["file_name"])
    df.to_csv("evaluation_results.csv")