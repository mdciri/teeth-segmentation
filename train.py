import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

from loader import TuftsDataset
from augmentation import get_transorms
from metric import MeanDiceScore
from loss import MeanDiceLoss
from engine import train_one_epoch, test_one_epoch

def train(device, model, train_loader, valid_loader, optimizer, criterion, metric, num_epochs, max_patience=10, checkpoint_dir="./checkpoint"):

    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, "model.pt")

    history = {
        "train": {
            "loss": [], "dice": []
        },
        "valid": {
            "loss": [], "dice": []
        },
    }

    dict_to_save = {
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "history": None
    }
    best_loss = torch.inf
    patience = 0

    for epoch in range(1, num_epochs+1):
        
        train_loss, train_dice = train_one_epoch(device, model, train_loader, optimizer, criterion, metric, epoch, num_epochs)
        valid_loss, valid_dice = test_one_epoch(device, model, valid_loader, criterion, metric)

        # update history
        history["train"]["loss"].append(train_loss)
        history["train"]["dice"].append(train_dice)
        history["valid"]["loss"].append(valid_loss)
        history["valid"]["dice"].append(valid_dice)

        # checkpoint 
        if valid_loss < best_loss:
            
            # reset patience
            patience = 0

            # save the new best model and optimizer parameters
            dict_to_save["epoch"] = epoch
            dict_to_save["model_state_dict"] = model.state_dict()
            dict_to_save["optimizer_state_dict"] = optimizer.state_dict()
            dict_to_save["history"] = history
            torch.save(dict_to_save, model_path)

            # update best loss
            print(f"@ epoch {epoch} val_loss decreased from {best_loss:.4f} to {valid_loss:.4f}. Model saved in {model_path}.\n")
            best_loss = valid_loss
        else:
            patience += 1
            print(f"@ epoch {epoch} val_loss did not decrease from {best_loss:.4f}. {patience} epochs of patience.\n")

            if patience == max_patience:
                print(f"val_loss did not decrease for {max_patience} consecutive epochs.")
                print("Model training has stopped!")
                break

    dict_to_save["history"] = history
    torch.save(dict_to_save, model_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Teeth model training script", add_help=False)
    parser.add_argument("-md", "--model_dir", type=str, help="model directory")
    parser.add_argument("-d", "--device", default="mps", type=str, help="GPU-ID position")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="GPU-ID position")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("-lr", "--learning_rate", default=1.e-4, type=float, help="learning rate")
    parser.add_argument("-ne", "--num_epochs", default=100, type=int, help="number of epochs")
    args = parser.parse_args()

    # load data file
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
    bright_range = (0.8, 1.2)
    rotation_range = (-np.pi/36, np.pi/36)
    scale_range = (0.8, 1.2)
    train_transform = get_transorms(
        new_shape, 
        bright_range=bright_range, 
        rotation_range=rotation_range, 
        scale_range=scale_range, 
        num_classes=num_classes
    )
    valid_transform = get_transorms(
        new_shape, 
        num_classes=num_classes
    )

    train_ds = TuftsDataset(jfile["train"], masking=True, transform=train_transform)
    valid_ds = TuftsDataset(jfile["valid"], masking=True, transform=valid_transform)

    # create dataloaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)

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

    # optimizer
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        lr = args.learning_rate
    )

    # loss function and dice metric
    metric = MeanDiceScore(softmax=True, weights=class_weights)
    criterion = MeanDiceLoss(softmax=True, weights=class_weights)

    # train model
    print(f"Training a model to segment {num_classes} classes:\n{class_names}\n")

    train(
        device, 
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        criterion, 
        metric, 
        args.num_epochs, 
        max_patience=10, 
        checkpoint_dir=args.model_dir
    )   