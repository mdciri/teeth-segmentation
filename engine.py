import os
import torch
import tqdm

def train_one_epoch(device, model, train_loader, optimizer, criterion, metric, epoch, num_epochs):

    model = model.to(device)
    model.train()

    len_dl = len(train_loader)
    epoch_loss, epoch_metric = 0, 0
    
    with tqdm.tqdm(train_loader, total=len_dl, unit="batch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as tepoch:
        for step, batch_data in enumerate(train_loader):

            ep_str = str(epoch).zfill(len(str(num_epochs)))
            tepoch.set_description(f"Epoch {ep_str}/{num_epochs}")

            inputs = batch_data["img"].to(device)
            targets = batch_data["seg"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            dice = metric(outputs, targets)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_metric += dice.item()

            tepoch.set_postfix(loss=epoch_loss/(step+1), dice=epoch_metric/(step+1))
    
    return epoch_loss/len_dl, epoch_metric/len_dl 

def test_one_epoch(device, model, test_loader, criterion, metric):

    len_dl = len(test_loader)
    epoch_loss, epoch_metric = 0, 0
    
    with torch.no_grad():
        model.eval()
        with tqdm.tqdm(test_loader, total=len_dl, unit="batch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as tepoch:
            for step, batch_data in enumerate(test_loader):

                tepoch.set_description(f"             ")
        
                inputs = batch_data["img"].to(device)
                targets = batch_data["seg"].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                dice = metric(outputs, targets)

                epoch_loss += loss.item()
                epoch_metric += dice.item()

                tepoch.set_postfix(val_loss=epoch_loss/(step+1), val_dice=epoch_metric/(step+1))

    return epoch_loss/len_dl, epoch_metric/len_dl
