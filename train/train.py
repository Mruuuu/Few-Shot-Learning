'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-24 16:17:50
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-04-09 13:23:12
FilePath: /mru/Few-Shot-Learning/train/train.py
Description: 

'''
# package
from tqdm import tqdm
import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------- train ---------------------
def train_episodic(model, epoch, total_epoch, data_loader, optimizer, criterion, device):

    # train
    model.train()

    # params
    epoch_loss, epoch_correct, total = 0, 0, 0

    with tqdm(data_loader, ncols=0, leave=False) as pbar:

        pbar.set_description(f"Epoch {epoch:03d}/{total_epoch:3d}")

        for (support_images, support_labels, query_images, query_labels, _) in pbar:

            # put on device (usually GPU)
            support_images = support_images.to(device, dtype=torch.float)
            query_images = query_images.to(device, dtype=torch.float)
            support_labels = support_labels.to(device, dtype=torch.long)
            query_labels = query_labels.to(device, dtype=torch.long)

            # predict
            model.compute_prototypes(support_images, support_labels)
            pred = model(query_images)

            # loss
            loss = criterion(pred, query_labels)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(pred.data, 1)
            correct_pred = (predicted == query_labels).sum().item()

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss & acc
            epoch_loss += loss.detach().cpu().numpy()
            epoch_correct += correct_pred
            total += query_labels.size(0)
    
    accuracy = epoch_correct / total

    return epoch_loss, accuracy


# --------------------- train ---------------------
def train_classical(model, epoch, total_epoch, data_loader, optimizer, criterion, device):

    # train
    model.train()

    # params
    epoch_loss, epoch_correct, total = 0, 0, 0

    with tqdm(data_loader, ncols=0, leave=False) as pbar:

        pbar.set_description(f"Epoch {epoch:03d}/{total_epoch:3d}")

        for (imgs, labels) in pbar:
                
            # put on device (usually GPU)
            imgs = imgs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # predict
            pred = model(imgs)

            # loss
            loss = criterion(pred, labels)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(pred.data, 1)
            correct_pred = (predicted == labels).sum().item()

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss & acc
            epoch_loss += loss.detach().cpu().numpy()
            epoch_correct += correct_pred
            total += labels.size(0)
    
    accuracy = epoch_correct / total

    return epoch_loss, accuracy
    

# --------------------- evaluate ---------------------
def evaluate(model, data_loader, device):

    # param
    correct = 0
    total = 0
    
    # switch to mode evaluation
    model.eval()

    with torch.no_grad():

        with tqdm(data_loader, ncols=0, leave=False) as pbar:

            pbar.set_description("Validation...")
            
            for (support_images, support_labels, query_images, query_labels, _) in pbar:
                
                # put on device (usually GPU)
                support_images = support_images.to(device, dtype=torch.float)
                query_images = query_images.to(device, dtype=torch.float)
                support_labels = support_labels.to(device, dtype=torch.long)
                query_labels = query_labels.to(device, dtype=torch.long)

                # predict
                model.compute_prototypes(support_images, support_labels)
                pred = model(query_images)

                # the label with the highest energy will be our prediction
                _, predicted = torch.max(pred.data, 1)
                correct_pred = (predicted == query_labels).sum().item()

                # accumulating correct term
                correct += correct_pred
                total += query_labels.size(0)
    
    accuracy = correct / total

    return accuracy


# --------------------- predict ---------------------
def predict(model, data_loader, device):

    # param
    pred_arr = []
    
    # switch to mode evaluation
    model.eval()

    with torch.no_grad():
            
        for (support_images, support_labels, query_images) in tqdm(data_loader):
            
            # squeezing data
            support_images = support_images.squeeze(0)
            support_labels = support_labels.squeeze(0)
            query_images = query_images.squeeze(0)

            # put on device (usually GPU)
            support_images = support_images.to(device, dtype=torch.float)
            query_images = query_images.to(device, dtype=torch.float)
            support_labels = support_labels.to(device, dtype=torch.long)

            # predict
            model.compute_prototypes(support_images, support_labels)
            pred = model(query_images)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(pred.data, 1)
            pred_arr += [i.item() for i in predicted]
    
    pred_data = {"Category": pred_arr}
    df_pred = pd.DataFrame(pred_data)

    return df_pred