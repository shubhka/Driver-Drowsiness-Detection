# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:18:26 2023

@author: Omen
"""

#%% Importing Libraries
import numpy as np
import torch
from torcheval.metrics.functional import binary_f1_score
import matplotlib.pyplot as plt

#%% Trainer
def trainer (model, criterion, optimizer, train_loader, val_loader, num_epochs=100, 
             device='cuda', min_batch_size=1000, max_epochs=10, dest='net.pth', plot_curves=True):
    min_val_loss = -np.inf
    
    train_losses = []
    train_aucs = []
    val_losses = []
    val_aucs = []
    
    counter = 0
    
    for e in range(num_epochs):
        train_loss = 0.0
        train_auc = 0.0
        i = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = torch.squeeze(model(inputs))
            labels = labels.type(torch.FloatTensor)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            with torch.no_grad():
                train_auc += binary_f1_score(outputs, labels)
            
            # Priniting every n mini batches
            if i % min_batch_size == (min_batch_size - 1):
                print('[Epoch %d, Batch %d] Loss: %.3f Accuracy: %.3f' %
                      (e+1, i+1, train_loss/min_batch_size, train_auc/min_batch_size))
                train_losses.append(train_loss/min_batch_size)
                train_aucs.append(train_auc/min_batch_size)
                train_loss = 0.0
                train_auc = 0.0
            
            i += 1
        
        with torch.no_grad():
            val_loss = 0.0
            val_auc = 0.0
            model.eval()
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = torch.squeeze(model(inputs))
                labels , outputs = labels.type(torch.FloatTensor),outputs.type(torch.FloatTensor)
                loss = criterion(outputs,labels)
                val_loss += loss.item()
                val_auc += binary_f1_score(outputs, labels)
            
        print(f'Epoch {e+1} Val Loss: {val_loss /len(val_loader)} \t\t Val Accuracy: {val_auc/len(val_loader)}')
        
        val_losses.append(val_loss / len(val_loader))
        val_aucs.append(val_auc / len(val_loader))
        
        # Saving the model
        if min_val_loss > val_loss:
            print(f'Validation Loss Decreased({min_val_loss / len(val_loader):.6f}---->{val_loss / len(val_loader):.6f}) \t Saving the Model')
            
            min_val_loss = val_loss
            counter = 0
            
            torch.save(model.state_dict(), dest)
            
        else:
            counter += 1
            
        if counter >= max_epochs:
            print(f'Training Stopped: \t\t Loss did not decrease for {max_epochs} epochs.')
            break
            
    if plot_curves :
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(np.arange(0, len(train_losses), int(len(train_losses)/(len(val_losses)))), val_losses, label='Val Loss')
        plt.xlabel('Mini-Batch')
        plt.ylabel('BCE Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.plot(train_aucs, label='Train Accuracy')
        plt.plot(np.arange(0, len(train_aucs), int(len(train_aucs)/(len(val_aucs)))), val_aucs, label='Val Accuracy')
        plt.xlabel('Mini-Batch')
        plt.ylabel('F1 Score')
        plt.legend()