import torch
import os
from torch import nn
import sys
sys.path.append('./')
from GPTModel import GPTModel
#from Generatenexttokens import Generatetext
from config.cfg import cfg


class calc_loss(nn.Module):
    def __init__(self):
        super().__init__()
    

    def calculate_loss(self,input_batch,target_batch,model,device):
        input_batch,target_batch=input_batch.to(device),target_batch.to(device)
        logits=model(input_batch)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
        return loss
    
    def calculate_loss_loader(self,data_loader,model,device,num_batches=None):
        total_loss=0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches=len(data_loader)
        else:
            num_batches=min(num_batches,len(data_loader))
        
        for i,(input_batch,target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss=self.calculate_loss(input_batch,target_batch,model,device)  # Use 'calc_loss' to reference the class
                total_loss+=loss.item()  # Convert loss to a scalar value
            else:
                break
        
        return total_loss/num_batches
