from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import torch
import math   

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = torch.log(1 - nn.Softmax(inputs)[targets])
        return loss.mean()
    
    
