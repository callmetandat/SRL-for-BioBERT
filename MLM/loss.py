from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import math   
class ModifiedLoss(_Loss):
    def __init__(self, alpha=1.0, name='Cross Entropy Loss', prob=None):
        super().__init__()
        
        self.alpha = alpha
        self.name = name
        self.ignore_index = -1
        self.prob = prob
    def forward(self, inp, target, prob = None):
        """
        This is the standard cross entropy loss as defined in pytorch.
        This loss should be used for single sentence or sentence pair classification tasks.

        To use this loss for training, set ``loss_type`` : **CrossEntropyLoss** in task file
        """
        # loss = -log(p) = -log(softmax(inp)[target])
        loss = math.log(1 - nn.Softmax(inp)[target])
        return loss