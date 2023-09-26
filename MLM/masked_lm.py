import copy
import torch
import torch.nn as nn
import logging
import numpy as np
from models import dropout 
from MLM.loss import ModifiedLoss
from transformers import AutoModelForMaskedLM, AutoTokenizer 

class modified_MLM(nn.Module):
    def __init__(self,):
        super(modified_MLM, self).__init__()
        self.dropout = dropout.Dropout(0.1)
        self.loss = ModifiedLoss()
        self.shared_model = AutoModelForMaskedLM.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    
    def forward(self, input_id):
        
        for i in range(len(outputs.logits)):
            losses = []
            logits = []
            
            labels = [-100] * len(input_id)
            
            input_id[i] = self.tokenizer.mask_token_id
            labels[i] = input_id[i]
            
            outputs = self.shared_model(**input_id)
            logit =  outputs.logits
            loss = self.loss(logit, labels)
            
            losses.append(loss)
            logits.append(logit)
        
        
    
    