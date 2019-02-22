#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """ Simple highway network
    """
    def __init__(self, f_embed):
        """ Initialize Highway Network
            @param embed_size (int): embedding size/dimension
        """
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(f_embed, f_embed)
        nn.init.xavier_normal_(self.W_proj.weight, gain=1)
        self.W_gate = nn.Linear(f_embed, f_embed)
        nn.init.xavier_normal_(self.W_gate.weight, gain=1)
        
    def forward(self, x_convout):
        """ Takes minibatch of x_convout and returns x_highway
            @param x_convout (Tensor): tensor of shape (batch_size, word_embed)
            
            @retuns x_highway (Tensor): tensor of shape (batch_size, word_embed)
        """
        x_proj = F.relu(self.W_proj(x_convout))
        x_gate = F.sigmoid(self.W_gate(x_convout))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1. - x_gate), x_convout)
        return x_highway

### END YOUR CODE 

