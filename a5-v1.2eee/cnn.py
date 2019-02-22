#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """ 1D Convolutional Network
    """
    def __init__(self, filters, char_embed, kernel=5):
        """ Initialize CNN
            @param filters (int): f_embed (final embedding size), or e_word in pdf
            @param char_embed (int): char embedding size
        """
        super(CNN, self).__init__()
        self.W_cnn = nn.Conv1d(char_embed, filters, kernel_size=kernel, stride=1)
        
    def forward(self, x_reshaped):
        """ Takes minibatch of x_reshaped (batch_size, char_embed, word_len) and returns x_convout (batch_size, f_embed)
        	Note: filters = f_embed (final embedding size in pdf, i.e. 'e_word')
            @param x_reshaped (Tensor): tensor of shape (batch_size, char_embed, word_len
                                        word_len is m_word in pdf handout
            
            @returns x_convout (Tensor): tensor of shape (batch_size, f_embed)
                                        NOTE: f_embed \neq char_embed
                                        e_char -> char_embed, f_embed -> word_embed (final embedding size)
        """
        x_conv = self.W_cnn(x_reshaped)
        x_convout = F.relu(torch.max(x_conv, dim=2)[0])
        return x_convout


### END YOUR CODE

