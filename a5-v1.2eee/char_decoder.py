#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        self.embed_size = char_embedding_size
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        pad_token_idx = target_vocab.char2id['<pad>']
        self.pad_token_idx = pad_token_idx

        self.charDecoder = nn.LSTM(input_size=self.embed_size, hidden_size=hidden_size, bias=True, bidirectional=False)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        nn.init.xavier_normal_(self.char_output_projection.weight, gain=1)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), embedding_dim=self.embed_size, padding_idx=pad_token_idx)
        self.loss_ = nn.CrossEntropyLoss(ignore_index=pad_token_idx, reduction='sum')
        self.st_softmax = nn.Softmax(dim=-1)

        ### END YOUR CODE


    
    def forward(self, input_, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        #sequence_length = input_.shape[0]

        X = self.decoderCharEmb(input_)
        dec_hidden, (last_hidden, last_cell) = self.charDecoder(X, dec_hidden)
        s_t = self.char_output_projection(dec_hidden)
        return s_t, (last_hidden, last_cell)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        char_sequence = char_sequence.contiguous()
        s_t, (last_hidden, last_cell) = self.forward(char_sequence[:-1], dec_hidden)
        s_t = s_t.permute(0, 2, 1) ## shape (batch, self.vocab_size, length)
        loss = self.loss_(s_t, char_sequence[1:])
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        all_chars = []
        batch_size = initialStates[0].shape[1]
        curr_chars = torch.tensor([self.target_vocab.start_of_word] * batch_size, device=device).unsqueeze(0) ##or get id? or char??

        for index in range(max_length):
            s_t, (last_hidden, last_cell) = self.forward(curr_chars, initialStates)
            initialStates = (last_hidden, last_cell)
            p_t = self.st_softmax(s_t) ##double check dim=-1 or 2 to make sure !!
            curr_chars = torch.argmax(p_t.squeeze(0), dim=-1).unsqueeze(0)
            all_chars.append(curr_chars.squeeze(0))

        all_char_indices = torch.stack(all_chars).permute(1, 0) ## (Tensor) of size (batch, max_length)
        batch_char_indices = torch.split(all_char_indices, 1, dim=0)
        decodedWords = []

        for batch_idx in batch_char_indices:
            word = ''
            batch_idx = batch_idx.squeeze(0)
            for char_idx in batch_idx:
                char_idx = int(char_idx)
                if char_idx == self.target_vocab.end_of_word:
                    break
                next_char = self.target_vocab.id2char[char_idx]
                word = word + str(next_char)
            if len(word) > max_length:
                word = word[:max_length]
            decodedWords.append(word)

        return decodedWords

        ### END YOUR CODE

