"""
Copyright 2021 Aditya Gomatam.

This file is part of music-transformer (https://github.com/spectraldoy/music-transformer), my project to build and
train a Music Transformer. music-transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. music-transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. music-transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for music-transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.
"""

import torch
from math import sqrt
from torch import nn
from hparams import hparams
from layers import DecoderLayer, abs_positional_encoding

"""
Implementation of Music Transformer model, using torch.nn.TransformerDecoder
based on Huang et. al, 2018, Vaswani et. al, 2017
"""


class MusicTransformer(nn.Module):
    """
    Transformer Decoder with Relative Attention. Consists of:
        1. Input Embedding
        2. Absolute Positional Encoding
        3. Stack of N DecoderLayers
        4. Final Linear Layer
    """
    def __init__(self,
                 d_model=hparams["d_model"],
                 num_layers=hparams["num_layers"],
                 num_heads=hparams["num_heads"],
                 d_ff=hparams["d_ff"],
                 max_rel_dist=hparams["max_rel_dist"],
                 max_abs_position=hparams["max_abs_position"],
                 vocab_size=hparams["vocab_size"],
                 bias=hparams["bias"],
                 dropout=hparams["dropout"],
                 layernorm_eps=hparams["layernorm_eps"]):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings. Set to 0 to compute normal attention
            max_abs_position (int): maximum absolute position for which to create sinusoidal absolute
                                    positional encodings. Set to 0 to compute pure relative attention
                                    make it greater than the maximum sequence length in the dataset if nonzero
            bias (bool, optional): if set to False, all Linear layers in the MusicTransformer will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model. Default: 0.1
            layernorm_eps (very small float, optional): epsilon for LayerNormalization. Default: 1e-6
        """
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist,
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )

        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass through the Music Transformer. Embeds x according to Vaswani et. al, 2017, adds absolute
        positional encoding if present, performs dropout, passes through the stack of decoder layers, and
        projects into the vocabulary space. DOES NOT SOFTMAX OR SAMPLE OUTPUT; OUTPUTS LOGITS.

        Args:
            x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
            mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None

        Returns:
            input batch after above steps of forward pass through MusicTransformer
        """
        # embed x according to Vaswani et. al, 2017
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # input dropout
        x = self.input_dropout(x)

        # pass through decoder
        x = self.decoder(x, memory=None, tgt_mask=mask)

        # final projection to vocabulary space
        return self.final(x)
