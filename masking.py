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
from hparams import device
from vocabulary import pad_token

"""
Implementations of masking functionality for training a transformer:
    padding_mask: mask <pad> tokens in input sequences
    look_ahead_mask: mask subsequent positions for masked self-attention calculation
    combined_mask: elementwise maximum of above two
"""


def create_padding_mask(inp, n=4):
    """
    Since some of the input sequences are padded with pad tokens (0), we need to mask out these parts of the input
    sequences so that the model does not treat it as input. The mask will be created as a tensor of the same shape as
    the input with ones in the positions that need to be masked.

    Args:
        inp: unembedded batch of input sequences of shape (batch_size, seq_len)
        n (int): number of dimensions to which to broadcast mask

    Returns:
        mask: tensor of ones of shape (batch_size, 1, ..., 1, seq_len) with ndim=n
              positions to mask are marked with ones
    """
    # find positions in inp equal to pad_token
    mask = torch.eq(inp, pad_token).float()

    # add extra dimensions
    return mask.view(*mask.shape[:-1], *[1 for _ in range(n-2)], mask.shape[-1]).to(inp.device)


def create_look_ahead_mask(seq_len):
    """
    Creates an upper triangular mask of ones of shape (seq_len, seq_len) for the calculation of Scaled Dot Product
    Attention, to prevent the transformer from looking ahead at future tokens, so that the next outputs of the
    model are based only on the current and previous tokens in the input sequence.

    Args:
        seq_len (int): input sequence length; the created mask is dependent only on the sequence length, not
                       on the sequence itself

    Returns:
        mask: upper triangular mask of ones of shape (seq_len, seq_len); easily broadcastable to n dimensions
              positions to mask are marked with ones
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.float().to(device)


def create_mask(inp, n=4):
    """
    The correct final mask for the input will be the maximum of the padding and look_ahead mask, as the elements that
    need to be zeroed are represented by 1's, and those that need to be preserved are represented by 0's.

    Args:
        inp: unembedded batch of input sequences of shape (batch_size, seq_len)
        n (int): number of dimensions to which to broadcast mask

    Returns:
        combined_mask: maximum of padding and look_ahead masks for inp;
                       tensor of ones of shape (batch_size, 1, ..., 1, seq_len, seq_len) with ndim=n
                       positions to mask are marked with ones
    """
    # padding mask
    padding_mask = create_padding_mask(inp, n=n)

    # look ahead mask, assuming seq_len is last dimension of inp
    look_ahead_mask = create_look_ahead_mask(inp.shape[-1])

    # final mask is the maximum of the two
    combined_mask = torch.max(padding_mask, look_ahead_mask)
    return combined_mask
