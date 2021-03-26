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
import torch.nn.functional as F
from torch import nn
from math import sqrt
from hparams import device

"""
Implementation of layers and functionality necessary to build Music Transformer model,
based on Huang et. al, 2018, Vaswani et. al, 2017
"""


def abs_positional_encoding(max_position, d_model, n=3):
    """
    Since the transformer does not use recurrence or convolution, we have to deliberately give it positional
    information. Though learned relative position embeddings will be added to the model, it is possible that absolute
    position encoding will aid it in predicting next tokens.

    Args:
        max_position (int): maximum position for which to calculate positional encoding
        d_model (int): Transformer hidden dimension size
        n (int): number of dimensions to which to broadcast output

    Returns:
        sinusoidal absolute positional encoding of shape d_model for max_position positions
    """
    # set of all positions to consider
    positions = torch.arange(max_position).float().to(device)

    # get angles to input to sinusoid functions
    k = torch.arange(d_model).float().to(device)
    coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
    angles = positions.view(-1, 1) @ coeffs.view(1, -1)

    # apply sin to the even indices of angles along the last axis
    angles[:, 0::2] = torch.sin(angles[:, 0::2])

    # apply cos to the odd indices of angles along the last axis
    angles[:, 1::2] = torch.cos(angles[:, 1::2])

    return angles.view(*[1 for _ in range(n-2)], max_position, d_model)


def skew(t):
    """
    Implements Huang et. al, 2018's skewing algorithm to correctly reorder the dot(Q, RelativePositionEmbeddings)
    matrix. This function generalizes to any shape and any number of dimensions. However, attention calculation
    requires shape (..., L, L).

    Algorithm:
        1. Pad T
        2. Reshape
        3. Slice

    Args:
        t (torch.Tensor): tensor to skew

    Returns:
        Srel: skewed t: nth column from the right is skewed into the nth diagonal under the main; same shape as t
    """
    # pad T
    padded = F.pad(t, [1, 0])

    # reshape to diagonalize the columns in the last 2 dimensions
    Srel = padded.reshape(-1, t.shape[-1] + 1, t.shape[-2])

    # final touches
    Srel = Srel[:, 1:]              # slice last L values
    Srel = Srel.reshape(*t.shape)   # reshape to shape of t
    return Srel


def rel_scaled_dot_prod_attention(q, k, v, e=None, mask=None):
    """
    A modification given by Shaw et. al, 2018, improved by Huang et. al, 2018, to the Scaled Dot-Product Attention
    mechanism given in Vaswani et. al, 2017, which allows the Transformer model to attend to all relevant elements of
    the input sequences as well as the relative distances between them.

    RelAttention = softmax( mask( QKT + skew(QET) ) / sqrt(d_k) ) V

    Args:
        q: Queries tensor of shape (..., seq_len_q, d_model)
        k: Keys tensor of shape (..., seq_len_k, d_model)
        v: Values tensor of shape (..., seq_len_k, d_model)
        e (optional): Relative Position Embeddings tensor of shape (seq_len_k, d_model)
        mask (optional): mask for input batch with ones indicating the positions to mask

    Returns:
        output attention of shape (..., seq_len_q, d_model)
    """
    QKt = torch.matmul(q, k.transpose(-1, -2))  # (..., seq_len_q, seq_len_k)

    if e is None:
        # assumes q.shape[:-2] == k.shape[:-2]
        Srel = torch.zeros(*q.shape[:-2], q.shape[-2], k.shape[-2], device=q.device)
    else:
        Srel = skew(torch.matmul(q, e.transpose(-1, -2)))  # (..., seq_len_q, seq_len_k)

    # find and scale attention logits
    dk = sqrt(k.shape[-1])
    scaled_attention_logits = (QKt + Srel) / dk  # (..., seq_len_q, seq_len_k)

    # add scaled mask to 0 out positions to mask in softmax
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # calculate attention by calculating attention weights by softmaxing on the last dimension
    # and then multiplying by v
    return torch.matmul(F.softmax(scaled_attention_logits, dim=-1), v)


class MultiHeadAttention(nn.Module):
    """
    MultiHead Relative Attention Block. Computes attention for input batch along num_heads "heads".
    In the process, attention weights are calculated num_heads times, which allows the network to
    extract information from the input batch through several different representations simultaneously
    """
    def __init__(self, d_model, num_heads, max_rel_dist, bias=True):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings; set to 0 to compute normal attention
            bias (bool, optional): if set to False, all Linear layers in the MHA block will not learn
                                   an additive bias. Default: True

        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_rel_dist = max_rel_dist

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible into num_heads heads")

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate Q from input
        self.wk = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate K from input
        self.wv = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate V from input

        self.E = nn.Embedding(self.max_rel_dist, self.d_model)      # relative position embeddings

        self.wo = nn.Linear(self.d_model, self.d_model, bias=True)  # final output layer

    @staticmethod
    def split_heads(x, num_heads, depth=None):
        """
        Helper function to split input x along num_heads heads

        Args:
            x: input tensor to split into heads; shape: (..., L, d_model); d_model = num_heads * depth
            num_heads (int): number of heads along which to calculate attention
            depth (int, optional): desired dimensionality at each head

        Returns:
            input tensor correctly reshaped and transposed to shape (..., num_heads, L, depth)
        """
        # get depth if None
        if depth is None:
            if x.shape[-1] % num_heads != 0:
                raise ValueError("d_model must be divisible into num_heads")
            depth = x.shape[-1] // num_heads

        # reshape and transpose x
        x = x.view(*x.shape[:-1], num_heads, depth)     # (..., L, num_heads, depth)
        return x.transpose(-2, -3)                      # (..., num_heads, L, depth)

    def get_required_embeddings(self, seq_len, max_len=None):
        """
        Helper function to get required non-positive relative position embeddings to calculate attention on
        input of length seq_len. Required relative position embeddings are:
            [last embedding from the right] * max(seq_len - max_len, 0) + Embeddings(max(max_len - seq_len, 0), max_len)

        Requires self.E (nn.Embedding): relative position embeddings ordered from E_{-max_len + 1} to E_0

        Args:
            seq_len (int): length of input sequence
            max_len (int, optional): maximum relative distance considered in relative attention calculation
                                     Default: E.num_embeddings

        Returns:
            required relative position embeddings from E
        """
        if max_len is None:
            max_len = self.E.num_embeddings

        # required relative position embeddings
        E_dev = self.E.weight.device
        first_emb = self.E(torch.arange(0, 1, device=E_dev)).clone()
        return torch.cat(
            [*[first_emb.clone() for _ in range(max(seq_len - max_len, 0))],
             self.E(torch.arange(max(max_len - seq_len, 0), max_len, device=E_dev))],
            dim=0
        )

    def forward(self, q, k, v, mask=None):
        """
        Computes Multi-Head Attention on input tensors Q, K, V

        Args:
            q: Queries tensor of shape (..., seq_len_q, d_model)
            k: Keys tensor of shape (..., seq_len_k, d_model)
            v: Values tensor of shape (..., seq_len_k, d_model)
            mask (optional): mask for input batch with ones indicating positions to mask. Default: None

        Returns:
            multi-head attention of shape (..., seq_len_q, d_model) for input batch
        """
        # get Q, K, V
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # get required embeddings from E
        seq_len_k = k.shape[-2]
        e = self.get_required_embeddings(seq_len_k, self.max_rel_dist)  # (seq_len_k, d_model)

        # split into heads
        q = self.split_heads(q, self.num_heads, self.depth)  # (batch_size, h, seq_len_q, depth)
        k = self.split_heads(k, self.num_heads, self.depth)  # (batch_size, h, seq_len_k, depth)
        v = self.split_heads(v, self.num_heads, self.depth)  # (batch_size, h, seq_len_k, depth)
        e = self.split_heads(e, self.num_heads, self.depth)  # (h, seq_len_k, depth)

        # compute MHA
        # attention shape: (batch_size, h, seq_len_q, depth); weights shape: (batch_size, h, seq_len_q, seq_len_k)
        rel_scaled_attention = rel_scaled_dot_prod_attention(q, k, v, e, mask=mask)

        # concatenate heads and pass through final layer
        rel_scaled_attention = rel_scaled_attention.transpose(-2, -3)  # (batch_size, seq_len_q, h, depth)
        sh = rel_scaled_attention.shape
        return self.wo(rel_scaled_attention.reshape(*sh[:-2], self.d_model))  # (batch_size, seq_len_q, d_model)


class PointwiseFFN(nn.Module):
    """
    Fully-connected Feedforward layer that follows the MHA block in each Transformer layer, which is simply a 2 layer
    Dense network with a ReLU in between
    """
    def __init__(self, d_model, d_ff, bias=True):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            d_ff (int): intermediate dimension of FFN blocks
            bias (bool, optional): if set to False, all Linear layers in the FFN block will not learn
                                   an additive bias. Default: True
        """
        super(PointwiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.main = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias)
        )

    def forward(self, x):
        return self.main(x)


class DecoderLayer(nn.Module):
    """
    Every TransformerDecoder layer consists of 2 sublayers:
        1. Masked Multi-Head Attention
        2. Pointwise Feedforward Network
    In the original Transformer, each sublayer further employs a residual connection followed by a LayerNorm on the last
    dimension. However, here the LayerNormalization will be placed before the residual connnection, as this Pre-LN
    architecture does not generally require an explicitly designed learning rate schedule.
    """
    def __init__(self, d_model, num_heads, d_ff, max_rel_dist, bias=True, dropout=0.1, layernorm_eps=1e-6):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings; set to 0 to compute normal attention
            bias (bool, optional): if set to False, all Linear layers in the Decoder will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model
            layernorm_eps (very small positive float, optional): epsilon for LayerNormalization
        """
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_rel_idst = max_rel_dist

        self.mha = MultiHeadAttention(d_model, num_heads, max_rel_dist, bias)
        self.ffn = PointwiseFFN(d_model, d_ff, bias)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass through decoder layer. Designed to be able to use torch's nn.TransformerDecoder as the final model,
        which is why memory and all parameters after tgt_mask are present but are unused.

        Args:
            tgt: input queries tensor from previous layer, named this way to use nn.TransformerDecoder
            tgt_mask (optional, must be explicitly specified as a kwarg): tensor of with 1's indicating positions to
                                                                          mask. Default: None

        Returns:
            output after passing through MHA and FFN blocks, along with intermediate layernorms and residual connections
        """
        # multi-head attention block
        attn_out = self.layernorm1(tgt)
        attn_out = self.mha(attn_out, attn_out, attn_out, mask=tgt_mask)
        attn_out = self.dropout1(attn_out)
        attn_out = tgt + attn_out

        # pointwise ffn block
        ffn_out = self.layernorm2(attn_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out
