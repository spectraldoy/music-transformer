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

from torch import cuda, device as d
from vocabulary import vocab_size

# get device
if cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = d(dev)

# default hparams for the model
hparams = {
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}

# hparams for TF model - significantly larger
hparams_large = {
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 1024,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}
