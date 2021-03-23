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

import argparse
import time
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from hparams import device
from masking import create_mask
from model import MusicTransformer

"""
Functionality to train a Music Transformer on a single CPU or single GPU

The transformer is an autoregressive model, which means that at the inference stage, it will make next predictions 
based on its previous outputs. However, while training, we can use teacher forcing - feeding the target into the 
model as previous output regardless of the true output of the model. This significantly cuts down on the compute 
required, while usually reducing loss (at the expense of generalizability of the model). Since we are training a 
generative model, the targets are simply the inputs shifted right by 1 position.
"""


def transformer_lr_schedule(d_model, step_num, warmup_steps=4000):
    """
    As per Vaswani et. al, 2017, the post-LayerNorm transformer performs vastly better a custom learning rate
    schedule. Though the PyTorch implementation of the Music-Transformer uses pre-LayerNorm, which has been observed
    not to require a custom schedule, this function is here for utility.

    Args:
        d_model: embedding / hidden dimenision of the transformer
        step_num: current training step
        warmup_steps: number of transformer schedule warmup steps. Set to 0 for a continuously decaying learning rate

    Returns:
        learning rate at current step_num
    """
    if warmup_steps <= 0:
        step_num += 4000
        warmup_steps = 4000
    step_num = step_num + 1e-6  # avoid division by 0

    if type(step_num) == torch.Tensor:
        arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    else:
        arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

    return (d_model ** -0.5) * arg


def loss_fn(prediction, target, criterion=F.cross_entropy):
    """
    Since some positions of the input sequences are padded, we must calculate the loss by appropriately masking
    padding values

    Args:
        prediction: output of the model for some input
        target: true value the model was supposed to predict
        criterion: vanilla loss criterion

    Returns:
        masked loss between prediction and target
    """
    mask = torch.ne(target, torch.zeros_like(target))           # ones where target is 0
    _loss = criterion(prediction, target, reduction='none')     # loss before masking

    # multiply mask to loss elementwise to zero out pad positions
    mask = mask.to(_loss.dtype)
    _loss *= mask

    # output is average over the number of values that were not masked
    return torch.sum(_loss) / torch.sum(mask)


def train_step(model: MusicTransformer, opt, sched, inp, tar):
    """
    Computes loss and backward pass for a single training step of the model

    Args:
        model: MusicTransformer model to train
        opt: optimizer initialized with model's parameters
        sched: scheduler properly initialized with opt
        inp: input batch
        tar: input batch shifted right by 1 position; MusicTransformer is a generative model

    Returns:
        loss before current backward pass
    """
    # forward pass
    predictions = model(inp, mask=create_mask(inp, n=inp.dim() + 2))

    # backward pass
    opt.zero_grad()
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    loss.backward()
    opt.step()
    sched.step()

    return float(loss)


def val_step(model: MusicTransformer, inp, tar):
    """
    Computes loss for a single evaluation / validation step of the model

    Args:
        model: MusicTransformer model to evaluate
        inp: input batch
        tar: input batch shifted right by 1 position

    Returns:
        loss of model on input batch
    """
    predictions = model(inp, mask=create_mask(inp, n=max(inp.dim() + 2, 2)))
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    return float(loss)


class MusicTransformerTrainer:
    """
    As the transformer is a large model and takes a while to train on a GPU, or even a TPU, I wrote this Trainer
    class to make it easier to load and save checkpoints with the model. The way I've designed it instantiates the
    model, optimizer, and scheduler within the class itself, as there are some problems with passing them in. But,
    to get these objects back just call:
        trainer.model
        trainer.optimizer
        trainer.scheduler

    This class also tracks the cumulative losses while training, which you can get back with:
        trainer.train_losses
        trainer.val_losses
    as lists of floats

    To save a checkpoint, call trainer.save()
    To load a checkpoint, call trainer.load( (optional) ckpt_path)
    """

    def __init__(self, hparams_, datapath, batch_size, warmup_steps=4000,
                 ckpt_path="music_transformer_ckpt.pt", load_from_checkpoint=False):
        """
        Args:
            hparams_: hyperparameters of the model
            datapath: path to the data to train on
            batch_size: batch size to batch the data
            warmup_steps: number of warmup steps for transformer learning rate schedule
            ckpt_path: path at which to save checkpoints while training; MUST end in .pt or .pth
            load_from_checkpoint (bool, optional): if true, on instantiating the trainer, this will load a previously
                                                   saved checkpoint at ckpt_path
        """
        # get the data
        self.datapath = datapath
        self.batch_size = batch_size
        data = torch.load(datapath).long().to(device)

        # max absolute position must be able to acount for the largest sequence in the data
        if hparams_["max_abs_position"] > 0:
            hparams_["max_abs_position"] = max(hparams_["max_abs_position"], data.shape[-1])

        # train / validation split: 80 / 20
        train_len = round(data.shape[0] * 0.8)
        train_data = data[:train_len]
        val_data = data[train_len:]
        print(f"There are {data.shape[0]} samples in the data, {len(train_data)} training samples and {len(val_data)} "
              "validation samples")

        # datasets and dataloaders: split data into first (n-1) and last (n-1) tokens
        self.train_ds = TensorDataset(train_data[:, :-1], train_data[:, 1:])
        self.train_dl = DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True)

        self.val_ds = TensorDataset(val_data[:, :-1], val_data[:, 1:])
        self.val_dl = DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=True)

        # create model
        self.model = MusicTransformer(**hparams_).to(device)
        self.hparams = hparams_

        # setup training
        self.warmup_steps = warmup_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )

        # setup checkpointing / saving
        self.ckpt_path = ckpt_path
        self.train_losses = []
        self.val_losses = []

        # load checkpoint if necessesary
        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()

    def save(self, ckpt_path=None):
        """
        Saves a checkpoint at ckpt_path

        Args:
            ckpt_path (str, optional): if None, saves the checkpoint at the previously stored self.ckpt_path
                                       else saves the checkpoints at the new passed-in path, and stores this new path at
                                       the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "validation_losses": self.val_losses,
            "warmup_steps": self.warmup_steps,
            "hparams": self.hparams
        }

        torch.save(ckpt, self.ckpt_path)
        return

    def load(self, ckpt_path=None):
        """
        Loads a checkpoint from ckpt_path
        NOTE: OVERWRITES THE MODEL STATE DICT, OPTIMIZER STATE DICT, SCHEDULER STATE DICT, AND HISTORY OF LOSSES

        Args:
            ckpt_path (str, optional): if None, loads the checkpoint at the previously stored self.ckpt_path
                                       else loads the checkpoints from the new passed-in path, and stores this new path
                                       at the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = torch.load(self.ckpt_path)

        del self.model, self.optimizer, self.scheduler

        # create and load model
        self.model = MusicTransformer(**ckpt["hparams"]).to(device)
        self.hparams = ckpt["hparams"]
        print("Loading the model...", end="")
        print(self.model.load_state_dict(ckpt["model_state_dict"]))

        # create and load load optimizer and scheduler
        self.warmup_steps = ckpt["warmup_steps"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # load loss histories
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["validation_losses"]

        return

    def fit(self, epochs):
        """
        Training loop to fit the model to the data stored at the passed in datapath. If KeyboardInterrupt at anytime
        during the training loop, and if progresss being printed, this method will save a checkpoint at the 
        passed-in ckpt_path

        Args:
            epochs: number of epochs to train for.

        Returns:
            history of training and validation losses for this training session
        """
        print_interval = epochs // 10 + int(epochs < 10)
        train_losses = []
        val_losses = []
        start = time.time()

        print("Beginning training...")

        try:
            for epoch in range(epochs):
                train_epoch_losses = []
                val_epoch_losses = []

                self.model.train()
                for train_inp, train_tar in self.train_dl:
                    loss = train_step(self.model, self.optimizer, self.scheduler, train_inp, train_tar)
                    train_epoch_losses.append(loss)

                self.model.eval()
                for val_inp, val_tar in self.val_dl:
                    loss = val_step(self.model, val_inp, val_tar)
                    val_epoch_losses.append(loss)

                # mean losses for the epoch
                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)

                # store complete history of losses in member lists and relative history for this session in output lists
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)

                if ((epoch + 1) % print_interval) == 0:
                    print(f"Epoch {epoch + 1} Time taken {round(time.time() - start, 2)} seconds "
                          f"Train Loss {train_losses[-1]} Val Loss {val_losses[-1]}")
                    # print("Checkpointing...")
                    # self.save()
                    # print("Done")
                    start = time.time()

        except KeyboardInterrupt:
            pass

        print("Checkpointing...")
        self.save()
        print("Done")

        return train_losses, val_losses


if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a Music Transformer on single tensor dataset of preprocessed MIDI files"
    )

    # trainer arguments
    parser.add_argument("datapath", help="path at which preprocessed MIDI files are stored as a single tensor after "
                                         "being translated into an event vocabulary")
    parser.add_argument("ckpt_path", help="path at which to load / store checkpoints while training; "
                                          "KeyboardInterrupt while training to checkpoint the model; MUST end in .pt "
                                          "or .pth", type=str)
    parser.add_argument("save_path", help="path at which to save the model's state dict and hyperparameters after "
                                          "training; model will only be saved if the training loop finishes before a "
                                          "KeyboardInterrupt; MUST end in .pt or .pth", type=str)
    parser.add_argument("epochs", help="number of epochs to train for", type=check_positive)
    parser.add_argument("-bs", "--batch-size", help="number of sequences to batch together to compute a single "
                                                    "training step while training; default: 32", type=check_positive)
    parser.add_argument("-l", "--load-checkpoint", help="flag to load a previously saved checkpoint from which to "
                                                        "resume training; default: False", action="store_true")
    parser.add_argument("-w", "--warmup-steps", help="number of warmup steps for transformer learning rate scheduler; "
                                                     "if loading from checkpoint, this will be overwritten by saved "
                                                     "value; default: 4000", type=int)

    # hyperparameters
    parser.add_argument("-d", "--d-model",
                        help="music transformer hidden dimension size; if loading from checkpoint "
                             "this will be overwritten by saved hparams; default: 128", type=check_positive)
    parser.add_argument("-nl", "--num-layers",
                        help="number of transformer decoder layers in the music transformer; if loading from "
                             "checkpoint, this will be overwritten by saved hparams; default: 3", type=check_positive)
    parser.add_argument("-nh", "--num-heads",
                        help="number of attention heads over which to compute multi-head relative attention in the "
                             "music transformer; if loading from checkpoint, this will be overwritten by saved "
                             "hparams; default: 8", type=check_positive)
    parser.add_argument("-dff", "--d-feedforward",
                        help="hidden dimension size of pointwise FFN layers in the music transformer; if loading from "
                             "checkpoint, this will be overwritten by saved hparams; default: 512", type=check_positive)
    parser.add_argument("-mrd", "--max-rel-dist",
                        help="maximum relative distance between tokens to consider in relative attention calculation "
                             "in the music transformer; if loading from checkpoint, this will be overwritten by saved "
                             "hparams; default: 1024", type=check_positive)
    parser.add_argument("-map", "--max-abs-position",
                        help="maximum absolute length of an input sequence; set this to a very large value to be able "
                             "to generalize to longer sequences than in the dataset; if a sequence longer than the "
                             "passed in value is passed into the dataset, max_abs_position is set to that value not "
                             "the passed in; if loading from checkpoint, this will be overwritten by saved hparams; "
                             "default: 20000", type=int)
    parser.add_argument("-vs", "--vocab-size",
                        help="length of the vocabulary in which the input training data has been tokenized. if "
                             "loading from checkpoint, this will be overwritten by saved hparams; default: 416 (size "
                             "of Oore et. al MIDI vocabulary)", type=check_positive)
    parser.add_argument("-nb", "--no-bias",
                        help="flag to not use a bias in the linear layers of the music transformer; if loading from "
                             "checkpoint, this will be overwritten by saved hparams; default: False",
                        action="store_false")
    parser.add_argument("-dr", "--dropout",
                        help="dropout rate for training the model; if loading from checkpoint, this will be "
                             "overwritten by saved hparams; default: 0.1")
    parser.add_argument("-le", "--layernorm-eps",
                        help="epsilon in layernorm layers to avoid zero division; if loading from checkpoint, "
                             "this will be overwritten by saved hparams; default: 1e-6")

    args = parser.parse_args()

    # fix optional parameters
    batch_size_ = 32 if args.batch_size is None else args.batch_size
    warmup_steps_ = 2000 if args.warmup_steps is None else args.warmup_steps

    # fix hyperparameters
    hparams["d_model"] = args.d_model if args.d_model else hparams["d_model"]
    hparams["num_layers"] = args.num_layers if args.num_layers else hparams["num_layers"]
    hparams["num_heads"] = args.num_heads if args.num_heads else hparams["num_heads"]
    hparams["d_ff"] = args.d_feedforward if args.d_feedforward else hparams["d_ff"]
    hparams["max_rel_dist"] = args.max_rel_dist if args.max_rel_dist else hparams["max_rel_dist"]
    hparams["max_abs_position"] = args.max_abs_position if args.max_abs_position else hparams["max_abs_position"]
    hparams["vocab_size"] = args.vocab_size if args.vocab_size else hparams["vocab_size"]
    hparams["bias"] = args.no_bias
    hparams["dropout"] = args.dropout if args.dropout else hparams["dropout"]
    hparams["layernorm_eps"] = args.layernorm_eps if args.layernorm_eps else hparams["layernorm_eps"]

    # set up the trainer
    print("Setting up the trainer...")
    trainer = MusicTransformerTrainer(hparams, args.datapath, batch_size_, warmup_steps_,
                                      args.ckpt_path, args.load_checkpoint)
    print()

    # train the model
    trainer.fit(args.epochs)

    # done training, save the model
    print("Saving...")
    save_file = {
        "state_dict": trainer.model.state_dict(),
        "hparams": trainer.hparams
    }
    torch.save(save_file, args.save_path)
    print("Done!")
