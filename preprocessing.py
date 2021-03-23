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

import os
import argparse
import torch
import torch.nn.functional as F
from random import randint, sample
from sys import exit
from vocabulary import *
from tokenizer import *

"""
Functionality to preprocess MIDI files translated into indices in the event vocabulary from command line
"""


def sample_end_data(seqs, lth, factor=6):
    """
    Randomly samples sequences of length ~lth from an input set of sequences seqs. Prepares data for augmentation.
    Returns a list. Deliberately samples from the end so that model learns to end.

    Args:
        seqs (list): list of sequences in the event vocabulary
        lth (int): approximate length to cut sequences into
        factor (int): factor to vary range of output lengths; Default: 6. Higher factor will narrow the output range

    Returns:
        input sequs cut to length ~lth
    """
    data = []
    for seq in seqs:
        lower_bound = max(len(seq) - lth, 0)
        idx = randint(lower_bound, lower_bound + lth // factor)
        data.append(seq[idx:])

    return data


def sample_data(seqs, lth, factor=6):
    """
    Randomly samples sequences of length ~lth from an input set of sequences seqs. Prepares data for augmentation.
    Returns a list.

    Args:
        seqs (list): list of sequences in the event vocabulary
        lth (int): approximate length to cut sequences into
        factor (int): factor to vary range of output lengths; Default: 6. Higher factor will narrow the output range

    Returns:
        input sequs cut to length ~lth
    """
    data = []
    for seq in seqs:
        length = randint(lth - lth // factor, lth + lth // factor)
        idx = randint(0, max(0, len(seq) - length))
        data.append(seq[idx:idx+length])
        
    return data


def aug(data, note_shifts=None, time_stretches=None, verbose=False):
    """
    Augments data up and down in pitch by note_shifts and faster and slower in time by time_stretches. Adds start
    and end tokens and pads to max sequence length in data

    Args:
        data (list of lists of ints): sequences to augment
        note_shifts (list): pitch transpositions to be made
        time_stretches (list): stretches in time to be made
        verbose (bool): set to True to periodically print augmentation progress

    Returns:
        input data with pitch transpositions and time stretches, concatendated to one tensor
    """
    if note_shifts is None:
        note_shifts = torch.arange(-2, 3)
    if time_stretches is None:
        time_stretches = [1, 1.05, 1.1]
    if any([i <= 0 for i in time_stretches]):
        raise ValueError("time_stretches must all be positive")

    # preprocess the time stretches
    if 1 not in time_stretches:
        time_stretches.append(1)
    ts = []
    for t in time_stretches:
        ts.append(t) if t not in ts else None
        ts.append(1 / t) if (t != 1 and 1 / t not in ts) else None
    ts.sort()
    time_stretches = ts

    # iteratively transpose and append the sequences
    note_shifted_data = []
    count = 0  # to print if verbose
    for seq in data:
        # data will be transposed by each shift in note_shifts
        for shift in note_shifts:
            # check torch tensor
            try:
                _shift = shift.item()
            except AttributeError:
                _shift = shift

            # iterate over and shift seq
            note_shifted_seq = []
            for idx in seq:
                _idx = idx + _shift  # shift the index

                # append only note values if changed, and don't go out of bounds of note events
                if (0 < idx <= note_on_events and 0 < _idx <= note_on_events) or \
                        (note_on_events < idx <= note_events and note_on_events < _idx <= note_events):
                    note_shifted_seq.append(_idx)
                else:
                    note_shifted_seq.append(idx)
            # verbose statement
            count += 1
            print(f"Transposed {count} sequences") if verbose else None
            # convert to tensor and append to data
            note_shifted_seq = torch.LongTensor(note_shifted_seq)
            note_shifted_data.append(note_shifted_seq)

    # now iterate over the note shifted data to stretch it in time
    time_stretched_data = []
    delta_time = 0  # helper
    count = 0  # to print if verbose
    for seq in note_shifted_data:
        # data will be stretched in time by each time_stretch
        for time_stretch in time_stretches:
            # iterate over and stretch time shift events in seq
            time_stretched_seq = []
            for idx in seq:
                if note_events < idx <= note_events + time_shift_events:
                    # acculumate stretched times
                    time = idx - (note_events - 1)
                    delta_time += round_(time * DIV * time_stretch)
                else:
                    time_to_events(delta_time, index_list=time_stretched_seq)
                    delta_time = 0
                    time_stretched_seq.append(idx)
            # verbose statement
            count += 1
            print(f"Stretched {count} sequences") if verbose else None
            # convert to tensor and append to data
            time_stretched_seq = torch.LongTensor(time_stretched_seq)
            time_stretched_data.append(time_stretched_seq)

    # preface and suffix with start and end tokens
    aug_data = []
    for seq in time_stretched_data:
        aug_data.append(F.pad(F.pad(seq, (1, 0), value=start_token), (0, 1), value=end_token))

    # pad all sequences to max length
    aug_data = torch.nn.utils.rnn.pad_sequence(aug_data, padding_value=pad_token).transpose(-1, -2)
    return aug_data


def randomly_sample_aug_data(aug_data, k, augs=25):
    """
    Randomly samples k sets of augmented data to cut down dataset

    Args:
        aug_data (torch.Tensor): augmented dataset
        k (int): coefficient such that k * augs samples are returned
        augs (int): total number of augmentations per sequence performed on original dataset
    """
    random_indices = sample(range(len(aug_data) // augs), k=k)
    out = torch.cat(
        [t[i * augs:i * augs + augs] for i in random_indices],
        dim=0
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocessing.py",
        description="Preprocess MIDI files into single tensor for ML"
    )
    parser.add_argument("source", help="source directory of MIDI files to preprocess")
    parser.add_argument("destination", help="destination path at which to save preprocessed data as a single tensor, "
                                            "including filename and extension")
    parser.add_argument("length", help="approximate sequence length to cut data into (length will be randomly sampled)",
                        type=int)
    parser.add_argument("-a", "--from-augmented-data", help="flag to specify whether or not the source contains "
                                                            "already augmented data",  action="store_true")
    parser.add_argument("-t", "--transpositions", help="list of pitch transpositions to make in data augmentation",
                        nargs="+", type=int)
    parser.add_argument("-s", "--time-stretches", help="list of stretches in time to make in data augmentation",
                        nargs="+", type=float)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")
    args = parser.parse_args()

    # fix source directory if necessary
    if args.source[-1] != "/":
        args.source += "/"

    # if source directory doesn't exist, exit
    if not os.path.isdir(args.source):
        print("Error: source must be an existing directory")
        exit(1)

    # fix save path if necessary
    if os.path.isdir(args.destination):
        if args.destination[-1] != "/":
            args.destination += "/"
        args.destination += "gnershk.pt"
    elif not (args.destination.endswith(".pt") or args.destination.endswith(".pth")):
        args.destination += ".pt"

    # turn length into int
    args.length = int(args.length)

    DATA = []
    PATH = args.source

    # load parsed midi files
    if not args.from_augmented_data:
        print("Translating midi files to event vocabulary (NOTE: may take a while)...") if args.verbose else None
        for file in os.listdir(PATH):
            try:
                idx_list = midi_parser(fname=PATH + file)[0]
                DATA.append(idx_list)
            except OSError:
                pass
        print("Done!") if args.verbose else None

    # randomly sample endings
    print("Randomly sampling and cutting data to length...") if args.verbose else None
    DATA = sample_data(DATA, lth=args.length) + sample_end_data(DATA, lth=args.length)
    print("Done!") if args.verbose else None

    # augment data
    if not args.from_augmented_data:
        print("Augmenting data (NOTE: may take even longer)...") if args.verbose else None
        DATA = aug(DATA, note_shifts=args.transpositions, time_stretches=args.time_stretches,
                   verbose=(args.verbose >= 2))
        print("Done!") if args.verbose else None
    
    # shuffle data
    DATA = DATA[torch.randperm(DATA.shape[0])]
    
    # save
    print("Saving...") if args.verbose else None
    torch.save(DATA, args.destination)
    print("Done!")
