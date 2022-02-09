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

"""
Vocabulary described in Oore et. al, 2018 and helper functions

Possible MIDI events being considered:
    128 note_on events
    128 note_off events
    125 time_shift events #time_shift = 1: 8 ms
    32  velocity events

Total midi events = 413

Indices in the vocabulary:
v[       0] = '<pad>'
v[  1..128] = note_on
v[129..256] = note_off
v[257..381] = time_shift
v[382..413] = velocity
v[414..415] = '<start>', '<end>'
"""

"""MANIFEST CONSTANTS"""

note_on_events = 128
note_off_events = note_on_events
note_events = note_on_events + note_off_events
time_shift_events = 125
velocity_events = 32

LTH = 1000  # max milliseconds; LTH ms = 125 time_shift_events
DIV = LTH // time_shift_events  # 1 time_shift = DIV milliseconds

BIN_STEP = 128 // velocity_events  # number of velocities per bin

# total events includes pad + start + end, but this is vocab_size
total_midi_events = note_on_events + note_off_events + time_shift_events + velocity_events

# create vocabulary
note_on_vocab = [f"note_on_{i}" for i in range(note_on_events)]
note_off_vocab = [f"note_off_{i}" for i in range(note_off_events)]
time_shift_vocab = [f"time_shift_{i}" for i in range(time_shift_events)]
velocity_vocab = [f"set_velocity_{i}" for i in range(velocity_events)]

vocab = ['<pad>'] + note_on_vocab + note_off_vocab + time_shift_vocab + velocity_vocab + ['<start>', '<end>']
vocab_size = len(vocab)

# useful tokens
pad_token = vocab.index("<pad>")
start_token = vocab.index("<start>")
end_token = vocab.index("<end>")


"""HELPER FUNCTIONS"""


def events_to_indices(event_list, _vocab=None):
    """
    converts event_list to list of indices in vocab
    """
    if _vocab is None:
        _vocab = vocab
    index_list = []
    for event in event_list:
        index_list.append(_vocab.index(event))
    return index_list


def indices_to_events(index_list, _vocab=None):
    """
    converts index_list to list of events in vocab
    """
    if _vocab is None:
        _vocab = vocab
    event_list = []
    for idx in index_list:
        event_list.append(_vocab[idx])
    return event_list


def velocity_to_bin(velocity, step=BIN_STEP):
    """
    Velocity in midi is an int between 0 and 127 inclusive, which is unnecessarily high resolution
    To reduce number of events in vocab, velocity is resolved into (128 / step) bins

    Returns:
        _bin (int): bin into which velocity is placed
    """
    if 128 % step != 0:
        raise ValueError("128 must be divisible by bins")
    if not (0 <= velocity <= 127):
        raise ValueError(f"velocity must be between 0 and 127, not {velocity}")

    # return bin into which velocity is placed
    _bin = velocity // step
    return _bin


def bin_to_velocity(_bin, step=BIN_STEP):
    """
    Convert binned velocity to midi velocity
    (i.e., convert from [0, velocity_events] -> [0, 127]
    """
    if not (0 <= _bin * step <= 127):
        raise ValueError(f"bin * step must be between 0 and 127 to be a midi velocity, not {_bin * step}")

    return int(_bin * step)


def time_to_events(delta_time, event_list=None, index_list=None, _vocab=None):
    """
    Translate accumulated delta_time between midi events into vocab using time_cutter
    event_list and index_list are passed by reference, so nothing is returned.
    Pass-by-reference is necessary to execute this function within a loop.

    Args:
        delta_time (int): time between midi events
        event_list (list): accumulated vocab event list during midi translation
        index_list (list): accumulated vocab index list during midi translation
        _vocab (list, optional): vocabulary list to translate into
    """
    if _vocab is None:
        _vocab = vocab
    time = time_cutter(delta_time)
    for i in time:
        # repeatedly create and append time events to the input lists
        idx = note_on_events + note_off_events + i
        if event_list is not None:
            event_list.append(_vocab[idx])
        if index_list is not None:
            index_list.append(idx)
    return


def time_cutter(time, lth=LTH, div=DIV):
    """
    As per Oore et. al, 2018, the time between midi events must be expressed as a sequence of finite-length
    time segments, so as to avoid considering every possible length of time in the vocab. This sequence can be
    expressed as k instances of a maximum time shift followed by a leftover time shift, i.e.,
    time = k * max_time_shift + leftover_time_shift
    where k = time // max_time_shift; leftover_time_shift = time % max_time_shift

    This function will translate the input time into indices in the vocabulary then cut it as above

    Args:
        time (int > 0): input milliseconds to translate and cut
        lth (int, optional): max milliseconds to consider for vocab, i.e., max_time_shift
        div (int, optional): number of ms per time_shift;
                   lth // div = num_time_shift_events

    Returns:
        time_shifts (list): list of time shifts into which time is cut
                            each time_shift is in range: (1, lth // div); 0 is not considered
    """
    if lth % div != 0:
        raise ValueError("lth must be divisible by div")

    time_shifts = []

    # assume time = k * lth, k >= 0; add k max_time_shifts (lth // div) to time_shifts
    for i in range(time // lth):
        time_shifts.append(round_(lth / div))   # custom round for consistent rounding of 0.5, see below
    leftover_time_shift = round_((time % lth) / div)
    time_shifts.append(leftover_time_shift) if leftover_time_shift > 0 else None

    return time_shifts


def round_(a):
    """
    Custom rounding function for consistent rounding of 0.5 to greater integer
    """
    b = a // 1
    decimal_digits = a % 1
    adder = 1 if decimal_digits >= 0.5 else 0
    return int(b + adder)
