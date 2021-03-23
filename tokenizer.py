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

import mido
from vocabulary import *
from torch import LongTensor

"""
Implementation of translators of MIDI files to and from the event-based
vocabulary representation of MIDI files according to Oore et al., 2018

NOTE: this functionality only works for single-track MIDI files
"""


def midi_parser(fname=None, mid=None):
    """
    Translates a single-track midi file (specifically piano) into Oore et. al, 2018 vocabulary

    Args:
        fname (str): path to midi file to load OR
        mid (mido.MidiFile): loaded midi file

    Returns:
        index_list (torch.Tensor): list of indices in vocab
        event_list (list): list of events in vocab
    """
    # take only one of fname or mid
    if not ((fname is None) ^ (mid is None)):
        raise ValueError("Input one of fname or mid, not both or neither")

    # load midi file
    if fname is not None:
        mid = mido.MidiFile(fname)

    # things needed for conversion
    delta_time = 0          # time between important midi messages
    event_list = []         # list of events in vocab
    index_list = []         # list of indices in vocab
    pedal_events = {}       # dict to handle pedal events
    pedal_flag = False      # flag to handle pedal events

    # translate midi file to event list
    for track in mid.tracks:
        for msg in track:
            # increase delta_time by msg time for all messages
            delta_time += msg.time

            # meta events are irrelevant
            if msg.is_meta:
                continue

            # process by message type
            t = msg.type
            vel = 0   # velocity

            if t == "note_on":  # key pressed
                # +1 or-1 everywhere accounts for <pad> token
                idx = msg.note + 1  # idx in vocab to help appending to output lists

                # get velocity to append after time events
                vel = velocity_to_bin(msg.velocity)

            elif t == "note_off":  # key released
                note = msg.note

                # if note_off while pedal down, add to pedal_events
                if pedal_flag:
                    if note not in pedal_events:
                        pedal_events[note] = 0
                    pedal_events[note] += 1
                    # to prevent adding more events to output lists, continue
                    continue
                else:  # else get idx to append to output lists
                    idx = note_on_events + note + 1
            # if pedal on or off and pedal_events is not empty
            elif t == "control_change":
                if msg.control == 64:
                    if msg.value >= 64:
                        # pedal down
                        pedal_flag = True
                    elif pedal_events:
                        # lift pedal
                        pedal_flag = False

                        # add the time events (0 is not a time shift, so all notes lifted at once is ok)
                        time_to_events(delta_time, event_list=event_list, index_list=index_list)
                        delta_time = 0

                        # perform note_offs that occurred when pedal was down now that pedal is up
                        for note in pedal_events:
                            idx = note_on_events + note + 1

                            # append a note_off event for all times note was released
                            for i in range(pedal_events[note]):
                                event_list.append(vocab[idx])
                                index_list.append(idx)
                        # restart pedal events dict
                        pedal_events = {}
                # to prevent adding more events to output lists, continue
                continue
            # if it's not a type of msg we care about, continue to avoid adding to output lists
            else:
                continue

            # process delta_time into events and indices in vocab
            time_to_events(delta_time, event_list=event_list, index_list=index_list)
            delta_time = 0  # reset delta_time to process subsequent messages

            # append velocity if note_on
            if t == "note_on":
                event_list.append(vocab[note_on_events + note_off_events + time_shift_events + vel + 1])
                index_list.append(note_on_events + note_off_events + time_shift_events + vel + 1)
            # append event and idx note events
            event_list.append(vocab[idx])
            index_list.append(idx)

    # return the lists of events
    return LongTensor(index_list), event_list


def list_parser(index_list=None, event_list=None, fname="bloop", tempo=512820):
    """
    Translates a set of events or indices in the Oore et. al, 2018 vocabulary into a midi file

    Args:
        index_list (list or torch.Tensor): list of indices in vocab OR
        event_list (list): list of events in vocab
        fname (str, optional): name for single track of midi file returned
        tempo (int, optional): tempo of midi file returned in µs / beat,
                               tempo_in_µs_per_beat = 60 * 10e6 / tempo_in_bpm

    Returns:
        mid (mido.MidiFile): single-track piano midi file translated from vocab
                             NOTE: mid IS NOT SAVED BY THIS FUNCTION, IT IS ONLY RETURNED
    """
    # take only one of event_list or index_list to translate
    if not ((index_list is None) ^ (event_list is None)):
        raise ValueError("Input one of index_list or event_list, not both or neither")

    # check index_list is ints, assuming 1d list
    if index_list is not None:
        try:
            # assume torch tensor
            if not all([isinstance(i.item(), int) for i in index_list]):
                raise ValueError("All indices in index_list must be int type")
        except AttributeError:
            # otherwise assume normal ,jst
            if not all([isinstance(i, int) for i in index_list]):
                raise ValueError("All indices in index_list must be int type")

    # check event_list is str, assuming 1d list and convert to index_list
    if event_list is not None:
        if not all(isinstance(i, str) for i in event_list):
            raise ValueError("All events in event_list must be str type")
        index_list = events_to_indices(event_list)

    # set up midi file
    mid = mido.MidiFile()
    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()

    # meta messages; meta time is 0 everywhere to prevent delay in playing notes
    meta_track.append(mido.MetaMessage("track_name").copy(name=fname, time=0))
    meta_track.append(mido.MetaMessage("smpte_offset"))
    # assume time_signature is 4/4
    time_sig = mido.MetaMessage("time_signature")
    time_sig = time_sig.copy(numerator=4, denominator=4, time=0)
    meta_track.append(time_sig)
    # assume key_signature is C
    key_sig = mido.MetaMessage("key_signature", time=0)
    meta_track.append(key_sig)
    # assume tempo is constant at input tempo
    set_tempo = mido.MetaMessage("set_tempo")
    set_tempo = set_tempo.copy(tempo=tempo, time=0)
    meta_track.append(set_tempo)
    # end of meta track
    end = mido.MetaMessage("end_of_track").copy(time=0)
    meta_track.append(end)

    # set up the piano; default channel is 0 everywhere; program=0 -> piano
    program = mido.Message("program_change", channel=0, program=0, time=0)
    track.append(program)
    # dummy pedal off message; control should be < 64
    cc = mido.Message("control_change", time=0)
    track.append(cc)

    # things needed for conversion
    delta_time = 0
    vel = 0

    # reconstruct the performance
    for idx in index_list:
        # if torch tensor, get item
        try:
            idx = idx.item()
        except AttributeError:
            pass
        # if pad token, continue
        if idx <= 0:
            continue
        # adjust idx to ignore pad token
        idx = idx - 1

        # note messages
        if 0 <= idx < note_on_events + note_off_events:
            # note on event
            if 0 <= idx < note_on_events:
                note = idx
                t = "note_on"
                v = vel  # get velocity from previous event
            # note off event
            else:
                note = idx - note_on_events
                t = "note_off"
                v = 127

            # create note message and append to track
            msg = mido.Message(t)
            msg = msg.copy(note=note, velocity=v, time=delta_time)
            track.append(msg)

            # reinitialize delta_time and velocity to handle subsequent notes
            delta_time = 0
            vel = 0

        # time shift event
        elif note_on_events + note_off_events <= idx < note_on_events + note_off_events + time_shift_events:
            # find cut time in range (1, time_shift_events)
            cut_time = idx - (note_on_events + note_off_events - 1)
            # scale cut_time by DIV (from vocabulary) to find time in ms; add to delta_time
            delta_time += cut_time * DIV

        # velocity event
        elif note_on_events + note_off_events + time_shift_events <= idx < total_midi_events:
            # get velocity for next note_on in range (0, 127)
            vel = bin_to_velocity(idx - (note_on_events + note_off_events + time_shift_events))

    # end the track
    end = mido.MetaMessage("end_of_track").copy(time=0)
    track.append(end)

    # append finalized track and return midi file
    mid.tracks.append(meta_track)
    mid.tracks.append(track)
    return mid
