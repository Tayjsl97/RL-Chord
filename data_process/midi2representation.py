"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import os
from music21 import *
import muspy
import pickle
from argparse import ArgumentParser


def duration_revise(duration):
    '''
        Correct abnormal durations.
    '''
    if duration%4==0 or duration%6==0:
        return duration
    else:
        patch_4=duration+(4-duration%4)
        patch_6=duration+(6-duration%6)
        if patch_4<patch_6:
            return patch_4
        else:
            return patch_6


def note_chord_align(note,chord):
    '''
        aligh melody note and chord
    '''
    note_i=0;chord_i=0
    new_note=[];
    note_len=len(note);chord_len=len(chord)
    while note_i<note_len and chord_i<chord_len:
        chord_dur_range=range(chord[chord_i][1],chord[chord_i][2]+1)
        if note[note_i][1] in chord_dur_range and note[note_i][2] in chord_dur_range:
            new_note.append(note[note_i])
            note_i+=1
        elif note[note_i][2] not in chord_dur_range and note[note_i][1]<chord[chord_i][2]:
            new_note.append([note[note_i][0],note[note_i][1],chord[chord_i][2]])
            note[note_i]=[note[note_i][0],chord[chord_i][2],note[note_i][2]]
            chord_i+=1
        else:
            chord_i+=1
    return new_note



def chord_dur_split(chord_onset_offset,path,type):
    '''
        Split durations greater than 96
    '''
    element = [4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 72, 96]
    new_chord=[]
    length = len(element)
    for i in chord_onset_offset:
        duration=i[2]-i[1]
        duration_split = []
        if duration not in element:
            while duration not in element:
                if duration < 4:
                    print("error: ",duration,", type: ", type,i[2],path)
                    return False
                for j in range(length):
                    if element[j] < duration and (j == length - 1 or (j < length - 1 and element[j + 1] > duration)):
                        duration_split.append(element[j])
                        duration -= element[j]
                        break
            duration_split.append(duration)
        else:
            duration_split.append(duration)
        start=i[1]
        for j in duration_split:
            new_chord.append([i[0],start,start+j])
            start=start+j
    return new_chord


def bar_position(music,positions,timr_r=96):
    '''
        transform the note position into the within-bar position
    '''
    new_positions=[]
    key_i=0
    position_i=0
    times=music.time_signatures
    while key_i < len(times):
        bar = (timr_r * times[key_i].numerator) // times[key_i].denominator
        start=times[key_i].time
        if key_i+1<len(music.time_signatures):
            end=times[key_i+1].time
        else:
            end=positions[-1]+1
        while position_i<len(positions):
            if start <= positions[position_i] < end:
                new_positions.append((positions[position_i]-start)%bar)
                position_i+=1
            else:
                break
        key_i+=1
    return new_positions


def midi2event(midi_path):
    '''
        transform a midi file into the event sequences
    '''
    event={}
    pitchs=[];durations=[];positions=[];chords=[]
    melody_onset_offset = [];chord_onset_offset = []
    streamm = converter.parse(midi_path)
    melodys=streamm[0]
    chordss=streamm[1]
    melody_part = instrument.partitionByInstrument(melodys)
    chord_part = instrument.partitionByInstrument(chordss)
    melody_notes = melody_part.parts[0].recurse()
    chord_notes = chord_part.parts[0].recurse()
    melody_length=len(melody_notes)
    chord_length=len(chord_notes)
    music = muspy.read_midi(midi_path, 'pretty_midi')
    melody_track = music.tracks[0].notes
    melody_i = 0;chord_i = 0
    chord_start = 0;
    note_start=0;
    while chord_i < chord_length:
        if not (isinstance(chord_notes[chord_i],note.Rest) or isinstance(chord_notes[chord_i],chord.Chord)):
            chord_i+=1
            continue
        if isinstance(chord_notes[chord_i], note.Rest):
            chord_tmp=[0]
        else:
            chord_tmp=[]
            chord_tmp.append(chord_notes[chord_i].pitches[0].octave)
            pitch_string=list(chord_notes[chord_i])
            pitch_value=[i.pitch.midi for i in pitch_string]
            normalOrder_tmp=[i%12 for i in pitch_value]
            if len(normalOrder_tmp)>4:
                print(normalOrder_tmp,midi_path)
            chord_tmp.append(normalOrder_tmp.index(min(normalOrder_tmp)))
            chord_tmp.extend(normalOrder_tmp)
        chord_end=chord_start+chord_notes[chord_i].duration.quarterLength*24
        chord_onset_offset.append([chord_tmp,int(chord_start),int(chord_end)])
        chord_start=chord_end
        chord_i+=1
    track_i=0
    while melody_i < melody_length and track_i<len(melody_track):
        pitch_value=0
        if not (isinstance(melody_notes[melody_i],note.Rest) or isinstance(melody_notes[melody_i],note.Note)):
            melody_i+=1
            continue
        if isinstance(melody_notes[melody_i],note.Rest):
            note_end=note_start+melody_notes[melody_i].duration.quarterLength*24
        elif track_i==len(melody_track)-1 or \
            (melody_i<melody_length-1 and isinstance(melody_notes[melody_i+1],note.Rest)):
            pitch_value = melody_notes[melody_i].pitch.midi
            if melody_track[track_i].duration==0:
                note_end = note_start + duration_revise(melody_track[track_i-1].duration)
            else:
                note_end = note_start+duration_revise(melody_track[track_i].duration)
            track_i += 1
        else:
            pitch_value = melody_notes[melody_i].pitch.midi
            note_end = melody_track[track_i+1].time
            track_i+=1
        melody_onset_offset.append([pitch_value,int(note_start),int(note_end)])
        note_start=note_end
        melody_i+=1
    chord_onset_offset=chord_dur_split(chord_onset_offset,midi_path,type="c")
    melody_onset_offset=note_chord_align(melody_onset_offset,chord_onset_offset)
    melody_onset_offset = chord_dur_split(melody_onset_offset,midi_path,type="m")
    if melody_onset_offset is False:
        return False
    note_i=0
    for chorddd in chord_onset_offset:
        chord_start=chorddd[1];chord_end=chorddd[2]
        while note_i<len(melody_onset_offset):
            pitch_value=melody_onset_offset[note_i][0]
            note_start=melody_onset_offset[note_i][1]
            note_end=melody_onset_offset[note_i][2]
            if note_start in range(chord_start,chord_end+1) and \
                note_end in range(chord_start,chord_end+1):
                pitchs.append(pitch_value)
                durations.append(note_end-note_start)
                positions.append(note_start)
                chords.append(chorddd[0])
                note_i+=1
            else:
                break
    positions=bar_position(music,positions)
    assert len(pitchs)==len(durations)==len(positions)==len(chords)
    event['pitchs']=pitchs
    event['durations']=durations
    event['bars']=positions
    event['chords']=chords
    return event


def store(events,path):
    '''
        event saving func
    '''
    new_path = args.events_path + args.dataset
    if path.find("train") != -1:
        new_path += "_train_"
    else:
        new_path += "_test_"
    new_path += str(len(events))+".data"
    print("new_path: ",new_path)
    file=open(new_path,'wb')
    pickle._dump(events,file)
    file.close()


def midis2events(path):
    '''
        transform every midi file into its event sequences
    '''
    midi_files = os.listdir(path)
    for midi_file in midi_files:
        midi_path = os.path.join(path, midi_file)
        if os.path.isdir(midi_path):
            events=[]
            midi_files = os.listdir(midi_path)
            for midi_file in midi_files:
                midi_file_one = os.path.join(midi_path, midi_file)
                if midi_file.endswith(".mid"):
                    event=midi2event(midi_file_one)
                    if event is not False:
                        events.append(event)
            store(events,midi_path)
        else:
            print("Don't have train/test midi path")



def slide_window(path,window,stride):
    '''
        Partition the event sequence into equal-length subsequences
    '''
    slide_path="data/slide_win"+str(window)+"_stride"+str(stride)
    if not os.path.exists(slide_path):
        os.makedirs(slide_path)
    event_files=os.listdir(path)
    for event_file in event_files:
        index = event_file.rfind("_")
        new_events = []
        event_path=os.path.join(args.events_path,event_file)
        file = open(event_path, 'rb')
        events = pickle.load(file)
        length = len(events)  # number of midi
        for i in range(length):
            flag=0
            event = events[i]
            assert len(event['pitchs']) == len(event['durations']) == len(event['bars']) == len(event['chords'])
            event_length = len(event['pitchs'])
            if event_length>=window:
                j = 0
                while j < event_length:
                    if j + window < event_length:
                        event_tmp = {}
                        event_tmp['pitchs'] = event['pitchs'][j:j + window]
                        event_tmp['durations'] = event['durations'][j:j + window]
                        event_tmp['bars'] = event['bars'][j:j + window]
                        event_tmp['bars'] = [x // 2 for x in event_tmp['bars']]
                        event_tmp['chords'] = event['chords'][j:j + window]
                        new_events.append(event_tmp)
                    if flag==1:
                        break
                    else:
                        j += stride
        new_path = os.path.join(slide_path, event_file[:index])
        print("new_path: ", new_path)
        file = open(new_path + "_" + str(len(new_events)) + ".data", 'wb')
        pickle._dump(new_events, file)
        file.close()


if __name__=="__main__":
    parser = ArgumentParser(description='midi2representation')
    parser.add_argument("--dataset", type=str, default='NMD', help="NMD or Wiki or TTD")
    parser.add_argument("--midi_path", type=str, default=None,
                        help="The midi folder contains several subfolders, "
                             "each storing training, validation, or testing MIDI files respectively.")
    parser.add_argument("--events_path", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride_size", type=int, default=8)
    args = parser.parse_args()

    midi_path=args.midi_path
    midis2events(midi_path)

    slide_window(args.event_path, int(args.window_size), int(args.stride_size))
