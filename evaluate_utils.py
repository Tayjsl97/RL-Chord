"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

from music21 import *
import muspy
import datetime
import pickle
import os
from data_process.midi2representation import midi2event,duration_revise,bar_position


def duration2type(duration):
    '''
        Mapping duration value to duration type
    '''
    dur2type={
        '4':0,
        '6':1,
        '8':2,
        '12':3,
        '16':4,
        '18':5,
        '24':6,
        '32':7,
        '36':8,
        '48':9,
        '72':10,
        '96':11
    }
    if duration<4:
        print("duration smaller than 4")
        return -1
    else:
        result=dur2type.get(str(duration),12)
        if result>11:
            print("wrong duration: ",result)
        return result


def duration_split(duration):
    '''
        split wrong duration
    '''
    element = [4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 72, 96]
    duration_split_list=[]
    while duration>0:
        for j in reversed(element):
            if j<=duration:
                duration_split_list.append(j)
                duration-=j
                break
    return duration_split_list


def melody2event(melody_midi):
    '''
        convert melody midi into pitch, duration and position sequences
    '''
    event = {}
    pitchs = [];durations = [];positions = [];
    melody_onset_end = [];
    streamm = converter.parse(melody_midi)
    melodys = streamm[0]
    melody_part = instrument.partitionByInstrument(melodys)
    melody_notes = melody_part.parts[0].recurse()
    melody_length = len(melody_notes)
    music = muspy.read_midi(melody_midi, 'pretty_midi')
    melody_track = music.tracks[0].notes
    melody_i = 0
    note_start = 0
    track_i = 0
    while melody_i < melody_length and track_i < len(melody_track):
        pitch_value = 0
        if not (isinstance(melody_notes[melody_i], note.Rest) or isinstance(melody_notes[melody_i], note.Note)):
            melody_i += 1
            continue
        if isinstance(melody_notes[melody_i], note.Rest):
            note_end = note_start + melody_notes[melody_i].duration.quarterLength * 24
        elif track_i == len(melody_track) - 1 or \
                (melody_i < melody_length - 1 and isinstance(melody_notes[melody_i + 1], note.Rest)):
            pitch_value = melody_notes[melody_i].pitch.midi
            if melody_track[track_i].duration == 0:  # the last note of triplet
                note_end = note_start + duration_revise(melody_track[track_i - 1].duration)
            else:
                note_end = note_start + duration_revise(melody_track[track_i].duration)
            track_i += 1
        else:
            pitch_value = melody_notes[melody_i].pitch.midi
            note_end = melody_track[track_i + 1].time
            track_i += 1
        melody_onset_end.append([pitch_value, int(note_start), int(note_end)])
        note_start = note_end
        melody_i += 1
    for i in melody_onset_end:
        if duration2type(i[2]-i[1])==12:
            duration_split_list=duration_split(i[2]-i[1])
            for j in range(len(duration_split_list)):
                if j>0:
                    positions.append(i[1]+sum(duration_split_list[0:j-1]))
                else:
                    positions.append(i[1])
                durations.append(duration_split_list[j])
                pitchs.append(i[0])
        else:
            pitchs.append(i[0])
            durations.append(i[2]-i[1])
            positions.append(i[1])
    positions=bar_position(music,positions)
    assert len(pitchs) == len(durations) == len(positions)
    event['pitchs'] = pitchs
    event['durations'] = durations
    positions=[x//2 for x in positions]
    event['positions'] = positions
    return event


def getTrainChordandDur(train_path):
    '''
        get chord and duration sequences of all training data
    '''
    chord_order_GT_list=[]
    duration_GT_list=[]
    train_files=os.listdir(train_path)
    for file in train_files:
        src=os.path.join(train_path,file)
        chord_order_GT=[]
        chord_order_new_GT = midi2event(src)['chords']
        duration_GT=midi2event(src)['durations']
        for i in range(len(chord_order_new_GT)):
            if len(chord_order_new_GT[i]) > 1:
                chord_new = chord_revise(chord_order_new_GT[i])
                chord_new = chord_transformation(chord_new)
                chord_order_new_new = [chord_new[0]]
                chord_order_new_new.extend(chord_new[2:])
                chord_order_GT.append(chord_order_new_new)
                chord_order_GT[i][0] = chord_order_GT[i][0] - 2
            else:
                chord_order_GT.append([0])
        if len(chord_order_GT)==len(duration_GT):
            chord_order_GT_list.append(chord_order_GT)
            duration_GT_list.append(duration_GT)
        else:
            print("length error")
    return chord_order_GT_list,duration_GT_list


def chord_transformation(chord):
    '''
        restore the inverted chord according to the chord inversion type
    '''
    new_chord=[chord[0],chord[1]]
    trans_chord=chord[2:]
    trans_chord=sorted(trans_chord)
    chord_len=len(trans_chord)
    if chord_len==3:
        if chord[1]==0:
            # root position
            new_chord.extend(trans_chord)
        elif chord[1]==2:
            # first inversion
            new_chord.extend([trans_chord[1],trans_chord[2],trans_chord[0]])
        elif chord[1]==1:
            # second inversion
            new_chord.extend([trans_chord[2],trans_chord[0],trans_chord[1]])
        else:
            new_chord.extend(trans_chord)
            print(chord)
            print("error  3")
    else:
        if chord[1]==0:
            # root position
            new_chord.extend(trans_chord)
        elif chord[1]==3:
            # first inversion
            new_chord.extend([trans_chord[1], trans_chord[2], trans_chord[3], trans_chord[0]])
        elif chord[1]==2:
            # second inversion
            new_chord.extend([trans_chord[2], trans_chord[3], trans_chord[0], trans_chord[1]])
        elif chord[1]==1:
            # third inversion
            new_chord.extend([trans_chord[3], trans_chord[0], trans_chord[1], trans_chord[2]])
        else:
            new_chord.extend(trans_chord)
            print(chord)
            print("error  4")
    return new_chord


def merge_chord(chords,positions,durations):
    '''
        merge the same chords within a bar
    '''
    new_chords=[]
    new_durations=[]
    assert len(chords)==len(positions)==len(durations)
    i=0
    while i < len(chords):
        start = i
        while i<len(chords)-1 and chords[i]==chords[i+1]:
            i+=1
        i+=1
        end=i
        slice_start=start
        for j in range(start,end):
            if j<end-1 and positions[j]>=positions[j+1]:
                new_chords.append(chords[start])
                new_durations.append(sum(durations[slice_start:j+1]))
                slice_start=j+1
            if j==end-1:
                new_chords.append(chords[start])
                new_durations.append(sum(durations[slice_start:end]))
    return new_chords,new_durations


def chord_revise(chord):
    '''
        Best matching principle, to convert the chords that are not triads or sevenths into triads or sevenths.
        chord: [octave, chord_inversion_type (cit), order1, order2, order3...],
        orders are the notes forming a chord divided by 12, with order1 representing the root note, order2 representing the third note...
    '''
    chord_type_3 = [[4, 3], [3, 4], [3, 3], [4, 4], [3, 5], [4, 5], [3, 6], [5, 4], [5, 3], [6, 3]]
    chord_type_4 = [[4, 3, 4], [4, 3, 3], [3, 4, 4], [3, 4, 3], [3, 3, 4], [3, 3, 3],
                    [3, 4, 1], [3, 3, 2], [4, 4, 1], [4, 3, 2], [3, 4, 2],
                    [4, 1, 4], [3, 2, 4], [4, 1, 3], [3, 2, 3], [4, 2, 3],
                    [1, 4, 3], [2, 4, 3], [1, 3, 4], [2, 3, 4], [2, 3, 3]]
    if len(chord)>1:
        new_chord = [chord[0],chord[1]]
        chord_order=chord[2:]
        chord_interval=[]
        for i in range(len(chord_order)-1):
            if chord_order[i+1]-chord_order[i]>0:
                chord_interval.append(chord_order[i+1]-chord_order[i])
            else:
                chord_interval.append(chord_order[i + 1]-chord_order[i]+12)
        # triad
        if len(chord_interval)==2:
            # wrong chord interval
            if chord_interval not in chord_type_3:
                for j in chord_type_3:
                    if chord_interval[0]==j[0]:
                        chord_order[2]=(chord_order[1]+j[1])%12
                        new_chord.extend(chord_order)
                        return new_chord
                for j in chord_type_3:
                    if chord_interval[1]==j[1]:
                        chord_order[0]=(chord_order[1]-j[0]+12)%12
                        new_chord.extend(chord_order)
                        return new_chord
                chord_order[1]=(chord_order[0]+4)%12
                chord_order[2]=(chord_order[1]+3)%12
                new_chord.extend(chord_order)
                return new_chord
            else:
                new_chord.extend(chord_order)
                return new_chord
        # seventh
        else:
            # wrong chord interval
            if chord_interval not in chord_type_4:
                min_differ=10
                for j in chord_type_4:
                    differ=abs(chord_interval[0]-j[0])+abs(chord_interval[1]-j[1])+abs(chord_interval[2]-j[2])
                    if differ<min_differ:
                        min_chord_interval = []
                        min_differ=differ
                        min_chord_interval.extend(j)
                if min_chord_interval[2]==chord_interval[2] and min_chord_interval[1]==chord_interval[1]:
                    chord_order[2] = (chord_order[3] - min_chord_interval[2]+12) % 12
                    chord_order[1] = (chord_order[2] - min_chord_interval[1]+12) % 12
                    chord_order[0] = (chord_order[1] - min_chord_interval[0]+12) % 12
                else:
                    chord_order[1]=(chord_order[0]+min_chord_interval[0])%12
                    chord_order[2]=(chord_order[1]+min_chord_interval[1])%12
                    chord_order[3]=(chord_order[2]+min_chord_interval[2])%12
                new_chord.extend(chord_order)
                return new_chord
            else:
                new_chord.extend(chord_order)
                return new_chord
    else:
        return chord


def revise_bar(chord, pitches, position, duration, time):
    '''
        Delete the ties across bars.
    '''
    i=0
    while i<len(duration):
        duration_sum = 0
        for j in range(i+1):
            duration_sum+=duration[j]
        bar_duration=0
        for j in range(len(time)):
            if duration_sum <time[j].time:
                bar_duration= (96 * time[j].numerator) // time[j].denominator
        if bar_duration==0:
            bar_duration = (96 * time[0].numerator) // time[0].denominator
        if duration[i]>bar_duration or \
                (duration[i]+duration[i-1]>bar_duration and (duration_sum-duration[i])%bar_duration!=0):
            if (duration_sum-duration[i])%bar_duration==0:
                duration_tmp=duration[i]
                while duration_tmp>=bar_duration:
                    duration.insert(i,bar_duration)
                    chord.insert(i,chord[i])
                    pitches.insert(i,pitches[i])
                    duration_tmp-=bar_duration
                    if duration_tmp!=0:
                        position.insert(i + 1, 0)
                    i+=1
                if duration_tmp!=0:
                    duration.insert(i,duration_tmp)
                    chord.insert(i, chord[i])
                    pitches.insert(i, pitches[i])
                    i+=1
                del duration[i],chord[i],pitches[i]
            else:
                duration_sum -= duration[i]
                duration.insert(i, bar_duration-(duration_sum)%bar_duration)
                pitches.insert(i, pitches[i])
                chord.insert(i, chord[i])
                position.insert(i+1,0)
                duration[i+1]-=(bar_duration-(duration_sum)%bar_duration)
                i+=1
        i+=1
    return chord,pitches,position,duration


def longSubStr(str1,str2,longest_length):
    '''
        compute the longest subsequence
    '''
    len1 = len(str1)
    len2 = len(str2)
    longest,start1,start2 = 0,0,0
    c = [[0 for i in range(len2+1)]for i in range(len1+1)]
    for i in range(len1+1):
        for j in range(len2+1):
            if i == 0 or j == 0:
                c[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                if not (isinstance(str1[i-1],int) and c[i-1][j-1]==0):
                    c[i][j] = c[i-1][j-1]+1
            else:
                c[i][j] = 0
            if (longest < c[i][j]):
                longest = c[i][j]
                start1 = i-longest
                start2 = j-longest
    subStr=str1[start1:start1+longest]
    # save time
    temp_num=0
    for i in subStr:
        if isinstance(i,int):
            temp_num+=i
    if temp_num<=longest_length or len(subStr)<=1:
        return 0
    num=0
    if isinstance(subStr[0],int):
        del subStr[0]
    if len(subStr)>=1:
        if not isinstance(subStr[-1],int):
            del subStr[-1]
    if len(subStr)<2:
        return 0
    else:
        for i in subStr:
            if isinstance(i,int):
                num+=i
        return num


def compute_corpus_level_copy(chord,duration,GT_chord_list,GT_dur_list):
    '''
        compute the length of the longest chord progression subsequence copied from the training set
    '''
    chord_with_dur=[]
    for i in range(len(chord)):
        chord_with_dur.append(chord[i])
        chord_with_dur.append(duration[i])
    longest_length=0
    start = datetime.datetime.now()
    for i in range(len(GT_chord_list)):
        GT_one=[]
        for j in range(len(GT_chord_list[i])):
            GT_one.append(GT_chord_list[i][j])
            GT_one.append(GT_dur_list[i][j])
        length=longSubStr(chord_with_dur,GT_one,longest_length)
        if length>longest_length:
            longest_length=length
    end = datetime.datetime.now()
    print('traverse one time is ', end - start)
    return longest_length/24


def hist_sim(hist_path,hist_GT_path):
    '''
        compute metrics chord histogram similarity (CHS)
    '''
    file = open(hist_GT_path, 'rb')
    GT_hist = pickle.load(file)
    file = open(hist_path, 'rb')
    compare_hist = pickle.load(file)
    CHS=0
    for i in range(len(GT_hist)):
        gt=GT_hist[i]
        com=compare_hist[i]
        similarity=0
        for j in range(len(gt)):
            if gt[j]==0 and com[j]==0:
                similarity += 1
            else:
                similarity+=(1-abs(gt[j]-com[j])/max(gt[j],com[j]))
        similarity /= len(gt)
        CHS+=similarity
    CHS /= len(GT_hist)
    print("CHS: ",CHS)