"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import math
import numpy as np

# refer to https://github.com/SunChungEn/automatic_melody_harmonization/blob/master/tonal.py
def tonal_centroid(notes):
    fifths_lookup = {9: [1.0, 0.0], 2: [math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
                     7: [math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0: [0.0, 1.0], 5: [math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)],
                     10: [math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3: [-1.0, 0.0], 8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                     1: [math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6: [0.0, -1.0], 11: [math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)],
                     4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3: [1.0, 0.0], 7: [1.0, 0.0], 11: [1.0, 0.0],
                           0: [0.0, 1.0], 4: [0.0, 1.0], 8: [0.0, 1.0],
                           1: [-1.0, 0.0], 5: [-1.0, 0.0], 9: [-1.0, 0.0],
                           2: [0.0, -1.0], 6: [0.0, -1.0], 10: [0.0, -1.0]}
    major_thirds_lookup = {0: [0.0, 1.0], 3: [0.0, 1.0], 6: [0.0, 1.0], 9: [0.0, 1.0],
                           2: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           5: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           8: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           11: [math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           4: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           7: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)],
                           10: [math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 = 1
    r2 = 1
    r3 = 0.5
    if notes:
        for note in notes:
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        for i in range(2):
            fifths[i] /= len(notes)
            minor[i] /= len(notes)
            major[i] /= len(notes)

    return fifths + minor + major


def chord2type(chord_interval):
    interval2type = {"[4, 3]":0,"[3, 4]":1,"[3, 3]":2,"[4, 4]":3,"[3, 5]":4,"[4, 5]":5,
                     "[3, 6]":6,"[5, 4]":7,"[5, 3]":8,"[6, 3]":9,"[4, 3, 4]":10,"[4, 3, 3]":11,
                     "[3, 4, 4]":12,"[3, 4, 3]":13,"[3, 3, 4]":14,"[3, 3, 3]":15,"[3, 4, 1]":16,
                     "[3, 3, 2]":17,"[4, 4, 1]":18,"[4, 3, 2]":19,"[3, 4, 2]":20,"[4, 1, 4]":21,
                     "[3, 2, 4]":22,"[4, 1, 3]":23,"[3, 2, 3]":24,"[4, 2, 3]":25,"[1, 4, 3]":26,
                     "[2, 4, 3]":27,"[1, 3, 4]":28,"[2, 3, 4]":29,"[2, 3, 3]":30,"[0]":-1}
    return interval2type[chord_interval]


def compute_metrics(chord_sequence,chord_duration,melody_position,melody_note,melody_duration,chord_num=31):
    '''
        Calculate objective metrics for chords generated for a given melody.
    '''
    chord_intervals=[]
    for chord in chord_sequence:
        chord_interval = []
        if len(chord) > 1:
            chordOrder = chord[1:]
            for i in range(len(chordOrder) - 1):
                if chordOrder[i + 1] - chordOrder[i] > 0:
                    chord_interval.append(chordOrder[i + 1] - chordOrder[i])
                else:
                    chord_interval.append(chordOrder[i + 1] - chordOrder[i] + 12)
        else:
            chord_interval.append(0)
        chord_intervals.append(chord_interval)
    chord_type=[chord2type(str(i)) for i in chord_intervals]

    # Histogram of chord interval
    chord_statistics = np.asarray([0 for i in range(chord_num)])
    sequence_length = len(chord_type)
    for i in range(len(chord_type)):
        if chord_type[i]==-1:
            sequence_length-=1
            continue
        chord_statistics[chord_type[i]] += 1

    # CTD
    y = 0
    CTD_length=len(chord_sequence) - 1
    for n in range(len(chord_sequence) - 1):
        if len(chord_sequence[n + 1])>1 and len(chord_sequence[n])>1:
            y += np.sqrt(
                np.sum((np.asarray(tonal_centroid(chord_sequence[n + 1][1:]))
                        - np.asarray(tonal_centroid(chord_sequence[n][1:]))) ** 2))
        else:
            CTD_length-=1
    if CTD_length==0:
        CTD=0
    else:
        CTD = y / CTD_length

    assert len(melody_position)==len(melody_note)==len(melody_duration)
    chord_i=0;melody_i=0;
    chord_zip=[];melody_zip=[];melody_duration_zip=[];melody_bar_zip=[];melody_bar_duration_zip=[]
    while chord_i<len(chord_sequence):
        if len(chord_sequence[chord_i])==1:
            chord_zip.append(chord_sequence[chord_i])
        else:
            chord_zip.append(chord_sequence[chord_i][1:])
        duration_sum=0
        chord_melody=[];m_duration=[]
        while melody_i<len(melody_note):
            duration_sum+=melody_duration[melody_i]
            if melody_note[melody_i]==0:
                chord_melody.append(-1)
            else:
                chord_melody.append(melody_note[melody_i]%12)
            m_duration.append(melody_duration[melody_i]/2)
            melody_i += 1
            if duration_sum==chord_duration[chord_i]:
                break
        start_i=chord_i;end_i=chord_i
        while start_i>0:
            if melody_position[start_i]<=melody_position[start_i-1]:
                break
            start_i-=1
        while end_i<len(melody_note)-1:
            if melody_position[end_i]>=melody_position[end_i+1]:
                break
            end_i+=1
        melody_bar_zip.append([i%12 for i in melody_note[start_i:end_i+1]])
        melody_bar_duration_zip.append([i/2 for i in melody_duration[start_i:end_i+1]])
        melody_zip.append(chord_melody)
        melody_duration_zip.append(melody_duration)
        chord_i+=1

    # DC
    DC = 0
    for melody_m, chord_m in zip(melody_bar_zip, chord_zip):
        m_i=0 #不在和弦里的音符数量
        for i in range(len(melody_m)):
            m = melody_m[i]
            if m not in chord_m:
                m_i+=1
        if m_i==len(melody_m):
            DC+=1

    # PCS
    score = 0
    count = 0
    for melody_m, duration_m, chord_m in zip(melody_zip, melody_duration_zip, chord_zip):
        if len(chord_m)==1:
            continue
        for m in range(len(melody_m)):
            if melody_m[m] != -1:
                score_m=0
                for c in chord_m:
                    if abs(melody_m[m] - c) == 0 or abs(melody_m[m] - c) == 3 or abs(melody_m[m] - c) == 4 \
                            or abs(melody_m[m] - c) == 7 or abs(melody_m[m] - c) == 8 or abs(melody_m[m] - c) == 9 \
                            or abs(melody_m[m] - c) == 5:
                        if abs(melody_m[m] - c) == 5:
                            score_m += 0
                        else:
                            score_m += 1
                    else:
                        score_m += -1
                score+=score_m*duration_m[m]
                count+=duration_m[m]
    if count == 0:
        PCS = 0
    else:
        PCS = score / count

    # MCTD
    y = 0
    y_m=0
    count = 0
    for melody_m, duration_m, chord_m in zip(melody_zip, melody_duration_zip, chord_zip):
        if len(chord_m)==1:
            continue
        for m in range(len(melody_m)):
            if melody_m[m] != -1:
                y += np.sqrt(np.sum((np.asarray(tonal_centroid([melody_m[m]])) - np.asarray(tonal_centroid(chord_m)))) ** 2)
            y_m+=y*duration_m[m]
            count += duration_m[m]
    if count == 0:
        MCTD = 0
    else:
        MCTD = y / count

    # CNR
    CNR=len(chord_sequence)/len(melody_note)
    return chord_statistics,CNR,CTD,DC,PCS,MCTD