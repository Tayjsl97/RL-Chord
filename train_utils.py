"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import math
import torch
from evaluate_utils import duration2type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chord2binary(chord):
    '''
        transform [octave, chord_inversion_type (cit), order1, order2, order3...] into 20-D multi-hot representation
    '''
    chord_3=[0]*3
    chord_4 = [0] * 4
    chord_13=[0]*13
    if len(chord)==1:
        chord_3[0]=1
    else:
        chord_3[chord[0]-1] = 1
        chord_4[chord[1]]=1
        for i in range(2,len(chord)):
            chord_13[chord[i]]=1
        if len(chord[2:])==3:
            chord_13[-1] = 1
    chord_3.extend(chord_4)
    chord_3.extend(chord_13)
    return chord_3


def batch_data_win(datas,batch_size,condition_window,seq_len):
    '''
        prepare one batch data
    '''
    one_batch = {}
    one_batch['condition'] = {'pitches': [], 'durations': [], 'positions': []}
    one_batch['note_t'] = {'pitches': [], 'durations': [], 'positions': []}
    one_batch['chords'] = []
    chord_real_all=[]
    chord_0 = [0] * 20
    chord_0[0] = 1
    chord_tmp=[]
    for i in range(batch_size):
        chord_tmp.append(chord_0)
    one_batch['chords'].append(chord_tmp)
    for t in range(seq_len):
        if (t - condition_window / 2) >= 0 and (t + condition_window / 2 - 1) < seq_len:
            window_start = int(t - condition_window / 2)
            window_end = int(t + condition_window / 2)
        elif t - condition_window / 2 < 0:
            window_start = 0
            window_end = int(condition_window)
        else:
            window_start = int(seq_len - condition_window)
            window_end = int(seq_len)
        pitch = []
        duration = []
        position = []
        chord = []
        chord_real=[]
        pitch_tt = []
        duration_tt = []
        position_tt = []
        for i in range(batch_size):
            pitch_temp=datas[i]['pitchs'][window_start:window_end]
            for j in range(len(pitch_temp)):
                if pitch_temp[j]!=0:
                    pitch_temp[j]-=47
            pitch.append(pitch_temp)
            d_temp=[]
            for hhh in datas[i]['durations'][window_start:window_end]:
                d_temp.append(duration2type(hhh))
            duration.append(d_temp)
            position.append(datas[i]['bars'][window_start:window_end])
            chord_before=datas[i]['chords'][t]
            chord_after=chord_revise(chord_before)
            chord.append(chord2binary(chord_after))
            chord_real_tmp=[chord_after[0]-2]
            chord_real_tmp.extend(chord_after[1:])
            chord_real.append(chord_real_tmp)
            pitch_t = [0] * 49;
            duration_t = [0] * 12;
            position_t = [0] * 72
            if datas[i]['pitchs'][t] == 0:
                pitch_t[0] = 1
            else:
                pitch_t[datas[i]['pitchs'][t] - 47] = 1
            dur = duration2type(datas[i]['durations'][t])
            duration_t[dur] = 1
            position_t[datas[i]['bars'][t]] = 1
            pitch_tt.append(pitch_t)
            duration_tt.append(duration_t)
            position_tt.append(position_t)
        one_batch['condition']['pitches'].append(pitch)
        one_batch['condition']['durations'].append(duration)
        one_batch['condition']['positions'].append(position)
        one_batch['note_t']['pitches'].append(pitch_tt)
        one_batch['note_t']['durations'].append(duration_tt)
        one_batch['note_t']['positions'].append(position_tt)
        one_batch['chords'].append(chord)
        chord_real_all.append(chord_real)
    return one_batch,chord_real_all


def rule_rewards(melody1,duration1,position1,chordOrder1,chord_t_1,Reward_R1,Reward_R2):
    '''
        Calculate harmony reward and chord progression reward
    '''
    #  interval consonance between the melody and chords
    duration_sum = 0
    score_between = 0
    chord_len=len(chordOrder1[2:])
    for ii in range(len(melody1)):
        score_b = 0
        for jj in chordOrder1[2:]:
            score_b += getHarmonicLevel(jj, melody1[ii]%12)
        score_between += score_b * type2duration(duration1[ii])
        duration_sum += type2duration(duration1[ii])
    if chord_len==0:
        score_between=0
    else:
        score_between = score_between / (duration_sum*chord_len)

    # consonance of the chord Itself
    score_self=0
    if len(chordOrder1)>1:
        chord_tmp=chordOrder1[2:]
    else:
        chord_tmp=chordOrder1
    chord_len = len(chord_tmp)
    for ii in range(chord_len):
        for jj in range(ii + 1, chord_len):
            score_self += getHarmonicLevel(chord_tmp[ii], chord_tmp[jj])
    score_self = score_self / (chord_len * chord_len)
    score = score_between + score_self

    # chord progression reward
    # repetition
    if position1[0]==0:
        if chordOrder1 == chord_t_1:
            repetition = -1
        else:
            repetition = 0
    else:
        if chordOrder1 == chord_t_1:
            repetition = 1
        else:
            repetition = 0

    # superstrong
    if len(chordOrder1) > 1 and len(chord_t_1) > 1:
        chord_now_pitch = chord2pitch(chordOrder1)
        chord_t_1_pitch = chord2pitch(chord_t_1)
        cnt=0
        for iii in chord_now_pitch:
            if iii in chord_t_1_pitch:
                cnt+=1
        if cnt == 0:
            superstrong = -1
        else:
            superstrong = 0
    else:
        superstrong = 0
    progression_penalty = repetition + superstrong
    return Reward_R1*score+Reward_R2*progression_penalty


def Reward(chord,chord_real,action1,action2,action4,action13,melody_pitch,melody_duration,melody_position,
           Reward_L,Reward_R1,Reward_R2,batch_size,seq_len,condition_window,harmony_rule_window,
           criterion_1,criterion_2,criterion_4,criterion_13):
    '''
        Compute the rewards (with the expection of mutual reward) for all actions
    '''
    # negative loss reward
    chord_gt = torch.Tensor(chord[1:]).to(device)
    chord_gt_1 = chord_gt.narrow(2, 0, 1)
    _,chord_gt_2 = chord_gt.narrow(2, 1, 2).topk(1)
    _,chord_gt_4 = chord_gt.narrow(2, 3, 4).topk(1)
    chord_gt_2 = torch.LongTensor(chord_gt_2.squeeze(-1).cpu().numpy()).to(device)
    chord_gt_4 = torch.LongTensor(chord_gt_4.squeeze(-1).cpu().numpy()).to(device)
    chord_gt_13 = chord_gt.narrow(2, 7, 13)
    action2_oh = torch.Tensor(nn.functional.one_hot(action2.squeeze(-1), 2).cpu().numpy()).to(device)
    action4_oh = torch.Tensor(nn.functional.one_hot(action4.squeeze(-1), 4).cpu().numpy()).to(device)
    l_2=[];l_4=[]
    for i in range(seq_len):
        l_2_tmp=criterion_2(action2_oh[i], chord_gt_2[i])
        l_4_tmp = criterion_4(action4_oh[i], chord_gt_4[i])
        l_2.append(l_2_tmp)
        l_4.append(l_4_tmp)
    l_1 = criterion_1(action1, chord_gt_1).sum(dim=-1)
    l_2 = torch.stack(l_2)
    l_4 = torch.stack(l_4)
    l_13 = criterion_13(action13 / 1000, chord_gt_13)
    loss_reward = -(l_1 + l_2 + l_4 + l_13).detach().to(device)

    # identical pitch reward
    _, indices13 = action13.topk(5)
    chord_order=torch.cat((action1,action2, action4, indices13),dim=-1).int().cpu().numpy().tolist()
    chordOrder = ChordOrder(chord_order)
    chord_gt=chord_real
    pitch_diff=[]
    for ii in range(seq_len):
        seq_t=[]
        for jj in range(batch_size):
            action_one = set(chord2pitch(chordOrder[ii][jj]))
            action_gt = set(chord2pitch(chord_gt[ii][jj]))
            seq_t.append(len(action_one&action_gt))
        pitch_diff.append(seq_t)
    pitch_diff=torch.Tensor(pitch_diff).detach().to(device)

    # compute rule reward including interval consonance, repetition, and superstrong
    melody_pitch_new=[]
    melody_duration_new=[]
    melody_position_new = []
    for i in range(seq_len):
        chord_index=i
        if chord_index - int((condition_window) / 2) >= 0 and chord_index + int((condition_window) / 2) <= seq_len:
            start = 4 - harmony_rule_window // 2
            bar_pos=4
        else:
            if chord_index < int((condition_window) / 2):
                if chord_index == 0:
                    start = 0
                    bar_pos=0
                else:
                    start = chord_index - harmony_rule_window // 2
                    bar_pos=chord_index
            else:
                chord_index_tmp = chord_index - seq_len + condition_window
                bar_pos = chord_index_tmp
                if chord_index_tmp+1<condition_window-1:
                    start = chord_index_tmp - harmony_rule_window // 2
                else:
                    end=condition_window-1
                    start=end-harmony_rule_window+1
        melody_pitch_new.append(melody_pitch[i].narrow(1,start,harmony_rule_window))
        melody_duration_new.append(melody_duration[i].narrow(1, start, harmony_rule_window))
        melody_position_new.append(melody_position[i].narrow(1, bar_pos, 1))
    melody_pitch_new=torch.stack(melody_pitch_new).cpu().numpy().tolist()
    melody_duration_new = torch.stack(melody_duration_new).cpu().numpy().tolist()
    melody_position_new= torch.stack(melody_position_new).cpu().numpy().tolist()
    rule_reward = []
    for seq in range(seq_len):
        reward_t = []
        for batch in range(batch_size):
            chord_t_1 = chord[seq][batch]
            chord_t_1 = binary2chord(chord_t_1)
            reward_t.append(rule_rewards(melody_pitch_new[seq][batch],melody_duration_new[seq][batch],melody_position_new[seq][batch],
                                        chordOrder[seq][batch],chord_t_1,Reward_R1,Reward_R2))
        rule_reward.append(reward_t)
    rule_reward=torch.Tensor(rule_reward).detach().to(device)
    total_reward = (Reward_L * loss_reward+pitch_diff+ rule_reward).detach().unsqueeze(-1)
    return total_reward


def multilabel_categorical_crossentropy(s_pred,cps_gt):
    '''
        Compute cps loss (Eq. 7) in [1].
        [1] Ji S, Yang X, Luo J, et al. RL-Chord: CLSTM-Based Melody Harmonization Using Deep Reinforcement Learning.
    '''
    s_pred = (1 - 2 * cps_gt) * s_pred
    s_pred_neg = s_pred - cps_gt * 1e8
    s_pred_pos = s_pred - (1 - cps_gt) * 1e8
    zeros = torch.zeros_like(s_pred[..., :1])
    s_pred_neg = torch.cat([s_pred_neg, zeros], dim=-1)
    s_pred_pos = torch.cat([s_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(s_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(s_pred_pos, dim=-1)
    return neg_loss + pos_loss


def timeSince(since):
    '''
        time interval from the start time until now
    '''
    now=time.time()
    s=now-since
    h=math.floor(s/3600)
    s-=h*3600
    m=math.floor(s/60)
    s-=m*60

    return '%dh_%dm_%ds' % (h, m, s)


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


def ChordOrder(chord_order):
    '''
        Restore the actual pitch ordering of chords according to the inversion type
    '''
    chord_order_new=[]
    for j in range(len(chord_order)):
        chord_order_t=[]
        for i in range(len(chord_order[0])):
            if chord_order[j][i][0]==1:
                chord_order_t.append([0])
                continue
            elif 12 in chord_order[j][i]:
                chord_order[j][i].remove(12)
                if chord_order[j][i][2]<3:
                    chord_order_t.append(chord_transformation(chord_order[j][i][1:-1]))
                else:
                    chord_order_t.append(chord_transformation(chord_order[j][i][1:]))
                continue
            else:
                chord_order_t.append(chord_transformation(chord_order[j][i][1:-1]))
        chord_order_new.append(chord_order_t)
    return chord_order_new


def getHarmonicLevel(note1,note2):
    '''
        Compute the score of interval consonance between two notes
    '''
    pd=abs(note1-note2)
    if pd==0:
        return 10
    elif pd==5 or pd==7:
        return 8
    elif pd==4 or pd==8:
        return 6
    elif pd==3 or pd==9:
        return 5
    elif pd==2 or pd==10:
        return 3
    elif pd==1 or pd==11:
        return 1
    else:
        return 0


def binary2chord(chord):
    '''
        transform multi-hot chord representation into [octave, chord_inversion_type (cit), order1, order2, order3...]
    '''
    chord_tmp = []
    if chord[0] == 1:
        chord_tmp.append(0)
    else:
        root = chord[1:3].index(1)
        type = chord[3:7].index(1)
        chord_pitch = chord[7:]
        pitch_set = [i for i in range(12) if chord_pitch[i] == 1]
        chord_tmp.append(root)
        chord_tmp.append(type)
        chord_tmp.extend(pitch_set)
        chord_tmp=chord_transformation(chord_tmp)
    return chord_tmp


def chord2pitch(chord):
    '''
        get the pitch set of the given multi-hot chord representation
    '''
    chords = []
    if len(chord) != 1:
        chord_tmp=[chord[0]]
        chord_tmp.extend(chord[2:])
        if chord_tmp[0] == 0:
            t = 2
        if chord_tmp[0] == 1:
            t = 3
        offset = 0
        for j in range(1, len(chord_tmp)):
            if j > 1 and chord_tmp[j] < chord_tmp[j - 1]:
                offset += 1
            pitch = 12 + (t + offset) * 12 + chord_tmp[j]
            chords.append(pitch)
    else:
        chords.append(0)
    return chords


def type2duration(d):
    '''
        Mapping duration type to duration value
    '''
    dur2type = {
        '0': 2,
        '1': 3,
        '2': 4,
        '3': 6,
        '4': 8,
        '5': 9,
        '6': 12,
        '7': 16,
        '8': 18,
        '9': 24,
        '10': 36,
        '11': 48
    }
    result = dur2type[str(d)]
    return result


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


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def plot_RL(num, rewards, path, model_name):
    '''
        plot the reward variation.
    '''
    clear_output(True)
    plt.title('reward: %s' % (rewards[-1]))
    plt.plot(num,rewards)
    plt.savefig(f"{path}/{model_name}_len{str(len(num))}.png")
    plt.close()
