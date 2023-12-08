"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
from model.BLSTM_Chord_MH import LSTM_Chord as BLSTM_Chord_MH
from model.BLSTM_Chord_OH import LSTM_Chord as BLSTM_Chord_OH
from model.CLSTM_Chord_MH import LSTM_Chord as CLSTM_Chord_MH
from model.CLSTM_Chord_OH import LSTM_Chord as CLSTM_Chord_OH
import pickle
import random
import muspy
from data_process.midi2representation import midi2event
from chord_metrics import compute_metrics
from evaluate_utils import melody2event, chord_revise, chord_transformation, revise_bar, \
    merge_chord,compute_corpus_level_copy, getTrainChordandDur, hist_sim, duration2type
import os
from argparse import ArgumentParser


def batch_data_win(datas,condition_window,seq_len, repre, chord_num):
    '''
        prepare one batch data (batch size = 1)
    '''
    one_batch = {}
    one_batch['condition'] = {'pitches': [], 'durations': [], 'positions': []}
    one_batch['chords'] = []
    if repre == "MH":
        chord_0 = [0] * 20
        chord_0[0] = 1
    elif repre=="OH":
        type0 = chord_type.index('[0]')
        chord_0 = [0] * chord_num
        chord_0[type0] = 1
    chord_tmp=[]
    chord_tmp.append(chord_0)
    one_batch['chords'].append(chord_tmp)
    for t in range(seq_len):
        if t - condition_window / 2 >= 0 and t + condition_window / 2 - 1 < seq_len:
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
        pitch_temp=datas['pitchs'][window_start:window_end]
        for i in range(len(pitch_temp)):
            if pitch_temp[i]!=0:
                pitch_temp[i]-=47
        pitch.append(pitch_temp)
        d_temp=[]
        for hhh in datas['durations'][window_start:window_end]:
            d_temp.append(duration2type(hhh))
        duration.append(d_temp)
        position.append(datas['bars'][window_start:window_end])
        one_batch['condition']['pitches'].append(pitch)
        one_batch['condition']['durations'].append(duration)
        one_batch['condition']['positions'].append(position)
    return one_batch


def generate_BLSTM(src,dis):
    '''
        generate chord for given melody (one sample)
    '''
    event=melody2event(src)
    seq_len=len(event["bars"])
    one_batch=batch_data_win(event,condition_window,seq_len)
    chord_order = []
    condition = one_batch['condition']
    hidden = None
    condition_pitch = torch.LongTensor(condition['pitches']).to(device)
    condition_duration = torch.LongTensor(condition['durations']).to(device)
    condition_position = torch.LongTensor(condition['positions']).to(device)
    condition_new = torch.cat([condition_pitch, condition_duration, condition_position], dim=-1).to(device)
    if args.repre == "MH":
        output_1, output_2, output_4, output_13, hidden = model(condition_new, hidden)
        for i in range(seq_len):
            if output_1[i][0].item() > 0.5:
                topi_1 = torch.Tensor([1]).to(device)
            else:
                topi_1 = torch.Tensor([0]).to(device)
            topv, topi_2 = output_2[i][0].topk(1)
            topv, topi_4 = output_4[i][0].topk(1)
            topv, topi_13 = output_13[i][0].topk(5)
            ctmp = torch.cat((topi_1, topi_2, topi_4, topi_13), dim=-1).int().cpu().numpy().tolist()
            if ctmp[0] == 1:
                chord_order.append([0])
            else:
                # judge whether it is a triad or a seventh chord
                if 12 in ctmp:
                    ctmp.remove(12)
                    if ctmp[2] < 3:
                        chord_order_new = chord_transformation(ctmp[1:-1])
                    else:
                        chord_order_new = chord_transformation(ctmp[1:])
                else:
                    chord_order_new = chord_transformation(ctmp[1:-1])
                chord_order_new = chord_revise(chord_order_new)
                chord_order_new = chord_transformation(chord_order_new)
                chord_order_new_new=[chord_order_new[0]]
                chord_order_new_new.extend(chord_order_new[2:])
                chord_order.append(chord_order_new_new)
    elif args.repre == "OH":
        output, hidden = model(condition_new, hidden)
        top, topi = output.topk(1)
        for i in range(seq_len):
            ctmp = chord_type[topi[i].item()]
            chord_order_tmp = []
            for ii in ctmp.split(', '):
                ii = ii.replace('[', '')
                ii = ii.replace(']', '')
                chord_order_tmp.append(int(ii))
            if chord_order_tmp[0] == 0:
                chord_order_new = [chord_order_tmp[0]]
            else:
                chord_order_new = [chord_order_tmp[0] - 2]
            chord_order_new.extend(chord_order_tmp[2:])
            chord_order.append(chord_order_new)
    elif args.repre == "GT":
        chord_order_new_GT = midi2event(src)['chords']
        for i in range(len(chord_order_new_GT)):
            if len(chord_order_new_GT[i])>1:
                chord_new = chord_revise(chord_order_new_GT[i])
                chord_new = chord_transformation(chord_new)
                chord_order_new_new = [chord_new[0]]
                chord_order_new_new.extend(chord_new[2:])
                chord_order.append(chord_order_new_new)
                chord_order[i][0]=chord_order[i][0]-2
            else:
                chord_order.append([0])
    music = muspy.read_midi(src, 'pretty_midi')
    times = music.time_signatures
    chord_order,event["pitchs"],event["bars"], event["durations"]=revise_bar(chord_order,event["pitchs"],event["bars"],event["durations"],times)
    # compute the length of the longest plagiarized subsequence
    beat_num=compute_corpus_level_copy(chord_order,event["durations"],GT_chord_list,GT_dur_list)
    corpus_level_copy_num_list.append(beat_num)
    new_chords,new_durations=merge_chord(chord_order,event["bars"],event["durations"])
    # compute metrics
    CHS,CTD,CTnCTR,PCS,MCTD,CNR=compute_metrics(new_chords,new_durations,event["bars"],event["pitchs"],event["durations"])
    # combine the generated chords with the original melody into a new midi
    if args.generate_midi:
        notesss=[]
        start_time=0
        for i in range(len(new_chords)):
            chord_t=new_chords[i]
            if len(chord_t)!=1:
                chords=[]
                if chord_t[0] == 0:
                    t = 2
                if chord_t[0] == 1:
                    t = 3
                offset=0
                for j in range(1,len(chord_t)):
                    if j>1 and chord_t[j]<chord_t[j-1]:
                        offset+=1
                    pitch=12+(t+offset)*12+chord_t[j]
                    notee=muspy.Note(time=start_time,pitch=pitch,
                                             duration=new_durations[i],velocity=music.tracks[0].notes[0].velocity)
                    chords.append(notee)
                notesss.extend(chords)
            start_time+=new_durations[i]
        chord_track=muspy.Track(program=0, is_drum=False, name='',notes=notesss)
        # music.tracks.append(chord_track) # keep the orignal chord
        # not keep the orignal chord
        music.tracks.insert(-1,chord_track)
        music.tracks=music.tracks[:-1]
        muspy.write_midi(dis, music)
    return CHS,CTD,CTnCTR,PCS,MCTD,CNR


def generate_CLSTM(src,dis):
    '''
        generate chord for given melody (one sample)
    '''
    event = melody2event(src)
    seq_len = len(event["bars"])
    one_batch = batch_data_win(event, condition_window, seq_len)
    chord_t_1 = torch.Tensor(one_batch['chords'][0]).to(device)
    hidden = None
    chord_order = []
    chord_order_GT = []
    for i in range(seq_len):
        if args.repre == "GT":
            break
        condition = one_batch['condition']
        note = one_batch['note_t']
        condition_t = {}
        note_t = {}
        condition_t['pitches'] = torch.LongTensor(condition['pitches'][i]).to(device)
        condition_t['durations'] = torch.LongTensor(condition['durations'][i]).to(device)
        condition_t['positions'] = torch.LongTensor(condition['positions'][i]).to(device)
        condition_tt = torch.cat([condition_t['pitches'], condition_t['durations'],
                                  condition_t['positions']], dim=1).to(device)
        note_t['pitches'] = torch.Tensor(note['pitches'][i]).to(device)
        note_t['durations'] = torch.Tensor(note['durations'][i]).to(device)
        note_t['positions'] = torch.Tensor(note['positions'][i]).to(device)
        note_tt = torch.cat([note_t['pitches'], note_t['durations'], note_t['positions']], dim=1).to(device)
        state = torch.cat([condition_tt, note_tt, chord_t_1], dim=-1).to(device)
        states = state.view(1, 1, -1)
        if args.repre == "OH":
            output, hidden = model(states, chord_num, hidden)
            top, topi = output.topk(1)
            ctmp = chord_type[topi.item()]
            chord_0 = [0] * chord_num
            chord_0[topi.item()] = 1
            chord_t_1 = torch.Tensor([chord_0]).to(device)
            chord_order_tmp = []
            for ii in ctmp.split(', '):
                ii = ii.replace('[', '')
                ii = ii.replace(']', '')
                chord_order_tmp.append(int(ii))
            if chord_order_tmp[0] == 0:
                chord_order_new = [chord_order_tmp[0]]
            else:
                chord_order_new = [chord_order_tmp[0] - 2]
            chord_order_new.extend(chord_order_tmp[2:])
            chord_order.append(chord_order_new)
        elif args.repre == "MH":
            output_1, output_2, output_4, output_13, hidden = model(states, hidden)
            if output_1[0][0].item() > 0.5:
                topi_1 = torch.Tensor([[[1]]]).to(device)
            else:
                topi_1 = torch.Tensor([[[0]]]).to(device)
            topv, topi_2 = output_2.topk(1)
            topv, topi_4 = output_4.topk(1)
            topv, topi_13 = output_13.topk(5)
            ctmp = torch.cat((topi_1, topi_2, topi_4, topi_13), dim=-1)[0][0].int().cpu().numpy().tolist()
            if ctmp[0] == 1:
                chord_order.append([0])
                chord_0 = [0] * 20
                chord_0[0] = 1;
                chord_t_1 = torch.Tensor([chord_0]).to(device)
            else:
                chord_t_1 = torch.Tensor(1, 20).zero_().to(device)
                chord_t_1[0][1 + ctmp[1]] = 1
                chord_t_1[0][3 + ctmp[2]] = 1
                if 12 in ctmp:
                    ctmp.remove(12)
                    if ctmp[2] < 3:
                        chord_order_new = chord_transformation(ctmp[1:-1])
                    else:
                        chord_order_new = chord_transformation(ctmp[1:])
                else:
                    chord_order_new = chord_transformation(ctmp[1:-1])
                chord_order_new = chord_revise(chord_order_new)
                chord_order_new = chord_transformation(chord_order_new)
                chord_order_new_new = [chord_order_new[0]]
                chord_order_new_new.extend(chord_order_new[2:])
                chord_order.append(chord_order_new_new)
                if len(chord_order_new_new[1:]) == 3:
                    chord_t_1[0][-1] = 1
                for p in chord_order_new_new[1:]:
                    chord_t_1[0][7 + p] = 1
    if args.repre == "GT":
        chord_order_new_GT = midi2event(src)['chords']
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
    music = muspy.read_midi(src, 'pretty_midi')
    times = music.time_signatures
    chord_order, event["pitchs"], event["bars"], event["durations"] = revise_bar(chord_order, event["pitchs"],
                                                                                 event["bars"], event["durations"],
                                                                                 times)
    # compute the length of the longest plagiarized subsequence
    beat_num = compute_corpus_level_copy(chord_order, event["durations"], GT_chord_list, GT_dur_list)
    corpus_level_copy_num_list.append(beat_num)
    new_chords, new_durations = merge_chord(chord_order, event["bars"], event["durations"])
    # compute metrics
    CHS, CTD, CTnCTR, PCS, MCTD, CNR = compute_metrics(new_chords, new_durations, event["bars"], event["pitchs"],
                                                      event["durations"])
    # combine the generated chords with the original melody into a new midi
    if args.generate_midi:
        notesss = []
        start_time = 0
        for i in range(len(new_chords)):
            chord_t = new_chords[i]
            if len(chord_t) != 1:
                chords = []
                if chord_t[0] == 0:
                    t = 2
                if chord_t[0] == 1:
                    t = 3
                offset = 0
                for j in range(1, len(chord_t)):
                    if j > 1 and chord_t[j] < chord_t[j - 1]:
                        offset += 1
                    pitch = 12 + (t + offset) * 12 + chord_t[j]
                    notee = muspy.Note(time=start_time, pitch=pitch,
                                       duration=new_durations[i], velocity=music.tracks[0].notes[0].velocity)
                    chords.append(notee)
                notesss.extend(chords)
            start_time += new_durations[i]
        chord_track = muspy.Track(program=0, is_drum=False, name='', notes=notesss)
        # music.tracks.append(chord_track) # keep the orignal chord
        # not keep the orignal chord
        music.tracks.insert(-1, chord_track)
        music.tracks = music.tracks[:-1]
        muspy.write_midi(dis, music)
    return CHS, CTD, CTnCTR, PCS, MCTD, CNR


def generate_compute_metrics(generate_path):
    '''
        compute metrics and generate chords for given melody
    '''
    CHS_ALL=[]
    CTD_aver=0
    CTnCTR_aver=0
    PCS_aver=0
    MCTD_aver=0
    CNR_aver=0
    cnt_64=0;cnt=0
    midi_files=os.listdir(args.test_data_path)
    r = random.random
    random.seed(13)
    random.shuffle(midi_files,random=r)
    while cnt<len(midi_files) and cnt_64<int(args.test_num):
        src_path=os.path.join(args.test_data_path, midi_files[cnt])
        music_length=len(midi2event(src_path)['pitchs'])
        # only calculate metrics for music with a length larger than 64
        if music_length>=64:
            dis_path = os.path.join(generate_path, midi_files[cnt])
            if args.model=="BLSTM":
                CHS, CTD, CTnCTR, PCS, MCTD, CNR = generate_BLSTM(src_path, dis_path)
            elif args.model=="CLSTM":
                CHS, CTD, CTnCTR, PCS, MCTD, CNR = generate_CLSTM(src_path, dis_path)
            CHS_ALL.append(CHS)
            CTD_aver+=CTD
            CTnCTR_aver+=CTnCTR
            PCS_aver+=PCS
            MCTD_aver+=MCTD
            CNR_aver+=CNR
            cnt_64+=1
        cnt+=1
    CTD_aver /= int(args.test_num)
    CTnCTR_aver /= int(args.test_num)
    PCS_aver /= int(args.test_num)
    MCTD_aver /= int(args.test_num)
    CNR_aver /= int(args.test_num)
    return CHS_ALL,CTD_aver,CTnCTR_aver,PCS_aver,MCTD_aver,CNR_aver


if __name__=='__main__':

    parser = ArgumentParser(description='evaluate LSTM')
    parser.add_argument("--dataset", type=str, default='NMD', help="NMD or Wiki")
    parser.add_argument("--seq_len", type=str, default='64', help="64 or 128")
    parser.add_argument("--model", type=str, default='BLSTM', help="BLSTM or CLSTM")
    parser.add_argument("--repre", type=str, default='MH', help="MH or OH or GT")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--generate_midi", type=bool, default=1)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--generate_path", type=str, default=None)
    parser.add_argument("--hist_path", type=str, default=None, help="save histogram for the CHS metric")
    parser.add_argument("--test_num", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load chord set
    if args.dataset == "NMD":
        file = open("./data/NMD_chord_type_116.data", 'rb')
        chord_num = 116
    elif args.dataset == "Wiki":
        file = open("./data/Wiki_chord_type_310.data", 'rb')
        chord_num = 310
    else:
        file = open("./data/TTD_chord_type_665.data", 'rb')
        chord_num = 665
    chord_type = list(pickle.load(file))
    file = open("./data/chord_interval_31.data", 'rb')
    chord_interval = pickle.load(file)

    # load model
    condition_window = 8
    input_size = 64
    hidden_size = 512
    models = {"BLSTM_MH": BLSTM_Chord_MH,
              "BLSTM_OH": BLSTM_Chord_OH,
              "CLSTM_MH": CLSTM_Chord_MH,
              "CLSTM_OH": CLSTM_Chord_OH}
    model=models[args.model+"_"+args.repre](condition_window, input_size, hidden_size, chord_num).to(device)
    model_path = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    load_model_path = model_path + args.load_model
    dict = torch.load(load_model_path, map_location=device)
    model.load_state_dict(dict['model'])
    model.eval()

    # get the real chord and duration sequences
    if not os.path.exists(f"data/{args.dataset}_GT_chord_order_list.data"):
        chord_order_GT_list,duration_GT_list=getTrainChordandDur(args.train_data_path)
        file = open(f"data/{args.dataset}_GT_chord_order_list.data", 'wb')
        pickle._dump(chord_order_GT_list, file)
        file.close()
        file = open(f"data/{args.dataset}_GT_duration_list.data", 'wb')
        pickle._dump(duration_GT_list, file)
        file.close()
    else:
        file = open(f"data/{args.dataset}_GT_chord_order_list.data", 'rb')
        GT_chord_list = pickle.load(file)
        file = open(f"data/{args.dataset}_GT_duration_list.data", 'rb')
        GT_dur_list = pickle.load(file)

    # compute metrics and generate music
    corpus_level_copy_num_list = []
    generate_path = f"{args.generate_path}/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}"
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)
    CHS_ALL, CTD_aver, CTnCTR_aver, PCS_aver, MCTD_aver, CNR_aver = generate_compute_metrics(generate_path)
    hist_path = f"{args.hist_path}/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.data"
    file = open(hist_path, 'wb')
    pickle._dump(CHS_ALL, file)
    file.close()
    hist_sim()
    print("CNR_aver: ",CNR_aver)
    print("CTD_aver: ",CTD_aver)
    print("DC_aver: ",CTnCTR_aver)
    print("PCS_aver: ",PCS_aver)
    print("MCTD_aver: ",MCTD_aver)
    print("corpus_level_copy_num_list: ", corpus_level_copy_num_list)
