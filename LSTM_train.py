"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
import os
import random
import pickle
import time
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from model.BLSTM_Chord_MH import LSTM_Chord as BLSTM_Chord_MH
from model.BLSTM_Chord_OH import LSTM_Chord as BLSTM_Chord_OH
from model.CLSTM_Chord_MH import LSTM_Chord as CLSTM_Chord_MH
from model.CLSTM_Chord_OH import LSTM_Chord as CLSTM_Chord_OH
from train_utils import timeSince,chord_revise,multilabel_categorical_crossentropy,chord2binary,duration2type


def batch_data_win(datas,batch_size,condition_window,seq_len):
    '''
        prepare one batch data
    '''
    one_batch = {}
    one_batch['condition'] = {'pitches': [], 'durations': [], 'positions': []}
    if args.model == "CLSTM":
        one_batch['note_t'] = {'pitches': [], 'durations': [], 'positions': []}
    one_batch['chords'] = []
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
        # print("start,end: ",window_start,window_end)
        pitch = []
        duration = []
        position = []
        chord = []
        if args.model=="CLSTM":
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
            for ddd in datas[i]['durations'][window_start:window_end]:
                d_temp.append(duration2type(ddd))
            duration.append(d_temp)
            position.append(datas[i]['bars'][window_start:window_end])
            chord_before=datas[i]['chords'][t]
            chord_after=chord_revise(chord_before)
            chord.append(chord2binary(chord_after))
            if args.model == "CLSTM":
                pitch_t = [0] * 49;
                duration_t = [0] * 12;
                position_t = [0] * 72
                if datas[i]['pitchs'][t] == 0:
                    pitch_t[0] = 1
                else:
                    pitch_t[datas[i]['pitchs'][t] - 47] = 1
                dur = duration2type(datas[i]['durations'][t])
                if dur in [-1, 12]:
                    print("Dur ERROR")
                else:
                    duration_t[dur] = 1
                position_t[datas[i]['bars'][t]] = 1
                pitch_tt.append(pitch_t)
                duration_tt.append(duration_t)
                position_tt.append(position_t)
        one_batch['condition']['pitches'].append(pitch)
        one_batch['condition']['durations'].append(duration)
        one_batch['condition']['positions'].append(position)
        if args.model == "CLSTM":
            one_batch['note_t']['pitches'].append(pitch_tt)
            one_batch['note_t']['durations'].append(duration_tt)
            one_batch['note_t']['positions'].append(position_tt)
        one_batch['chords'].append(chord)
    return one_batch


def train(one_batch,type,seq_len):
    '''
        train one batch
    '''
    if type=="train":
        LSTM_baseline.train()
    else:
        LSTM_baseline.eval()
    condition = one_batch['condition']
    if args.model == "CLSTM":
        note = one_batch['note_t']
        note_pitch = torch.Tensor(note['pitches']).to(device)
        note_duration = torch.Tensor(note['durations']).to(device)
        note_position = torch.Tensor(note['positions']).to(device)
        note_new = torch.cat([note_pitch, note_duration, note_position], dim=-1).to(device)
    chord = one_batch['chords']
    hidden=None
    condition_pitch = torch.LongTensor(condition['pitches']).to(device)  # 64*320*8
    condition_duration = torch.LongTensor(condition['durations']).to(device)
    condition_position = torch.LongTensor(condition['positions']).to(device)
    condition_new = torch.cat([condition_pitch, condition_duration, condition_position], dim=-1).to(device)
    if args.model == "CLSTM":
        chord_new = torch.Tensor(chord[:-1]).to(device)
        condition_new = torch.cat([condition_new, note_new, chord_new], dim=-1).to(device)
    chord_gt = torch.Tensor(chord[1:]).to(device)
    if args.repre=="MH":
        output_1, output_2, output_4, output_13, hidden = LSTM_baseline(condition_new, hidden)
        chord_gt_1 = chord_gt.narrow(2, 0, 1)
        _, chord_gt_2 = chord_gt.narrow(2, 1, 2).topk(1)
        _, chord_gt_4 = chord_gt.narrow(2, 3, 4).topk(1)
        chord_gt_2 = torch.LongTensor(chord_gt_2.squeeze(-1).cpu().numpy()).to(device)
        chord_gt_4 = torch.LongTensor(chord_gt_4.squeeze(-1).cpu().numpy()).to(device)
        chord_gt_13 = chord_gt.narrow(2, 7, 13)
        l_1 = 0
        l_2 = 0
        l_4 = 0
        l_13 = 0
        for k in range(seq_len):
            l_1 += criterion_1(output_1[k], chord_gt_1[k]).squeeze(-1)
            l_2 += criterion_2(output_2[k], chord_gt_2[k])
            l_4 += criterion_4(output_4[k], chord_gt_4[k])
            l_13 += criterion_13(output_13[k], chord_gt_13[k]).mean()
        loss = (l_1 + l_2 + l_4 + l_13)
    elif args.repre=="OH":
        output, hidden = LSTM_baseline(condition_new, hidden)
        _, chord_gt = chord_gt.topk(1)  # 64*320*4
        chord_gt = torch.LongTensor(chord_gt.squeeze(-1).cpu().numpy()).to(device)
        loss = 0
        for k in range(seq_len):
            loss += criterion(output[k], chord_gt[k]).squeeze(-1)
    if type=="train":
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(LSTM_baseline.parameters(), 5)
        optimizer.step()
    return loss.item()#/seq_len


def trainIter():
    '''
        training loop
    '''
    max_test_loss = 1000
    last_aver = 0
    lr = args.learning_rate
    lr_cnt = 0
    train_writer = SummaryWriter(f"runs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}-train")
    test_writer = SummaryWriter(f"runs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}-test")
    seq_len = int(args.seq_len)
    # load training data
    file = open(train_path, 'rb')
    train_data = pickle.load(file)
    file = open(test_path, 'rb')
    test_data = pickle.load(file)
    train_num = len(train_data)
    test_num = len(test_data)
    for epoch in range(epoch_already+1,Epoch):
        f = open(f'./logs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.txt', 'a')
        print("-----------------------------epoch ", epoch, "------------------------------")
        f.write('\n-----------------------------epoch %d------------------------------' % (epoch))
        # training
        total_loss=0
        total_total_loss=0
        train_start_idx=0
        random.shuffle(train_data)
        while train_start_idx + batch_size <=train_num:
            one_batch=batch_data_win(train_data,batch_size,condition_window,seq_len)
            loss=train(one_batch,"train",seq_len)
            total_loss+=loss*batch_size
            total_total_loss+=loss*batch_size
            train_start_idx+=batch_size
            if train_start_idx%(print_every)==0:
                print('epoch train:%d, %s(%d %d%%) %.10f' % (
                    epoch, timeSince(start_time), train_start_idx, train_start_idx / ((train_num // batch_size) * batch_size) * 100,
                    total_loss / print_every))
                f.write('\nepoch:%d, %s(%d %d%%) %.10f' % (
                    epoch, timeSince(start_time), train_start_idx, train_start_idx / ((train_num // batch_size) * batch_size) * 100,
                    total_loss / print_every))
                print("--------------------------------------------------------------")
                total_loss = 0
        # validation
        test_start_idx = 0
        test_total_loss= 0
        total_loss=0
        random.shuffle(test_data)
        while test_start_idx + batch_size <= test_num:
            one_batch=batch_data_win(test_data,batch_size,condition_window,seq_len)
            loss=train(one_batch,"test",seq_len)
            total_loss += loss*batch_size
            test_total_loss+=loss*batch_size
            test_start_idx+=batch_size
            if test_start_idx%print_every==0:
                print('epoch test:%d, %s(%d %d%%) %.10f' % (
                    epoch, timeSince(start_time), test_start_idx, test_start_idx / ((test_num // batch_size) * batch_size) * 100,
                    total_loss / print_every))
                f.write('\nepoch test:%d, %s(%d %d%%) %.10f' % (
                    epoch, timeSince(start_time), test_start_idx,test_start_idx / ((test_num // batch_size) * batch_size) * 100,
                    total_loss / print_every))
                print("--------------------------------------------------------------")
                total_loss=0
        print('epoch: %d, time: %s, train loss : %.6f, test loss : %.6f, learning rate: %.6f, batch_size: %d' % (
        epoch, timeSince(start_time), total_total_loss / train_start_idx, test_total_loss/test_start_idx, lr,batch_size))
        f.write('\nepoch: %d, time: %s, train loss : %.6f, test loss : %.6f, learning rate: %.6f, batch_size: %d' % (
        epoch, timeSince(start_time), total_total_loss / train_start_idx, test_total_loss/test_start_idx, lr,batch_size))
        train_average = total_total_loss / train_start_idx
        test_average = test_total_loss / test_start_idx
        train_writer.add_scalar('loss', train_average, epoch)
        test_writer.add_scalar('loss', test_average, epoch)
        # lr scheduler
        if test_average>last_aver:
            if lr_cnt<10:
                lr_cnt+=1
            else:
                if lr<0.001:
                    lr=lr/2
                else:
                    lr=lr/10
                    batch_size = batch_size // 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print("change: ",lr,batch_size)
                f.write('\nchange: %.6f %d' % (lr,batch_size))
        else:
            lr_cnt=0
        last_aver=test_average
        # save model per epoch
        if test_average < max_test_loss:
            f.write('\nepoch: %d save min test loss model' % (epoch))
            state = {'model': LSTM_baseline.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_path + f"{args.dataset}-{args.model}-{args.repre}-{args.seq_len}_min_test_loss.pth")
            max_test_loss = test_average
        model_save_path = model_path + f"{args.dataset}-{args.model}-{args.repre}-{args.seq_len}-epoch" + str(epoch) + "-" + str(test_average) + ".pth"
        state = {'model': LSTM_baseline.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state,model_save_path)
        f.close()


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='train LSTM model')
    parser.add_argument("--dataset",type=str,default='NMD', help="NMD or Wiki")
    parser.add_argument("--seq_len", type=str, default='64', help="64 or 128")
    parser.add_argument("--model", type=str, default='BLSTM',help="BLSTM or CLSTM")
    parser.add_argument("--repre", type=str, default='MH', help="MH or OH")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--batch_size", type=str, default="64")
    parser.add_argument("--epoch", type=str, default='100')
    parser.add_argument("--learning_rate", type=str, default='0.001')
    args = parser.parse_args()

    f = open(f'./logs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.txt', 'a')
    f.write('\nHyperparameters: \n%s' % (str(args)))
    f.close()

    # data
    if args.dataset=="NMD":
        chord_num=116
        if int(args.seq_len)==64:
            train_path = "./data/slide_win64_stride8/NMD_train_104423.data"
            test_path = "./data/slide_win64_stride8/NMD_test_7962.data"
        else:
            train_path = "./data/slide_win128_stride8/NMD_train_58425.data"
            test_path = "./data/slide_win128_stride8/NMD_test_3894.data"
    elif args.dataset=="Wiki":
        chord_num=310
        if int(args.seq_len) == 64:
            train_path="./data/slide_win64_stride8/Wiki_train_202119.data"
            test_path="./data/slide_win64_stride8/Wiki_test_19751.data"
        else:
            train_path = "./data/slide_win128_stride8/Wiki_train_88775.data"
            test_path = "./data/slide_win128_stride8/Wiki_test_8295.data"

    # hyperparameter
    condition_window = 8  # the window size for melody condition
    input_size = 64
    hidden_size = 512
    print_every = 10240
    batch_size = int(args.batch_size)
    Epoch = int(args.epoch)

    # define model
    models={"BLSTM_MH":BLSTM_Chord_MH,
            "BLSTM_OH":BLSTM_Chord_OH,
            "CLSTM_MH":CLSTM_Chord_MH,
            "CLSTM_OH":CLSTM_Chord_OH}
    LSTM_baseline = models[args.model+"_"+args.repre](condition_window, input_size, hidden_size, chord_num).to(device)
    model_path = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    optimizer = torch.optim.SGD(LSTM_baseline.parameters(), lr=float(args.learning_rate), momentum=0.9)

    # load model
    epoch_already = -1
    if args.load_model is not None:
        load_model_path = model_path + args.load_model
        dict = torch.load(load_model_path, map_location=device)
        LSTM_baseline.load_state_dict(dict['model'])
        epoch_already = dict['epoch']
        optimizer.load_state_dict(dict['optimizer'])

    # train model
    if args.repre=="MH":
        criterion_1 = nn.BCELoss().to(device)
        criterion_2 = nn.NLLLoss().to(device)
        criterion_4 = nn.NLLLoss().to(device)
        criterion_13 = multilabel_categorical_crossentropy
    elif args.repre=="OH":
        criterion = nn.NLLLoss().to(device)

    start_time = time.time()
    trainIter()
