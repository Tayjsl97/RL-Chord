"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
import random
import pickle
import time
import os
from model.DQN_Chord import DQN_Chord
from model.Mutual_Chord import Mutual_Chord
from model.CLSTM_Chord_MH import LSTM_Chord
from model.attention_classifier import Attention_Classifier
from argparse import ArgumentParser
from train_utils import timeSince,ChordOrder,binary2chord,plot_RL,chord_revise,\
    multilabel_categorical_crossentropy,batch_data_win,rule_rewards,chord_transformation


def Reward(chord,action1,action2,action4,action13,melody_pitch,melody_duration,melody_position,batch_size):
    '''
        Compute the rewards (with the expection of mutual reward) for all actions.
        Note that CF-Chord do not have the ground-truth reward.
    '''
    # compute rule reward including interval consonance, repetition, and superstrong
    values13, indices13 = action13.topk(5)
    chord_order=torch.cat((action1,action2, action4, indices13),dim=-1).int().cpu().numpy().tolist()
    chordOrder = ChordOrder(chord_order)
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
    total_reward = (rule_reward).detach().unsqueeze(-1)
    return total_reward


def sample_batch_data(one_batch,seq_len,batch_size,type):
    '''
        Sample one batch data
    '''
    if type=="train":
        current_model.train()
    else:
        current_model.eval()
    condition = one_batch['condition']
    note = one_batch['note_t']
    chord = one_batch['chords']
    hidden=None
    chord_0 = chord[0]
    condition_pitch=torch.LongTensor(condition['pitches']).to(device)
    condition_duration = torch.LongTensor(condition['durations']).to(device)
    condition_position = torch.LongTensor(condition['positions']).to(device)
    duration=torch.LongTensor(one_batch['duration']).to(device)
    chord_seq=[];states_seq=[];action_seq=[]
    action1_seq=[];action2_seq=[];action4_seq=[];action13_seq=[]
    mc1_seq=[];mc2_seq=[];mc4_seq=[];mc13_seq=[]
    c1_seq = [];c2_seq = [];c4_seq = [];c13_seq = []
    chord_root=[]
    for i in range(seq_len):
        condition_pitch_t=torch.LongTensor(condition['pitches'][i]).to(device)
        condition_duration_t = torch.LongTensor(condition['durations'][i]).to(device)
        condition_position_t=torch.LongTensor(condition['positions'][i]).to(device)
        condition_new=torch.cat([condition_pitch_t,condition_duration_t,condition_position_t],dim=-1).unsqueeze(0).to(device)
        note_pitch = torch.Tensor(note['pitches'][i]).to(device)
        note_duration = torch.Tensor(note['durations'][i]).to(device)
        note_position = torch.Tensor(note['positions'][i]).to(device)
        note_new = torch.cat([note_pitch, note_duration, note_position], dim=-1).unsqueeze(0).to(device)
        chord_t=torch.Tensor(chord_0).unsqueeze(0).to(device)
        states = torch.cat([condition_new, note_new, chord_t], dim=-1).to(device)
        states_seq.append(states.squeeze(0))
        states_c = torch.cat([note_position, chord_t.squeeze(0)], dim=-1).unsqueeze(0).to(device)
        output_1, output_2, output_4, output_13, hidden = current_model.act(states, hidden)
        mc_output_1, mc_output_2, mc_output_4, mc_output_13, hidden = mutual_mc(states, hidden)
        c_output_1, c_output_2, c_output_4, c_output_13, hidden = mutual_c(states_c, hidden)
        action1 = torch.Tensor(1,batch_size, 1).zero_().to(device)
        chord_tmp=[]
        for j in range(batch_size):
            if output_1[0][j][0].item() > 0.5:
                action1[0][j][0] = 1
        _,action2 = output_2.topk(1)
        _,action4 = output_4.topk(1)
        _,action13 = output_13.topk(5)
        ctmp = torch.cat([action1, action2, action4, action13], dim=-1).squeeze()
        action1_seq.append(action1.squeeze(0));action2_seq.append(action2.squeeze(0))
        action4_seq.append(action4.squeeze(0));action13_seq.append(output_13.squeeze(0))
        c_root=[]
        for j in range(batch_size):
            ctmp_t = ctmp[j].int().cpu().numpy().tolist()
            if ctmp_t[0] == 1:
                chord_t = [0] * 20
                chord_t[0] = 1;
                c_root.append(0)
            else:
                chord_t = [0] * 20
                chord_t[1 + ctmp_t[1]] = 1
                chord_t[3 + ctmp_t[2]] = 1
                if 12 in ctmp_t:
                    ctmp_t.remove(12)
                    if ctmp_t[2] < 3:
                        chord_order_new = chord_transformation(ctmp_t[1:-1])
                    else:
                        chord_order_new = chord_transformation(ctmp_t[1:])
                else:
                    chord_order_new = chord_transformation(ctmp_t[1:-1])
                chord_order_new = chord_revise(chord_order_new)
                chord_order_new = chord_transformation(chord_order_new)
                chord_order_new_new = [chord_order_new[0]]
                chord_order_new_new.extend(chord_order_new[2:])
                if len(chord_order_new_new[1:]) == 3:
                    chord_t[-1] = 1
                for p in chord_order_new_new[1:]:
                    chord_t[7 + p] = 1
                root_pitch=12 + (chord_order_new_new[0]+2) * 12+chord_order_new_new[1]
                if root_pitch != 0:
                    root_pitch -= 35
                c_root.append(root_pitch)
            chord_tmp.append(chord_t)
        chord_root.append(c_root)
        chord_0=chord_tmp  # update the input
        chord_seq.append(chord_tmp)
        action11=action1
        _,action22=output_2.topk(1)
        _,action44=output_4.topk(1)
        _,action1313=output_13.topk(4)
        actions = torch.cat([action11, action22, action44, action1313], dim=-1).squeeze()
        action_seq.append(actions)
        _, index13 = output_13.topk(13)
        mc1 = action1 * mc_output_1 + (1 - action1) * (1 - mc_output_1)
        mc2 = mc_output_2.gather(2, action2.long())
        mc4 = mc_output_4.gather(2, action4.long())
        mc13 = mc_output_13.gather(2, index13.long()).narrow(-1, 0, 4).mean(-1).unsqueeze(-1)
        c1 = action1 * c_output_1 + (1 - action1) * (1 - c_output_1)
        c2 = c_output_2.gather(2, action2.long())
        c4 = c_output_4.gather(2, action4.long())
        c13 = c_output_13.gather(2, index13.long()).narrow(-1, 0, 4).mean(-1).unsqueeze(-1)
        mc1_seq.append(mc1.squeeze(0));mc2_seq.append(mc2.squeeze(0));mc4_seq.append(mc4.squeeze(0));mc13_seq.append(mc13.squeeze(0))
        c1_seq.append(c1.squeeze(0));c2_seq.append(c2.squeeze(0));c4_seq.append(c4.squeeze(0));c13_seq.append(c13.squeeze(0))

    # get rewards of all steps
    chord=chord_seq
    mc1 = torch.stack(mc1_seq);mc2 = torch.stack(mc2_seq);mc4 = torch.stack(mc4_seq);mc13 = torch.stack(mc13_seq)
    c1 = torch.stack(c1_seq);c2 = torch.stack(c2_seq);c4 = torch.stack(c4_seq);c13 = torch.stack(c13_seq)
    mutual_reward = (torch.log(mc1) - torch.log(c1) + mc2 + mc4 + mc13 - c2 - c4 - c13).detach()
    action1=torch.stack(action1_seq);action2=torch.stack(action2_seq);action4=torch.stack(action4_seq);action13=torch.stack(action13_seq)
    rewards = Reward(chord,action1,action2,action4,action13,condition_pitch, condition_duration,condition_position,batch_size)
    rewards+=mutual_reward

    # save for loss calculation
    states = torch.stack(states_seq)
    actions = torch.stack(action_seq)
    actions = actions.narrow(0, 0, seq_len - 1)
    rewards = rewards.narrow(0, 0, seq_len - 1)/100
    chord_root=torch.LongTensor(chord_root).to(device)
    return states,actions,rewards,chord_root,duration


def DQN_train(states, actions, rewards,chord_root,duration):
    '''
        train one batch
    '''
    loss1 = 0;loss2 = 0;loss4 = 0;loss13 = 0
    current_hidden=None
    target_hidden=None
    for i in range(seq_len-1):
        state = states[i].unsqueeze(0)
        next_state = states[i+1].unsqueeze(0)
        action = actions[i].unsqueeze(0)
        reward = rewards[i].unsqueeze(0)
        action2 = action.narrow(-1, 1, 1)
        action4 = action.narrow(-1, 2, 1)
        action13 = action.narrow(-1, 3, 4)
        q_values1, q_values2, q_values4, q_values13, current_hidden = current_model(state,current_hidden)
        next_q_values1, next_q_values2, next_q_values4, next_q_values13, target_hidden = target_model(next_state,target_hidden)
        q_value2 = q_values2.gather(2, action2.long()).squeeze(0)
        q_value4 = q_values4.gather(2, action4.long()).squeeze(0)
        q_value13 = q_values13.gather(2, action13.long()).squeeze(0)
        next_q_value2, _ = next_q_values2.topk(1)
        next_q_value4, _ = next_q_values4.topk(1)
        next_q_value13, _ = next_q_values13.topk(4)
        reward=reward.squeeze(0)
        next_q_value1 = next_q_values1
        expected_q_value1 = reward + gamma * next_q_value1.squeeze(0)
        expected_q_value2 = reward + gamma * next_q_value2.squeeze(0)
        expected_q_value4 = reward + gamma * next_q_value4.squeeze(0)
        expected_q_value13 = reward + gamma * next_q_value13.squeeze(0)
        q_value1 = q_values1.squeeze(0)
        loss1 += (q_value1 - expected_q_value1.detach()).pow(2).squeeze(-1).mean()
        loss2 += (q_value2 - expected_q_value2.detach()).pow(2).squeeze(-1).mean()
        loss4 += (q_value4 - expected_q_value4.detach()).pow(2).squeeze(-1).mean()
        loss13 += (q_value13 - expected_q_value13.detach()).mean(1).pow(2).mean()
    RL_loss=loss1+loss2+loss4+loss13
    # calculate classification loss.
    with torch.no_grad():
        mid_feature, output = classifier(None, duration, chord_root)
        gt = torch.LongTensor([1] * batch_size).to(device)
        class_loss = class_criterion(output, gt)
        pre = output.topk(1)[1].squeeze(-1)
        diff = torch.eq((pre - gt),0)
        acc=sum(diff).item()/len(gt)
    beta = 0.3
    loss = beta * RL_loss + (1 - beta) * class_loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(current_model.parameters(), 10)
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()

    return RL_loss,class_loss,acc,beta


def _test(test_mini_data):
    '''
        test one batch
    '''
    one_batch,real = batch_data_win(test_mini_data, batch_size, condition_window, seq_len)
    states,actions,rewards=sample_batch_data(one_batch, real,seq_len, batch_size,"test")
    return rewards


def trainIter_DQN():
    '''
        training loop
    '''
    # load data
    file = open(data_path, 'rb')
    train_data = pickle.load(file)
    train_length = len(train_data)
    pic_path = f"./pics/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    best_reward = best_reward_init
    step = step_init
    for epoch in range(epoch_already+1,Epoch):
        f=open(f'./logs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.txt','a')
        f.write('\n------------------------epoch %d-------------------------' % (epoch))
        print("--------------------epoch: ",epoch,"---------------------")
        random.shuffle(train_data)
        RL_total_loss=0
        class_total_loss=0
        total_loss = 0
        total_acc=0
        total_reward=0
        train_start_idx=0
        batch_cnt=0
        while train_start_idx + batch_size <= train_length:
            train_batch_data=train_data[train_start_idx:train_start_idx+batch_size]
            one_batch = batch_data_win(train_batch_data, batch_size, condition_window, seq_len)
            states,actions,rewards,chord_root,duration=sample_batch_data(one_batch,seq_len, batch_size,"train")
            RL_loss,class_loss,acc,beta = DQN_train(states,actions,rewards,chord_root,duration)
            loss=RL_loss+class_loss
            RL_total_loss+=RL_loss.item()
            class_total_loss+=class_loss.item()
            total_loss+=loss.item()
            total_acc+=acc
            total_reward+=sum(rewards).mean().item()
            train_start_idx+=batch_size
            batch_cnt+=1
            if batch_cnt % target_n==0:
                target_model.load_state_dict(current_model.state_dict())
            if batch_cnt % Train_i == 0:
                print('epoch train:%d, %s(%d %d%%) reward: %.6f loss: %.6f RL_loss: %.6f class_loss: %.6f acc: %.6f beta: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / ((train_length // batch_size) * batch_size) * 100,
                    total_reward / Train_i,total_loss / Train_i,RL_total_loss/Train_i,
                    class_total_loss/Train_i,total_acc/Train_i,beta))
                f.write('\nepoch train:%d, %s(%d %d%%) reward: %.6f loss: %.6f RL_loss: %.6f class_loss: %.6f acc: %.6f beta: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / ((train_length // batch_size) * batch_size) * 100,
                    total_reward / Train_i, total_loss / Train_i,RL_total_loss/Train_i,
                    class_total_loss/Train_i,total_acc/Train_i,beta))
                print("---------------------------------------------------------------------------")
                aver_reward=total_reward/Train_i
                aver_rewards.append(aver_reward)
                batch_num.append(step)
                step=step+1
                plot_RL(batch_num, aver_rewards, pic_path, args.model)
                print("aver_reward: ", aver_reward)
                f.write('\naver_reward: %.6f' % (aver_reward))
                if best_reward is None or best_reward < aver_reward:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, aver_reward))
                        f.write("\nBest reward updated: %.3f -> %.3f" % (best_reward, aver_reward))
                    best_reward = aver_reward
                total_loss=0
                total_reward=0
                RL_total_loss = 0
                class_total_loss = 0
                total_acc=0
        f.close()
        model_save_path = model_path+"epoch%d_reward%.3f_beta%.3f.pth" % (epoch, best_reward,beta)
        states = {
            'model': current_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'aver_rewards': aver_rewards,
            'batch_nums': batch_num,
            'epoch': epoch
        }
        torch.save(states, model_save_path)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description='train LSTM model')
    parser.add_argument("--dataset", type=str, default='NMD', help="NMD or Wiki")
    parser.add_argument("--seq_len", type=str, default='64', help="64 or 128")
    parser.add_argument("--model", type=str, default='CF')
    parser.add_argument("--repre", type=str, default='MH')
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--mutual_c", type=str, default=None)
    parser.add_argument("--mutual_mc", type=str, default=None)
    parser.add_argument("--classifier_path", type=str, default=None)
    parser.add_argument("--batch_size", type=str, default="64")
    parser.add_argument("--epoch", type=str, default='100')
    parser.add_argument("--learning_rate", type=str, default='0.001')
    args = parser.parse_args()

    f = open(f'./logs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.txt', 'a')
    f.write('\nHyperparameters: \n%s' % (str(args)))
    f.close()

    data_path = "./data/slide_win64_stride8/CF_8519.data"

    # hyperparameter
    condition_window = 8
    harmony_rule_window = 3
    input_size = 64
    hidden_size = 512
    gamma = 0.99
    Reward_R1 = 1  # reward weight for interval consonance reward
    Reward_R2 = 1  # reward weight for chord progression reward
    Train_i = 10  # batch interval for printing training info
    target_n = 3  # batch interval for updating target_model's parameters
    batch_size = int(args.batch_size)
    Epoch = int(args.epoch)
    seq_len = int(args.seq_len)

    # define model
    current_model = DQN_Chord(condition_window=condition_window, input_size=input_size, hidden_size=512).to(device)
    target_model = DQN_Chord(condition_window=condition_window, input_size=input_size, hidden_size=512).to(device)
    target_model.load_state_dict(current_model.state_dict())
    mutual_c = Mutual_Chord(input_size=input_size, hidden_size=512).to(device)
    mutual_mc = LSTM_Chord(condition_window=condition_window, input_size=input_size, hidden_size=512).to(device)
    classifier = Attention_Classifier(input_size=20, hidden_size=512).to(device)
    optimizer = torch.optim.SGD(current_model.parameters(), lr=float(args.learning_rate), momentum=0.9)
    model_path = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load model
    if args.mutual_c is not None and args.mutual_mc is not None:
        mutual_c_resume = f"./saved_models/Wiki-Mutual-{args.repre}-{args.seq_len}/{args.mutual_c}"
        mutual_mc_resume = f"./saved_models/Wiki-CLSTM-{args.repre}-{args.seq_len}/{args.mutual_mc}"
        mutual_c_dict = torch.load(mutual_c_resume, map_location=device)
        mutual_mc_dict = torch.load(mutual_mc_resume, map_location=device)
        mutual_c.load_state_dict(mutual_c_dict['model'])
        mutual_mc.load_state_dict(mutual_mc_dict['model'])
        mutual_c.eval()
        mutual_mc.eval()
    epoch_already = -1
    aver_rewards = []
    batch_num = []
    step_init = 0
    best_reward_init = None
    if args.load_model is not None:
        load_model_path = f"./saved_models/Wiki-DQN-MH-{args.seq_len}/" + args.load_model
        dict = torch.load(load_model_path, map_location=device)
        current_model.load_state_dict(dict['model'])
        current_model = current_model.to(device)
        target_model.load_state_dict(current_model.state_dict())
        optimizer.load_state_dict(dict['optimizer'])
        epoch_already = dict['epoch']
        aver_rewards = dict['aver_rewards']
        batch_num = dict['batch_nums']
        step_init = len(aver_rewards)
        best_reward_init = max(aver_rewards)
    if args.classifier_path is not None:
        c_dict = torch.load(args.classifier_path, map_location=device)
        classifier.load_state_dict(c_dict['model'])
        classifier = classifier.to(device)

    # train model
    criterion_1 = nn.BCELoss(reduction='none').to(device)
    criterion_2 = nn.NLLLoss(reduction='none').to(device)
    criterion_4 = nn.NLLLoss(reduction='none').to(device)
    criterion_13 = multilabel_categorical_crossentropy
    class_criterion = nn.NLLLoss().to(device)
    current_model.train()
    classifier.eval()
    start_time = time.time()
    trainIter_DQN()

