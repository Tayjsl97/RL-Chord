"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
import torch.nn as nn
import random
import pickle
import os
import time
from argparse import ArgumentParser
from model.DQN_Chord import DQN_Chord
from model.Mutual_Chord import Mutual_Chord
from model.CLSTM_Chord_MH import LSTM_Chord
from train_utils import timeSince,plot_RL,multilabel_categorical_crossentropy,\
    batch_data_win,Reward


def sample_batch_data(one_batch,chord_real,batch_size,type):
    '''
        sample one batch data
    '''
    if type=="train":
        current_model.train()
    else:
        current_model.eval()
    condition = one_batch['condition']
    note = one_batch['note_t']
    chord = one_batch['chords']
    hidden=None
    condition_pitch=torch.LongTensor(condition['pitches']).to(device)
    condition_duration = torch.LongTensor(condition['durations']).to(device)
    condition_position=torch.LongTensor(condition['positions']).to(device)
    condition_new=torch.cat([condition_pitch,condition_duration,condition_position],dim=-1).to(device)
    note_pitch = torch.Tensor(note['pitches']).to(device)
    note_duration = torch.Tensor(note['durations']).to(device)
    note_position = torch.Tensor(note['positions']).to(device)
    note_new = torch.cat([note_pitch, note_duration, note_position], dim=-1).to(device)
    chord_new=torch.Tensor(chord[:-1]).to(device)
    states = torch.cat([condition_new, note_new, chord_new], dim=-1).to(device)
    states_c = torch.cat([note_position, chord_new], dim=-1).to(device)
    output_1, output_2, output_4, output_13, hidden = current_model.act(states, hidden)
    mc_output_1, mc_output_2, mc_output_4, mc_output_13, hidden = mutual_mc(states, hidden)
    c_output_1, c_output_2, c_output_4, c_output_13, hidden = mutual_c(states_c, hidden)

    # compute MLE loss
    chord_gt = torch.Tensor(chord[1:]).to(device)
    chord_gt_1 = chord_gt.narrow(2, 0, 1)
    _, chord_gt_2 = chord_gt.narrow(2, 1, 2).topk(1)
    _, chord_gt_4 = chord_gt.narrow(2, 3, 4).topk(1)
    chord_gt_2 = torch.LongTensor(chord_gt_2.squeeze(-1).cpu().numpy()).to(device)
    chord_gt_4 = torch.LongTensor(chord_gt_4.squeeze(-1).cpu().numpy()).to(device)
    chord_gt_13 = chord_gt.narrow(2, 7, 13)
    l_1 = 0;
    l_2 = 0;
    l_4 = 0;
    l_13 = 0
    for k in range(seq_len):
        l_1 += mle_criterion_1(output_1[k], chord_gt_1[k]).squeeze(-1)
        l_2 += mle_criterion_2(output_2[k], chord_gt_2[k])
        l_4 += mle_criterion_4(output_4[k], chord_gt_4[k])
        l_13 += mle_criterion_13(output_13[k], chord_gt_13[k]).mean()
    MLE_loss = l_1 + l_2 + l_4 + l_13

    # compute mutual reward
    action1 = torch.Tensor(seq_len,batch_size, 1).zero_().to(device)
    for i in range(seq_len):
        for j in range(batch_size):
            if output_1[i][j][0].item() > 0.5:
                action1[i][j][0] = 1
    _,action2 = output_2.topk(1)
    _,action4 = output_4.topk(1)
    action13 = output_13
    _, index13 = action13.topk(13)
    mc1 = action1 * mc_output_1 + (1 - action1) * (1 - mc_output_1)
    mc2 = mc_output_2.gather(2, action2.long())
    mc4 = mc_output_4.gather(2, action4.long())
    mc13 = mc_output_13.gather(2, index13.long()).narrow(-1, 0, 4).mean(-1).unsqueeze(-1)
    c1 = action1 * c_output_1 + (1 - action1) * (1 - c_output_1)
    c2 = c_output_2.gather(2, action2.long())
    c4 = c_output_4.gather(2, action4.long())
    c13 = c_output_13.gather(2, index13.long()).narrow(-1, 0, 4).mean(-1).unsqueeze(-1)
    mutual_reward = (torch.log(mc1) - torch.log(c1) + mc2 + mc4 + mc13 - c2 - c4 - c13).detach()

    # get rewards of all steps
    rewards = Reward(chord, chord_real, action1, action2, action4, action13, condition_pitch, condition_duration,
                     condition_position, Reward_L,Reward_R1,Reward_R2, batch_size, seq_len, condition_window, harmony_rule_window,
                     criterion_1, criterion_2, criterion_4,criterion_13)
    rewards += mutual_reward
    rewards = rewards.narrow(0, 0, seq_len - 1) / 100

    action11 = action1
    _, action22 = output_2.topk(1)
    _, action44 = output_4.topk(1)
    _, action1313 = output_13.topk(4)
    actions = torch.cat([action11, action22, action44, action1313], dim=-1).squeeze()
    actions = actions.narrow(0, 0, seq_len - 1)
    return states,actions,rewards,MLE_loss


def DQN_train(states, actions, rewards, MLE_loss):
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
    beta = 0.7
    loss = beta * RL_loss + (1 - beta) * MLE_loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(current_model.parameters(), 10)
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()

    return RL_loss,beta

def _test(test_mini_data):
    '''
        test one batch
    '''
    one_batch,real = batch_data_win(test_mini_data, batch_size, condition_window, seq_len)
    states,actions,rewards,MLE_loss=sample_batch_data(one_batch, real, batch_size,"test")
    return rewards,MLE_loss


def trainIter_DQN():
    '''
        training loop
    '''
    # load data
    file = open(train_path, 'rb')
    train_data = pickle.load(file)
    file = open(test_path, 'rb')
    test_data = pickle.load(file)
    train_length = len(train_data)
    test_length = len(test_data)
    pic_path = f"./pics/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    best_reward = best_reward_init
    test_step = test_step_init
    for epoch in range(epoch_already+1,Epoch):
        f=open(f'./logs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.txt','a')
        f.write('\n------------------------epoch %d-------------------------' % (epoch))
        print("--------------------epoch: ",epoch,"---------------------")
        random.shuffle(train_data)
        total_loss = 0
        total_mle_loss=0
        total_reward=0
        train_start_idx=0
        batch_cnt=0
        while train_start_idx + batch_size <= train_length:
            train_batch_data=train_data[train_start_idx:train_start_idx+batch_size]
            one_batch,chord_real = batch_data_win(train_batch_data, batch_size, condition_window, seq_len)
            states,actions,rewards,MLE_loss=sample_batch_data(one_batch, chord_real,seq_len, batch_size,"train")
            loss,beta = DQN_train(states,actions,rewards,MLE_loss)
            total_loss+=loss.item()
            total_mle_loss+=MLE_loss.item()
            total_reward+=sum(rewards).mean().item()
            train_start_idx+=batch_size
            batch_cnt+=1
            if batch_cnt % target_n==0:
                target_model.load_state_dict(current_model.state_dict())
            if batch_cnt % Train_i == 0:
                print('epoch train:%d, %s(%d %d%%) reward: %.10f loss: %.10f mle: %.10f beta: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / ((train_length // batch_size) * batch_size) * 100,
                    total_reward / Train_i,total_loss / Train_i,total_mle_loss/Train_i,beta))
                f.write('\nepoch train:%d, %s(%d %d%%) reward: %.10f loss: %.10f mle: %.10f beta: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / ((train_length // batch_size) * batch_size) * 100,
                    total_reward / Train_i, total_loss / Train_i,total_mle_loss / Train_i,beta))
                print("---------------------------------------------------------------------------")
                total_loss=0
                total_mle_loss=0
                total_reward=0
            if batch_cnt%Test_i==0:
                test_start_idx = 0
                random.shuffle(test_data)
                test_aver_reward = 0
                total_test_MLE_loss=0
                test_cnt = 0
                while test_start_idx + batch_size <= test_length:
                    # print("test_cnt: ",test_cnt)
                    test_mini_data = test_data[test_start_idx:test_start_idx + batch_size]
                    test_reward,test_MLE_loss = _test(test_mini_data)
                    total_test_MLE_loss+=test_MLE_loss.item()
                    test_aver_reward += sum(test_reward).mean().item()
                    test_start_idx += batch_size
                    test_cnt += 1
                total_test_MLE_loss /=test_cnt
                test_aver_reward /= test_cnt
                test_rewards.append(test_aver_reward)
                test_batch_num.append(test_step)
                test_step= test_step+1
                plot_RL(test_batch_num, test_rewards,pic_path,args.model)
                print("test_aver_reward: ", test_aver_reward,"test MLE_loss: ",total_test_MLE_loss)
                f.write('\ntest_aver_reward: %.6f, test MLE_loss: %.6f' % (test_aver_reward,total_test_MLE_loss))
                if best_reward is None or best_reward < test_aver_reward:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, test_aver_reward))
                        f.write("\nBest reward updated: %.3f -> %.3f" % (best_reward, test_aver_reward))
                    best_reward = test_aver_reward
        f.close()
        model_save_path = model_path+"epoch%d_reward%.3f_mle_loss%.3f_beta%.3f.pth" % (epoch, best_reward,total_test_MLE_loss,beta)
        states = {
            'model': current_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'test_rewards': test_rewards,
            'test_nums': test_batch_num,
            'epoch': epoch
        }
        torch.save(states, model_save_path)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description='train LSTM model')
    parser.add_argument("--dataset", type=str, default='NMD', help="NMD or Wiki or TTD")
    parser.add_argument("--seq_len", type=str, default='64', help="64 or 128")
    parser.add_argument("--model", type=str, default='DQN')
    parser.add_argument("--repre", type=str, default='MH')
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--mutual_c", type=str, default=None)
    parser.add_argument("--mutual_mc", type=str, default=None)
    parser.add_argument("--batch_size", type=str, default="64")
    parser.add_argument("--epoch", type=str, default='100')
    parser.add_argument("--learning_rate", type=str, default='0.001')
    args = parser.parse_args()

    f = open(f'./logs/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.txt', 'a')
    f.write('\nHyperparameters: \n%s' % (str(args)))
    f.close()

    # data
    if args.dataset == "NMD":
        chord_num = 116
        if int(args.seq_len) == 64:
            train_path = "./data/slide_win64_stride8/NMD_train_104423.data"
            test_path = "./data/slide_win64_stride8/NMD_test_7962.data"
        else:
            train_path = "./data/slide_win128_stride8/NMD_train_58425.data"
            test_path = "./data/slide_win128_stride8/NMD_test_3894.data"
    elif args.dataset == "Wiki":
        chord_num = 310
        if int(args.seq_len) == 64:
            train_path = "./data/slide_win64_stride8/Wiki_train_202119.data"
            test_path = "./data/slide_win64_stride8/Wiki_test_19751.data"
        else:
            train_path = "./data/slide_win128_stride8/Wiki_train_88775.data"
            test_path = "./data/slide_win128_stride8/Wiki_test_8295.data"

    # hyperparameter
    condition_window = 8
    harmony_rule_window = 3
    input_size = 64
    hidden_size=512
    gamma = 0.99
    Reward_L = 1  # reward weight for negative loss reward
    Reward_R1 = 1  # reward weight for interval consonance reward
    Reward_R2 = 1  # reward weight for chord progression reward
    Test_i = 1000  # batch interval for printing testing info
    Train_i = 200  # batch interval for printing training info
    target_n = 3  # batch interval for updating target_model's parameters
    batch_size = int(args.batch_size)
    Epoch = int(args.epoch)
    seq_len = int(args.seq_len)

    # define model
    mutual_c = Mutual_Chord(input_size, hidden_size).to(device)
    mutual_mc = LSTM_Chord(condition_window, input_size, hidden_size).to(device)
    current_model = DQN_Chord(condition_window, input_size, hidden_size).to(device)
    target_model = DQN_Chord(condition_window, input_size, hidden_size).to(device)
    target_model.load_state_dict(current_model.state_dict())
    model_path = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    optimizer=torch.optim.SGD(current_model.parameters(),lr=float(args.learning_rate),momentum=0.9)

    # load model
    if args.mutual_c is not None and args.mutual_mc is not None:
        mutual_c_resume = f"./saved_models/{args.dataset}-Mutual-{args.repre}-{args.seq_len}/{args.mutual_c}"
        mutual_mc_resume = f"./saved_models/{args.dataset}-CLSTM-{args.repre}-{args.seq_len}/{args.mutual_mc}"
        mutual_c_dict = torch.load(mutual_c_resume, map_location=device)
        mutual_mc_dict = torch.load(mutual_mc_resume, map_location=device)
        mutual_c.load_state_dict(mutual_c_dict['model'])
        mutual_mc.load_state_dict(mutual_mc_dict['model'])
        mutual_c.eval()
        mutual_mc.eval()
    epoch_already = -1
    test_rewards = []
    test_batch_num = []
    test_step_init = 0
    best_reward_init = None
    if args.load_model is not None:
        load_model_path = model_path + args.load_model
        dict = torch.load(load_model_path, map_location=device)
        current_model.load_state_dict(dict['model'])
        target_model.load_state_dict(current_model.state_dict())
        optimizer.load_state_dict(dict['optimizer'])
        epoch_already = dict['epoch']
        test_rewards = dict['test_rewards']
        test_batch_num = dict['test_nums']
        test_step_init = len(test_rewards)
        best_reward_init = max(test_rewards)

    # train model
    criterion_1 = nn.BCELoss(reduction='none').to(device)
    criterion_2 = nn.NLLLoss(reduction='none').to(device)
    criterion_4 = nn.NLLLoss(reduction='none').to(device)
    criterion_13 = multilabel_categorical_crossentropy
    mle_criterion_1 = nn.BCELoss().to(device)
    mle_criterion_2 = nn.NLLLoss().to(device)
    mle_criterion_4 = nn.NLLLoss().to(device)
    mle_criterion_13 = multilabel_categorical_crossentropy

    start_time = time.time()
    trainIter_DQN()

