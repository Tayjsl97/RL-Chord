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
from torch.distributions import Categorical,Bernoulli,Multinomial
from model.PPO_Chord import PPO_Chord
from model.Mutual_Chord import Mutual_Chord
from model.CLSTM_Chord_MH import LSTM_Chord
from train_utils import timeSince,plot_RL,multilabel_categorical_crossentropy,Reward,batch_data_win,normalize


def sample_batch_data(one_batch,chord_real,type):
    '''
        Sample one batch data
    '''
    if type=="train":
        model.train()
    else:
        model.eval()
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
    output_1, output_2, output_4, output_13, hidden, value = model(states, hidden)
    mc_output_1, mc_output_2, mc_output_4, mc_output_13, hidden = mutual_mc(states, hidden)
    c_output_1, c_output_2, c_output_4, c_output_13, hidden = mutual_c(states_c, hidden)

    # compute mutual reward
    dist1 = Bernoulli(output_1)
    dist2 = Categorical(output_2)
    dist4 = Categorical(output_4)
    dist13 = Multinomial(1000,output_13)
    action1 = dist1.sample().squeeze().unsqueeze(-1).to(device)
    action2 = dist2.sample().squeeze().unsqueeze(-1).to(device)
    action4 = dist4.sample().squeeze().unsqueeze(-1).to(device)
    action13 = dist13.sample().squeeze().to(device)
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
    rewards = Reward(chord,chord_real,action1,action2,action4,action13, condition_pitch, condition_duration,condition_position,
                     Reward_L, Reward_R1, Reward_R2, batch_size, seq_len, condition_window, harmony_rule_window,
                     criterion_1, criterion_2, criterion_4, criterion_13)
    rewards+=mutual_reward

    # save for loss calculation
    action = torch.cat([action1, action2, action4, action13], dim=-1).squeeze()
    log_prob1 = dist1.log_prob(action1)
    log_prob2 = dist2.log_prob(action2.squeeze(-1)).view(-1, batch_size, 1)
    log_prob4 = dist4.log_prob(action4.squeeze(-1)).view(-1, batch_size, 1)
    log_prob13 = dist13.log_prob(action13).unsqueeze(-1)
    log_prob1=log_prob1.narrow(0, 0, seq_len - 1).detach()
    log_prob2=log_prob2.narrow(0,0,seq_len-1).detach()
    log_prob4=log_prob4.narrow(0,0,seq_len-1).detach()
    log_prob13=log_prob13.narrow(0,0,seq_len-1).detach()
    actions=action.narrow(0,0,seq_len-1)
    values=value.detach()
    rewards=rewards.narrow(0,0,seq_len-1)
    states=states.narrow(0,0,seq_len-1)
    log_probs=[log_prob1,log_prob2,log_prob4,log_prob13]
    log_probs=torch.cat(log_probs,dim=-1).detach()
    return states,actions,values,log_probs,rewards


def compute_gae(rewards, values,gamma, lam):
    '''
        Compute GAE [1]
        [1] Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control
        using generalized advantage estimation[J]. arXiv preprint arXiv:1506.02438, 2015.
    '''
    gae=0
    returns=[]
    for step in reversed(range(len(rewards))):
        delta=rewards[step]+gamma*values[step+1]-values[step]
        gae=delta+gamma*lam*gae
        returns.insert(0,gae+values[step])
    return returns


def PPO_update(chord,states,actions,log_probs,returns,advantage):
    '''
        Randomly divide the batch_size into smaller mini_batch_size,
        train for ppo_epochs times, and take the average loss over ppo_epochs as the loss for this batch.
    '''
    chord_gt = torch.Tensor(chord[1:]).to(device)
    ppo_loss=0
    ppo_mle_loss=0
    log_probs_1, log_probs_2, log_probs_4, log_probs_13 = torch.split(log_probs, [1, 1, 1, 1], dim=-1)
    for ppo_epoch in range(ppo_epochs):
        r=random.random
        random.seed(random.randint(0,100))
        index=list(range(batch_size))
        random.shuffle(index,random=r)
        mini_start_idx = 0
        loss_aver=0
        mle_loss_aver=0
        cnt=0
        while mini_start_idx + mini_batch_size <= batch_size:
            mini_index=index[mini_start_idx:mini_start_idx+mini_batch_size]
            mini_states=[];mini_actions=[];
            mini_log_probs_1=[];mini_log_probs_2=[];mini_log_probs_4=[];mini_log_probs_13=[];
            mini_returns=[];mini_advantage=[]
            mini_chord=[];
            for j in range(seq_len-1):
                mini_states_tmp = []; mini_actions_tmp = []
                mini_log_probs_1_tmp = []; mini_log_probs_2_tmp = [];
                mini_log_probs_4_tmp = [];mini_log_probs_13_tmp = []
                mini_returns_tmp = [];mini_advantage_tmp = []
                mini_chord_tmp=[]
                for i in mini_index:
                    mini_states_tmp.append(states[j][i])
                    mini_actions_tmp.append(actions[j][i])
                    mini_log_probs_1_tmp.append(log_probs_1[j][i])
                    mini_log_probs_2_tmp.append(log_probs_2[j][i])
                    mini_log_probs_4_tmp.append(log_probs_4[j][i])
                    mini_log_probs_13_tmp.append(log_probs_13[j][i])
                    mini_returns_tmp.append(returns[j][i])
                    mini_advantage_tmp.append(advantage[j][i])
                    mini_chord_tmp.append(chord_gt[j][i])
                mini_states.append(torch.stack(mini_states_tmp, 0))
                mini_actions.append(torch.stack(mini_actions_tmp, 0))
                mini_log_probs_1.append(torch.stack(mini_log_probs_1_tmp, 0))
                mini_log_probs_2.append(torch.stack(mini_log_probs_2_tmp, 0))
                mini_log_probs_4.append(torch.stack(mini_log_probs_4_tmp, 0))
                mini_log_probs_13.append(torch.stack(mini_log_probs_13_tmp, 0))
                mini_returns.append(torch.stack(mini_returns_tmp, 0))
                mini_advantage.append(torch.stack(mini_advantage_tmp, 0))
                mini_chord.append(torch.stack(mini_chord_tmp, 0))
            mini_states=torch.stack(mini_states)
            mini_actions=torch.stack(mini_actions)
            mini_log_probs_1 = torch.stack(mini_log_probs_1)
            mini_log_probs_2=torch.stack(mini_log_probs_2)
            mini_log_probs_4=torch.stack(mini_log_probs_4)
            mini_log_probs_13=torch.stack(mini_log_probs_13)
            mini_returns=torch.stack(mini_returns)
            mini_advantage=torch.stack(mini_advantage)
            mini_chord=torch.stack(mini_chord)
            mini_log_probs=[mini_log_probs_1,mini_log_probs_2,mini_log_probs_4,mini_log_probs_13]
            loss,mle_loss,beta = PPO_train(mini_states, mini_actions, mini_log_probs, mini_returns, mini_advantage, mini_chord, E_CLIP)
            loss_aver+=loss
            mle_loss_aver+=mle_loss
            mini_start_idx+=mini_batch_size
            cnt+=1
        loss_aver=loss_aver/cnt
        mle_loss_aver=mle_loss_aver/cnt
        ppo_loss+=loss_aver
        ppo_mle_loss+=mle_loss_aver
    return ppo_loss/ppo_epochs,ppo_mle_loss/ppo_epochs,beta


def PPO_train(states, actions, old_log_probs, returns, advantage, chord_gt, clip_param):
    '''
        train one batch
    '''
    old_log_probs_1 = old_log_probs[0]
    old_log_probs_2=old_log_probs[1]
    old_log_probs_4=old_log_probs[2]
    old_log_probs_13=old_log_probs[3]
    hidden=None
    output_1,output_2, output_4, output_13, hidden, value = model(states, hidden)

    # compute MLE loss
    chord_gt_1 = chord_gt.narrow(2, 0, 1)
    _, chord_gt_2 = chord_gt.narrow(2, 1, 2).topk(1)
    _, chord_gt_4 = chord_gt.narrow(2, 3, 4).topk(1)
    chord_gt_2 = torch.LongTensor(chord_gt_2.squeeze(-1).cpu().numpy()).to(device)
    chord_gt_4 = torch.LongTensor(chord_gt_4.squeeze(-1).cpu().numpy()).to(device)
    chord_gt_13 = chord_gt.narrow(2, 7, 13)
    l_1=0;l_2=0;l_4=0;l_13=0
    for k in range(seq_len-1):
        l_1 += mle_criterion_1(output_1[k],chord_gt_1[k]).squeeze(-1)
        l_2 += mle_criterion_2(output_2[k], chord_gt_2[k])
        l_4 += mle_criterion_4(output_4[k], chord_gt_4[k])
        l_13 += mle_criterion_13(output_13[k],chord_gt_13[k]).mean()
    MLE_loss=l_1+l_2+l_4+l_13

    # compute PPO loss
    dist1 = Bernoulli(output_1)
    dist2 = Categorical(output_2)
    dist4 = Categorical(output_4)
    dist13 = Multinomial(1000,output_13)
    action1,action2, action4, action13 = torch.split(actions, [1, 1, 1, 13], dim=-1)
    new_log_probs1 = dist1.log_prob(action1)
    new_log_probs2 = dist2.log_prob(action2.squeeze(-1)).view(-1, mini_batch_size, 1)
    new_log_probs4 = dist4.log_prob(action4.squeeze(-1)).view(-1, mini_batch_size, 1)
    new_log_probs13 = dist13.log_prob(action13).unsqueeze(-1)
    ratio_1 = (new_log_probs1 - old_log_probs_1).exp()
    ratio_2 = (new_log_probs2 - old_log_probs_2).exp()
    ratio_4 = (new_log_probs4 - old_log_probs_4).exp()
    ratio_13 = (new_log_probs13 - old_log_probs_13).exp()
    surr1_1 = ratio_1 * advantage
    surr2_1 = torch.clamp(ratio_1, 1.0 - clip_param, 1.0 + clip_param) * advantage
    surr1_2 = ratio_2*advantage
    surr2_2 = torch.clamp(ratio_2, 1.0 - clip_param, 1.0 + clip_param)*advantage
    surr1_4 = ratio_4*advantage
    surr2_4 = torch.clamp(ratio_4, 1.0 - clip_param, 1.0 + clip_param)*advantage
    surr1_13 = ratio_13*advantage
    surr2_13 = torch.clamp(ratio_13, 1.0 - clip_param, 1.0 + clip_param)*advantage
    actor_loss_1 = - torch.min(surr1_1, surr2_1).mean()
    actor_loss_2 = - torch.min(surr1_2, surr2_2).mean()
    actor_loss_4 = - torch.min(surr1_4, surr2_4).mean()
    actor_loss_13 = - torch.min(surr1_13, surr2_13).mean()
    actor_loss = actor_loss_1 + actor_loss_2 + actor_loss_4 + actor_loss_13
    critic_loss = nn.MSELoss()(returns,value)
    entropy1 = dist1.entropy().mean()
    entropy2 = dist2.entropy().mean()
    entropy4 = dist4.entropy().mean()
    entropy13 = 0
    entropy = entropy1 + entropy2 + entropy4 + entropy13
    rl_loss = C_1 * critic_loss + actor_loss - C_2 * entropy
    beta=0.7
    loss=beta*rl_loss+(1-beta)*MLE_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return rl_loss,MLE_loss,beta


def _test(test_mini_data):
    '''
        test one batch
    '''
    one_batch,real = batch_data_win(test_mini_data, batch_size, condition_window, seq_len)
    states, actions, values, log_probs, rewards, MLE_loss = sample_batch_data(one_batch,real, batch_size, "test")
    return rewards,MLE_loss


def trainIter_PPO():
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
        total_mle_loss = 0
        total_reward=0
        train_start_idx=0
        batch_cnt=0
        while train_start_idx + batch_size <= train_length:
            train_batch_data=train_data[train_start_idx:train_start_idx+batch_size]
            one_batch,chord_real = batch_data_win(train_batch_data, batch_size, condition_window, seq_len)
            states, actions, values, log_probs, rewards= \
                sample_batch_data(one_batch, chord_real,seq_len, batch_size,"train")
            returns = compute_gae(rewards, values,G_GAE,L_GAE)
            returns=torch.stack(returns).detach()
            values = values.narrow(0, 0, seq_len - 1)
            advantage = [returns[i] - values[i] for i in range(returns.shape[0])]
            advantage = [normalize(advantage[j]) for j in range(len(advantage))]
            advantage=torch.stack(advantage)
            loss,mle_loss,beta=PPO_update(one_batch['chords'],states,actions,log_probs,returns,advantage)
            total_loss += loss.item()
            total_mle_loss += mle_loss.item()
            total_reward += sum(rewards).mean().item()
            train_start_idx += batch_size
            batch_cnt+=1
            if batch_cnt%Train_i==0:
                print('epoch train:%d, %s(%d %d%%) reward: %.10f loss: %.10f mle: %.10f beta: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / ((train_length // batch_size) * batch_size) * 100, total_reward/ Train_i,
                    total_loss / Train_i,total_mle_loss / Train_i,beta))
                f.write('\nepoch:%d, %s(%d %d%%) reward: %.10f loss: %.10f mle: %.10f beta: %.6f' % (
                    epoch, timeSince(start_time), train_start_idx,
                    train_start_idx / ((train_length // batch_size) * batch_size) * 100, total_reward/ Train_i,
                    total_loss / Train_i,total_mle_loss / Train_i,beta))
                print("--------------------------------------------------------------")
                total_loss = 0
                total_mle_loss = 0
                total_reward = 0
            if batch_cnt%Test_i==0:
                test_start_idx=0
                random.shuffle(test_data)
                test_aver_reward=0
                total_test_MLE_loss = 0
                test_cnt=0
                while test_start_idx + mini_batch_size <= test_length:
                    test_mini_data = test_data[test_start_idx:test_start_idx + mini_batch_size]
                    test_reward,test_MLE_loss=_test(test_mini_data)
                    total_test_MLE_loss += test_MLE_loss.item()
                    test_aver_reward += sum(test_reward).mean().item()
                    test_start_idx+=mini_batch_size
                    test_cnt+=1
                total_test_MLE_loss /= test_cnt
                test_aver_reward /= test_cnt
                test_rewards.append(test_aver_reward)
                test_batch_num.append(test_step)
                test_step=test_step+1
                plot_RL(test_batch_num,test_rewards,pic_path, args.model)
                print("test_aver_reward: ",test_aver_reward,"test MLE_loss: ",total_test_MLE_loss)
                f.write('\ntest_aver_reward: %.6f  test MLE_loss: %.6f: ' % (test_aver_reward, total_test_MLE_loss))
                if best_reward is None or best_reward<test_aver_reward:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, test_aver_reward))
                        f.write("\nBest reward updated: %.3f -> %.3f" % (best_reward, test_aver_reward))
                    best_reward = test_aver_reward
        f.close()
        model_save_path = model_path +"epoch%d_reward%.3f_mle_loss%.3f_beta%.3f.pth" % (epoch,best_reward,total_test_MLE_loss,beta)
        states = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'test_rewards': test_rewards,
            'test_nums': test_batch_num,
            'epoch': epoch
        }
        torch.save(states, model_save_path)


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description='train LSTM model')
    parser.add_argument("--dataset", type=str, default='NMD', help="NMD or Wiki or TTD")
    parser.add_argument("--seq_len", type=str, default='64', help="64 or 128")
    parser.add_argument("--model", type=str, default='PPO')
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
    G_GAE = 0.99  # gamma param for GAE
    L_GAE = 0.95  # lambda param for GAE
    E_CLIP = 0.2  # clipping coefficient
    C_1 = 0.1  # critic loss coefficient
    C_2 = 0.001  # entropy coefficient
    Reward_L = 1  # reward weight for negative loss reward
    Reward_R1 = 1  # reward weight for interval consonance reward
    Reward_R2 = 1  # reward weight for chord progression reward
    Test_i = 120  # batch interval for printing testing info
    Train_i = 40  # batch interval for printing training info
    batch_size = int(args.batch_size)
    Epoch = int(args.epoch)
    seq_len = int(args.seq_len)
    ppo_epochs = 2
    mini_batch_size = batch_size//5

    # define model
    model = PPO_Chord(condition_window, input_size, hidden_size).to(device)
    mutual_c = Mutual_Chord(input_size, hidden_size).to(device)
    mutual_mc = LSTM_Chord(condition_window, input_size, hidden_size).to(device)
    model_path = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum=0.9)

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
        model.load_state_dict(dict['model'])
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

    start_time=time.time()
    trainIter_PPO()

