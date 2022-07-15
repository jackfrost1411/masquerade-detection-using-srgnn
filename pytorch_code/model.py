#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step #1
        self.hidden_size = hidden_size #100
        self.input_size = hidden_size * 2 #200
        self.gate_size = 3 * hidden_size #300
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size)) #300x200
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size)) #300x100
        self.b_ih = Parameter(torch.Tensor(self.gate_size)) #300
        self.b_hh = Parameter(torch.Tensor(self.gate_size)) #300
        self.b_iah = Parameter(torch.Tensor(self.hidden_size)) #100
        self.b_oah = Parameter(torch.Tensor(self.hidden_size)) #100

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True) #100 -> 100
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True) #100 -> 100
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True) #100 -> 100

    def GNNCell(self, A, hidden):
#         print("A:",A.shape) # [200x 15x 30] for 15 max nodes in this batch
#         print(A[:, :, :A.shape[1]].shape, self.linear_edge_in(hidden).shape, self.b_iah.shape) #torch.Size([200, 15, 15]) torch.Size([200, 15, 100]) torch.Size([100])
#         print(hidden.shape) #[200,15,100]

        #For all 200 batches, For all 15 nodes we will multiply each hidden dimension (can be interpreted as feature) i.e. (15x15) x (15x100) = (15x100) 
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
#         print(input_in.shape) #torch.Size([200, 15, 100])
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
#         print(inputs.shape) #here inputs shape = torch.Size([200, 15, 200])

        #'inputs' shape dimension(2) = 2 x 'hidden' shape dimension(2)
        
        #But because of shape of w_ih and w_hh the shapes of gi and gh gets same.
        gi = F.linear(inputs, self.w_ih, self.b_ih) #inputs x (w_ih)^T + b_ih
        gh = F.linear(hidden, self.w_hh, self.b_hh) #hidden x (w_hh)^T + b_hh
        #gh and gi are of same shape
        
        #(3 chunks and 2 is for specifying split in second dimension)
        i_r, i_i, i_n = gi.chunk(3, 2) #E.g., gi(200, 6, 300) -> (200, 6, 100), (200, 6, 100), (200, 6, 100)
        h_r, h_i, h_n = gh.chunk(3, 2) #E.g., gh(200, 6, 300) -> (200, 6, 100), (200, 6, 100), (200, 6, 100)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        
        newgate = torch.tanh(i_n + resetgate * h_n)
        
        #hy = newgate + inputgate * (hidden - newgate)
        #According to the paper formula
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step): #Run for number of steps specified for GNN here, 1
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size) 
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        #hidden -> Output from the Gated GNN
        
        #ht -> last computed command embedding for all batches
        #The sum of mask embedding over first dimension will give you the index of the last computed command embedding of
        #length 100 in a batch (1 out of 100) of 16(commands)x100(each command's embedding)
        #seq_hidden states are computed in a sorted manner becuase it is alias input based and 
        #last extra appended 0's commands' embeddings (if any) would be last in 16 x 100.. it would be at 15th or 16th
        #position for example
        #the number of actual commands = total 1's in the mask embedding -> their sum is = the last executed command's index
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        
        #SOFT ATTENTION MECHANISM
        #Here, ht -> Vn or Sl from the equation 6.
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        #Here, hidden -> Vi from the equation 6.
        q2 = self.linear_two(hidden) # batch_size x seq_length x latent_size
        #alpha from equation 6.
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        
        #a -> Sg from equation 6.
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        #Calculating hybrid Sh
        if not self.nonhybrid:
            #a -> Sh from equation 7.
            a = self.linear_transform(torch.cat([a, ht], 1))
        
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        
        #Score for each command for being the next command
        #100x100 x 100x1398 -> 100x1398 (1398 command scores of being the next command for each batch)
        scores = torch.matmul(a, b.transpose(1, 0)) 
        return scores

    def forward(self, inputs, A):        
        #Will return the hidden state of shape:
        #inputs.shape[0] (that is the batch size) x inputs.shape[1] (sequence of commands length) x embeddinghidden_size(=100)
        #print("Input shape:", inputs.shape)
        hidden = self.embedding(inputs)
        #print("Before gnn:" ,hidden.shape)
        hidden = hidden
        
        hidden = self.gnn(A, hidden)
        #print("After gnn:",hidden.shape) #Same as before gnn shape
        
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    #print(alias_inputs)
    #To cuda
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    
    hidden = model(items, A)
    hidden = hidden.cuda()
    
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    seq_hidden = seq_hidden.cuda()
    #print(hidden.shape, seq_hidden.shape)
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(epoch, model, train_data, test_data):
    #print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = [0.0]
    slices = train_data.generate_batch(model.batch_size)
    pbar = tqdm(zip(slices, np.arange(len(slices))))
    train_hit, train_mrr, train_acc = [], [], []
    for i, j in pbar:
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
#         print(targets.shape, scores.shape)
        
        train_sub_scores = scores.topk(5)[1]
        train_sub_scores = trans_to_cpu(train_sub_scores).detach().numpy()
#         print(np.array(train_sub_scores).shape)
        
        for score, target, mask in zip(train_sub_scores, targets, train_data.mask):
            train_hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                train_mrr.append(0)
            else:
                train_mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
#                 print(score, target)
#                 print((np.where(score == target - 1)[0][0] + 1))
            
            if score[0]==target-1: train_acc += [1]
            else: train_acc+=[0]
        
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += [loss.item()]
        #f j % int(len(slices) / 5 + 1) == 0:
        pbar.set_description('Epoch %d:[%d/%d] Train Loss:%.3f Train Accuracy:%.3f Train Precision@5:%.3f Train MRR@5:%.3f' % (epoch, j, len(slices), np.mean(total_loss), np.mean(train_acc), np.mean(train_hit), np.mean(train_mrr)))
#     model.scheduler.step()
#         print(np.array(train_hit).shape)
    
    train_hit = np.mean(train_hit) * 100
    train_mrr = np.mean(train_mrr) * 100
    train_acc = np.mean(train_acc) * 100
    train_loss = np.mean(total_loss)
    
    #print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, acc = [], [], []
    total_loss = [0.0]
    slices = test_data.generate_batch(model.batch_size)
    pbar = tqdm(zip(slices, np.arange(len(slices))))
    for i, j in pbar:
        targets, scores = forward(model, i, test_data)
#         print(np.array(targets).shape, np.array(scores).shape)
        sub_scores = scores.topk(5)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
#         print(np.array(sub_scores).shape)
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
#                 print((np.where(score == target - 1)[0][0] + 1))

            if score[0]==target-1: acc += [1]
            else: acc+=[0]
                
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        total_loss += [loss.item()]
        pbar.set_description('Epoch %d:[%d/%d] Test Loss:%.3f Test Accuracy:%.3f Test Precision@5:%.3f Test MRR@5:%.3f' % (epoch, j, len(slices), np.mean(total_loss), np.mean(acc), np.mean(hit), np.mean(mrr)))
#         print(np.array(hit).shape)
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    acc = np.mean(acc) * 100
    loss = np.mean(total_loss)
    
    return train_hit, train_mrr, train_acc, train_loss, hit, mrr, acc, loss