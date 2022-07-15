#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import torch
print(torch.cuda.is_available())
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='Dataset name')
parser.add_argument('--batchSize', type=int, default=100, help='Input batch size')
parser.add_argument('--n_node', type=int, default=100, required=True, help='Number of unique commands i.e. size of the Embedding layer')
parser.add_argument('--hiddenSize', type=int, default=100, help='Hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='Learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='The number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='Gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='The number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='Only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='Validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='Split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
   
    n_node = opt.n_node #Number of unique commands in the merged users dataset
    
    model = SessionGraph(opt, n_node)
    model = model.cuda()
    
    start = time.time()
    best_result = [0,0,0]
    best_epoch = [0,0,0]
    bad_counter = 0
    
    for epoch in range(opt.epoch):
        train_hit, train_mrr, train_acc, train_loss, test_hit, test_mrr, test_acc, test_loss = train_test(epoch, model, train_data, test_data)
        print("Train Loss:%.4f Train Accuracy:%.4f Train Precision@5:%.4f Train MRR@5:%.4f Test Loss:%.4f Test Accuracy:%.4f Test Precision@5:%.4f Test MRR@5:%.4f\n"% (train_loss, train_acc, train_hit, train_mrr, test_loss, test_acc, test_hit, test_mrr))
        flag = 0
        if test_hit >= best_result[0]:
            best_result[0] = test_hit
            best_epoch[0] = epoch
            flag = 1
        if test_mrr >= best_result[1]:
            best_result[1] = test_mrr
            best_epoch[1] = epoch
            flag = 1
        if test_acc >= best_result[2]:
            best_result[2] = test_acc
            best_epoch[2] = epoch
            flag = 1
        
        bad_counter += 1 - flag
#         if bad_counter >= opt.patience:
#             break
            
    print('Best Result:')
    print('\tRecall@20:\t%.4f\tMRR@5:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
