#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""
import os
os.chdir('/home/dshah47/.local/lib/python3.7/site-packages')
import networkx as nx
os.chdir('../../../../SR-GNN2Users/pytorch_code')
import numpy as np
import torch
from torch.nn import Module
import multiprocessing as mp
from tqdm import tqdm
import random
import sys, pdb, math, time

def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph



def data_masks(all_usr_pois, item_tail):
    #Getting lengths of the sequences in the data
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    
    #Appending zeros to the sequences to match the maximum length sequence
    us_pois = [upois + item_tail * (len_max - le) for upois, le in tqdm(zip(all_usr_pois, us_lens))]
    
    #Preparing masks like - 1's for the actual data and 0's for the appended zeros
    us_msks = [[1] * le + [0] * (len_max - le) for le in tqdm(us_lens)]
    
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0] #The training sequences
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs) #New sequences with appended 0's
        self.mask = np.asarray(mask) #Mask with 1's for the data and 0's for the appended zeros
        self.len_max = len_max
        self.targets = np.asarray(data[1], dtype=np.float32)
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        #We are shuffling the training or testing data
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    def generate_batch2(self, batch_size, n_batch):
        #We are shuffling the training or testing data
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
#         n_batch = int(self.length / batch_size)
#         if self.length % batch_size != 0:
#             n_batch += 1
        slices = []
        for _ in range(n_batch):
            slices += [random.choices(np.arange(self.length), k=batch_size)]

        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input))) #np.unique - Returns unique sorted array
        
        max_n_node = np.max(n_node) #Max length after "uniquifying" the sequences
        
        #inputs is 100 sequences(#batchsize) 
        #u_input is a sequence that we'll convert into a graph by creating an Adjacency matrix u_A 
        #Will create u_A for each of 100 sequences and append into A.
        for u_input in inputs:
            #This removes duplicates and sorts the u_input so we can later use it for indexing purposes using np.where()
            node = np.unique(u_input)
            
            #Different from what we did in data_masks function - here we have uniquified the data
            #so we now have different maximum length, so we append zeros according to new max length
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            
            u_A = np.zeros((max_n_node, max_n_node)) #Max_n_node x Max_n_node
            
            #Creating the graph matrix 
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                
                #Edges are maintained here because we are using indices here from the uniquified and 'sorted'
                #sequences, so don't worry that if the commands were sorted then the edges would also alter
                #that won't happen because we are getting indices based on u_input only (original sequence)
                #"node" is just so that we have unique commands and in sorted way so we can have fixed matrix repre.
                u_A[u][v] = 1
            
            u_sum_in = np.sum(u_A, 0) #Columnwise summation for incoming edges
            u_sum_in[np.where(u_sum_in == 0)] = 1 #Every zero is now 1 and other 2's/3's remain same
            u_A_in = np.divide(u_A, u_sum_in) #Each edge's weight = occurence of that edge/outdegree of the edges's startnode
            
            u_sum_out = np.sum(u_A, 1) #Same thing repeat for rowwise i.e. outgoing edges
            u_sum_out[np.where(u_sum_out == 0)] = 1 #
            u_A_out = np.divide(u_A.transpose(), u_sum_out) #
            
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A) #The adjacency matrix A of shape: d (max_n_node) x 2d (2 * max_n_node)
            
            #Inputs indices from "node" with the zeros included. (before we broke out the loop 
            #whenever the first zero encountered)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            #print("Items:", items)
            #print("Alias inputs:", alias_inputs)
        return alias_inputs, A, items, mask, targets
