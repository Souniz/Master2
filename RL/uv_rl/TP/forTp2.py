#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:39:44 2020

@author: berar
"""
import numpy as np


class Grid_Solv:
    def __init__(self,GRID_SIZE=4):
        self.GRID_SIZE=GRID_SIZE
        self.TERMINAL_STATES = [0, GRID_SIZE*GRID_SIZE-1]
        self.states = np.arange(GRID_SIZE*GRID_SIZE)
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.discount = 1.
        # The unifom policy 
        self.uniform_policy = {s : { a : 
                                    1/len(self.actions) for a in self.actions } for s in self.states}

    def next_state(self,grid_size, state, action):
        i,j = np.unravel_index(state, (grid_size, grid_size))
        if action == 'UP':
            i = np.maximum(0,i-1)
        elif action == 'DOWN':
            i = np.minimum(i+1,grid_size-1)
        elif action == 'RIGHT':
            j = np.minimum(j+1,grid_size-1)
        elif action == 'LEFT':
            j = np.maximum(0,j-1)    
        new_state = np.ravel_multi_index((i,j), (grid_size, grid_size))
        return new_state
    
    def is_done(self,state, terminal_states):
        return state in terminal_states
   # Transition is coded as a dictionary of dictionary 
    def transition(self):

        P = {}
        for s in range(len(self.states)):
            P[s] = {a : () for a in self.actions}
            if s in self.TERMINAL_STATES:
                # if terminal state, stay where you are
                # instead of next_state
                reward = 0.
                for action in self.actions:
                    P[s][action] = (s, reward, True)
            else:
                # transition
                reward = -1.
                for action in self.actions:
                    next_s = self.next_state(self.GRID_SIZE, s, action)
                    P[s][action] = (next_s,reward,self.is_done(next_s, self.TERMINAL_STATES))
        return P
        


