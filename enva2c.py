# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:35:30 2019

@author: abhil
"""


from A2C import A2C
from env_nn import PortfolioNN_a2c


class PortfolioA2C(A2C):
    def __init__(self, *args, **kw):

        self.min_epsilon = kw['min_epsilon']
        self.initial_epsilon = kw['initial_epsilon']
        self.epsilon_decay_steps = kw['epsilon_decay_steps']
        self.action_space_dim = kw['action_space_dim']
        for key in ['action_space_dim','min_epsilon', 'initial_epsilon', 'epsilon_decay_steps']:
            if key in kw: del kw[key]

        super(PortfolioA2C, self).__init__(*args, **kw)
        
        for key in ['action_space_map','max_time_spent_in_episode','num_iterations','sample_every_N_transitions','batchsize','copy_over_target_every_M_training_iterations', 'buffer_size', 'min_buffer_size_to_train', 'models_path']:
            if key in kw: del kw[key]

        self.state_space_dim = self.env.observation_space.shape
        self.Q = PortfolioNN_a2c(self.state_space_dim,self.action_space_dim, self.gamma, **kw)
        self.Q_target = PortfolioNN_a2c(self.state_space_dim,self.action_space_dim, self.gamma, **kw)

    def sample_random_action(self):
        '''
        Uniform random
        '''
        return np.random.choice(self.action_space_dim)

    # def epsilon(self, epoch=None, total_steps=None):
    #     return 1./(total_steps/100 + 3)
    def epsilon(self, epoch=None, total_steps=None):
        if epoch >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            alpha = epoch / float(self.epsilon_decay_steps)
            current_epsilon = self.initial_epsilon * (1-alpha) + self.min_epsilon * (alpha)
            return current_epsilon
