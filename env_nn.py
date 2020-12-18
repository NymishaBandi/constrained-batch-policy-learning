import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model as KerasModel
from keras.layers import Input, Dense, Flatten, concatenate, dot, MaxPooling2D,Dropout
from keras.layers.merge import Add, Multiply
from keras.losses import mean_squared_error
from keras import optimizers
from keras import regularizers
from keras.callbacks import Callback, TensorBoard
from exact_policy_evaluation import ExactPolicyEvaluator
from keras_tqdm import TQDMCallback
from model import Model
from keras import backend as K
from skimage import color
import os
from keras.layers.convolutional import Conv2D
from collections import deque
import tensorflow as tf

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(42)


class PortfolioNN(Model):
    def __init__(self, input_shape, dim_of_actions, gamma, convergence_of_model_epsilon=1e-10, model_type='mlp', position_of_holes=None, position_of_goals=None, num_frame_stack=None, frame_skip= None, pic_size = None, freeze_cnn_layers=False,**kw):
        '''
        An implementation of fitted Q iteration

        num_inputs: number of inputs
        num_outputs: number of outputs
        dim_of_actions: dimension of action space
        convergence_of_model_epsilon: small float. Defines when the model has converged.
        '''
#        super(PortfolioNN, self).__init__()
        super().__init__()
        self.convergence_of_model_epsilon = convergence_of_model_epsilon 
        self.model_type = model_type
        self.dim_of_actions = dim_of_actions
        self.dim_of_state = input_shape
        self.freeze_cnn_layers = freeze_cnn_layers
        self.model = self.create_model(input_shape)
        self.all_actions_func=None
        #debug purposes
        from config_portfolio import env
        self.policy_evalutor = ExactPolicyEvaluator(None, gamma, env=env, num_frame_stack=num_frame_stack, frame_skip = frame_skip, pic_size = pic_size)


    def create_model(self, input_shape):
        
##        self.model = Sequential()
##
##        self.model.add(
##            Conv2D(filters=32, kernel_size=(1, 3), input_shape=(self.nb_classes, self.window_length, 1),
##                   activation='relu'))
##        self.model.add(Dropout(0.5))
##        self.model.add(Conv2D(filters=32, kernel_size=(1, self.window_length - 2), activation='relu'))
##        self.model.add(Dropout(0.5))
##        self.model.add(Flatten())
##        self.model.add(Dense(64, activation='relu'))
##        self.model.add(Dropout(0.5))
##        self.model.add(Dense(64, activation='relu'))
##        self.model.add(Dropout(0.5))
##        self.model.add(Dense(self.nb_classes, activation='softmax'))
##        self.model.compile(loss='categorical_crossentropy',
##                           optimizer=Adam(lr=1e-3),
##                           metrics=['accuracy'])
##        
##        self.graph = tf.get_default_graph()
#       
#        
        if self.model_type == 'cnn':
#            print("input_shape",input_shape)
            inp = Input(shape=input_shape, name='inp')
#            print("input shape",inp.shape)
#            action_mask = Input(shape=(self.dim_of_actions,1), name='mask')
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=np.random.randint(2*32))

#            conv1 = Conv2D(32, (1,3), trainable=not self.freeze_cnn_layers, activation='relu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(inp)
#            dropout1 = Dropout(0.5)(conv1)
#            conv2 = Conv2D(32, (1,input_shape[2]-2), trainable=not self.freeze_cnn_layers, activation='relu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(dropout1)
#            dropout2 = Dropout(0.5)(conv2)
#            flat1 = Flatten(name='flattened')(dropout2)
#            dense1 = Dense(64, activation='relu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(flat1)
#            dropout3 = Dropout(0.5)(dense1)
#            dense2 = Dense(64, activation='relu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(dropout2)
#            dropout4 = Dropout(0.5)(dense2)
#            all_actions = Dense(self.dim_of_actions, name='all_actions', activation="softmax",kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(dropout4)
#            
            
            conv1 = Conv2D(32, (1,3),activation='relu')(inp)
#            print("conv1",conv1.shape)
            dropout1 = Dropout(0.5)(conv1)
            conv2 = Conv2D(32, (1,input_shape[1]-2), activation='relu')(dropout1)
#            print("conv2",conv2.shape)
            dropout2 = Dropout(0.5)(conv2)
            flat1 = Flatten(name='flattened')(dropout2)
#            print("flat1",flat1.shape)
            dense1 = Dense(64, activation='relu')(flat1)
            dropout3 = Dropout(0.5)(dense1)
#            print("dropout3",dropout3.shape)
            dense2 = Dense(64, activation='relu')(dropout3)
            dropout4 = Dropout(0.5)(dense2)
#            print("dropout4",dropout4.shape)
            all_actions = Dense(self.dim_of_actions, name='all_actions', activation="relu")(dropout4)
#            print("all_actions",all_actions.shape)
            output = [all_actions]
#            print("output",output[0])

#            model = KerasModel(inputs=[inp, action_mask], outputs=output)
            model = KerasModel(inputs=[inp], outputs=output)

#            rmsprop = optimizers.RMSprop(lr=0.0005, rho=0.95, epsilon=1e-08, decay=0.0)
            model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['accuracy'])
            self.all_actions_func = K.function([model.get_layer('inp').input], [model.get_layer('all_actions').output])
            # if self.freeze_cnn_layers:
            #     conv1.trainable = False
            #     conv2.trainable = False
            #     pool1.trainable = False # isnt this always true?
            #     pool2.trainable = False # isnt this always true?
            #     flat1.trainable = False # isnt this always true?

#            
#            self.all_actions_func = K.function([model.get_layer('inp').input], [model.get_layer('all_actions').output])
#            # self.all_actions_func = None
#        if self.model_type == 'cnn':
#            model = Sequential()
#            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=np.random.randint(2*32))
##            model.add(Dense(64, activation='tanh', input_shape=(input_shape[0],input,),kernel_initializer=init(), bias_initializer=init()))
##            model.add(Dense(num_outputs, activation='linear',kernel_initializer=init(), bias_initializer=init()))
##            # adam = optimizers.Adam(clipnorm=1.)
##            model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
#            
#
#            model.add(
#                Conv2D(filters=32, kernel_size=(1, 3), input_shape=(input_shape[0], input_shape[1], input_shape[2]),
#                       activation='relu'))
#            model.add(Dropout(0.5))
#            model.add(Conv2D(filters=32, kernel_size=(1, input_shape[1] - 2), activation='relu'))
#            model.add(Dropout(0.5))
#            model.add(Flatten())
#            model.add(Dense(64, activation='relu'))
#            model.add(Dropout(0.5))
#            model.add(Dense(64, activation='relu'))
#            model.add(Dropout(0.5))
#            model.add(Dense(self.dim_of_actions, activation='softmax'))
#            model.compile(loss='categorical_crossentropy',
#                               optimizer=optimizers.Adam(lr=1e-3),
#                               metrics=['accuracy'])
        else:
            raise NotImplemented

        return model


    def fit(self, X, y, verbose=0, batch_size=512, epochs=1000, evaluate=False, tqdm_verbose=True, additional_callbacks=[], **kw):
        if isinstance(X,(list,)):
            X = (np.reshape(X[0],-1), X[1])
        else:
            X = (X[:,0], X[:,1])
        
        self.callbacks_list = additional_callbacks + [EarlyStoppingByConvergence(epsilon=self.convergence_of_model_epsilon, diff =1e-10, verbose=verbose)]#, TQDMCallback(show_inner=False, show_outer=tqdm_verbose)]
        self.model.fit(X,y,verbose=verbose==2, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks_list, **kw)

        if evaluate:
            return self.evaluate()
        else:
            return None

    def representation(self, *args, **kw):
         
        if self.model_type == 'mlp':
            if len(args) == 1:
                return np.eye(self.dim_of_state)[np.array(args[0]).astype(int)]
            elif len(args) == 2:
                return np.hstack([np.eye(self.dim_of_state)[np.array(args[0]).astype(int)], np.eye(self.dim_of_actions)[np.array(args[1]).astype(int)] ])
            else:
                raise NotImplemented
        elif self.model_type == 'cnn':
            if len(args) == 1:
                position = np.eye(self.dim_of_state)[np.array(args[0]).astype(int)].reshape(-1,self.grid_shape[0],self.grid_shape[1])
                X, surrounding = self.create_cnn_rep_helper(position)
                return [X, surrounding]
            elif len(args) == 2:
#                print(np.array(args[0]))
                position = np.eye(self.dim_of_state)[np.array(args[0]).astype(int)].reshape(-1,self.grid_shape[0],self.grid_shape[1])
                X, surrounding = self.create_cnn_rep_helper(position)
                return [X, surrounding, np.eye(self.dim_of_actions)[np.array(args[1]).astype(int)] ]
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def create_cnn_rep_helper(self, position):
        how_many = position.shape[0]
        holes = np.repeat(self.position_of_holes[np.newaxis, :, :], how_many, axis=0)
        goals = np.repeat(self.position_of_goals[np.newaxis, :, :], how_many, axis=0)

        ix_x, ix_y, ix_z = np.where(position)
        surrounding = self.is_next_to([self.position_of_holes, self.position_of_goals], ix_y, ix_z)

        return np.sum([position*.5, holes*1, goals*(-1)], axis = 0)[:,:,:,np.newaxis], np.hstack(surrounding)

    def is_next_to(self, obstacles, x, y):
        # obstacles must be list
        assert np.all(np.array([obstacle.shape for obstacle in obstacles]) == obstacles[0].shape)
        surround = lambda x,y: [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]

        ret = []
        for idx in range(len(x)):
            neighbors = []
            for a,b in surround(x[idx], y[idx]):
                # only works if all obstacles are same shape
                neighbor = np.vstack([obstacle[a, b] for obstacle in obstacles]) if 0 <= a < obstacles[0].shape[0] and 0 <= b < obstacles[0].shape[1] else np.array([0.]*len(obstacles)).reshape(1,-1).T
                neighbors.append(neighbor)

            ret.append(np.hstack(neighbors))

        return np.stack(ret, axis=1)

    def predict(self, X, **kw):
#        return self.model.predict(self.representation(X,a))
#        print(X.shape)
        return self.model.predict(X)

    def all_actions(self, X, **kw):
        # X_a = ((x_1, a_1)
               # (x_1, a_2)
               #  ....
               # (x_1, a_m)
               # ...
               # (x_N, a_1)
               # (x_N, a_2)
               #  ...
               #  ...
               # (x_N, a_m))
#        print(X.shape)
#        X = np.array(X).reshape(-1)

#        X_a = self.cartesian_product(X, 17)
#        print(X.shape)
#        if isinstance(X,list):
#            X=np.ravel(X)
#        
#        print(X)
        if len(X.shape)==3:
            X_a = np.expand_dims(X,axis=0)
        else:
            X_a=X
#            X_a=X.reshape(self.dim_of_state[0],self.dim_of_state[1],self.dim_of_state[2],np.newaxis)
#        X_a=np.reshape(X,(self.dim_of_state[0],self.dim_of_state[1],self.dim_of_state[2],))

        # Q_x_a = ((Q_x1_a1, Q_x1_a2,... Q_x1_am)
                 # (Q_x2_a1, Q_x2_a2,... Q_x2_am)
                 # ...
                 # (Q_xN_a1, Q_xN_a2,... Q_xN_am)
        # by reshaping using C ordering
#        print("X_a shape",X_a.shape)
        Q_x_a = self.predict(X_a)#.reshape(1,self.dim_of_actions,order='C')
        return Q_x_a


class PortfolioNN_a2c(Model):
    def __init__(self, input_shape, dim_of_actions, gamma=0.95, convergence_of_model_epsilon=1e-10, model_type='mlp', position_of_holes=None, position_of_goals=None, num_frame_stack=None, frame_skip= None, pic_size = None, freeze_cnn_layers=False,**kw):
        '''
        An implementation of fitted Q iteration

        num_inputs: number of inputs
        num_outputs: number of outputs
        dim_of_actions: dimension of action space
        convergence_of_model_epsilon: small float. Defines when the model has converged.
        '''
        super().__init__()
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = gamma
        self.tau   = .125
        self.convergence_of_model_epsilon = convergence_of_model_epsilon 
        self.model_type = model_type
#        print(dim_of_actions)
        self.dim_of_actions = dim_of_actions
        self.dim_of_state = input_shape
#        print(self.dim_of_state )
        self.freeze_cnn_layers = freeze_cnn_layers

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #
        
        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32, 
        	[None, dim_of_actions[0]]) # where we will feed de/dC (from critic)
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
        	actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #        
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, 
        	self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
#        self.sess.run(tf.initialize_all_variables())
      
#        self.model = self.create_model(input_shape)
        self.all_actions_func=None
        #debug purposes
        from config_portfolio import env
        self.policy_evalutor = ExactPolicyEvaluator(None, gamma, env=env, num_frame_stack=num_frame_stack, frame_skip = frame_skip, pic_size = pic_size)

    
    def create_actor_model(self):
        state_input = Input(shape=self.dim_of_state)
        conv1 = Conv2D(32, (1,3),activation='relu')(state_input)
        dropout1 = Dropout(0.5)(conv1)
        conv2 = Conv2D(32, (1,self.dim_of_state[1]-2), activation='relu')(dropout1)
        dropout2 = Dropout(0.5)(conv2)
        flat1 = Flatten(name='flattened')(dropout2)
        dense1 = Dense(64, activation='relu')(flat1)
        dropout3 = Dropout(0.5)(dense1)
        dense2 = Dense(64, activation='relu')(dropout3)
        dropout4 = Dropout(0.5)(dense2)
        all_actions = Dense(self.dim_of_actions[0], name='all_actions', activation="relu")(dropout4)
        output = [all_actions]
        
        model = KerasModel(inputs=state_input, outputs=output)
        adam  = optimizers.Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model
    
    def create_critic_model(self):
        state_input = Input(shape=self.dim_of_state)
        conv1 = Conv2D(32, (1,3),activation='relu')(state_input)
        dropout1 = Dropout(0.5)(conv1)
        conv2 = Conv2D(32, (1,self.dim_of_state[1]-2), activation='relu')(dropout1)
        dropout2 = Dropout(0.5)(conv2)
        flat1 = Flatten(name='flattened')(dropout2)
        dense1 = Dense(64, activation='relu')(flat1)
        dropout3 = Dropout(0.5)(dense1)
        dense2 = Dense(24, activation='relu')(dropout3)
        
        
        action_input = Input(shape=self.dim_of_actions)
        action_h1    = Dense(24)(action_input)
        
        merged    = Add()([dense2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = KerasModel(inputs=[state_input,action_input], outputs=output)
        
        adam  = optimizers.Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model


    def fit(self, X, y, verbose=0, batch_size=512, epochs=1000, evaluate=False, tqdm_verbose=True, additional_callbacks=[], **kw):
        if isinstance(X,(list,)):
            X = (np.reshape(X[0],-1), X[1])
        else:
            X = (X[:,0], X[:,1])
        
        self.callbacks_list = additional_callbacks + [EarlyStoppingByConvergence(epsilon=self.convergence_of_model_epsilon, diff =1e-10, verbose=verbose)]#, TQDMCallback(show_inner=False, show_outer=tqdm_verbose)]
        self.model.fit(X,y,verbose=verbose==2, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks_list, **kw)

        if evaluate:
            return self.evaluate()
        else:
            return None

    def representation(self, *args, **kw):
         
        if self.model_type == 'mlp':
            if len(args) == 1:
                return np.eye(self.dim_of_state)[np.array(args[0]).astype(int)]
            elif len(args) == 2:
                return np.hstack([np.eye(self.dim_of_state)[np.array(args[0]).astype(int)], np.eye(self.dim_of_actions)[np.array(args[1]).astype(int)] ])
            else:
                raise NotImplemented
        elif self.model_type == 'cnn':
            if len(args) == 1:
                position = np.eye(self.dim_of_state)[np.array(args[0]).astype(int)].reshape(-1,self.grid_shape[0],self.grid_shape[1])
                X, surrounding = self.create_cnn_rep_helper(position)
                return [X, surrounding]
            elif len(args) == 2:
#                print(np.array(args[0]))
                position = np.eye(self.dim_of_state)[np.array(args[0]).astype(int)].reshape(-1,self.grid_shape[0],self.grid_shape[1])
                X, surrounding = self.create_cnn_rep_helper(position)
                return [X, surrounding, np.eye(self.dim_of_actions)[np.array(args[1]).astype(int)] ]
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def create_cnn_rep_helper(self, position):
        how_many = position.shape[0]
        holes = np.repeat(self.position_of_holes[np.newaxis, :, :], how_many, axis=0)
        goals = np.repeat(self.position_of_goals[np.newaxis, :, :], how_many, axis=0)

        ix_x, ix_y, ix_z = np.where(position)
        surrounding = self.is_next_to([self.position_of_holes, self.position_of_goals], ix_y, ix_z)

        return np.sum([position*.5, holes*1, goals*(-1)], axis = 0)[:,:,:,np.newaxis], np.hstack(surrounding)

    def is_next_to(self, obstacles, x, y):
        # obstacles must be list
        assert np.all(np.array([obstacle.shape for obstacle in obstacles]) == obstacles[0].shape)
        surround = lambda x,y: [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]

        ret = []
        for idx in range(len(x)):
            neighbors = []
            for a,b in surround(x[idx], y[idx]):
                # only works if all obstacles are same shape
                neighbor = np.vstack([obstacle[a, b] for obstacle in obstacles]) if 0 <= a < obstacles[0].shape[0] and 0 <= b < obstacles[0].shape[1] else np.array([0.]*len(obstacles)).reshape(1,-1).T
                neighbors.append(neighbor)

            ret.append(np.hstack(neighbors))

        return np.stack(ret, axis=1)
   
    
    def predict(self, X, **kw):
#        return self.model.predict(self.representation(X,a))
#        print(X.shape)

        return self.model.predict(X)

    def all_actions(self, X, **kw):
        # X_a = ((x_1, a_1)
               # (x_1, a_2)
               #  ....
               # (x_1, a_m)
               # ...
               # (x_N, a_1)
               # (x_N, a_2)
               #  ...
               #  ...
               # (x_N, a_m))
#        print(X.shape)
#        X = np.array(X).reshape(-1)

#        X_a = self.cartesian_product(X, 17)
#        print(X.shape)
#        if isinstance(X,list):
#            X=np.ravel(X)
#        
#        print(X)
        if len(X.shape)==3:
            X_a = np.expand_dims(X,axis=0)
        else:
            X_a=X
#            X_a=X.reshape(self.dim_of_state[0],self.dim_of_state[1],self.dim_of_state[2],np.newaxis)
#        X_a=np.reshape(X,(self.dim_of_state[0],self.dim_of_state[1],self.dim_of_state[2],))

        # Q_x_a = ((Q_x1_a1, Q_x1_a2,... Q_x1_am)
                 # (Q_x2_a1, Q_x2_a2,... Q_x2_am)
                 # ...
                 # (Q_xN_a1, Q_xN_a2,... Q_xN_am)
        # by reshaping using C ordering
#        print("X_a shape",X_a.shape)
        Q_x_a = self.predict(X_a)#.reshape(1,self.dim_of_actions,order='C')
        return Q_x_a
    
    
class PortfolioNN_model(nn.Module):
    def __init__(self, input_shape, dim_of_actions,lr=1e-6,n_epochs = 1000 ):
        super().__init__()
        self.dim_of_state = input_shape
        self.dim_of_actions = dim_of_actions
        self.lr = lr
        self.n_epochs = n_epochs 
        self.conv1=nn.Conv2d(self.dim_of_state[0] ,32, kernel_size=(1, 3))
        self.hid1 = nn.Dropout(0.2)
        self.hid2 = nn.Conv2d(32,32,  kernel_size=(5, 2))
        self.hid3 = nn.Dropout(0.5)
        self.hid4 = nn.Linear(32,64)
        self.hid5 = nn.Dropout(0.5)
        self.hid6 = nn.Linear(64,12)
        self.conv2 = nn.Conv2d(self.dim_of_state[0] ,12, kernel_size=(1, 1))
        self.hid8 = nn.Linear(12,12)
        self.hid9 = nn.Linear(24,1)
        self.hid10=nn.Tanh()
        self.model_params=self.parameters()
    
    
    def forward(self,X,a):
        state = X
        state = self.hid1(F.relu(self.conv1(state)))
        state = self.hid3(F.relu(self.hid2(state)))
#        print(state.shape)
        state = self.hid5(F.relu(self.hid4(state.view(-1,32))))
        state =self.hid6(state)
        
        action = a.unsqueeze(0).view(-1,self.dim_of_state[0] ,1,1)
        action = F.relu(self.conv2(action)).view(1,-1)
        action = self.hid8(action)
#        print(state,action)
        combined = torch.cat((state.view(action.size(0), -1),
                          action.view(action.size(0), -1)), dim=1)
        
        Q = self.hid10(self.hid9(combined))[0][0]
#        print(Q)        
        return Q


    def min_over_a_cont(self,X):
#        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        Q=[]
        actions=[]

        for i in range(len(X)):
            X_i = torch.FloatTensor(X[i]).view(-1,self.dim_of_state[0] ,5,4) 
            a = torch.rand(self.dim_of_actions, requires_grad=True, dtype=torch.float, device=device)

            optimizer = optim.SGD([a], lr=self.lr)
            for epoch in range(self.n_epochs):
                Q_x = self.forward(X_i,a)
                Q_x.backward()
                optimizer.step()
                optimizer.zero_grad()
                a=Variable(torch.clamp(a, 0, 1),requires_grad=True)
                assert Q_x.detach().numpy()!=np.nan, X_i
            Q.append(Q_x.detach().numpy())  
            actions.append(a.detach().numpy())
        assert sum([1 for j in self.parameters() if (j.detach().numpy()!=j.detach().numpy()).any()])==0,"nan"
        
        return Q,actions
    
    
    def fit_generator(self, generator,model_params,lr=1e-6, steps_per_epoch=1000, epochs=512, evaluate=False, tqdm_verbose=True, additional_callbacks=[], **kw):
        
        optimizer = optim.SGD(model_params, lr=self.lr)
        criterion = nn.MSELoss()
#        a=torch.zeros(0, 0)
        for epoch in range(epochs):
            
            for _,(X,a,cost) in enumerate(generator):
                pred=[]
                for i in range(len(X)):
                    X_i = torch.FloatTensor(X[i]).view(-1,self.dim_of_state[0] ,5,4)
                    a_i = torch.FloatTensor(a[i])

                    prediction = self.forward(X_i,a_i)
                    pred.append(prediction)
                cost=torch.FloatTensor(cost)
                p = torch.stack(pred)
                optimizer.zero_grad()
                loss = criterion(p,cost)
                loss.backward()
                optimizer.step() 
                
        
            