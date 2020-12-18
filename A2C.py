import keras
import numpy as np
from replay_buffer_portfolio import Buffer
import time
from keras.callbacks import ModelCheckpoint
import os
from collections import deque
import keras.backend as K
import random
import deepdish as dd
import tensorflow as tf


class A2C(object):
    def __init__(self, env, 
                       gamma, 
                       model_type='mlp', 
                       action_space_map = None,
                       num_iterations = 5000, 
                       sample_every_N_transitions = 10,
                       batchsize = 1000,
                       copy_over_target_every_M_training_iterations = 100,
                       max_time_spent_in_episode = 100,
                       buffer_size = 10000,
                       num_frame_stack=1,
                       min_buffer_size_to_train=1000,
                       frame_skip = 1,
                       pic_size = (96, 96),
                       models_path = None,
                       ):

        self.models_path = models_path
        self.env = env
        self.num_iterations = num_iterations
        self.gamma = gamma 
        self.frame_skip = frame_skip
        _ = self.env.reset()
        if self.env.env_type in ['car']: 
            self.env.render()
            _, r, _, _ = self.env.step(action_space_map[0])
            self.buffer = deque(maxlen=2000)
        else:
            self.buffer = deque(maxlen=2000)
        self.prev_states = deque(maxlen=2000)
        self.action = deque(maxlen=2000)
        self.reward = deque(maxlen=2000)
        self.next_states = deque(maxlen=2000)
        self.is_done = deque(maxlen=2000)
#            self.buffer = Buffer(buffer_size=buffer_size, num_frame_stack=num_frame_stack, min_buffer_size_to_train=min_buffer_size_to_train, pic_size = (1,), n_costs = (1,))        
        self.sample_every_N_transitions = sample_every_N_transitions
        self.batchsize = batchsize
        self.copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations
        self.max_time_spent_in_episode = max_time_spent_in_episode
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.initialize_all_variables())


    def min_over_a(self, *args, **kw):
        return self.Q.min_over_a(*args, **kw)


    def all_actions(self, *args, **kw):
        return self.Q.all_actions(*args, **kw)

    # def representation(self, *args, **kw):
    #     return self.Q.representation(*args, **kw)

    def learn(self):
        
        more_callbacks = [ModelCheckpointExtended(self.models_path)]
        self.time_steps = 0
        training_iteration = -1
        perf = Performance()
        main_tic = time.time()
        training_complete = False
        for i in range(self.num_iterations):
            if training_complete: continue
            tic = time.time()
            x = self.env.reset()
            if self.env.env_type in ['car']: self.env.render()
#            self.buffer.start_new_episode(x)
            done = False
            time_spent_in_episode = 0
            episode_cost = 0
            cur_state=x
            while not done and time_spent_in_episode<50:
                #if self.env.env_type in ['car']: self.env.render()
                
                time_spent_in_episode += 1
                self.time_steps += 1
                if time_spent_in_episode%100==0:
                    print(time_spent_in_episode)
                use_random = np.random.rand(1) < self.epsilon(epoch=i, total_steps=self.time_steps)
                if use_random:
                    action=self.env.action_space.sample()
#                    print("random action",action)
                else:
                    cur_state=np.expand_dims(cur_state,axis=0)
#                    print(cur_state.shape)
                    action= self.Q.actor_model.predict(cur_state)[0]
#                    print("epsilon action",action)
                    if np.isnan(action).any():
                        action=self.env.action_space.sample()

                cost = []
                for _ in range(self.frame_skip):
                    if done: continue
                    new_state, rewards, done, _ = self.env.step(action)
                    costs=rewards[0]*(-1)
                    cost.append(costs)

                if self.frame_skip>1:
                    cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
                episode_cost += costs
                self.remember(cur_state, action, cost, new_state, done)
                self.train()
                
                cur_state = new_state
            if self.env.env_type == 'car': 
                perf.append(float(self.env.tile_visited_count)/len(self.env.track))
            else:
                perf.append(episode_cost/time_spent_in_episode)

            if (i % 1) == 0:
                print ('Episode %s' % i)
                print(episode_cost)
                episode_time = time.time()-tic
                print ('Total Time: %s. Episode time: %s. Time/Frame: %s' % (np.round(time.time() - main_tic,2), np.round(episode_time, 2), np.round(episode_time/time_spent_in_episode, 2)))
                print ('Episode frames: %s. Total frames: %s. Total train steps: %s' % (time_spent_in_episode, self.time_steps, training_iteration))
                if self.env.env_type in ['car']:
                    print ('Performance: %s/%s. Score out of 1: %s. Average Score: %s' %  (self.env.tile_visited_count, len(self.env.track), perf.last(), perf.get_avg_performance()))
                else:
                    print ('Score out of 1: %s. Average Score: %s' %  (perf.last(), perf.get_avg_performance()))
                print ('*'*20)
            if perf.reached_goal():
                print("goal reached")
                #return more_callbacks[0].all_filepaths[-1]
                training_complete = True#return self.Q #more_callbacks[0].all_filepaths[-1]
        self.save(os.path.join(os.getcwd(),'datasets','%s_data_{0}.h5' % self.env.env_type))
        
    def remember(self, cur_state, action, reward, new_state, done):
        self.prev_states.append([cur_state])
        self.action.append([action])
        self.reward.append([reward])
        self.next_states.append([new_state])
        self.is_done.append([done])
        self.buffer.append([cur_state, action, reward, new_state, done])

    def train(self):
        batch_size = 32
        if len(self.buffer) < batch_size:
            return
        
        rewards = []
        samples = random.sample(self.buffer, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            if len(cur_state.shape)==3:
                cur_state = np.expand_dims(cur_state,axis=0)
#            print(cur_state.shape)
            predicted_action = self.Q.actor_model.predict(cur_state)
            grads = self.sess.run(self.Q.critic_grads, feed_dict={
                self.Q.critic_state_input:  cur_state,
                self.Q.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.Q.optimize, feed_dict={
                self.Q.actor_state_input: cur_state,
                self.Q.actor_critic_grad: grads
            })
            
    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                new_state=np.expand_dims(new_state,axis=0)
                target_action = self.Q.target_actor_model.predict(new_state)
                future_reward = self.Q.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
#            print(reward)
            if len(cur_state.shape)==3:
                cur_state=np.expand_dims(cur_state,axis=0)
            action=action.reshape((1,len(self.env.action_space.sample())))
#            print(action,cur_state.shape,action.shape,reward.shape)
            
            self.Q.critic_model.fit(x=[cur_state, action], y=reward, verbose=0)
    
    
    def _update_actor_target(self):
        actor_model_weights  = self.Q.actor_model.get_weights()
        actor_target_weights = self.Q.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.Q.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.Q.critic_model.get_weights()
        critic_target_weights = self.Q.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.Q.critic_target_model.set_weights(critic_target_weights)
        
    
    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()
        
        
    def save(self, path):
        #data = {'frames':self.frames, 'prev_states':self.prev_states, 'next_states':self.next_states, 'rewards':self.rewards, 'is_done':self.is_done, 'actions':self.actions}
        #for data, key in zip([self.frames, self.prev_states, self.next_states, self.rewards, self.is_done, self.actions],['frames', 'prev_astates', 'next_states', 'costs', 'is_done', 'actions'])
        #       dd.io.save(path % key, data)
#        count = min(self.capacity, self.counter)
#        dd.io.save(path.format('frames'), self.frames[:count])
        dd.io.save(path.format('prev_states'), self.prev_states)
        dd.io.save(path.format('next_states'), self.next_states)
        dd.io.save(path.format('rewards'), self.reward)
        dd.io.save(path.format('is_done'), self.is_done)
        dd.io.save(path.format('actions'), self.action)
    
    
    def __call__(self,*args):
        return self.Q.__call__(*args)

    def __deepcopy__(self, memo):
        return self

class Performance(object):
    def __init__(self):
        self.goal = .85
        self.avg_over = 20
        self.costs = []

    def reached_goal(self):
        if self.get_avg_performance() >= self.goal:
            return True
        else:
            return False

    def append(self, cost):
        self.costs.append(cost)

    def last(self):
        return np.round(self.costs[-1], 3)

    def get_avg_performance(self):
        num_iters = min(self.avg_over, len(self.costs))
        return np.round(sum(self.costs[-num_iters:])/ float(num_iters), 3)


class ModelCheckpointExtended(ModelCheckpoint):
    def __init__(self, filepath, max_to_keep=5, monitor='loss', *args, **kw):
        super(ModelCheckpointExtended, self).__init__(filepath, *args, **kw)
        self.max_to_keep = max_to_keep
        self.all_filepaths = []

    def on_epoch_end(self, epoch, logs=None):
        
        super(ModelCheckpointExtended, self).on_epoch_end(epoch, logs)
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        self.all_filepaths.append(filepath)
        if len(self.all_filepaths) > self.max_to_keep:
            try:
                os.remove(self.all_filepaths.pop(0))
            except:
                pass


# class Buffer(object):
#     def __init__(self, buffer_size=10000):
#         self.data = []
#         self.size = buffer_size
#         self.idx = -1

#     def append(self, datum):
#         self.idx = (self.idx + 1) % self.size
        
#         if len(self.data) > self.idx:
#             self.data[self.idx] = datum
#         else:
#             self.data.append(datum)

#     def sample(self, N):
#         N = min(N, len(self.data))
#         rows = np.random.choice(len(self.data), size=N, replace=False)
#         return np.array(self.data)[rows]



