# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:10:17 2019

@author: abhil
"""

import deepdish as dd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import time
from portfolio import PortfolioEnv
import numpy as np
from data import read_stock_history, index_to_date, date_to_index, normalize



def discounted_sum(costs, discount):
    '''
    Calculate discounted sum of costs
    '''
    y = signal.lfilter([1], [1, -discount], x=costs[::-1])
    return y[::-1][0]

# Data setup 
#dones = dd.io.load(os.path.join('finance_done.h5'))[:500]
#costs = dd.io.load(os.path.join('finance_c.h5'))[:500]
#g = dd.io.load(os.path.join('finance_g.h5'))[:500]
#dones = np.hstack([0,1+np.where(dones)[0]])
#episodes = []
#for low_, high_ in zip(dones[:-1], dones[1:]):
#    new_episode ={
#        'c': costs[low_:high_, 0].reshape(-1),
#        'g': g[low_:high_, 0].reshape(-1),
#    }
#    
#    episodes.append(new_episode)
#discounted_costs = np.array([[discounted_sum(x['c'],.95),discounted_sum(x['g'],.95)]  for x in episodes])

data = dd.io.load('portfolio_policy_improvement.h5')

max_iterations = 9
iterations = range(len(data['g_eval'][0][:max_iterations]))
constraint_names = ['range']
constraint_upper_bound = [1]



####Plot the derandomized policy
def derandomize(data, constraints, min_iteration):
	
	fqe_c = np.array(data['c_eval'][0])[:,-1]
	fqe_g_0 = np.array(data['g_eval'][0])[:,-1]
	out = []
	for iteration in range(min_iteration, len(fqe_c)):

		df_tmp = pd.DataFrame(np.hstack([np.arange(min_iteration,iteration+1).reshape(1,-1).T, fqe_c[min_iteration:(iteration+1)].reshape(1,-1).T, fqe_g_0[min_iteration:(iteration+1)].reshape(1,-1).T ]), columns=['iteration', 'fqe_c', 'fqe_g_0'])
		df_tmp = df_tmp[(df_tmp['fqe_g_0'] < constraints[0])]
		try:
			argmin = np.argmin(np.array(df_tmp['fqe_c']))
			it = int(df_tmp.iloc[argmin]['iteration'])
		except:
			argmin = 0
			it = 0
		out.append(np.hstack([iteration, np.hstack([data['c_exacts'][it],  np.array(data['g_exacts'])[it,:-1]]) ]))

	return pd.DataFrame(out, columns=['iterations', 'c_derandomized', 'g_0_derandomized'])

df_derandom = derandomize(data, np.array(constraint_upper_bound)*.8, 0)
plt.plot(df_derandom['iterations'], df_derandom['c_derandomized'],color="blue", linestyle='-',markersize=7)

####Plot CRP
max_time_spent_in_episode = 100
history, abbreviation = read_stock_history(filepath=r'datasets/stocks_history_target_2.h5')
history = history[:, :, :4]
nb_classes = len(history) + 1
print(history.shape)
num_training_time = history.shape[1]
target_stocks = ['CSCO','QCOM','PCLN','CELG','AMGN','FOX','FISV','EXPE','FAST','ESRX'] #['CSCO','QCOM','PCLN','CELG','AMGN']
target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
for i, stock in enumerate(target_stocks):
    target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
    
#env = PortfolioEnv(history,abbreviation)
env = PortfolioEnv(target_history,target_stocks)
data_crp = []
for i in range(20):
  tic = time.time()
  x = env.reset()
  done = False
  time_steps = 0
  episode_cost = 0
  while not done:
      punishment=0
      time_steps += 1
      cur_state=x
      if len(cur_state.shape)==3:
          cur_state = np.expand_dims(cur_state,axis=0)
      action = np.array([1/len(target_stocks)]*len(env.action_space.sample()))
      cost = []
      x_prime, rewards, done, _ = env.step(action)
      costs=rewards[0]*-1
      if costs>0:
          punishment=1
      cost.append((costs))
      if done:
          break
      episode_cost += cost[0] + punishment
      c = (cost[0] + punishment).tolist()

      g = rewards[1:][0]
      data_crp.append( [action,
                        x_prime, 
                        np.hstack([c,g]).reshape(-1).tolist(),
                        done]
                        ) 

  if (i % 10) == 0:
    print ('Epoch: %s' % i )                        

c=[]
g=[]
for i in data_crp:
  c.append(i[2][0])
  g.append(i[2][1:])
  
rew_crp=np.mean(c)
c_crp=-1*np.mean(c)
g_crp=np.nanmean(g)


####Plot A2c without constraints(???)
cost = dd.io.load(r'datasets/finance_data_rewards.h5')
rew_a2c = -1*np.sum(cost)



####Plot the new algorithm
max_iterations=10
iterations = range(len(data['g_eval'][0][:max_iterations]))
portfolio_results=pd.read_csv('portfolio_results_temp.csv')
c_avg = portfolio_results['c_exact_avg'][:max_iterations]
#c_avg=np.cumsum(np.array(data['c_eval_actuals'])[:max_iterations,-10:,:][:,-1,0])
c_values = np.array(data['c_eval']).tolist()[0][:max_iterations] # shape = (iteration #, k, performance)
last_c = [np.array(i).mean() for i in c_values]
#last = np.cumsum(c_values[:,-1,0])/np.arange(1,1+len(c_values[:,-1,0]))#*100
#evaluation = np.array(pd.DataFrame(c_values[:,-1,0]).expanding().mean()).reshape(-1)
#std = np.array(pd.DataFrame(c_values[:,-1,0]).expanding().std()).reshape(-1)
fig, ax = plt.subplots()
ax.plot(iterations, [-1*i for i in c_avg],color='purple', label ='CPOT',linestyle='-',markersize=7)
ax.plot(iterations, [rew_crp for i in iterations],label='CRP', color='yellow')
ax.plot(iterations, [rew_a2c for i in iterations],label='A2C', color='black')
ax.legend()
ax.set_xlabel('Iterations')
ax.set_ylabel('Reward')

print("optimization graph")


###Constraint plot - VaR
g_values = np.array(data['g_eval']).tolist()[0][:max_iterations] # shape = (iteration #, k, performance)
#last = [np.cumsum(i).mean()/len(i) for i in g_values]
last_g = [np.array(i).mean() for i in g_values]
#evaluation = np.array(pd.DataFrame(c_values[:,-1,0]).expanding().mean()).reshape(-1)
#std = np.array(pd.DataFrame(c_values[:,-1,0]).expanding().std()).reshape(-1)
fig, ax = plt.subplots()
ax.plot(iterations, last_g,label='CPOT',color='purple', linestyle='-',markersize=7)
ax.plot(iterations, [0.05 for i in iterations],label='constraint', color='red', linestyle='-',markersize=7)
ax.plot(iterations, [g_crp for i in iterations],label='CRP', color='yellow')
ax.legend()
ax.set_xlabel('Iterations')
ax.set_ylabel('VaR (Value at Risk)')
print("constraint graph")
