

import numpy as np
np.random.seed(3141592)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from optimization_problem import Program
from fittedq import *
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import *
from exact_policy_evaluation import ExactPolicyEvaluator
from keras.models import load_model
from keras import backend as K
from enva2c import *
import deepdish as dd
import time
import os
np.set_printoptions(suppress=True)
from config_portfolio import *
import subprocess
import torch
#import config_car as car


env_name='portfolio'
def main(env_name, headless):
    
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #### Get a decent policy. 
    #### Called pi_old because this will be the policy we use to gather data
    policy_old = None
    old_policy_path = os.path.join(model_dir, old_policy_name)
    if env_name == 'portfolio':
        policy_old = PortfolioA2C(env, 
                                gamma,
                                action_space_dim=action_space_dim, 
                                model_type=model_type,
                                max_time_spent_in_episode=max_time_spent_in_episode,
                                num_iterations = num_iterations,
                                sample_every_N_transitions = sample_every_N_transitions,
                                batchsize = batchsize,
                                copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations,
                                buffer_size = buffer_size,
                                min_epsilon = min_epsilon, 
                                initial_epsilon = initial_epsilon,
                                epsilon_decay_steps = epsilon_decay_steps,
                                num_frame_stack=num_frame_stack,
                                min_buffer_size_to_train=min_buffer_size_to_train,
                                models_path = os.path.join(model_dir,'test_weights.{epoch:02d}-{loss:.2f}.hdf5'),
                                  )



    else:
        raise
    
    if not os.path.isfile(old_policy_path):
        print ('Learning a policy using DQN')
        policy_old.learn()
        policy_old.Q.actor_model.save(os.path.join(model_dir,"pi_old_portfolio_actor.hdf5"))
        policy_old.Q.critic_model.save(os.path.join(model_dir,"pi_old_portfolio_critic.hdf5"))
    else:
        print ('Loading a policy')
#        policy_old.Q.model = load_model(old_policy_path)
        policy_old.Q.actor_model = load_model(os.path.join(model_dir,"pi_old_portfolio_actor.hdf5"))
        policy_old.Q.critic_model =load_model(os.path.join(model_dir,"pi_old_portfolio_critic.hdf5"))

    #### Problem setup
    if env_name == 'portfolio':
        state_space_dim=env.observation_space.shape
        best_response_algorithm = PortfolioFittedQIteration(state_space_dim, 
                                                      action_space_dim, 
                                                      max_Q_fitting_epochs, 
                                                      gamma, 
                                                      model_type=model_type,
                                                      num_frame_stack=num_frame_stack,
                                                      initialization=policy_old)
#                                                      freeze_cnn_layers=freeze_cnn_layers)# for _ in range(2)]
        fitted_off_policy_evaluation_algorithm = PortfolioFittedQEvaluation(state_space_dim, 
                                                                      action_space_dim, 
                                                                      max_eval_fitting_epochs, 
                                                                      gamma, 
                                                                      model_type=model_type,
                                                                      num_frame_stack=num_frame_stack)# for _ in range(2*len(constraints_cared_about) + 2)] 
        exact_policy_algorithm = ExactPolicyEvaluator(action_space_map=None, gamma=gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size) #, constraint_thresholds=constraint_thresholds, constraints_cared_about=constraints_cared_about)
    
    else:
        raise

    online_convex_algorithm = ExponentiatedGradient(lambda_bound, len(constraints), eta, starting_lambda=starting_lambda)
    
    problem = Program(constraints, 
                      action_space_dim, 
                      best_response_algorithm, 
                      online_convex_algorithm, 
                      fitted_off_policy_evaluation_algorithm, 
                      exact_policy_algorithm, 
                      lambda_bound, 
                      epsilon, 
                      env, 
                      max_number_of_main_algo_iterations,
                      num_frame_stack,
                      pic_size,)    

    lambdas = []
    policies = []

    #### Collect Data
    try:
        print ('Loading Prebuilt Data')
        batch_idxs = np.random.choice(len(dd.io.load(r".\datasets\finance_a.h5")), sample_size,replace=False)
        tic = time.time()
        if env_name == 'portfolio': 
            tic = time.time()
            action_data = dd.io.load(r".\datasets\finance_a.h5")
    #            frame_data = dd.io.load()
            done_data = dd.io.load(r".\datasets\finance_done.h5")
            next_state_data = dd.io.load(r".\datasets\finance_next_states.h5")
            current_state_data = dd.io.load(r".\datasets\finance_prev_states.h5")
            c = dd.io.load(r".\datasets\finance_c.h5")
            g = dd.io.load(r".\datasets\finance_g.h5")
    
            problem.dataset.data = {'prev_states': [current_state_data[i] for i in batch_idxs],
                            'next_states': [next_state_data[i] for i in batch_idxs],
                            'a': [action_data[i] for i in batch_idxs],
                            'c':[c[i] for i in batch_idxs],
                            'g':[g[i] for i in batch_idxs],
                            'done': [done_data[i] for i in batch_idxs]
                            }
            print ('Preprocessed g. Time elapsed: %s' % (time.time() - tic))
            
        else:
          raise 
    except:
        print ('Failed to load')
        print ('Recreating dataset')
        dataset_size = 0 
        main_tic = time.time()
        for i in range(max_epochs):
            tic = time.time()
            x = env.reset()
            problem.collect(x, start=True)
            dataset_size += 1
            done = False
            time_steps = 0
            episode_cost = 0
            while not done:
                punishment=0
                time_steps += 1
                cur_state=x
                if len(cur_state.shape)==3:
                    cur_state = np.expand_dims(cur_state,axis=0)
                action = policy_old.Q.actor_model.predict(cur_state)[0]
                cost = []
                for _ in range(frame_skip):
                    x_prime, rewards, done, info = env.step(action)
                    costs=rewards[0]*-1
                    if costs>0:
                        punishment=1
                    cost.append((costs))
                    if done:
                        break
                if frame_skip>1:
                    cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
                    
                episode_cost += cost[0] + punishment
                c = (cost[0] + punishment).tolist()

                g = rewards[1:][0]
                if len(g) < len(constraints): g=np.hstack([g,0])
                problem.collect( info['action'],
                                 x_prime, 
                                 np.hstack([c,g]).reshape(-1).tolist(),
                                 done
                                 ) 
                dataset_size += 1
                x = x_prime
            if (i % 1) == 0:
                print ('Epoch: %s' % i )
                print ('Dataset size: %s Time Elapsed: %s. Total time: %s' % (dataset_size, time.time() - tic, time.time()-main_tic))
                if env_name in ['car']: 
                    print ('Performance: %s/%s = %s' %  (env.tile_visited_count, len(env.track), env.tile_visited_count/float(len(env.track))))
                else:
                    print('performance: %s/%s =%s'% (episode_cost,time_steps,float(episode_cost)/float(time_steps)))
                print ('*'*20 )
        problem.finish_collection(env_name)



     ### Solve Batch Constrained Problem
    
    iteration = 0
    while not problem.is_over(policies, lambdas, infinite_loop=infinite_loop, calculate_gap=calculate_gap, results_name=results_name, policy_improvement_name=policy_improvement_name):
        iteration += 1
        K.clear_session()
        for i in range(1):
           
            # policy_printer.pprint(policies)
            print ('*'*20)
            print ('Iteration %s, %s' % (iteration, i))
            print()
            if len(lambdas) == 0:
                # first iteration
                lambdas.append(online_convex_algorithm.get())
                print ('lambda_{0}_{2} = {1}'.format(iteration, lambdas[-1], i))
            else:
                # all other iterations
                lambda_t = problem.online_algo()
                lambdas.append(lambda_t)
                print ('lambda_{0}_{3} = online-algo(pi_{1}_{3}) = {2}'.format(iteration, iteration-1, lambdas[-1], i))

            lambda_t = lambdas[-1]
            #FQI here
            pi_t, values = problem.best_response(lambda_t, desc='FQI pi_{0}_{1}'.format(iteration, i), exact=exact_policy_algorithm)
            torch.save(pi_t.state_dict(), os.path.join(model_dir,"pi_final.hdf5"))
#            pi_t.model_params.save(os.path.join(model_dir,"pi_final.hdf5"))
            #FQE
            problem.update(pi_t, values, iteration) #Evaluate C(pi_t), G(pi_t) and save

if __name__ == "__main__":
    main('portfolio',0)