
#### Setup Gym 
from portfolio import PortfolioEnv
import numpy as np
from data import read_stock_history, index_to_date, date_to_index, normalize

max_time_spent_in_episode = 100
history, abbreviation = read_stock_history(filepath=r'datasets/stocks_history_target_2.h5')
history = history[:, :, :4]
nb_classes = len(history) + 1
print(history.shape)
num_training_time = history.shape[1]
target_stocks = ['CSCO','QCOM','PCLN','CELG','AMGN','FOX','FISV','EXPE','FAST','ESRX']
target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
for i, stock in enumerate(target_stocks):
    target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
    
#env = PortfolioEnv(history,abbreviation)
env = PortfolioEnv(target_history,target_stocks)

#### Hyperparam
gamma = 0.9
max_epochs = 50 # max number of epochs over which to collect data
#max_epochs = 50
max_Q_fitting_epochs = 50 #max number of epochs over which to converge to Q^\ast.   Fitted Q Iter
max_eval_fitting_epochs = 50 #max number of epochs over which to converge to Q^\pi. Off Policy Eval
lambda_bound = 30. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
deviation_from_old_policy_eps = .95 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
action_space_dim = env.action_space.shape # action space dimension
#state_space_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2] # state space dimension

state_space_dim = env.observation_space

eta =5. #50. # param for exponentiated gradient algorithm
initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states

max_number_of_main_algo_iterations = 15 # After how many iterations to cut off the main algorithm
model_type = 'cnn'
old_policy_name = 'pi_old_portfolio_actor.hdf5'
constraints = [0.05, 0] # the constraint limits
#constraint_thresholds = [0.05]
#constraints_cared_about = [0]
starting_lambda = 'uniform'

## Old policy Param(DQN/A2C)
num_iterations = 100
#num_iterations = 5
sample_every_N_transitions = 10
batchsize = 5000
copy_over_target_every_M_training_iterations = 100
buffer_size = 10000
#buffer_size = 100
num_frame_stack=1
min_buffer_size_to_train=0
frame_skip = 1
pic_size = (17, 5, 4)
min_epsilon = .02
initial_epsilon = .3
epsilon_decay_steps = 1000 #num_iterations
#min_buffer_size_to_train = 2000
min_buffer_size_to_train = 20

# Other
stochastic_env = False

sample_size=2000
calculate_gap = True # Run Main algo. If False, it skips calc of primal-dual gap
infinite_loop = True # Stop script if reached primal-dual gap threshold
policy_improvement_name = 'portfolio_policy_improvement.h5'
results_name = 'portfolio_results_temp.csv'