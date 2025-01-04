from importlib import reload

import os
CWD = os.getcwd()

import gym
import eval 
from gym_backgammon.envs.backgammon import WHITE, BLACK
from models import TDGammon
from agents import TDAgent
import matplotlib.pyplot as plt

env = gym.make('gym_backgammon:backgammon-v0', disable_env_checker=True) # initialize environment

def compare_consecutive_models(input_units, hidden_units, run_name, snapshots, iters, i, n_episodes):

    model1 = TDGammon(input_units=input_units, hidden_units=hidden_units, lr=0.1, lambda_=0.7, env=env) # initialize model
    model1.load(snapshot_path=CWD+'/trained_models/'+run_name+'/'+snapshots[i]) # load the weights
    black_agent = TDAgent(BLACK, name=f"{run_name}-{iters[i]}", 
                          model = model1) # initialize the agent
    model2 = TDGammon(input_units=input_units, hidden_units=hidden_units, lr=0.1, lambda_=0.7, env=env) # initialize model
    model2.load(snapshot_path=CWD+'/trained_models/'+run_name+'/'+snapshots[i+1]) # load the weights
    white_agent = TDAgent(WHITE, name=f"{run_name}-{iters[i+1]}", 
                        model = model2) # initialize the agent
    agents = {BLACK: black_agent, WHITE: white_agent}
    return eval.evaluate_agents(agents=agents, env=env, n_episodes=n_episodes)[WHITE] / n_episodes 


def plot_training_performance(run_name, n_episodes=100):

    param_file = CWD+'/trained_models/' + run_name + '/parameters.txt'
    with open(param_file) as f: # get the model architecture from parameters file
        for line in f.readlines():
            if line[:11] == 'input_units':
                input_units = int(line[12:])
            elif line[:12] == 'hidden_units':
                hidden_units = int(line[13:])

    snapshots = [file for file in os.listdir(CWD+'/trained_models/'+run_name) if file not in ['log.txt', 'parameters.txt', '.DS_Store']] #filtering
    snapshots = sorted(snapshots, key=lambda x: '0'+x[-9] if x[-10] == '_' else x[-10:-8])[:-1] #sorting

    iters = ['0'+snapshots[i][-9] if snapshots[i][-10] == '_' else snapshots[i][-10:-8] for i in range(len(snapshots))]
    score = []
    
    for i in range(len(snapshots)-1):
        
        score.append(compare_consecutive_models(input_units=input_units, hidden_units=hidden_units, run_name=run_name, 
                                   snapshots=snapshots, iters=iters, i=i, n_episodes=n_episodes))

    plt.plot(range(2, len(snapshots)+1), score)
    plt.xlabel("Snapshot (10k) iters")
    plt.xticks(range(2, 15))
    plt.ylabel("Win rate iter i vs iter i-1")
        
        