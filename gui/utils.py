import os, sys, yaml
import numpy as np
import gym

CWD = os.getcwd()
sys.path.insert(0, CWD + "/gym-backgammon")
sys.path.insert(0, CWD + "/agents")
sys.path.insert(0, CWD + "/gui")

config = yaml.safe_load(open(f"{CWD}/config.yaml"))

from models import TDGammon
from agents import TDAgent, HumanAgent
from gym_backgammon.envs.backgammon import WHITE, BLACK
from gui.gui import GUI

def args_gui(args):

    host = args.host
    port = args.port
    difficulty = args.difficulty
    first_player = args.first_player

    if difficulty not in config['difficulty']:
        raise Exception(f"Accepted difficulty levels are{config['difficulty']}.")
    
    if first_player == 'random':
        first_player = np.random.choice(config['player'])

    elif first_player not in config['player']:
        raise Exception(f"Accepted first player must be random or {config['player']}.")
        
    env = gym.make('gym_backgammon:backgammon-v0', disable_env_checker=True) # initialize environment

    model = TDGammon(hidden_units=config['hidden_units'][args.difficulty], lr=0.1, lambda_=0.7) # initialize model
    model.load(snapshot_path=CWD+config['model_folder']+config['model_file'][args.difficulty]+'.zip') # load the weights

    # if first_player == 'AI':
    #     agents = {BLACK: HumanAgent(BLACK), WHITE: TDAgent(WHITE, model=model)}
    # else:
    #     agents = {BLACK: TDAgent(BLACK, model=model), WHITE: HumanAgent(WHITE)}

    agents = {BLACK: TDAgent(BLACK, model=model), WHITE: HumanAgent(WHITE)}
    
    gui = GUI(env=env, host=host, port=port, agents=agents)
    gui.run()
