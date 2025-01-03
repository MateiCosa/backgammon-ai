import os, sys
import gym

from models import TDGammon
from agents import TDAgent, RandomAgent
from eval import evaluate_agents

CWD = os.getcwd()
sys.path.append(CWD + '/gym-backgammon/')

from gym_backgammon.envs.backgammon import WHITE, BLACK

def write_model(path, **kwargs):
    with open(f"{path}/parameters.txt", 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

def check_path(path):
    if os.path.exists(path):
        return True
    else:
       raise Exception("Path does not exist.")

def args_train(args):

    # training run
    name = args.name
    save_step = args.save_step
    save_path = args.save_path
    n_episodes = args.episodes
    seed = args.seed
    n_ply = args.n_ply

    # architecture
    model_type = args.type
    input_units = args.input_units
    hidden_units = args.hidden_units

    # hyperprarameters
    lambda_ = args.lambda_
    lr = args.lr

    if model_type == 'nn':
        model = TDGammon(input_units=input_units, hidden_units=hidden_units, lr=lr, lambda_=lambda_, seed=seed)
        env = gym.make('gym_backgammon:backgammon-v0', disable_env_checker=True) 
    else:
        raise NotImplementedError

    if not args.save_path:
        raise Exception("Empty save path is not a valid input.")
    if check_path(args.save_path):
        save_path = args.save_path

        write_model(
            save_path, save_path=args.save_path, command_line_args=args, type=model_type, input_units=input_units, hidden_units=hidden_units, alpha=model.lr, lambda_=model.lambda_,
            n_episodes=n_episodes, save_step=save_step, run_name=name, env=env.spec.id, seed=seed, modules=[module for module in model.modules()]
        )

    model.train(env=env, n_episodes=n_episodes, save_path=save_path, save_step=save_step, n_ply=n_ply, run_name=name)

def args_eval(args):

    # parameters
    model1_type = args.model1_type
    model1_path = args.model1_path
    model1_n_ply = args.model1_n_ply
    model1_hidden_units = args.model1_hidden_units
    model2_type = args.model2_type
    model2_path = args.model2_path
    model2_n_ply = args.model2_n_ply
    model2_hidden_units = args.model2_hidden_units
    n_episodes = args.n_episodes

    if model1_type not in ['random', 'nn']:
        raise Exception("Model type must be random or nn.")
    if model2_type not in ['random', 'nn']:
        raise Exception("Model type must be random or nn.")
    
    env = gym.make('gym_backgammon:backgammon-v0', disable_env_checker=True) 
    
    if model1_type == 'random':
        black_agent = RandomAgent(BLACK)
    else:
        model1 = TDGammon(hidden_units=model1_hidden_units, lr=0.1, lambda_=0.7) # initialize model
        model1.load(snapshot_path=CWD+model1_path) # load the weights
        black_agent = TDAgent(BLACK, model = model1, n_ply=model1_n_ply, train=False) # initialize the agent

    if model2_type == 'random':
        white_agent = RandomAgent(WHITE)
    else:
        model2 = TDGammon(hidden_units=model2_hidden_units, lr=0.1, lambda_=0.7) # initialize model
        model2.load(snapshot_path=CWD+model2_path) # load the weights
        white_agent = TDAgent(WHITE, model = model2, n_ply=model2_n_ply, train=False) # initialize the agent

    agents = {BLACK: black_agent, WHITE: white_agent}

    score = evaluate_agents(agents=agents, env=env, n_episodes=n_episodes)