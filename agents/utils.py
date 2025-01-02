import os, sys
import gym

from models import TDGammon

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
