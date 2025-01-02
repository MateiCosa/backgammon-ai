import argparse
import utils

def formatter(prog):
    return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=100, width=180)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD-Gammon', formatter_class=lambda prog: formatter(prog))
    subparsers = parser.add_subparsers(help='Train TD-Network | Evaluate Agent(s) | Plot Wins')

    parser_train = subparsers.add_parser('train', help='Train TD-Network', formatter_class=lambda prog: formatter(prog))
    parser_train.add_argument('--save_path', help='Save directory location', type=str, default=None)
    parser_train.add_argument('--save_step', help='Save the model every n episodes/games', type=int, default=0)
    parser_train.add_argument('--episodes', help='Number of episodes/games', type=int, default=200000)
    parser_train.add_argument('--lr', help='Learning rate', type=float, default=1e-2)
    parser_train.add_argument('--input_units', help='Input units', type=int, default=198)
    parser_train.add_argument('--hidden_units', help='Hidden units', type=int, default=40)
    parser_train.add_argument('--lambda_', help='Credit assignment parameter', type=float, default=0.7)
    parser_train.add_argument('--name', help='Name of the experiment', type=str, default='baseline')
    parser_train.add_argument('--type', help='Model type', choices=['nn'], type=str, default='nn')
    parser_train.add_argument('--seed', help='Seed used to reproduce results', type=int, default=0)
    parser_train.add_argument('--n_ply', help='Number of plies considered for move selection', type=int, default=1)

    try:
        parser_train.set_defaults(func=utils.args_train)
    except Exception as error:
        print(f"{error}")

    parser_gui = subparsers.add_parser('gui', help='Start GUI', formatter_class=lambda prog: formatter(prog))
    parser_gui.add_argument('--host', help='Host running the web gui', default='localhost')
    parser_gui.add_argument('--port', help='Port listening for command', default=8002, type=int)
    parser_gui.add_argument('--model', help='Model used by the opponent', required=True, type=str)
    parser_gui.add_argument('--hidden_units', help='Hidden units of the model loaded', required=False, type=int, default=40)
    parser_gui.add_argument('--type', help='Model type', choices=['nn'], type=str, default='nn')

    try:
        parser_gui.set_defaults(func=utils.args_gui)
    except Exception as erorr:
        print(f"{error}")
    
    args = parser.parse_args()
    args.func(args)

    