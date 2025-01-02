import argparse
from gui.utils import args_gui

def formatter(prog):
    return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=100, width=180)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Backgammon AI', formatter_class=lambda prog: formatter(prog))

    parser.add_argument('--host', help='Host running the web gui.', default='localhost')
    parser.add_argument('--port', help='Port listening for command.', default=8002, type=int)
    parser.add_argument('--difficulty', help='Determines the model used by the opponent. Accepted options are: beginner, intermediate, expert, champion.', default='intermediate', type=str)
    parser.add_argument('--first_player', help='First player: random, human or AI.', default='random', type=str)

    try:
        parser.set_defaults(func=args_gui)
    except Exception as error:
        print(f"{error}")
    
    args = parser.parse_args()
    args.func(args)

    