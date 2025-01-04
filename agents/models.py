import numpy as np
import torch
import torch.nn as nn

import time, datetime

from itertools import count

import os, sys
CWD = os.getcwd()
sys.path.append(CWD + '/gym-backgammon/')

from gym_backgammon.envs.backgammon import WHITE, BLACK
from agents import RandomAgent, TDAgent
from eval import evaluate_agents, MAX_PLAYS

torch.set_default_tensor_type('torch.DoubleTensor')

BASE_INPUT_FEATURES = 198 # standard td-gammon board encoding

class TDBackbone(nn.Module):

    def __init__(self, lr, lambda_,  seed=0):

        super(TDBackbone, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.lr = lr # learning rate
        self.lambda_ = lambda_  # trace-decay parameter
        self.start_episode = 0 # start episode
    
        self.eligibility_traces = None # eligibility traces
        self.init_eligibility_traces()
    
    def init_eligibility_traces(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in self.parameters()]

    def save_snapshot(self, snapshot_path, step, run_name):
        path = snapshot_path + f"/{run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f')}_{step+1}.zip"
        torch.save({'step': step + 1, 'model_state_dict': self.state_dict(), 'eligibility': self.eligibility_traces}, path)
        print(f"\nSnapshot saved: {path}")

    def train(self, env, n_episodes, save_path=None, save_step=0, n_ply = 1, run_name='baseline'):
        
        score = {WHITE: 0, BLACK: 0} # score table
        model = self # network
        agents = {WHITE: TDAgent(WHITE, model=model, n_ply=n_ply), BLACK: TDAgent(BLACK, model=model, n_ply=n_ply)} # setup for self-play; note: both agents use the same model

        episode_durations = np.zeros(n_episodes)
        training_start_time = time.time() 
        unfinished_games = 0 # number of games that did not end within MAX_PLAYS plays
        total_plays = 0 # total number of plays

        assert n_episodes > 0

        for episode in range(n_episodes):

            self.init_eligibility_traces() # reset eligibility traces
            agent_color, first_roll, curr_state = env.reset() # reset environment
            agent = agents[agent_color] # set first player

            episode_start_time = time.time() # get the episode start time

            for i in range(MAX_PLAYS):

                roll = agent.roll_dice() if i > 0 else first_roll # get the dice

                curr_eval = self(curr_state) # evaluate the current state

                actions = env.get_valid_actions(roll) # get valid actions
                action = agent.choose_action(actions, env) # choose action according to the model
                next_state, reward, end_game, winner = env.step(action)
                next_eval = self(next_state) # evaluate the next state

                if end_game:
                        
                    loss = round(self.update_weights(curr_eval, reward).item(), 2) # final loss
                    score[agent.color] += 1 # increment winner's score
                    episode_duration = time.time() - episode_start_time # compute episode duration
                    episode_durations[episode] = episode_duration # save the current episode duration
                    total_plays += i # add the number of plays of the current episode

                    white_win_perc = score[WHITE] / (episode + 1 - unfinished_games) * 100
                    black_win_perc = score[BLACK] / (episode + 1 - unfinished_games) * 100

                    log_msg = f"Game={episode+1} | Winner={winner} | after {i} plays | Final loss: {loss}||"
                    log_msg +=  f" Wins: {agents[WHITE].name}={score[WHITE]}({white_win_perc:<5.1f}%) |"
                    log_msg += f" {agents[BLACK].name}={score[BLACK]}({black_win_perc:<5.1f}%) |"
                    log_msg += f" Duration={episode_duration:<.3f} sec"
                    print(log_msg)
                
                    break
                else:
                    loss = self.update_weights(curr_eval, next_eval)

                agent_color = env.get_opponent_agent() # get the opposite color
                agent = agents[agent_color] # switch current agent
                curr_state = next_state # move to the next state

            if episode_durations[episode] == 0: # no winner within 1000 plays
                episode_durations[episode] = time.time() - episode_start_time
                unfinished_games += 1
                total_plays += 1

            if save_path and save_step > 0 and episode > 0 and (episode + 1) % save_step == 0:
                self.save_snapshot(snapshot_path=save_path, step=episode, run_name=run_name)
                model = self
                agents_to_evaluate = {WHITE: TDAgent(WHITE, model=model, n_ply=n_ply), BLACK: RandomAgent(BLACK)}
                print("Snapsot evaluation:")
                evaluate_agents(agents_to_evaluate, env, n_episodes=50)
                print()

        average_game_duration = round(episode_durations.sum() / n_episodes, 3)
        average_game_length = round(total_plays / n_episodes, 2) 
        total_duration = datetime.timedelta(seconds=int(time.time() - training_start_time))         

        print(f"\nAverage duration per game: {average_game_duration} seconds")
        print(f"Average game length: {average_game_length} plays | Total Duration: {total_duration}")

        if save_path:
            self.save_snapshot(snapshot_path=save_path, step=n_episodes - 1, run_name=run_name)

            with open(f'{save_path}/log.txt', 'a') as file:
                file.write(f"Average duration per game: {average_game_duration} seconds")
                file.write(f"\nAverage game length: {average_game_length} plays | Total Duration: {total_duration}")

        env.close()

    def load(self, snapshot_path):

        snapshot = torch.load(snapshot_path)
        self.load_state_dict(snapshot['model_state_dict'])

class TDGammon(TDBackbone):
    def __init__(self, lr, lambda_, seed=0, input_units=198, hidden_units=80, output_units=1, env = None):

        super(TDGammon, self).__init__(lr, lambda_, seed=seed)

        self.hidden = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_units, output_units),
            nn.Sigmoid()
        )

        for p in self.parameters():
            nn.init.zeros_(p)
        
        self.env = env
        self.extra_features = (input_units > BASE_INPUT_FEATURES)

    def blot_exposure(self):

        white_target, black_target= 0, 0
        board, bar = self.env.game.board, self.env.game.bar

        for dice in range(1, 7):

            found = False
            point = 0
            if bar[BLACK] > 0 and board[dice - 1] == (1, WHITE):
                white_target += 1
                found = True
            while not found and point < 24:
                if board[point] == (1, WHITE) and board[point - dice][1] == BLACK:
                    white_target += 1
                    found = True
                    continue
                point += 1

            found = False
            point = 0
            if bar[WHITE] > 0 and board[24 - dice] == (1, BLACK):
                black_target += 1
                found = True
            while not found and point < 24:
                if board[point - dice] == (1, BLACK) and board[point][1] == WHITE:
                    black_target += 1
                    found = True
                    continue
                point += 1

        return [(12-white_target)*white_target/36, (12 - black_target)*black_target/36]
    
    def blockade_strength(self):
        board = self.env.game.board
        bar = self.env.game.bar
        white_counter = [0]*25
        black_counter = [0]*25
        for dice in range(1, 7):
            if bar[WHITE] > 0 and board[24 - dice][1] == BLACK and board[24 - dice][0] > 1:
                white_counter[-1] += 1/6
            if bar[BLACK] > 0 and board[dice - 1][1] == WHITE and board[24 - dice][0] > 1:
                black_counter[-1] += 1/6
            for point in range(dice, 24):
                if board[point][1] == WHITE and board[point - dice] == BLACK:
                    if board[point][0] > 1:
                        black_counter[point - dice] += 1/6
                    if board[point][0] > 1:
                        white_counter[point] += 1/6
        return white_counter, black_counter
            
    def forward(self, x):

        if self.extra_features: # include extra features
            exposure = self.blot_exposure()
            white_counter, black_counter = self.blockade_strength()
            x = x + exposure + white_counter + black_counter

        x = torch.from_numpy(np.array(x))
        x = self.hidden(x)
        x = self.output(x)
        return x

    def update_weights(self, p, p_next):
        self.zero_grad() 
        p.backward()

        with torch.no_grad():

            td_error = p_next - p
            for i, weights in enumerate(self.parameters()):

                # z <- lambda * z + (grad w w.r.t P_t)
                self.eligibility_traces[i] = self.lambda_ * self.eligibility_traces[i] + weights.grad

                # w <- w + alpha * td_error * z
                new_weights = weights + self.lr * td_error * self.eligibility_traces[i]
                weights.copy_(new_weights)

        return td_error

