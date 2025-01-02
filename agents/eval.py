import numpy as np
import torch
import torch.nn as nn

import time, datetime

import os, sys
CWD = os.getcwd()
sys.path.append(CWD + '/gym-backgammon/')

from gym_backgammon.envs.backgammon import WHITE, BLACK
from agents import RandomAgent, TDAgent

MAX_PLAYS = 1000

def evaluate_agents(agents, env, n_episodes):

    score = {WHITE: 0, BLACK: 0} # score table
    eval_start_time = time.time() # eval start time
    unfinished_games = 0 # number of games that did not end within MAX_PLAYS plays
    total_plays = 0 # total number of plays
    episode_durations = np.zeros(n_episodes) # time durations of each episode
    unfinished_games = 0 # number of games that did not end within MAX_PLAYS plays
    total_plays = 0 # total number of plays

    assert n_episodes > 0

    for episode in range(n_episodes):

        agent_color, first_roll, curr_state = env.reset() # reset environment
        agent = agents[agent_color] # set first player
        episode_start_time = time.time() # get the episode start time

        for i in range(MAX_PLAYS):

            roll = agent.roll_dice() if i > 0 else first_roll # get the dice
            actions = env.get_valid_actions(roll) # get valid actions
            action = agent.choose_action(actions, env) # choose action according to the model
            next_state, reward, end_game, winner = env.step(action)
        
            if end_game:
                    
                score[agent.color] += 1 # increment winner's score
                episode_duration = time.time() - episode_start_time # compute episode duration
                episode_durations[episode] = episode_duration # save the current episode duration
                total_plays += i # add the number of plays of the current episode

                white_win_perc = score[WHITE] / (episode + 1 - unfinished_games) * 100
                black_win_perc = score[BLACK] / (episode + 1 - unfinished_games) * 100

                log_msg = f"Game={episode+1} | Winner={winner} | after {i} plays ||"
                log_msg +=  f" Wins: {agents[WHITE].name}={score[WHITE]}({white_win_perc:<5.1f}%) |"
                log_msg += f" {agents[BLACK].name}={score[BLACK]}({black_win_perc:<5.1f}%) |"
                log_msg += f" Duration={episode_duration:<.3f} sec"
                print(log_msg)
            
                break

            agent_color = env.get_opponent_agent() # get the opposite color
            agent = agents[agent_color] # switch current agent
            curr_state = next_state # move to the next state
        
        if episode_durations[episode] == 0: # no winner within 1000 plays
            episode_durations[episode] = time.time() - episode_start_time
            unfinished_games += 1
            total_plays += 1

    average_game_duration = round(episode_durations.sum() / n_episodes, 3)
    average_game_length = round(total_plays / n_episodes, 2) 
    total_duration = datetime.timedelta(seconds=int(time.time() - eval_start_time))         

    print(f"\nAverage duration per game: {average_game_duration} seconds")
    print(f"Average game length: {average_game_length} plays | Total Duration: {total_duration}")

    return score