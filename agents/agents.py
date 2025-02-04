import numpy as np

# import sys
# import os
# CWD = os.getcwd()
# sys.path.append(CWD + "/gym-backgammon")
# print(CWD + "/gym-backgammon")
from gym_backgammon.envs.backgammon import WHITE, COLORS

np.random.seed(0)

DIE_OUTCOMES = np.arange(1, 7)

class Agent():

    def __init__(self, color):

        self.color = color
        self.name = f"Agent({COLORS[color]})"
    
    def roll_dice(self):
        die1 = np.random.choice(DIE_OUTCOMES)
        die2 = np.random.choice(DIE_OUTCOMES)
        if self.color == WHITE:
            die1 *= -1
            die2 *= -1
        return (die1, die2)
    
    def choose_action(self, actions, env):
        raise NotImplementedError
    
class HumanAgent(Agent):

    def __init__(self, color):

        super().__init__(color)
        self.name = f"Human({COLORS[color]})"

    def choose_action(self, actions, env):
        pass

class RandomAgent(Agent):

    def __init__(self, color):

        super().__init__(color)
        self.name = f"Random({COLORS[color]})"

    def choose_action(self, actions, env):
        return list(actions)[np.random.randint(0, len(actions))] if actions else None

class TDAgent(Agent):

    def __init__(self, color, model, name = "", n_ply=1, train=True):
        super().__init__(color)
        if name:
            self.name = f"TD({name + '-' + COLORS[color]})"
        else:
            self.name = f"TD({COLORS[color]})"
        self.model = model
        self.n_ply = n_ply
        self.train = train
    
    def choose_action(self, actions, env):
        
        best_action = None

        if not actions:
            return best_action
        
        actions_list = list(actions)
        eval_arr = np.zeros(len(actions_list))
        counter = env.counter
        env.counter = 0
        init_state = env.game.save_state()

        if self.train:
            opponent_roll = self.roll_dice()
            opponent_roll = (-opponent_roll[0], -opponent_roll[1])
        else:
            opponent_dice = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)]
            if self.roll_dice()[0] > 0:
                for i, dice in enumerate(opponent_dice):
                    opponent_dice[i] = (-dice[0], -dice[1])
            prob_dice = 2 * np.ones(21)
            prob_dice[[0, 6, 11, 15, 18, 20]] /= 2
            prob_dice /= 36 

        best_response_reward = -np.inf if self.color == WHITE else np.inf

        # action simulation
        for i, action in enumerate(actions_list):
            state, reward, done, info = env.step(action) # play action and receive feedback
            prev_state = env.game.save_state()
            eval_arr[i] = self.model(state) # predict value of resulting position

            if self.n_ply == 1:
                pass
                
            elif self.n_ply == 2:

                env.get_opponent_agent()

                if self.train:
                    opponent_actions_list = env.get_valid_actions(opponent_roll) # get opponent's valid actions
                    for j, action in enumerate(opponent_actions_list):
                        next_state, next_reward, next_done, next_info = env.step(action) # play action and receive feedback
                        next_eval = self.model(next_state) # evaluate next postiion
                        eval_arr[i] = min(eval_arr[i], next_eval) if self.color == WHITE else max(eval_arr[i], next_eval) # opponent chooses best response
                        if self.color == WHITE:
                            if best_response_reward > eval_arr[i]:
                                break
                        else:
                            if best_response_reward < eval_arr[i]:
                                break
                        env.game.restore_state(prev_state) # revert to initial state
                else:
                    evals = np.zeros(21)
                    for k, dice_roll in enumerate(opponent_dice):
                        opponent_actions_list = env.get_valid_actions(dice_roll) # get opponent's valid actions
                        for j, action in enumerate(opponent_actions_list):
                            next_state, next_reward, next_done, next_info = env.step(action) # play action and receive feedback
                            next_eval = self.model(next_state) # evaluate next postiion
                            evals[k] = min(evals[k], next_eval) if self.color == WHITE else max(evals[k], next_eval) # opponent chooses best response
                            env.game.restore_state(prev_state) # revert to initial state
                    eval_arr[i] = np.dot(evals, prob_dice) # expected value of postion resulting from agent's action over every opponent dice roll

                env.get_opponent_agent() # back to current agent's turn

            else:
                raise NotImplementedError
            
            if self.color == WHITE:
                if best_response_reward < eval_arr[i]:
                    best_response_reward = eval_arr[i]
                    best_action_index = i
            else:
                if best_response_reward > eval_arr[i]:
                    best_response_reward = eval_arr[i]
                    best_action_index = i

            env.game.restore_state(init_state) # revert to initial state

        # action selection
        best_action = actions_list[best_action_index] # best action
        env.counter = counter # reset to orginal counter

        return best_action
    
