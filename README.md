# Backgammon AI

## Scope 

This repo attempts to build an AI system capable of consistently defeating non-professional human players in the game of backgammon. The project encompasses (a) a training and testing pipeline implemented in Python & PyTorch accessible via command line and (b) a graphical user interface (GUI) implemented in Python (backend) and HTML, CSS, and JavaScript (frontend) that can be used to play games against the trained models. The environment for training the AI agents and simulating the game is an [OpenAI-style gym environment](https://github.com/dellalibera/gym-backgammon) that we used without significant modifications.

## Getting Started 

### Prerequisites

We recommend creating a virtual Python envirnoment and then importing the necessary packages using the following command:

`pip3 install -r requirements.txt`

*Note:* We used Python 3.11, but any other relatively recent version should work as well.

### Installation

First, you should clone this repo:

`git clone https://github.com/MateiCosa/backgammon-ai.git`

Secondly, you need to clone the repo containing the gym environment following the instructions of dellalibera:

`git clone https://github.com/dellalibera/gym-backgammon.git`
`cd gym-backgammon/`
`pip3 install -e .`

Altough not necessary, for certain variations of the models, changing `self.max_length_episode = np.inf` in the `backgammon_env.py` will allow agents to perform deep searches that consider a large number of positions.

## Usage

### Training AI agents

The training code is contained in the agents subsirectory:

`cd agents`

We provide three examples of training runs, `v0`, `v1`, `v2`, each with different characteristics. To train a *TD-Gammon*-inspired model with no additional features and 1-ply evaluation, one can use

`python3 ./main.py train --save_path ./trained_models/v0 --save_step 10000 --episodes 150000 --name v0 --type nn --lr 0.1 --hidden_units 80 --n_ply 1`

or, equivalently, 

`chmod -x train_v0.sh`
`sh train_v0.sh`

By setting `n_ply 2`, one obtains our proposed v0 model which implements a 2-ply evaluation policy. *Note:* due to computational restrictions, during training, **only** one dice roll is sampled for the opponent. For evaluation, the full 2-ply approach (i.e., all dice rolls and all opponent actions) is implemented.

Finally, to incorporate extra features based on *blot exposure* and *blockade strength*, one can set `input_units 250`. This version alters the architecture of the neural net to include hand-crafted features on top of the 198-dimensional encoding of the position.

### AI agent evaluation

We also provide several tools for evaluating the quality of the trained models. Using the command line, one can compare any two models saved in the subdirectory `trained_models` using the following command:

`python3 main.py eval --n_episodes <number_of_episodes> --model1_name <name> --model1_type <nn|random> --model1_path </trained_models/run_name/snapshot.zip> --model1_input_units <198|250> --model1_n_ply <1|2> --model2_name <name> --model2_type <nn|random> --model2_path </trained_models/run_name/snapshot.zip> --model2_input_units <198|250> --model2_n_ply <1|2>`

Furthermore, we provide plotting functionality for understanding the behavior of a single training run by displaying the win rate of a snapshot model against its predecessor. For code examples and an analysis of our models, please check `benchmark.ipynb`.

### Gameplay

To play the game, you can use `main.py` script in the project directory. For instance:

`python3 main.py --host localhost --port 8002 --difficulty expert`

The previous command launches the GUI and allows the human player to compete against an AI model. Four difficulty modes correspond to the models trained and stored in `/agents/game_models`. To replace them with your models, simply add the desired model in the folder, rename it `custom.zip` then launch the game with `--difficulty custom` after adding the correct parameters in the config file.

The *champion* difficulty level should provide a real challenge for any amateur player. Note that the AI may take a few moments to move due to the 2-ply search. This should not be a major inconvenience. For a virtually instant moving time you may downgrade to the *expert* difficulty.

Enjoy the game & good luck! :) 

## Technial notes

### Overview of RL techniques

We followed the highly successful TD-Gammon architecture proposed by Tesauro 1992. This method applies *temporal difference learning* which relies on *self-play* and clever *credit assignment* to learn a good evaluation function. The evaluation function learner is composed of a simple fully connected network with one hidden layer. After trying increasing the number of hidden layers we noticed no significant improvement. 

By having two agents using the same model play against each other, the model is trained to predict the probability of winning given the current position. After every move, the network's weights are updated in order to minimize the difference between evaluations of consecutive positions, thereby attempting to learn a smooth function which ideally incorporates all possible future moves.

### Extending the search depth

The idea of 2-ply evaluation means considering not only the evaluation of the positions resulting from the current agent’s move, but also the opponent’s best response. While conceptually simple and recommended by the literature, this approach raises the problem of a high computational cost given by the large branching factor of backgammon: for every subsequent position, there are 21 possible die rolls and for each roll there are on average of 20 legal moves. Even with early stopping approaches like alpha-beta pruning, a single training episode would last upwards of 7s, making this approach unfeasible. The main bottleneck is the move generation component of the environment. We chose to only sample one die roll and find the opponent’ best response through a pruning approach, cutting the per-episode time to around 0.5s. Unfortunately, this model was unable to outperform the baseline, hinting at the fact the importance of exploration due to the stochasticity of the environment.

Performing full-fledged 2-ply search is, however, a great way to improve the performance of any trained model. With every move taking under a second, it does not greatly affected the enjoyment of gameplay. Therefore, in evaluation mode one can take the already learned model and use it to evaluate all possible outcomes within the next two moves, selecting the best one in expectation.

### Adding hand-crafted features

TODO

## References
  
[1] Backgammon rules - https://usbgf.org/backgammon-basics-how-to-play/

[2] TD-Gammon paper - https://link.springer.com/chapter/10.1007/978-1-4757-2379-3_11

[3] Implementation details for TD-Gammon - http://incompleteideas.net/book/first/ebook/node108.html

[4] Reinforcement Learning: An Introduction - http://incompleteideas.net/book/the-book-2nd.html

[5] Alpha-beta pruning - https://en.wikipedia.org/wiki/Alpha–beta_pruning

[6] OpenAI-style environemnt: https://github.com/dellalibera/gym-backgammon

[7] TD-Gammon inspiration repo: https://github.com/dellalibera/td-gammon

[8] GUI inspiration repo: https://github.com/aknandi/backgammon/tree/master
