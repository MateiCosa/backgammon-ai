# Report on Backgammon AI Project

This project focuses on building an AI system capable of defeating non-professional human players in the game of Backgammon. Implemented primarily in Python using PyTorch, the project consists of two main components: a training and testing pipeline accessible via command-line tools and a graphical user interface (GUI) implemented in Python (backend) and HTML, CSS, and JavaScript (frontend). The training environment is based on a slightly modified OpenAI-style gym environment.

The project is structured into key modules:

***Training and Evaluation Pipeline:*** This handles the implementation of reinforcement learning techniques, such as temporal difference (TD) learning, and supports 1-ply and 2-ply evaluations to improve model performance. The pipeline includes customizable neural network architectures that can be trained directly from the command line and a script for evaluating different models against each other. These tools are located in the agents folder.

***Gameplay Interface:*** A GUI that enables users to play against AI models trained with different configurations. This interface supports multiple difficulty levels and allows users to replace pre-trained models with custom-trained ones. Located in the GUI folder, the interface includes HTML, CSS, and JavaScript files. It provides a user-friendly experience by offering possible moves as buttons during each turn and features dice animations to enhance the gameplay experience.

***Feature Engineering:*** Optimized features such as Blot Exposure and Blockade Strength are integrated to improve the AI’s decision-making efficiency without sacrificing computational speed.

### Challenges and Solutions

***Computational Cost of 2-ply Evaluation:*** Implementing 2-ply search posed challenges due to Backgammon's large branching factor. To mitigate this, we adopted a sampling approach during training, which significantly reduced computation time while maintaining reasonable performance.

***Feature Optimization:*** Calculating Blot Exposure and Blockade Strength using all dice combinations proved computationally expensive. We redefined these measures to use single dice rolls, reducing the average simulation time from 5 seconds to 0.3 seconds per move. This optimization maintained high correlation with the original measures and enhanced scalability.

***Training Stability:*** Exploring additional neural network layers did not improve model performance, emphasizing the importance of simpler architectures for this task. The implementation of hand-crafted features further enhanced prediction accuracy.

# Innovations and Ideas

***Optimized Feature Engineering:*** By streamlining Blot Exposure and Blockade Strength calculations, we achieved a balance between computational efficiency and informational richness, enabling more effective large-scale training.

***Deep Search Implementation:*** Incorporating full-fledged 2-ply evaluations during gameplay improved the AI’s tactical depth without compromising the user experience.

***Flexible Difficulty Levels:*** The inclusion of customizable difficulty settings allows for a broad user base, from casual players to enthusiasts seeking a challenging experience.

This project demonstrates the potential of reinforcement learning in strategic games and highlights the importance of computational efficiency and thoughtful feature design. Future work could explore additional strategies to improve AI performance in competitive settings while maintaining real-time responsiveness.