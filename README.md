# Deep Q-Learning Pong

This project implements Deep Q-Learning to play the classic game Pong. The Deep Q-Network (DQN) is built using Keras, and the environment is managed using OpenAI's Gym.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/Deep_Q_Learning_Pong.git
    cd Deep_Q_Learning_Pong
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training a Model from Scratch

To train a model from scratch, use the `initial_agent_run.py` script.

1. Ensure your environment is set up and dependencies are installed.
2. Run the script:
    ```sh
    python initial_agent_run.py
    ```
   This will initialize the environment and train the agent for a specified number of games, storing transitions and learning from them to improve performance. The model will be saved if it achieves a better average score than previously recorded.

### Refining the Agent

To further refine the agent using an existing model, follow these steps:

1. Ensure the pretrained model is in the appropriate directory.
2. Run the `agent_refinement_env.py` script:
    ```sh
    python agent_refinement_env.py
    ```
   This script will load the existing model and continue training, fine-tuning the agent's performance.
   
### Testing the Pretrained Agent

To test the agent using a pretrained model, use the `test_env.py` script.

1. Ensure the `q_eval` model is in the appropriate directory.
2. Run the script:
    ```sh
    python test_env.py
    ```
   This will load the pretrained `q_eval` model and execute it in the Gym environment with rendering enabled, displaying the agent's performance in a window.

## Files

- `initial_agent_run.py`: Script to train the agent from scratch.
- `agent_refinement_env.py`: Script to refine the agent using an existing model.
- `test_env.py`: Script to test the agent using a pretrained model.
- `dqn_keras.py`: Deep Q-Network implementation using Keras.
- `utils.py`: Utility functions for environment setup.

## Contributing

Feel free to submit pull requests to improve the project or fix any issues.
