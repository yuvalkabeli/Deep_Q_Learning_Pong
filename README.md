# Deep Q-Learning Pong

This project implements Deep Q-Learning to play the classic game Pong. The Deep Q-Network (DQN) is built using Keras, and the environment is managed using OpenAI's Gym.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yuvalkabeli/Deep_Q_Learning_Pong.git
    cd Deep_Q_Learning_Pong
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Train the agent:
    ```sh
    python agent_training_env.py
    ```

2. Refine the agent:
    ```sh
    python agent_refinement_env.py
    ```

3. Run the initial agent (optional):
    ```sh
    python initial_agent_run.py
    ```
    > Use this if you don't want to run the entire training process, as it takes a while to train the agent. This script uses a pretrained model generated from `agent_training_env.py`.

## Running the Initial Agent

To run the initial agent using the pretrained `q_eval` model, you need to configure the agent object with the path to the initial files and set `load_checkpoint` to `True`. 

1. Ensure the `q_eval` model is in the appropriate directory.
2. Modify the `initial_agent_run.py` script to include the path to the pretrained model:
    ```python
    agent = Agent(alpha=0.0001, gamma=0.99, n_actions=6, epsilon=1.0, batch_size=64, input_dims=[210, 160, 4], chkpt_dir='models/')
    agent.load_models()  # Make sure load_checkpoint is set to True within the Agent class
    ```
3. Run the script:
    ```sh
    python initial_agent_run.py
    ```
This will load the pretrained `q_eval` model and execute it in the Gym environment, displaying the agent's performance in a window.

## Files

- `agent_refinement_env.py`: Environment for agent refinement.
- `agent_training_env.py`: Environment for agent training.
- `dqn_keras.py`: Deep Q-Network implementation using Keras.
- `initial_agent_run.py`: Script to run the initial agent.
- `utils.py`: Utility functions.

## Contributing

Feel free to submit pull requests to improve the project or fix any issues.

