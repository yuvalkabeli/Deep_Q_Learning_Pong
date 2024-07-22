import numpy as np
from dqn_keras import Agent
from utils import make_env

# This script tests the performance of a DQN agent on the Pong-v5 environment using a pretrained model.

if __name__ == '__main__':
    # Initialize the environment
    env = make_env('ALE/Pong-v5',render_mode="human")

    num_games = 100  # Number of games to test the agent
    agent = Agent(gamma=0.99, epsilon=0.0, alpha=0.0001,
                  input_dims=(4, 80, 80), n_actions=6, mem_size=25000,
                  eps_min=0.02, batch_size=32, replace=1000, eps_dec=1e-5)

    # Load pretrained model
    agent.load_models()

    scores = []  # Initialize scores list
    n_steps = 0  # Initialize step counter

    # Testing loop
    for i in range(num_games):
        done = False
        observation = env.reset()  # Reset environment
        score = 0  # Initialize score for the current game
        while not done:
            action = agent.choose_action(observation)  # Choose action
            observation_, reward, terminated, truncated, info = env.step(action)  # Step in environment
            done = terminated or truncated  # Check if game is done
            n_steps += 1
            score += reward  # Accumulate score
            env.render()  # Render environment
            observation = observation_  # Update observation

        scores.append(score)  # Append score

        avg_score = np.mean(scores[-100:])  # Calculate average score over the last 100 games
        print(f'episode: {i}/{num_games}, score: {score}, average score {avg_score:.3f}, steps {n_steps}')