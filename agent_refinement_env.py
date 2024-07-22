import numpy as np
from dqn_keras import Agent
from utils import make_env

# This script refines an existing agent to play Pong using a Deep Q-Network (DQN).
# After running this file, you will receive a refined model.

if __name__ == '__main__':
    # Initialize the environment
    env = make_env('ALE/Pong-v5')

    num_games = 500  # Number of games to train the agent
    best_score = -21  # Initialize the best score
    agent = Agent(gamma=0.99, epsilon=0.5, alpha=0.00005,
                  input_dims=(4, 80, 80), n_actions=6, mem_size=25000,
                  eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5)

    # Load pretrained model if available
    agent.load_models()

    scores, eps_history = [], []  # Initialize scores and epsilon history
    n_steps = 0  # Initialize step counter

    # Training loop
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
            agent.store_transition(observation, action, reward, observation_, int(done))  # Store transition
            agent.learn()  # Learn from the transition
            observation = observation_  # Update observation

        scores.append(score)  # Append score

        avg_score = np.mean(scores[-50:])  # Calculate average score over the last 50 games
        print('episode: ', f"{i}/{num_games}", 'score: ', score,
              'average score %.3f' % avg_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
        
        # Save model if the average score improves
        if avg_score > best_score and i > 10:
            print('avg score %.2f better than best score %.2f, saving model' % (avg_score, best_score))
            agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)  # Append epsilon to history

