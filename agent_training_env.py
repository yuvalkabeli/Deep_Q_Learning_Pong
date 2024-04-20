import numpy as np
from dqn_keras import Agent
from utils import  make_env


# After runnning this file, you will recieve an initial model.
if __name__ == '__main__':
    env = make_env('ALE/Pong-v5')

    num_games = 500
    best_score = -10
    agent = Agent(gamma=0.99, epsilon=0.01, alpha=0.0001,
                  input_dims=(4,80,80), n_actions=6, mem_size=25000,
                  eps_min=0.01, batch_size=32, replace=1000, eps_dec=1e-5,
                   q_eval_fname='q_eval.keras', q_target_fname='q_next.keras')
    
    agent.load_models()


    scores, eps_history = [], []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()
            observation = observation_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', f"{i}/{num_games}",'score: ', score,
             ' average score %.3f' % avg_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
        if avg_score > best_score and i > 10:
            print('avg score %.2f better than best score %.2f, saving model' % (avg_score, best_score))
            agent.save_models()
            best_score = avg_score
        
        

        eps_history.append(agent.epsilon)