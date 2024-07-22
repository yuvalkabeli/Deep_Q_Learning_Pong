from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import disable_interactive_logging
from tensorflow import debugging, config
import numpy as np

class ReplayBuffer(object):
    """
    Replay Buffer to store transitions of the agent during training.

    Attributes:
        mem_size (int): Maximum size of the buffer.
        mem_cntr (int): Counter to keep track of the number of transitions.
        state_memory (np.ndarray): Memory to store states.
        new_state_memory (np.ndarray): Memory to store next states.
        action_memory (np.ndarray): Memory to store actions.
        reward_memory (np.ndarray): Memory to store rewards.
        terminal_memory (np.ndarray): Memory to store terminal flags (done).

    Methods:
        store_transition(state, action, reward, state_, done): Stores a transition in the buffer.
        sample_buffer(batch_size): Samples a batch of transitions from the buffer.
    """
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        """
        Stores a transition in the buffer.
        
        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            state_ (np.ndarray): The next state.
            done (bool): Flag indicating if the episode is done.
        """
        if state[0].shape == (4, 80, 80):
            state = state[0]
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Samples a batch of transitions from the buffer.git remote -v
        
        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: Batch of states, actions, rewards, next states, and terminal flags.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims):
    """
    Builds a Deep Q-Network using Keras.

    Args:
        lr (float): Learning rate.
        n_actions (int): Number of possible actions.
        input_dims (tuple): Dimensions of the input.
        fc1_dims (int): Number of neurons in the first fully connected layer.

    Returns:
        keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=(*input_dims,), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model

class Agent(object):
    """
    Agent that interacts with and learns from the environment using a Deep Q-Network.

    Attributes:
        action_space (list): List of possible actions.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        eps_dec (float): Rate of decay for epsilon.
        eps_min (float): Minimum value for epsilon.
        batch_size (int): Size of the batch used in training.
        replace (int): Frequency to replace target network weights.
        q_target_model_file (str): File path for saving the target model.
        q_eval_model_file (str): File path for saving the evaluation model.
        learn_step (int): Counter for learning steps.
        memory (ReplayBuffer): Replay buffer to store transitions.
        q_eval (keras.models.Sequential): Evaluation Q-network.
        q_next (keras.models.Sequential): Target Q-network.

    Methods:
        replace_target_network(): Replaces the target network weights with the evaluation network weights.
        store_transition(state, action, reward, new_state, done): Stores a transition in the replay buffer.
        choose_action(observation): Chooses an action based on the current observation.
        learn(): Samples a batch of transitions and updates the Q-network.
        save_models(): Saves the Q-network models.
        load_models(): Loads the Q-network models.
    """
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace, input_dims, eps_dec=0.996, eps_min=0.01, mem_size=1000000, q_eval_fname='model/q_eval.keras', q_target_fname='model/q_next.keras'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname
        self.learn_step = 0
        disable_interactive_logging()
        config.experimental_run_functions_eagerly(True)
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 512)
        self.q_next = build_dqn(alpha, n_actions, input_dims, 512)

    def replace_target_network(self):
        """
        Replaces the target network weights with the evaluation network weights.
        """
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        """
        Stores a transition in the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            new_state (np.ndarray): The next state.
            done (bool): Flag indicating if the episode is done.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        """
        Chooses an action based on the current observation.

        Args:
            observation (np.ndarray): The current observation.

        Returns:
            int: The action chosen.
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            if observation[0][0].shape == (80, 80):
                observation = observation[0]
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        """
        Samples a batch of transitions and updates the Q-network.
        """
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()

            q_eval = self.q_eval.predict(state)
            q_next = self.q_next.predict(new_state)

            q_target = np.array(q_eval)
            indices = np.arange(self.batch_size)
            q_target[indices, action] = reward + self.gamma * np.max(q_next, axis=1) * (1 - done)

            try:
                self.q_eval.train_on_batch(state, q_target)
            except BaseException as e:
                print("Exception during training:", e)

            self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)
            self.learn_step += 1

    def save_models(self):
        """
        Saves the Q-network models.
        """
        self.q_eval.save(self.q_eval_model_file)
        self.q_next.save(self.q_target_model_file)
        print('... saving models ...')

    def load_models(self):
        """
        Loads the Q-network models.
        """
        self.q_eval = load_model(self.q_eval_model_file)
        self.q_next = load_model(self.q_target_model_file)
        print('... loading models ...')
