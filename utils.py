import numpy as np
import gymnasium as gym

class SkipEnv(gym.Wrapper):
    """
    Wrapper to skip frames in the environment to speed up training.

    Attributes:
        _skip (int): Number of frames to skip.
    
    Methods:
        step(action): Executes an action in the environment, skipping frames.
        reset(**kwargs): Resets the environment and observation buffer.
    """
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Executes an action in the environment, skipping frames.
        
        Args:
            action (int): The action to be executed.
        
        Returns:
            tuple: Observations, total reward, termination flag, truncation flag, and info.
        """
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            t_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, t_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment and observation buffer.
        
        Returns:
            np.ndarray: The initial observation.
        """
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class PreProcessFrame(gym.ObservationWrapper):
    """
    Wrapper to preprocess frames from the environment.
    
    Methods:
        observation(obs): Processes the observation frame.
        process(frame): Converts the frame to grayscale and resizes it.
    """
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80,80,1), dtype=np.uint8)

    def observation(self, obs):
        """
        Processes the observation frame.
        
        Args:
            obs (np.ndarray): The original observation frame.
        
        Returns:
            np.ndarray: The processed frame.
        """
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        """
        Converts the frame to grayscale and resizes it.
        
        Args:
            frame (np.ndarray): The original frame.
        
        Returns:
            np.ndarray: The processed frame.
        """
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + 0.114*new_frame[:,:,2]
        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)
        return new_frame.astype(np.uint8)

class MoveImgChannel(gym.ObservationWrapper):
    """
    Wrapper to move image channels to the first dimension.
    
    Methods:
        observation(observation): Moves the image channels.
    """
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.observation_space.shape[-1],
                   self.observation_space.shape[0],
                   self.observation_space.shape[1]),
            dtype=np.float32)

    def observation(self, observation):
        """
        Moves the image channels.
        
        Args:
            observation (np.ndarray): The original observation.
        
        Returns:
            np.ndarray: The observation with moved channels.
        """
        return np.moveaxis(observation, 2, 0)

class ScaleFrame(gym.ObservationWrapper):
    """
    Wrapper to scale frame values to the range [0, 1].
    
    Methods:
        observation(obs): Scales the observation frame.
    """
    def observation(self, obs):
        """
        Scales the observation frame.
        
        Args:
            obs (np.ndarray): The original observation frame.
        
        Returns:
            np.ndarray: The scaled frame.
        """
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    """
    Wrapper to stack frames in a buffer for the environment.
    
    Attributes:
        buffer (np.ndarray): Buffer to store stacked frames.
    
    Methods:
        reset(**kwargs): Resets the environment and buffer.
        observation(observation): Adds a new observation to the buffer.
    """
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32)
    
    def reset(self, **kwargs):
        """
        Resets the environment and buffer.
        
        Returns:
            tuple: Initial observation and an empty dictionary.
        """
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset()), {}

    def observation(self, observation):
        """
        Adds a new observation to the buffer.
        
        Args:
            observation (np.ndarray): The new observation.
        
        Returns:
            np.ndarray: The updated buffer.
        """
        if observation[0][0].shape == (80, 80):
            observation = observation[0]
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation[0]
        return self.buffer

def make_env(env_name, render_mode='rgb_array'):
    """
    Creates the gym env for the model to run on.
    
    Args:
        env_name (str): Gym environment name.
        render_mode (str):`human` option opens up visual game. defaults to `rgb_array`.
    
    Returns:
        ScaleFrame: Constructor for the observation wrapper..
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)
