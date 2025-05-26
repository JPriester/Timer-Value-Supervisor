import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class SupervisorLineEnv(gym.Env):
    def __init__(self, TIME_STEPS=200, 
                 initialize_state_randomly=True, 
                 SAMPLING_TIME_SECONDS=0.01, 
                 INITIAL_STATE=np.array([0], dtype=np.float32), 
                 SET_POINTS=np.array([-1, 1], dtype=np.float32),
                 NOISE_MAGNITUDE = 0):
        super(SupervisorLineEnv, self).__init__()
        self._max_episode_steps = TIME_STEPS
        self.TIME_STEPS = TIME_STEPS
        self.SAMPLING_TIME_SECONDS = SAMPLING_TIME_SECONDS
        self.initialize_state_randomly = initialize_state_randomly
        self.INITIAL_STATE = INITIAL_STATE
        self.NUMBER_OF_qS = 2
        self.NOISE_MAGNITUDE = NOISE_MAGNITUDE

        self.MINIMAL_VALUE_POSITION_X, self.MAXIMAL_VALUE_POSITION_X = -3., 3.
        self.SET_POINTS = SET_POINTS
        self.setpoint_radius = 0.05
        self.action_space = spaces.Discrete(int(self.NUMBER_OF_qS))
        
        self.observation_space = spaces.Box(
                    low=np.array([self.MINIMAL_VALUE_POSITION_X, self.MINIMAL_VALUE_POSITION_X], dtype=np.float32),
                    high=np.array([self.MAXIMAL_VALUE_POSITION_X, self.MAXIMAL_VALUE_POSITION_X], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32)

    
    def get_reward(self):  
        position_error = self.get_observations(self.x)
        reward_time_step = -np.min(abs(position_error))*self.SAMPLING_TIME_SECONDS 
        return reward_time_step
    
    def check_set_point(self):
        distance_to_target = self.x-self.SET_POINTS
        smallest_distance_to_target = distance_to_target[np.argmin(np.abs(distance_to_target))]
        if abs(smallest_distance_to_target) <= self.setpoint_radius:
            reached_set_point = True
        else:
            reached_set_point = False
        return reached_set_point
    
    def get_observations(self, x):     
        observations = x-self.SET_POINTS    
        return observations
    
    def get_control_action(self, x, q):
        action = self.SET_POINTS[q]-x
        return action
    


    def step(self, action):
        self.noise_sign = -1*self.noise_sign
        self.noisy_x = self.x + self.noise_sign*self.NOISE_MAGNITUDE

        q_supervisor = action
        control_action = self.get_control_action(self.noisy_x, q_supervisor)
        self.x = self.x + control_action*self.SAMPLING_TIME_SECONDS
        # update observation
        self.state = self.get_observations(self.noisy_x)
        # get reward
        reward_time_step = self.get_reward()
        self.TIME_STEPS_left -= 1
        
        # Check if simulation is terminated
        if self.TIME_STEPS_left <= 0: 
            terminated = True
        else:
            terminated = False 
        # Set placeholder for info
        info = {}
        return self.state, reward_time_step, terminated, False, info


    def get_random_state(self):
        self.x = np.random.uniform(self.MINIMAL_VALUE_POSITION_X, self.MAXIMAL_VALUE_POSITION_X, 1).astype(np.float32) #*1
        self.state = self.get_observations(self.x)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.noise_sign = 1
                
        if self.initialize_state_randomly == True:
            self.get_random_state()
        else:
            self.x = self.INITIAL_STATE

            self.state = self.get_observations(self.x)
        # set the total number of episode TIME_STEPS
        self.TIME_STEPS_left = self.TIME_STEPS
        self.noisy_x = self.x + self.noise_sign*self.NOISE_MAGNITUDE
        
        info = {}
        return self.state, info
    
if __name__ == "__main__":
    check_env(SupervisorLineEnv())
