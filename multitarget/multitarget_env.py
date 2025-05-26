import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from plot_rewardmap import plot_reward

class MultiTargetsEnv(gym.Env):
    def __init__(self, TIME_STEPS=200, 
                 initialize_state_randomly=True, 
                 SAMPLING_TIME_SECONDS=0.01, 
                 INITIAL_STATE=np.array([0, 0], dtype=np.float32), 
                 SET_POINTS=np.array([[-1, 0.2], [-0.1, 1], [0.9, -.1], [-0.4, -0.9]], dtype=np.float32),
                 NOISE_MAGNITUDE = 0,
                 distance_reward=False):
        super(MultiTargetsEnv, self).__init__()
        self._max_episode_steps = TIME_STEPS
        self.TIME_STEPS = TIME_STEPS
        self.SAMPLING_TIME_SECONDS = SAMPLING_TIME_SECONDS
        self.initialize_state_randomly = initialize_state_randomly
        self.INITIAL_STATE = INITIAL_STATE
        self.NUMBER_OF_qS = 4
        self.NOISE_MAGNITUDE = NOISE_MAGNITUDE
        self.distance_reward=distance_reward

        self.MINIMAL_VALUE_POSITION, self.MAXIMAL_VALUE_POSITION = np.ones(2)*-1.5, np.ones(2)*1.5
        self.SET_POINTS = SET_POINTS
        self.setpoint_radius = 0.05
        self.action_space = spaces.Discrete(int(self.NUMBER_OF_qS))
        
        self.observation_space = spaces.Box(
                    low=self.MINIMAL_VALUE_POSITION,
                    high=self.MAXIMAL_VALUE_POSITION,
                    shape=(2,),
                    dtype=np.float32)

    
    def get_reward(self, position):  
        distance_to_target = np.linalg.norm(position - self.SET_POINTS, axis=1)
        smallest_distance_to_target = np.min(distance_to_target)

        x, y = position
        if self.distance_reward == True:
            penalty = 1
        else:
            penalty = 1 + 0.5 * np.sin(6 * x) * np.cos(5 * y)  # Creates wave-like barriers

        reward_time_step = -(smallest_distance_to_target * (1*penalty)) * self.SAMPLING_TIME_SECONDS
        return reward_time_step



    def check_set_point(self):
        distance_to_target = np.linalg.norm(self.position-self.SET_POINTS, axis=1)
        smallest_distance_to_target = distance_to_target[np.argmin(np.abs(distance_to_target))]
        if abs(smallest_distance_to_target) <= self.setpoint_radius:
            reached_set_point = True
        else:
            reached_set_point = False
        return reached_set_point
    
    def get_observations(self, position):     
        observations = position
        return observations
    
    def get_control_action(self, position, q):
        action = self.SET_POINTS[q]-position
        return action
    


    def step(self, action):
        self.noise_sign = -1*self.noise_sign

        q_supervisor = action
        control_action = self.get_control_action(self.noisy_position, q_supervisor)
        self.position = self.position + control_action*self.SAMPLING_TIME_SECONDS

        # update observation
        self.noisy_position = self.position + self.noise_sign*self.NOISE_MAGNITUDE
        self.observation = self.get_observations(self.noisy_position)
        # get reward
        reward_time_step = self.get_reward(self.position)
        self.TIME_STEPS_left -= 1
        

        # Check if set point is reached:
        reached_set_point = self.check_set_point()
        # Check if simulation is terminated
        if self.TIME_STEPS_left <= 0: #or reached_set_point:
            terminated = True
        else:
            terminated = False 
        # Set placeholder for info
        info = {}
        return self.observation, reward_time_step, terminated, False, info


    def get_random_state(self):
        self.position = np.random.uniform(self.MINIMAL_VALUE_POSITION, self.MAXIMAL_VALUE_POSITION, 2).astype(np.float32) #*1
        self.observation = self.get_observations(self.position)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.noise_sign = np.array([1, -1], dtype=np.float32)
                
        if self.initialize_state_randomly == True:
            self.get_random_state()
        else:
            self.position = self.INITIAL_STATE

            self.observation = self.get_observations(self.position)
        # set the total number of episode TIME_STEPS
        self.TIME_STEPS_left = self.TIME_STEPS
        self.noisy_position = self.position + self.noise_sign*self.NOISE_MAGNITUDE
        
        info = {}
        # print('state in env', self.observation)
        return self.observation, info
    
if __name__ == "__main__":
    check_env(MultiTargetsEnv())
    plot_reward(MultiTargetsEnv())
