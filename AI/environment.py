from gymnasium import Env
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt
import random
import numpy as np

class Environment(Env):
    def __init__(self):
        self.states: int = 22
        self.goal_states: int = 1
        self.addictive_states: int = 15
        self.neutral_states: int = 6
        self.actions: int = 9
        self.starting_state: int = 3 
        self.punishment_end_addictive: int = -4
        self.punishment_in_addictive: float = -1.2
        self.reward_addictive: int = 10
        self.reward_healthy: int = 1
        self.duration_safe_phase: int = 50
        self.duration_addictive_phase:int = 1000
        self.action_space = Discrete(self.actions)  
        self.observation_space = Discrete(self.states) 
        self.current_state: int = self.starting_state
        self.current_phase: str = "safe_phase"
        self.current_step: int = 0
        
    def step(self, action: int) -> tuple:
        next_state = self.calculate_next_state(self.current_state, action)
        reward = self.calculate_reward(self.current_state, next_state, action)
        if self.duration_safe_phase == 0:
            self.current_phase = "addictive_phase"
        if next_state is not None:
            done = (self.duration_safe_phase == 0 and self.duration_addictive_phase == 0) or self.current_step == 1050
        else:
            done = True
        self.current_state = next_state
        self.current_step += 1
        return self.current_state, reward, done
    
    def calculate_next_state(self, current_state: int, action: int) -> int:
        if current_state is not None:
            if self.duration_safe_phase > 2 and self.current_phase == "safe_phase":
                if current_state == 3 and 0<= action <= 5:
                    self.duration_safe_phase -= 1
                    return random.choice([1, 2, 3, 4, 5, 6])
                elif 1 <= current_state <= 6 and 0 <= action <= 5:
                    self.duration_safe_phase -= 1
                    return 3
                elif current_state == 1 and action == 6:
                    self.duration_safe_phase -= 1
                    return 0
                elif current_state == 0 and action == 6:
                    self.duration_safe_phase -= 1
                    return 3
                elif current_state == 3 and action == 7:
                    self.duration_safe_phase -= 1
                    return 3
                else:
                    return current_state
            elif self.duration_safe_phase == 2:
                self.duration_safe_phase -= 1
                return 6
            elif self.duration_safe_phase == 1:
                if current_state == 6 and action == 7:
                    self.duration_safe_phase -= 1
                    return 7
                else:
                    return current_state
            elif self.duration_addictive_phase > 2 and self.current_phase == "addictive_phase":
                if 7 <= current_state <= 21  and (action == 7 or action == 8):
                    if current_state != 7 and current_state != 21 and current_state != 14 and current_state != 19:
                        self.duration_addictive_phase -= 1
                        return random.choice([current_state - 1, current_state + 1])
                    elif current_state == 14 or current_state == 19 and current_state != 7 and current_state != 21 and action == 8:
                        self.duration_addictive_phase -= 1
                        return random.choice([4, current_state - 1, current_state + 1])
                    elif current_state != 14 and current_state == 19 and current_state != 7 and current_state != 21:
                        self.duration_addictive_phase -= 1
                        return random.choice([4, current_state - 1, current_state + 1])
                    elif current_state == 7 and current_state != 14 and current_state != 19 and current_state != 21:
                        self.duration_addictive_phase -= 1
                        return random.choice([current_state + 1, 21])
                    elif current_state == 21 and current_state != 7 and current_state != 14 and current_state != 19:
                        self.duration_addictive_phase -= 1
                        return random.choice([current_state - 1, 7])
                else:
                    return current_state    
            elif self.duration_addictive_phase == 2 and self.current_phase == "addictive_phase":
                self.duration_addictive_phase -= 1
                return random.choice([14, 19])
            elif self.duration_addictive_phase == 1 and self.current_phase == "addictive_phase":        
                if (current_state == 14 or current_state == 19) and action == 8:
                    self.duration_addictive_phase -= 1
                    return 3
                else:
                    return current_state
            else:
                return current_state
        else:
            self.done = True
            return current_state
        
    def calculate_reward(self, current_state: int, next_state: int, action: int) -> float:
        if current_state is not None and next_state is not None and action is not None:
            if self.current_phase == "safe_phase" and current_state == 0 and next_state == 3:
                return self.reward_healthy
            elif self.current_phase == "safe_phase" and current_state == 6 and next_state == 7:
                return self.reward_addictive
            elif 7 <= current_state <= 21 and 7 <= next_state <= 21 and (action == 7 or action == 8):
                return self.punishment_in_addictive
            elif self.current_phase == "addictive_phase" and current_state == 14 or current_state == 19 and next_state == 3:
                return self.punishment_end_addictive
            else:
                return 0.0
        else:
            return 0.0
    def reset(self):
        self.current_state = self.starting_state
        self.current_phase = "safe_phase"
        self.current_step = 0
        return self.current_state

class EnvironmentModel:
    def __init__(self, env: Environment):
        self.env = env
        self.num_states: int = env.states
        self.num_actions: int = env.actions
        self.transition_matrix = np.ones((self.num_states, self.num_actions, self.num_states)) /self.num_states  
        self.value_estimate = np.zeros(self.num_states)  
        self.value_counts = np.zeros(self.num_states)  
        self.reward_function = np.zeros((self.num_states, self.num_actions))
        
    def update(self, current_state: int, action: int, next_state: int):
        if next_state is not None and current_state is not None and action is not None:
            self.transition_matrix[current_state, action, next_state] += 1
            self.reward_function[current_state, action] += self.predict_reward(current_state, action, next_state)
            self.transition_matrix[current_state, action, :] /= np.sum(self.transition_matrix[current_state, action, :])
            self.calculate_values_estimate(current_state)

    def calculate_values_estimate(self, state: int):
        for action in range(self.num_actions):
            for next_state in range(self.num_states):
                self.value_estimate[state] += self.predict_reward(state, action, next_state)
        
    def predict(self, current_state: int, action: int) -> int:
        next_state = self.env.calculate_next_state(current_state, action)
        if next_state is not None:
            return next_state
        else:
            return current_state
    
    def predict_reward(self, current_state: int, action: int, next_state):
        reward = self.env.calculate_reward(current_state, next_state, action)
        if reward is not None:
            return reward
        else:
            return 0.0
        
    def successor_representation(self, current_state: int, action: int, next_state: int) -> np.ndarray:
        successor_rep = np.zeros(self.num_states)
        successor_rep[next_state] = 1
        return successor_rep
