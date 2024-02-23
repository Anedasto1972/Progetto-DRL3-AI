import numpy as np
from environment import EnvironmentModel
from environment import Environment
from collections import defaultdict

env = Environment()

class ModelBasedAgent:
    def __init__(self, temperature=1, prioritized_sweeping_steps=50):
        self.num_states: int = env.states
        self.num_actions: int = env.actions
        self.temperature = temperature
        self.prioritized_sweeping_steps = prioritized_sweeping_steps
        self.values= defaultdict(lambda: np.zeros(self.num_actions))
        self.H_values = np.zeros(self.num_states)
        self.V_values = np.zeros(self.num_states)
        self.model = EnvironmentModel(env)

    def update(self, action: int, state: int):
        for _ in range(self.prioritized_sweeping_steps):
            exp_values = np.exp(self.H_values - np.max(self.H_values)) / self.temperature
            total_exp = np.sum(exp_values)
            probabilities = exp_values / total_exp if total_exp != 0 else np.ones_like(exp_values) / len(exp_values)
            state_to_update = np.random.choice(self.num_states, p = probabilities)
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    reward: float = self.model.predict_reward(state_to_update, a, s)
                    if reward != None: 
                        self.values[state_to_update , a ] += self.model.transition_matrix[state_to_update, a, s] * (reward + self.calculate_value_of_state(s))
            M = np.max(self.values[state_to_update])     
            delta = np.abs(self.calculate_value_of_state(state_to_update) - M)
            self.V_values[state_to_update] = self.calculate_value_of_state(state_to_update)
            self.V_values[state_to_update] = M
            h_values = np.zeros(self.num_states)
            for s in range(self.num_states):
                h_values[s] = delta * np.max(1 - self.model.transition_matrix[state_to_update, action, s])
            self.H_values[state_to_update] = h_values[state_to_update]
            for s in range(self.num_states):
                if s != state_to_update:
                    self.H_values[s] = max(h_values[s], self.H_values[s])
            
            
    def calculate_value_of_state(self, state: int) -> float:
        return self.model.value_estimate[state]
    
    def take_action(self, state: int):
        return np.argmax(self.values[state])
    
    def step(self, state):
        action = self.take_action(state)
        next_state, reward, done = env.step(action)
        self.model.update(state, action, next_state)
        self.update(action, state)
        return action

class ModelFreeAgent:
    def __init__(self, learning_rate=0.05, discount_factor=0.9):
        self.num_states: int = env.states
        self.num_actions: int = env.actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        
    def choose_action(self, state: int):
        return int(np.argmax(self.q_values[state]))

    def update_q_values(self, state: int, action: int, next_state: int, reward: float):
        if reward is not None:
            future_q_value = np.max(self.q_values[next_state])
            temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[state][action])
            self.q_values[state][action] = (self.q_values[state][action] + self.learning_rate * temporal_difference)
        
    def step(self, state: int) -> int:
        action = self.choose_action(state)
        next_state, reward, done = env.step(action)
        self.update_q_values(state , action, next_state, reward, done)
        return action


class HybridAgent:
    def __init__(self, model_based_agent, model_free_agent, beta_values, exploration_rate=0.1):
        self.model_based_agent: ModelBasedAgent = model_based_agent
        self.model_free_agent: ModelFreeAgent = model_free_agent
        self.num_states = env.states
        self.num_actions = env.actions
        self.beta_values = beta_values
        self.exploration_rate = exploration_rate
        self.q_values_hybrid = defaultdict(lambda: np.zeros(self.num_actions))
        self.model = EnvironmentModel(env)

    def choose_action(self, state: int):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.model_based_agent.num_actions)
        else:
            beta = np.random.choice(self.beta_values)
            q_values_mb = self.model_based_agent.values[state]
            q_values_mf = self.model_free_agent.q_values[state]
            q_values_hybrid = beta * q_values_mb + (1 - beta) * q_values_mf
            return np.argmax(q_values_hybrid[state])
    
    def step(self, state: int):
        action_mb = self.model_based_agent.step(state)
        action_mf = self.model_free_agent.step(state)
        next_state, reward, done = env.step(action_mb)
        if not done:
            self.model_free_agent.update_q_values(state, action_mf, next_state, reward, done)
        action_hybrid = self.choose_action(next_state)
        return action_hybrid
