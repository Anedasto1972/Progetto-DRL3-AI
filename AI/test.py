from environment import Environment, EnvironmentModel
from agent import HybridAgent, ModelBasedAgent, ModelFreeAgent

env = Environment()
model = EnvironmentModel(env)
model_based_agent = ModelBasedAgent()
model_free_agent = ModelFreeAgent()
beta_values = [0.5, 0.7] 
hybrid_agent = HybridAgent(model_based_agent, model_free_agent, beta_values)
num_episodes = 1

for episode in range(num_episodes):
    state = env.reset()
    
    while True:
        action = hybrid_agent.step(state)
        next_state, reward, done = env.step(action)
        successor_rep = model.successor_representation(state, action, next_state)
        print(successor_rep)
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        print(experience)
        action_mf = model_free_agent.step(state)
        action_mb = model_based_agent.step(state)
        model_based_agent.update(action, state)
        model_free_agent.update_q_values(state, action, next_state, reward, done)
        state = next_state
        if done:
            break
