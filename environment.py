from sequence_generator import Single_Signal_Generator
import numpy as np

def find_ideal(p, just_once):
	if not just_once:
		diff = np.array(p[1:]) - np.array(p[:-1])
		return sum(np.maximum(np.zeros(diff.shape), diff))
	else:
		best = 0.
		i0_best = None
		for i in range(len(p)-1):
			best = max(best, max(p[i+1:]) - p[i])

		return best

class Market:

    def __init__(self, sampler, last_n_timesteps, buy_cost, risk_averse=0.):

        self.no_of_actions = 3
        self.action_labels = ["don't buy /  sell", "buy", "hold"]
        self.sampler = sampler
        self.last_n_timesteps = last_n_timesteps
        self.buy_cost = buy_cost
        self.risk_averse = risk_averse
        self.state_shape = (last_n_timesteps, 1)
        self.start_index = last_n_timesteps - 1
        self.current_index = self.start_index
        self.last_index = None
        self.reset()
	
    def reset(self, rand_price=True):
        self.isAvailable = True

        sample_2d = self.sampler.sample()
        sample_1d = np.reshape(sample_2d[:,0], sample_2d.shape[0])

        self.sample_2d = sample_2d.copy()
        self.normalized_values = sample_1d/sample_1d[0]*100
        self.last_index = self.normalized_values.shape[0] - 1

        self.max_profit = find_ideal(self.normalized_values[self.start_index:], False)
        self.current_index = self.start_index

        return self.get_state(), self.get_valid_actions()

    def get_state(self, t=None):
        state = self.sample_2d[self.current_index - self.last_n_timesteps + 1: self.current_index + 1, :].copy()

        for i in range(state.shape[1]):
            norm = np.mean(state[:,i])
            state[:,i] = (state[:,i]/norm - 1.)*100

        return state

    def get_valid_actions(self):
        if self.isAvailable:
            return [0, 1]	# don't buy, buy
        else:
            return [0, 2]	# sell , hold


    def get_noncash_reward(self, t=None, empty=None):
        reward = self.normalized_values[self.current_index+1] - self.normalized_values[self.current_index]

        if self.isAvailable:
            reward -= self.buy_cost

        if reward < 0:
            reward *= (1. + self.risk_averse)

        return reward


    def step(self, action):

        if action == 0:		# don't buy / sell
            reward = 0.
            self.isAvailable = True
        elif action == 1:	# buy
            reward = self.get_noncash_reward()
            self.isAvailable = False
        elif action == 2:	# hold
            reward = self.get_noncash_reward()
        else:
            raise ValueError('no such action: '+str(action))

        self.current_index += 1

        return self.get_state(), reward, self.current_index == self.last_index, self.get_valid_actions()

if __name__ == '__main__':
    gen = Single_Signal_Generator(180, (10, 40), (5, 80), 0.5)
    env = Market(gen, 40, 3.3)
    env.reset()