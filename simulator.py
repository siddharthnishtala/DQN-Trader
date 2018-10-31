from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Simulator:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def play_one_episode(self, epsilon, training=True):

        self.agent.epsilon = epsilon

        state, valid_actions = self.env.reset()
        done = False
        prev_cum_reward = 0

        while not done:

            action = self.agent.act(state, valid_actions)
            next_state, reward, done, valid_actions = self.env.step(action)

            prev_cum_reward += reward

            if training:
                self.agent.remember((state, action, reward, next_state, done, valid_actions))
                self.agent.replay()

            state = next_state

        return prev_cum_reward, self.env.max_profit

    def train(self, no_of_episodes_train, epsilon_decay=0.995, min_epsilon=0.01, epsilon=1, progress_report=100):

        exploration_episode_rewards = []
        safe_episode_rewards = []
        exploration_max_episode_rewards = []
        safe_max_episode_rewards = []

        print("-"*60)
        print("Training")
        print("-"*60)

        for episode_no in tqdm(range(1, no_of_episodes_train+1)):

            exploration_episode_reward, exploration_max_episode_reward  = self.play_one_episode(epsilon, training=True)
            exploration_episode_rewards.append(exploration_episode_reward)
            exploration_max_episode_rewards.append(exploration_max_episode_reward)

            safe_episode_reward, safe_max_episode_reward = self.play_one_episode(0, training=False)
            safe_episode_rewards.append(safe_episode_reward)
            safe_max_episode_rewards.append(safe_max_episode_reward)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            if episode_no % progress_report == 0:
                fig = plt.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.plot(exploration_episode_rewards, 'red')
                ax1.plot(exploration_max_episode_rewards, 'blue')

                ax2 = fig.add_subplot(2, 1, 2)
                ax2.plot(safe_episode_rewards, 'red')
                ax2.plot(safe_max_episode_rewards, 'blue')

                fig.savefig('training_progress_' + str(episode_no) + '_episodes.png')

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(exploration_episode_rewards, 'red')
        ax1.plot(exploration_max_episode_rewards, 'blue')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(safe_episode_rewards, 'red')
        ax2.plot(safe_max_episode_rewards, 'blue')
        
        fig.savefig('training_progress_' + str(no_of_episodes_train) + '_episodes.png')

    def test(self, no_of_episodes_test):

        test_episode_rewards = []
        max_rewards = []

        print("-"*60)
        print("Testing")
        print("-"*60)

        for episode_no in tqdm(range(1, no_of_episodes_test+1)):

            test_episode_reward, max_reward = self.play_one_episode(0, training=False)
            test_episode_rewards.append(test_episode_reward)
            max_rewards.append(max_reward)

        plt.figure()
        plt.hist(test_episode_rewards, bins=10)
        plt.savefig('test_results.png')

        positive_percentage = sum(x > 0 for x in test_episode_rewards)/len(test_episode_rewards)

        print("-"*60)
        print("Mean Reward:", np.mean(test_episode_rewards))
        print("Mean Max Reward:", np.mean(max_rewards))
        print("Positive Reward Percentage:", positive_percentage)
        print("-"*60)
