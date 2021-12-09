import numpy as np
import time
import random
import matplotlib
import matplotlib.pyplot as plt

class BernoulliBandit():
    
    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0


class MultiArmedBandit():
    
    def __init__(self, n, probas=None):
        self.n = n
        self.bandits = BernoulliBandit(n, probas=probas)
        self.bandits_reward = [0] * n
        self.bandits_n = [0] * n
        self.total_reward = 0
        self.total_n = 0
        self.reward_list = []

    def update(self, index):
        reward = self.bandits.generate_reward(index)
        self.bandits_reward[index] = (self.bandits_reward[index] * self.bandits_n[index] + reward)/(self.bandits_n[index] + 1)
        self.bandits_n[index] += 1
        self.total_reward += reward
        self.reward_list.append(self.total_reward)
        self.total_n += 1

    def random_choose(self):
        index = random.randint(0,self.n-1)
        self.update(index)

    def max_greedy_choose(self):
        index = np.argsort(-np.array(self.bandits_reward))[0]
        self.update(index)

    def epsilon_greedy_choose(self, epsilon=0.2):
        if np.random.random() < epsilon:
            self.random_choose()
        else:
            self.max_greedy_choose()

    def ucb_choose(self, cp=0.2):
        ucb_list = []
        for i in range(self.n):
            ucb = self.bandits_reward[i] + 2*cp*np.sqrt(2*np.log(self.total_n+1)/(self.bandits_n[i]+1))
            ucb_list.append(ucb)
        index = np.argsort(-np.array(ucb_list))[0]
        self.update(index)

def plt_reward(reward_list):
    x = [_ for _ in range(len(reward_list[0]))]
    y1 = reward_list[0]
    y2 = reward_list[1]
    y3 = reward_list[2]
    y4 = reward_list[3]
    plt.plot(x, y1, marker='o', ms=1, label="random")
    plt.plot(x, y2, marker='o', ms=1, label="max_greedy")
    plt.plot(x, y3, marker='o', ms=1, label="epsilon_greedy")
    plt.plot(x, y4, marker='o', ms=1, label="ucb")

    plt.xticks(rotation=45)
    plt.xlabel("次数")
    plt.ylabel("方法")
    plt.title("多臂赌博机")
    plt.legend(loc="upper left")

    plt.show()



if __name__ == "__main__":
    reward_list = []
    multiArmedBandit = MultiArmedBandit(10, [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    for i in range(10000):
        multiArmedBandit.random_choose()
    reward_list.append(multiArmedBandit.reward_list)
    print(multiArmedBandit.total_reward)

    multiArmedBandit = MultiArmedBandit(10, [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    for i in range(10000):
        multiArmedBandit.max_greedy_choose()
    reward_list.append(multiArmedBandit.reward_list)
    print(multiArmedBandit.total_reward)

    multiArmedBandit = MultiArmedBandit(10, [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    for i in range(10000):
        multiArmedBandit.epsilon_greedy_choose()
    reward_list.append(multiArmedBandit.reward_list)
    print(multiArmedBandit.total_reward)

    multiArmedBandit = MultiArmedBandit(10, [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    for i in range(10000):
        multiArmedBandit.ucb_choose()
    reward_list.append(multiArmedBandit.reward_list)
    print(multiArmedBandit.total_reward)

    plt_reward(reward_list)