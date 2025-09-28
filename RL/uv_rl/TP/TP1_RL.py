import numpy as np
import matplotlib.pyplot as plt
from forTP1 import Bandit

class myAlgo_greedy:
    def __init__(self,bandi,T):
        self.mybandi=bandi
        self.T=T
        self.N=(self.T/self.mybandi.K)**(2/3)*2
        self.count=np.zeros(self.mybandi.K)
        self.rewards=[]

    def greddy_and_epsi_greedy(self,epsilon=None):
        qt=np.zeros(self.mybandi.K)
        # Exploration phase
        for b in range(self.mybandi.K):
            for n in range(1,int(self.N+1)):
                reward = self.mybandi.get_arm(b)
                self.rewards.append(reward)
                qt[b] += (reward - qt[b]) / n
        
       # choose best hand
        if epsilon is None or np.random.rand() > epsilon:
          
          best_bandi = np.argmax(qt)
        else:
            best_bandi= np.random.randint(self.mybandi.K)

        #Exploitation phase
        for _ in range(int(self.T-self.N)):
            reward = self.mybandi.get_arm(best_bandi)
            self.rewards.append(reward)

        return sum(self.rewards), best_bandi



class SuccessiveElimination:
    def __init__(self, bandi, delta=0.1):
        self.bandi = bandi
        self.counts = np.zeros(self.bandi.K)    
        self.values = np.zeros(self.bandi.K)    
        self.actice_bandi = list(range(self.bandi.K))
        self.delta = delta
        self.rewards = []
        self.T=T

    def ucb(self, bandi):
        if self.counts[bandi] == 0:
            return float('inf')
        rt = 2 * np.log(self.T) / self.counts[bandi]
        return self.values[bandi] + rt

    def lcb(self, bandi):
        if self.counts[bandi] == 0:
            return -float('inf')
        rt = 2 * np.log(self.T) / self.counts[bandi]
        return self.values[bandi] - rt


    def run(self):
        t = 0
        while t < self.T and len(self.actice_bandi) > 1:
            
            for band in self.actice_bandi.copy():
                reward = self.bandi.get_arm(band)
                self.counts[band] += 1
                n = self.counts[band]
                self.values[band] += (reward - self.values[band]) / n
                self.rewards.append(reward)
                t += 1
                if t >= self.T:
                    break

            
            for band in self.actice_bandi.copy():
                for other_bandi in self.actice_bandi:
                    if band != other_bandi and self.ucb(band) < self.lcb(other_bandi):
                        if band in self.actice_bandi:
                            self.actice_bandi.remove(band)
                            break

        #Exploitation
        if self.actice_bandi:
            best_bandi = self.actice_bandi[0]
            while t < self.T:
                reward = self.bandi.get_arm(best_bandi)
                self.counts[best_bandi] += 1
                n = self.counts[best_bandi]
                self.values[best_bandi] += (reward - self.values[best_bandi]) / n
                self.rewards.append(reward)
                t += 1

        return sum(self.rewards), self.actice_bandi[0]


class UCB1:
    def __init__(self, bandit,T):
        self.bandit = bandit
        self.T=T        
        self.K = bandit.K
        self.counts = np.zeros(self.K)    
        self.values = np.zeros(self.K)   
        self.total_rewards = []
        self.actions = []

    def run(self):
      
        for band in range(self.K):
            reward = self.bandit.get_arm(band)
            self.counts[band] += 1
            self.values[band] += reward
            self.total_rewards.append(reward)
            self.actions.append(band)

        for t in range(self.K, self.T):
            
            ucb_values = np.zeros(self.K)
            for band in range(self.K):
                rt = 2 * np.log(self.T) / self.counts[band]
                ucb_values[band] = self.values[band]/self.counts[band] + rt

            action = np.argmax(ucb_values)
            reward = self.bandit.get_arm(action)

            self.counts[action] += 1
            self.values[action] += reward
            self.total_rewards.append(reward)
            self.actions.append(action)

        return sum(self.total_rewards)





if __name__ == "__main__":
    K=10
    T=500
    bandi=Bandit(K)
    algo=myAlgo_greedy(bandi,T)
    print(algo.greddy_and_epsi_greedy(0.8))
    
    suc=SuccessiveElimination(bandi,0.2)
    print(suc.run())

    uc=UCB1(bandi,T)
    print(uc.run())