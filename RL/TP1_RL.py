import numpy as np
import matplotlib.pyplot as plt
from forTP1 import Bandit

class myAlgo:
    def __init__(self,K,T):
        self.mybandi=Bandit(K)
        self.T=T
        self.N=(self.T/self.mybandi.K)**(2/3)*2

    def uniform_explor(self):
        qt=[]
        for _ in range(self.mybandi.K):
            qt.append([])
        for n in range(int(self.N)):
            for k in range(self.mybandi.K):
                
                  reward=self.mybandi.get_arm(k)
                  qt[k].append(reward)
        mean_qt=[np.mean(np.array(i)) for i in qt]
        
        k_opti=np.argmax(mean_qt)
        
        round=self.T-self.N 
        return np.sum([self.mybandi.get_arm(k_opti) for i in range(int(round))])

    def 


if __name__ == "__main__":
    K=10
    T=500
    algo=myAlgo(K,T)
    print(algo.uniform_explor())
