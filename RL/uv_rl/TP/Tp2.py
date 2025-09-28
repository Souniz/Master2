from forTp2 import Grid_Solv
import numpy as np

grid_eval= Grid_Solv(4)

def policy_evaluatio (pi):
    v=np.zeros(grid_eval.GRID_SIZE*grid_eval.GRID_SIZE)
    delta=-1
    while delta<0:
        for s in grid_eval.states:
            delta=0
            

if __name__ == "__main__":

    policy_evaluatio(2)
