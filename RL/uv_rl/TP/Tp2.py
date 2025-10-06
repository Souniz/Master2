from forTp2 import Grid_Solv
import numpy as np


def policy_evaluation(grid_eval,gamma=1,teta=0.02):

    p=grid_eval.transition()
    V=np.zeros(grid_eval.GRID_SIZE*grid_eval.GRID_SIZE)

    while True:
        delta=0
        for s in grid_eval.states:
            v=0
            for a,pi_a in grid_eval.uniform_policy[s].items():
                s_prim,r,done=p[s][a]
                v+=pi_a* (r+gamma*V[s_prim])

            delta=max(delta,abs(v-V[s]))
            V[s]=v
        if delta<teta:
            break
    return V

def policy_improvement(grid_eval,gamma=1,teta=0.02):
    p=grid_eval.transition()
    V=policy_evaluation(grid_eval,gamma=1,teta=0.02)
    pi_imp={}
    for s in grid_eval.states:
         q={}
         for a in grid_eval.actions :
             s_prim,r,done=p[s][a]
             q[a]=r+gamma*V[s_prim]
            
         best_a= max(q, key=q.get)
         pi_imp[s]={a:1 if a==best_a else 0 for a in grid_eval.actions}
    return pi_imp

def policy_iteration(grid_eval,gamma=1,teta=0.02):

    policy_eval=policy_evaluation(grid_eval)
    policy_imp=policy_improvement(grid_eval)
    pi={}
    while True:
        policy_stable=True
        for s in grid_eval.states:
            q={}
            old_action=max(policy_imp[s],key=policy_imp[s].get)
            for a in grid_eval.actions :
                s_prim,r,done=grid_eval.transition()[s][a]
                q[a]=r+gamma*policy_eval[s_prim]
            best_a= max(q, key=q.get)
            pi[s]=best_a
            if old_action != best_a:
                policy_stable=False
        if policy_stable==False:
            grid_eval.uniform_policy=policy_imp
            policy_eval=policy_evaluation(grid_eval)
            policy_imp=policy_improvement(grid_eval)
        return pi,policy_eval
    

def value_iteration(grid_eval,gamma=1,teta=0.02):
    p=grid_eval.transition()
    V=np.zeros(grid_eval.GRID_SIZE*grid_eval.GRID_SIZE)

    while True:
        delta=0
        
        for s in grid_eval.states:

            if s in grid_eval.TERMINAL_STATES:
                continue
            q_values = []
            v = V[s]
            for a in grid_eval.actions:
                s_prim,r,done =p[s][a]
                q_values.append(r + gamma * V[s_prim])
            V[s] = max(q_values)
            delta = max(delta, abs(v-V[s]))
            if delta < teta:
                break
    return V


if __name__ == "__main__":
    grid_eval= Grid_Solv(4)
    s=value_iteration(grid_eval)
    print(s)