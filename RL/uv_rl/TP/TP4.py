from ForTP3 import generate_episode,get_actions,get_states_from_Maze,actions,next_state
from Maze_generating_interface import App
import numpy as np

def my_TD_0(Maze,num_episodes,gamma=1.0,alpha=0.1):
    S=np.argwhere(Maze !=2)
    V={tuple(i):0 for i in S}
    for count_episode in range(num_episodes):
        states_list, exit_state, init_states = get_states_from_Maze(Maze)
        init_state = init_states[np.random.randint(len(init_states))]
        s=init_state
       
        while np.all(s==exit_state) !=True:
            action=get_actions(s,actions,Maze)
            new_state,reward=next_state(s, action, exit_state, Maze)
            V[tuple(s)]=V[tuple(s)]+alpha*(reward+gamma*V[tuple(new_state)]-V[tuple(s)])
            s=new_state
            print(V)
    return V
if 1:
    app = App()
    app.mainloop()
    Maze=app.A
 
    V = my_TD_0(Maze,num_episodes=1, gamma=0.9)
    








