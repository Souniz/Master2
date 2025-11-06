from ForTP3 import generate_episode,get_actions,get_states_from_Maze,actions
from Maze_generating_interface import App
import numpy as np


def First_visit_MC_predic(Maze, gamma=1.0, num_episodes=10):
    states = [tuple(s) for s in np.argwhere(Maze == 1)]

    V = {s: 0.0 for s in states}
    returns = {s: [] for s in states}
    
    for count_episode in range(num_episodes):
       
        states_list, exit_state, init_states = get_states_from_Maze(Maze)
        init_state = init_states[np.random.randint(len(init_states))]
        episode = generate_episode(init_state, actions, Maze, exit_state, itermax=10)
        G = 0.0
        visited_states = set()
        
        for (s, action, reward, new_state) in reversed(episode):
            s = tuple(s)
            G = gamma * G + reward
            if s not in visited_states:
                visited_states.add(s)
                returns[s].append(G)
                V[s] = np.mean(returns[s])
        
        print(f"Épisode {count_episode+1}/{num_episodes} terminé.")
    
    return V


if __name__=="__main__":
    app = App()
    app.mainloop()
    Maze=app.A

    V = First_visit_MC_predic(Maze, gamma=0.9, num_episodes=20)
    print(V)


    





    
