import numpy as np
import matplotlib.pyplot as plt
from grid_world import negative_grid
from MC import print_policy,print_values,max_dict
from SARSA import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ("U","D","L","R")

if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.8)
    print("Rewards:")
    print_values(grid.rewards,grid)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
    #to keep in count the number of times Q of s and sa have been changed
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0
        
    t = 1.0
    deltas = []
    for it in range(10000):
        #increase t to reduce epsilon to encourage greedy behavior in later episodes
        if it%100 == 0:
            t += 0.01
        if it%2000 == 0:
            print(it)

        s = (2,0) #starting state
        grid.set_state(s)
        a,_ = max_dict(Q[s])
        biggest_change = 0
        while not grid.game_over():
            a= random_action(a,eps=0.5/t)
            r = grid.move(a)
            s2 = grid.current_state()

            old_qsa = Q[s][a]

            a2,max_q_s2a2 = max_dict(Q[s2])
            Q[s][a] = Q[s][a] + ALPHA*(r+GAMMA*max_q_s2a2 - Q[s][a])
            biggest_change = max(biggest_change,np.abs(old_qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s,0) + 1
            s = s2
            a = a2
        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()

    policy={}
    V={}
    #visualizations
    for s in grid.actions.keys():
        a,max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q
    print("Update Counts:")
    total = np.sum(list(update_counts.values()))
    for k,v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts,grid)


    print("Values:")
    print_values(V,grid)
    print("Policy:")
    print_policy(policy,grid)