import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid,negative_grid
from MC import print_policy,print_values,max_dict

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ("U","D","L","R")


#epsilon greedy policy
def random_action(a,eps=0.1):
    p = np.random.random()
    if p<(1-eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

if __name__ =="__main__":
    grid = negative_grid(step_cost=-0.9)
    print("Rewards:")
    print_values(grid.rewards,grid)
    Q={}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a]=0

    #we need to track how many times the Q values have been chanced for states and states actions
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0
    
    t = 1.0
    deltas = []
    for it in range(10000):
        #to increase value of t so that epsilon value becomes lower at later episodes
        if it%100==0:
            t += 0.01
        if it%2000 == 0:
            print(it)
        
        s = (2,0) #initial starting point
        grid.set_state(s)

        a = max_dict(Q[s])[0]
        a = random_action(a,eps=0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2,eps=0.5/t)
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + ALPHA*(r+GAMMA*Q[s2][a2] - Q[s][a])
            biggest_change = max(biggest_change,np.abs(old_qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s,0) + 1
            s = s2
            a = a2
        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()

    policy = {}
    value = {}

    for s in grid.actions.keys():
        a,max_q = max_dict(Q[s])
        policy[s] = a
        value[s] = max_q
    #visualizations
    print("Update counts:")
    total = np.sum(list(update_counts.values()))
    for k,v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts,grid)

    print("Values:")
    print_values(value,grid)
    print("Policy:")
    print_policy(policy,grid)