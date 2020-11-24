import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid,negative_grid

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U","D","L","R")
#getting returns for all states and actions
def play_game(grid,policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    states_actions_rewards =[(s,a,0)]
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0
    while True:
        r = grid.move(a)
        num_steps+=1
        s = grid.current_state()
        if s in seen_states:  
            #this step is to make sure agent doesnt stay in same states for a longer while
            reward = -10 / num_steps
            states_actions_rewards.append((s,None,reward))
            break
        elif grid.game_over():
            states_actions_rewards.append((s,None,r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s,a,r))
        seen_states.add(s)
    #converting all rewards into returns for each state action pair
    G = 0
    states_actions_retruns = []
    first = True
    for s,a,r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_retruns.append((s,a,G))
        G = r + GAMMA*G
    states_actions_retruns.reverse()
    return states_actions_retruns

#function to get max value and action which has max value

def max_dict(d):
    max_key = None
    max_val = float("-inf")
    for k,v in d.items():
        if v>max_val:
            max_val = v
            max_key = k
    return max_key,max_val


#VISUALISATION FUNCTIONS
def print_policy(P,g):
    for i in range(g.rows):
        print("------------------------")
        for j in range(g.cols):
            a = P.get((i,j),' ')
            print("%s|" % a,end = "")
        print("")

def print_values(V,g):
    for i in range(g.rows):
        print("--------------------------")
        for j in range(g.cols):
            v = V.get((i,j),0)
            if v>=0:
                print("%.2f|" % v,end="")
            else:
                print("%.2f|" % v,end="")
            print("")


if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.8)
    print("Rewards: \n")
    print_values(grid.rewards,grid)
    policy={}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s,a)] = []
        else:
            pass
    deltas = []
    for t in range(2000):
        if t%100 == 0:
            print(t)
        bigg = 0
        states_actions_returns = play_game(grid,policy)
        seenstates_actions_returns = set()
        for s,a,G in states_actions_returns:
            sa = (s,a)
            if sa not in seenstates_actions_returns:
                old_Q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                bigg = max(bigg,np.abs(old_Q - Q[s][a]))
                seenstates_actions_returns.add(sa)
        deltas.append(bigg)

        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
    plt.plot(deltas)
    plt.show()
    print("\n Final Policy: \n")
    print_policy(policy,grid)
    V={}
    for s,Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    
    print("\n Final Values: \n")
    print_values(V,grid)