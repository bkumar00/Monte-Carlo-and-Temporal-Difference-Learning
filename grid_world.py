import numpy as np
import matplotlib.pyplot as plt

action_space = ("U","D","L","R")

class Grid:
    
    def __init__(self,rows,cols,start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
    
    def set(self,reward,action):
        self.rewards = reward
        self.actions = action
    
    def set_state(self,s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return(self.i , self.j)
    
    def is_terminal(self,s):
        return s not in self.actions

    def get_next_state(Self,s,a):
        i,j = s[0],s[1]
        if a in self.actions[(i,j)]:
            if a == 'U':
                i-=1
            elif a == 'D':
                i+=1
            elif a == 'L':
                j-=1
            elif a == 'R':
                j+=1
        return i,j

    def move(self,a):
        if a in self.actions[(self.i,self.j)]:
            if a == 'U':
                self.i-=1
            elif a == 'D':
                self.i+=1
            elif a == 'L':
                self.j-=1
            elif a == 'R':
                self.j+=1
        return self.rewards.get((self.i,self.j),0)
    
    def undo_move(self,action):
        if action == 'U':
            self.i+=1
        elif action == 'D':
            self.i -=1
        elif action == 'R':
            self.j +=1
        elif action == 'L':
            self.j -=1
        assert(self.current_state() in self.all_states())
    
    def game_over(self):
        return (self.i,self.j) not in self.actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    g = Grid(3,4,(2,0))
    rewards={(0,3):1,(1,3):-1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),(0, 2): ('L', 'D', 'R'),(1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),(2, 0): ('U', 'R'),(2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),(2, 3): ('L', 'U')

    }
    g.set(rewards,actions)
    return g

def negative_grid(step_cost=-0.1):
    g = standard_grid()
    g.rewards.update({(0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,})
    return g

def play_game(agent,env):
    pass
     