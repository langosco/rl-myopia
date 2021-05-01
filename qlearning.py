import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ACTION_SPACE = [0, 1] # [DONT PUSH, PUSH]
epsilon = 0.1 # parameter for epsilon-greedy q-learning

def act(action_values):
    """
    args:
        action_values: array of q-values for actions
        
    returns:
        action, one of 1 (push) or -1 (don't push)
    """
    greedy_action = ACTION_SPACE[np.argmax(action_values)]
    random_action = np.random.choice(ACTION_SPACE)
    return np.random.choice([random_action, greedy_action], p=[epsilon, 1-epsilon])


def reward(previous_action, action):
    return -action + 10 * previous_action


def run_episode(action_values, previous_action):
    a = act(action_values)
    r = reward(previous_action, a)
    return a, r


def update_running_mean(num_values, current_mean, next_value):
    return (num_values * current_mean + next_values) / (num_values + 1)
    

def append_to_log(dct, update_dict):
    """appends update_dict to dict, entry-wise. Creates list entry
    if it doesn't exist.
    """
    for key, newvalue in update_dict.items():
        dct.setdefault(key, []).append(newvalue)
    return dct
    

class QLearning:
    def __init__(self, schedule: callable = lambda n: 1/n):
        """
        args:
            schedule: 1/n corresponds to computing the average of 
        empirical q-values. Constant step size in (0, 1) corresponds 
        to taking a weighted average instead, with values close to 1
        weighting recent values more.
        """
        self.action_values = [0, -1]
        self.num_updates = [1, 1]
        self.rundata = {}
        self.weighted_avg = True
        self.schedule = schedule
    
    def log(self, **data):
        append_to_log(self.rundata, data)
    
    def update(self, action, reward):
        """keep track of mean action-values"""
        stepsize = self.schedule(self.num_updates[action])
        self.action_values[action] += stepsize * (reward - self.action_values[action])
        self.num_updates[action] += 1
        self.log(rewards=reward,
                 actions=action,
                 push_values=self.action_values[1],
                 dp_values=self.action_values[0])

        
def running_mean(a, window=30):
    return np.convolve(a, np.ones(window)/window, mode='valid')



def run_and_plot():
    print(f"Training agent using {epsilon}-greedy Q-learning...")
    qvalues = QLearning()
    a = 1 

    for _ in tqdm(range(50000)):
        a, r = run_episode(qvalues.action_values, a)
        qvalues.update(a, r)


    # plotting
    fig, axs = plt.subplots(2, 1, figsize=[10, 10])
    axs = axs.flatten()

    ax = axs[0]
    ax.plot(qvalues.rundata['push_values'], label='push')
    ax.plot(qvalues.rundata['dp_values'], label='dont push')
    ax.set_ylabel("learned action-value")
    ax.set_xlabel("training iteration")
    ax.set_title("Q-values during training")
    ax.legend()


    ax = axs[1]
    push_freq = sum(qvalues.rundata['actions']) / len(qvalues.rundata['actions'])
    push_freq

    ax.bar(["don't push", "push"], [1-push_freq, push_freq])
    ax.set_ylabel("Frequency")
    ax.text(0, 1 - push_freq, f"{round(1-push_freq, 3)}", ha='center', va='bottom')
    ax.text(1, push_freq, f"{round(push_freq, 3)}", ha='center', va='bottom')
    ax.set_title("Actions taken during all of training")

    plt.show()

if __name__=="__main__":
    run_and_plot()
