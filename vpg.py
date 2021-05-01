from jax import random, nn, grad, jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

ACTION_SPACE = jnp.array([1, -1]) # [PUSH, DONT PUSH]


def rep_action(action):
    """Transform action space.
    Maps 1 --> 1 and -1 --> 0"""
    return (action + 1) // 2


def act(rngkey, theta):
    """
    returns:
        action, one of 1 (push) or -1 (don't push)
    """
    prob_push = nn.sigmoid(theta)
    probs = jnp.array([prob_push, 1-prob_push])
    return random.choice(rngkey, ACTION_SPACE, p=probs)


# def alt_reward(previous_action, action):
#     return -action + 10 * previous_action # different from how I described it in the doc


def reward(previous_action, action):
    action = rep_action(action)
    previous_action = rep_action(previous_action)
    return -action + 10*previous_action


@jit
def run_episode(rngkey, theta, previous_action):
    a = act(rngkey, theta)
    r = reward(previous_action, a)
    return a, r


def action_logprob(theta, action):
    """returns log-likelihood of action
    given theta: p_\theta(a)"""
    return nn.log_sigmoid(theta*action) 


@jit
def single_ep_policy_grad(theta, action, reward):
    return reward * grad(action_logprob)(theta, action)


def append_to_log(dct, update_dict):
    """appends update_dict to dict, entry-wise. Creates list entry
    if it doesn't exist.
    """
    for key, newvalue in update_dict.items():
        dct.setdefault(key, []).append(newvalue)
    return dct


def update_baseline(baseline, reward):
    """weighted avg of empirical rewards"""
    return baseline + 0.05 * (reward - baseline)


def run_and_plot():
    key = random.PRNGKey(0)
    eta = .01
    theta = 3.
    a = 1.
    baseline = 0


    print("Running...")
    data = {}
    for _ in range(10000):
        key, subkey = random.split(key)
        a, r = run_episode(subkey, theta, a)
        policy_grad = single_ep_policy_grad(theta, a, r - baseline)
        baseline = update_baseline(baseline, r)

        aux = {
            "action": a,
            "reward": r,
            "theta": theta,
            "grad": policy_grad,
        }
        
        theta = theta + eta * policy_grad
        append_to_log(data, aux)


    # plotting
    fig, axs = plt.subplots(2, 1, figsize=[8, 8])
    axs = axs.flatten()

    axs[0].plot(data['theta'], label='theta')
    axs[1].plot(nn.sigmoid(data['theta']), label='prob(push)')
    for ax in axs:
        ax.legend()
    plt.show()

    return


if __name__=="__main__":
    run_and_plot()
