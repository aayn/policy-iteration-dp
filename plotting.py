import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def vplot(V, max_cars, suffix=''):
    """Plots heatmap of a given value function.
    
    Code taken and (slightly) modified from Shangtong Zhang's repo:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/3827b0a1fd7c5ac3e838dd61151550940a8ff73c/chapter04/car_rental.py#L156"""
    value = np.zeros((max_cars + 1, max_cars + 1))
    for s, v in V.items():
        value[s[0]][s[1]] = v

    fig = sns.heatmap(np.flipud(value))
    fig.set_ylabel('# cars at first location', fontsize=15)
    fig.set_yticks(list(reversed(range(max_cars + 1))))
    fig.set_xlabel('# cars at second location', fontsize=15)
    fig.set_title('optimal value', fontsize=15)
    plt.savefig(f'data/images/Vplot{suffix}.png')
    plt.close()


def piplot(π, max_cars, suffix=''):
    """Plots heatmap of a given policy.
    
    Code taken and (slightly) modified from Shangtong Zhang's repo:
    https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/3827b0a1fd7c5ac3e838dd61151550940a8ff73c/chapter04/car_rental.py#L156"""
    policy = np.zeros((max_cars + 1, max_cars + 1))
    for s, a in π.items():
        policy[s[0]][s[1]] = a
    
    fig = sns.heatmap(np.flipud(policy), vmin=-5, vmax=5, center=0)
    fig.set_ylabel('# cars at first location', fontsize=15)
    fig.set_yticks(list(reversed(range(max_cars + 1))))
    fig.set_xlabel('# cars at second location', fontsize=15)
    plt.savefig(f'data/images/piplot{suffix}.png')