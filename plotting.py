import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def vplot(V, max_cars):
    value = np.zeros((max_cars + 1, max_cars + 1))
    for s, v in V.items():
        value[s[0]][s[1]] = v

    fig = sns.heatmap(np.flipud(value), cmap="YlGnBu")
    fig.set_ylabel('# cars at first location', fontsize=15)
    fig.set_yticks(list(reversed(range(max_cars + 1))))
    fig.set_xlabel('# cars at second location', fontsize=15)
    fig.set_title('optimal value', fontsize=15)
    plt.savefig('Vplot.png')
    plt.close()


def piplot(π, max_cars):
    policy = np.zeros((max_cars + 1, max_cars + 1))
    for s, a in π.items():
        policy[s[0]][s[1]] = a
    
    fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu")
    fig.set_ylabel('# cars at first location', fontsize=15)
    fig.set_yticks(list(reversed(range(max_cars + 1))))
    fig.set_xlabel('# cars at second location', fontsize=15)
    plt.savefig('piplot.png',)