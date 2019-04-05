import numpy as np
import pickle
from policy import make_policy
from environment import states, state_value_return, actions
from plotting import vplot, piplot

MAX_CARS = 20
# Set to True if running for Exercise 4.7
EX_4_7 = True


def policy_iteration(model_return, theta=0.1):
    S = states(MAX_CARS)
    print(S)
    V = {s: 0.0 for s in S}
    π = make_policy(S)
    
    policy_stable = False
    while not policy_stable:
        # Evaluation
        Δ = theta + 1
        while Δ > theta:
            Δ = 0
            for s in S:
                v = V[s]
                V[s] = model_return(s, π[s], V, EX_4_7)
                Δ = max(Δ, abs(v - V[s]))
            print(Δ)

        # Policy Improvement
        policy_stable = True
        for s in S:
            old_action = π[s]
            # Valid actions
            A = actions(s)
            π[s] = max(A, key=lambda a: model_return(s, a, V))
            print(π[s], old_action)
            if old_action != π[s]:
                policy_stable = False
    
    return V, π

if __name__ == '__main__':
    V, π = policy_iteration(state_value_return)
    # with open('data/Vpi_47.pkl', 'wb') as pfile:
        # pickle.dump([V, π], pfile)
    with open('data/Vpi_47.pkl', 'rb') as pfile:
        V, π = pickle.load(pfile)
    vplot(V, MAX_CARS, suffix='_47')
    piplot(π, MAX_CARS, suffix='_47')