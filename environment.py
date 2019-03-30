from itertools import product
from functools import partial
import numpy as np
import futils as λ
from utils import poisson


def states(max_cars):
    return λ.cart_prod(range(max_cars + 1), range(max_cars + 1))


def actions(max_cars_moved):
    return tuple(range(-max_cars_moved, max_cars_moved + 1))


GAMMA = 0.9
POISSON_THRESHOLD = 11
req_loc1_p = partial(poisson, l=3)
req_loc2_p = partial(poisson, l=4)
ret_loc1_p = partial(poisson, l=3)
ret_loc2_p = partial(poisson, l=2)

def state_value_return(s, a, V):
    s, cost = act(s, a)
    init_cars1, init_cars2 = s
    returns = 0.0

    for req_loc1, req_loc2, ret_loc1, ret_loc2 in product(*([range(POISSON_THRESHOLD)] * 4)):
        cars_given_loc1 = min(req_loc1, init_cars1)
        cars_given_loc2 = min(req_loc2, init_cars2)
        
        money_earned = 10 * (cars_given_loc1 + cars_given_loc2)

        cars_loc1 = min(init_cars1 - cars_given_loc1 + ret_loc1, 20)
        cars_loc2 = min(init_cars2 - cars_given_loc2 + ret_loc2, 20)
        
        s_ = (cars_loc1, cars_loc2)

        prob = req_loc1_p(req_loc1) * req_loc2_p(req_loc2) * ret_loc1_p(ret_loc1) * ret_loc2_p(ret_loc2)
        returns += prob * (money_earned - cost + GAMMA * V[s_])
    
    return returns


def act(s, a):
    s1 = s[0] - a
    s2 = s[1] + a
    return (s1, s2), 2 * abs(a)


if __name__ == '__main__':
    V = {s: 0.0 for s in states(20)}
    print(state_value_return([20, 5], -5, V))