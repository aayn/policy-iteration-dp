from itertools import product
from functools import partial
import numpy as np
import utils as λ


def states(max_cars):
    "Gives all possible states given the maximum number of cars."
    return λ.product_t(range(max_cars + 1), range(max_cars + 1))


def actions(s, max_cars_moved=5):
    """Gives allowed actions for a given state.
    
    s: state.
    max_cars_moved: Maximum no. of cars that can be moved from one
    place to another."""
    return range(max(-s[1], -max_cars_moved), min(max_cars_moved + 1, s[0] + 1))

# Discount factor
GAMMA = 0.9
# Maximum value of Poisson random variables. Values that are greater
# or equal to POISSON_THRESHOLD have a probability of 0.
POISSON_THRESHOLD = 11


def state_value_return(s, a, V, ex_4_7=False):
    """Calculates V(s) as the discounted sum over all possible rewards
    and next states.
    
    Set ex_4_7 to True if using for exercise 4.7."""
    if ex_4_7:
        s, a = employee_help(s, a)
    s, cost = act(s, a)
    parking_cost = 0
    init_cars1, init_cars2 = s
    returns = 0.0

    # Each iteration has 4 values corresponding to (cars requested at location
    # 1, cars requested at location 2, cars returned at location 1, cars
    # returned at location 2).
    for req_loc1, req_loc2, ret_loc1, ret_loc2 in product(*([range(POISSON_THRESHOLD)] * 4)):
        cars_given_loc1 = min(req_loc1, init_cars1)
        cars_given_loc2 = min(req_loc2, init_cars2)
        
        money_earned = 10 * (cars_given_loc1 + cars_given_loc2)

        cars_loc1 = min(init_cars1 - cars_given_loc1 + ret_loc1, 20)
        cars_loc2 = min(init_cars2 - cars_given_loc2 + ret_loc2, 20)
        
        s_ = (cars_loc1, cars_loc2)
        if ex_4_7:
            parking_cost = parking_lot_cost(s_)

        prob = λ.poisson(req_loc1, l=3) * λ.poisson(req_loc2, l=4) * λ.poisson(ret_loc1, l=3) * λ.poisson(ret_loc2, l=2)
        returns += prob * (money_earned - cost - parking_cost + GAMMA * V[s_])
    
    return returns


def act(s, a):
    "Returns new state and action, given a state and action."
    s1 = λ.first(s) - a
    s2 = λ.second(s) + a
    return (s1, s2), 2 * abs(a)


def employee_help(s, a):
    "Use employee to send one car to location 2 for free."
    s1, s2 = s
    if a > 0 and s1 > 0:
        s2 += 1
        s1 -= 1
        a -= 1
    return (s1, s2), a


def parking_lot_cost(s):
    "Calculate additional cost for extra parking lots."
    additional_cost = sum(map(lambda num_cars: 4 if num_cars > 10 else 0, s))
    return additional_cost
