import matplotlib.pyplot as plt
import numpy as np

import utils as ut
import MPC_attempt_3 as mpc


#Estimates the region of attraction numerically by generating random samples and seeing if the problem is still feasible
def roa_sampled(samples_n, bounds, Ad, Bd, Q, R, P, N, u_lb, u_ub, D, c_lb, c_ub, A_inf, b_inf):

    upper_bound, lower_bound = bounds #limits of the calculations

    #generate 10k random combinations of initial states
    x0_samples = np.random.uniform(lower_bound, upper_bound, (samples_n, 10))

    feasible_results = []
    unfeasible_results = []
    i = 0 #counter to know the progress of the loop
    for x0 in x0_samples:
        print("i =", i)
        #solve the MPC for each set of initial conditions
        x_bar, u_bar = ut.solve_condensed_mpc(x0, Ad, Bd, Q, R, P, N, u_lb, u_ub, D, c_lb, c_ub, True, A_inf, b_inf)
        if u_bar is not None:
            feasible_results.append(x0)  # This is a feasible point
        else:
            unfeasible_results.append(x0)

        i += 1

    print("feasible results length", len(feasible_results))
    print("unfeasible results length", len(unfeasible_results))


    return feasible_results, unfeasible_results

#creates 2d projections of the RoA of two states at a time as calculated from the above fuction
def plot_RoA_projections(feasible_results, unfeasible_results, state1, state2):
    feat = np.array(feasible_results)
    unfeat = np.array(unfeasible_results)

    plt.figure(figsize=(8, 6))

    if unfeat.size > 0:
        plt.scatter(unfeat[:, state1], unfeat[:, state2], color='red', alpha=0.3, label='Unfeasible')
    if feat.size > 0:
        plt.scatter(feat[:, state1], feat[:, state2], color='green', marker='x', label='Feasible')

    plt.xlabel(f"State {state1}")
    plt.ylabel(f"State {state2}")
    plt.legend()
    plt.show()


bounds = [-1, 1] #radians
samples_n = 500

feasible_results, unfeasible_results = roa_sampled(samples_n, bounds, mpc.Ad, mpc.Bd, mpc.Q, mpc.R, mpc.P, mpc.N, mpc.u_lb, mpc.u_ub, mpc.D, mpc.c_lb, mpc.c_ub, mpc.A_inf, mpc.b_inf)
plot_RoA_projections(feasible_results, unfeasible_results, 5, 6)

