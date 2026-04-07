import Test_numerical_solution as model
import numpy as np
import utils as ut
from control import dare
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd


#linearized state space
Ac = model.A
Bc = model.B
Ad, Bd = ut.matrix_discretization(Ac, Bc)
#Initial Condition
x0 = model.x0
#Prediction Horizon
N = 40

#dimensions
dim_x = Ac.shape[0]
dim_u = Bc.shape[1]

#Cost Matrices
Q = np.eye(10)
R = np.eye(2)
R[0,0] = 10
Q[5,5] =10 #penalty on the rope velocity


#Terminal Ingredients
P, _, K = dare(Ad, Bd, Q, R) #terminal cost matrix and LQR gain

# T,S = ut.gen_prediction_matrices(Ad, Bd, N)
# H, h = ut.gen_cost_matrices(Q, R, P, T, S, x0, N)

#constraints
Torque_limit = 5
u_lb = np.array([-Torque_limit, -Torque_limit]) #Torque lower limits
u_ub = np.array([Torque_limit, Torque_limit]) #Torque upper limits
# D = np.zeros((1, dim_x)) #selector for theta constraint
# D[0, 2] = 1.0
D = np.eye(dim_x)


c_lb = -50 #theta lower constraint
c_ub = 50 #theta upper constraint
inf_bound = 1e5 #artificial constraints for unconstrained states for stability

#state constraints different format
lb_x = np.array([-inf_bound, -inf_bound, c_lb, -inf_bound, -inf_bound, -inf_bound, -inf_bound, -inf_bound, -inf_bound, -inf_bound])
ub_x = np.array([inf_bound, inf_bound, c_ub, inf_bound, inf_bound, inf_bound, inf_bound, inf_bound, inf_bound, inf_bound])

# x_bar, u_bar = ut.solve_mpc_condensed(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub)

#TERMINAL SET
A_inf, b_inf = ut.terminal_set(Ad, Bd, K, lb_x, u_lb, ub_x, u_ub)

# print("A_inf shape", A_inf.shape)
# print("B_inf shape", b_inf.shape)

#REGION OF ATTRACTION
# Xns = ut.computeXn(Ad, Bd, K, N, lb_x, ub_x, u_lb, u_ub)
# print("XNs", Xns)

#Simulation for 100 time steps
N_sim = 400
x_hist = np.zeros((N_sim + 1, dim_x)) #the measurable state for each step of the simulation
u_hist = np.zeros((N_sim, dim_u)) #the input applied to each step of the simulation
x_hist[0, :] = x0 #initial state

#log the states and inputs of the simulation
x_sim = np.zeros((N_sim + 1, dim_x))
u_sim = np.zeros((N_sim, dim_u))

#Apply MPC for N_sim steps
for t in range(N_sim):
    # print(f"Time step: {t}")
    # x_bar, u_bar = ut.solve_mpc_condensed(Ad, Bd, Q, R, P, x_hist[t, :], N, u_lb, u_ub)
    # x_bar, u_bar = ut.solve_mpc(Ad, Bd, Q, R, P, N, D, c_lb, c_ub, x_hist[t, :], False)
    # x_bar, u_bar = ut.solve_condensed_mpc(x_hist[t, :], Ad, Bd, Q, R, P, N, u_lb, u_ub, D, c_lb, c_ub, True, A_inf, b_inf)
    x_bar, u_bar = ut.solve_condensed_mpc(x_hist[t, :], Ad, Bd, Q, R, P, N, u_lb, u_ub, D, lb_x, ub_x, True, A_inf,
                                          b_inf)

    u0 = u_bar[0, :] # Take the first control input
    u_hist[t, :] = u0
    x_hist[t + 1, :] = Ad @ x_hist[t, :] + Bd @ u0 # Forward simulation

    x_sim[t] = x_hist[t + 1, :]
    u_sim[t] = u_hist[t, :]


"""EXCEL TO SEE TERMINAL SET"""
df_A = pd.DataFrame(A_inf)
df_b = pd.DataFrame(b_inf)

# Create an Excel writer
with pd.ExcelWriter('terminal_sets.xlsx', engine='xlsxwriter') as writer:
    df_A.to_excel(writer, sheet_name='A_inf', index=False)
    df_b.to_excel(writer, sheet_name='b_inf', index=False)
print("Excel File with terminal set equations created!")
"""--------------------------------------------"""

#PLOTTING (and scheming)
# Assuming x_hist and N_sim are already defined from your simulation
time_steps = np.arange(N_sim + 1)
state_labels = ["φ", "δ", "θ", "γ", "β", "φ_dot", "δ_dot", "θ_dot", "γ_dot", "β_dot"]

fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True)
fig.suptitle('Evolution of 10-Dimensional State Space', fontsize=16)
axes_flat = axes.flatten()

for i in range(dim_x):
    ax = axes_flat[i]
    ax.plot(time_steps, x_hist[:, i], label=state_labels[i], color='tab:blue', linewidth=1.5)
    ax.set_title(state_labels[i])
    ax.grid(True, linestyle='--', alpha=0.7)

    if i >= 5:
        ax.set_xlabel('Time Step')
    if i % 5 == 0:
        ax.set_ylabel('Magnitude')

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


