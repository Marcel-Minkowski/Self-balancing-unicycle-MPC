import Test_numerical_solution as model
from scipy.linalg import expm
import numpy as np
import utils as ut
from control import dare
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import cvxpy as cp

def matrix_discretization(Ac, Bc, dt = 0.1):
    dim_x = Ac.shape[0]
    dim_u = Bc.shape[1]

    ABc = np.zeros((dim_x + dim_u, dim_x + dim_u))
    ABc[:dim_x, :dim_x] = Ac
    ABc[:dim_x, dim_x:] = Bc
    expm_ABc = expm(ABc*dt)
    Ad = expm_ABc[:dim_x, :dim_x]
    Bd = expm_ABc[:dim_x, dim_x:]

    # print(f"Ad:\n{Ad}")
    # print(f"Bd:\n{Bd}")

    return Ad, Bd


#linearized state space
Ac = model.A
Bc = model.B
#Initial Condition
x0 = model.x0
#Prediction Horizon
N = 20

#dimensions
dim_x = Ac.shape[0]
dim_u = Bc.shape[1]

#Cost Matrices
Q = np.eye(10)
R = np.eye(2)

Ad, Bd = matrix_discretization(Ac, Bc)

#Terminal Ingredients
P, _, K = dare(Ad, Bd, Q, R) #terminal cost matrix

T,S = ut.gen_prediction_matrices(Ad, Bd, N)
# print("T shape", T.shape)
# print("S shape", S.shape)

H, h = ut.gen_cost_matrices(Q, R, P, T, S, x0, N)

#constraints
u_lb = np.array([-5, -5])
u_ub = np.array([5, 5])
D = np.zeros((1, dim_x))
D[0, 2] = 1.0

c_lb = -50
c_ub = 50

x_bar, u_bar = ut.solve_mpc_condensed(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub)

# print("x_bar", x_bar)
# print("u_bar", u_bar)

#Simulation for 100 time steps
N_sim = 200
x_hist = np.zeros((N_sim + 1, dim_x)) #the state we have for each step of the simulation
u_hist = np.zeros((N_sim, dim_u)) #the input applied to each step of the simulation
x_hist[0, :] = x0

#log the states and inputs of the simulation
x_sim = np.zeros((N_sim + 1, dim_x))
u_sim = np.zeros((N_sim, dim_u))

for t in range(N_sim):
    # print(f"Time step: {t}")
    # x_bar, u_bar = ut.solve_mpc_condensed(Ad, Bd, Q, R, P, x_hist[t, :], N, u_lb, u_ub)
    x_bar, u_bar = ut.solve_mpc(Ad, Bd, Q, R, P, N, D, c_lb, c_ub, x_hist[t, :], False)
    u0 = u_bar[0, :] # Take the first control input
    u_hist[t, :] = u0
    x_hist[t + 1, :] = Ad @ x_hist[t, :] + Bd @ u0 # Forward simulation

    x_sim[t] = x_hist[t + 1, :]
    u_sim[t] = u_hist[t, :]


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

