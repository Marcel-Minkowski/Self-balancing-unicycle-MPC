import numpy as np
from scipy.linalg import expm
from quadprog import solve_qp
import Test_numerical_solution as model

np.set_printoptions(precision=5, suppress=True)


# Import the continuous-time linearized system from your other file
A_c = np.array(model.A, dtype=float)
B_c = np.array(model.B, dtype=float)

# State and input dimensions
dim_x = A_c.shape[0]
dim_u = B_c.shape[1]

# Sampling time
dt = 0.02


# Discretize the continuous-time system using the same trick as in the course
def discretize_system(Ac, Bc, dt):
    ABc = np.zeros((dim_x + dim_u, dim_x + dim_u))
    ABc[:dim_x, :dim_x] = Ac
    ABc[:dim_x, dim_x:] = Bc

    expm_ABc = expm(ABc * dt)

    Ad = expm_ABc[:dim_x, :dim_x]
    Bd = expm_ABc[:dim_x, dim_x:]

    return Ad, Bd


# Generate the condensed prediction matrices
# X_bar = T x0 + S U_bar
def gen_prediction_matrices(Ad, Bd, N):
    T = np.zeros((dim_x * (N + 1), dim_x))
    S = np.zeros((dim_x * (N + 1), dim_u * N))

    power_matrices = [np.eye(dim_x)]
    for k in range(N):
        power_matrices.append(power_matrices[k] @ Ad)

    for k in range(N + 1):
        T[k * dim_x:(k + 1) * dim_x, :] = power_matrices[k]

        for j in range(N):
            if k > j:
                S[k * dim_x:(k + 1) * dim_x, j * dim_u:(j + 1) * dim_u] = \
                    power_matrices[k - j - 1] @ Bd

    return T, S


# Build the quadratic cost matrices
def gen_cost_matrices(Q, R, P, T, S, x0, N):
    Q_bar = np.zeros((dim_x * (N + 1), dim_x * (N + 1)))
    Q_bar[:dim_x * N, :dim_x * N] = np.kron(np.eye(N), Q)
    Q_bar[-dim_x:, -dim_x:] = P

    R_bar = np.kron(np.eye(N), R)

    H = S.T @ Q_bar @ S + R_bar
    f = S.T @ Q_bar @ T @ x0

    return H, f


# Build a selector matrix that extracts one chosen state from every predicted state vector
# If theta_index = 2, then this extracts theta from x0, x1, ..., xN
def build_state_selector(state_index, N):
    E = np.zeros((N + 1, dim_x * (N + 1)))

    for k in range(N + 1):
        E[k, k * dim_x + state_index] = 1.0

    return E


# Build inequality constraints in the form:
# A_ineq * U_bar <= b_ineq
def gen_constraint_matrices(T, S, x0, N, u_min, u_max, theta_index, theta_min, theta_max):
    nU = dim_u * N

    # ------------------------------------------------------------
    # 1. Input constraints:
    #    u_min <= u_k <= u_max   for all k
    #
    # Stacked form:
    #    U_bar <= U_max_bar
    #   -U_bar <= -U_min_bar
    # ------------------------------------------------------------
    Iu = np.eye(nU)

    u_min_bar = np.tile(u_min, N)
    u_max_bar = np.tile(u_max, N)

    A_u = np.vstack([
        Iu,
        -Iu
    ])

    b_u = np.hstack([
        u_max_bar,
        -u_min_bar
    ])

    # ------------------------------------------------------------
    # 2. Theta state constraints:
    #    theta_min <= theta_k <= theta_max   for all predicted steps
    #
    # We know:
    #    X_bar = T x0 + S U_bar
    #
    # Extract theta from all predicted states:
    #    theta_bar = E X_bar = E(T x0 + S U_bar)
    #
    # So:
    #    E S U_bar <= theta_max - E T x0
    #   -E S U_bar <= -(theta_min - E T x0)
    # ------------------------------------------------------------
    E = build_state_selector(theta_index, N)

    ETx0 = E @ (T @ x0)
    ES = E @ S

    theta_max_vec = theta_max * np.ones(N + 1)
    theta_min_vec = theta_min * np.ones(N + 1)

    A_theta = np.vstack([
        ES,
        -ES
    ])

    b_theta = np.hstack([
        theta_max_vec - ETx0,
        -(theta_min_vec - ETx0)
    ])

    # ------------------------------------------------------------
    # Combine all inequality constraints
    # ------------------------------------------------------------
    A_ineq = np.vstack([
        A_u,
        A_theta
    ])

    b_ineq = np.hstack([
        b_u,
        b_theta
    ])

    return A_ineq, b_ineq


# Solve the constrained MPC problem
def solve_mpc(Ad, Bd, Q, R, P, x0, N, u_min, u_max, theta_index, theta_min, theta_max):
    T, S = gen_prediction_matrices(Ad, Bd, N)
    H, f = gen_cost_matrices(Q, R, P, T, S, x0, N)

    A_ineq, b_ineq = gen_constraint_matrices(
        T, S, x0, N, u_min, u_max, theta_index, theta_min, theta_max
    )

    # quadprog expects:
    #   min 1/2 z^T G z - a^T z
    #   subject to C^T z >= b
    #
    # Our inequalities are in the form:
    #   A_ineq z <= b_ineq
    #
    # Convert:
    #   -A_ineq z >= -b_ineq
    #
    # So:
    #   C^T = -A_ineq
    #   C = -A_ineq.T
    #   b = -b_ineq
    H = 0.5 * (H + H.T)
    C = -A_ineq.T
    b = -b_ineq

    sol = solve_qp(H, -f, C, b)[0]

    u = sol[:dim_u]
    u_bar = sol.reshape((N, dim_u))

    return u, u_bar


# Discretize your system
Ad, Bd = discretize_system(A_c, B_c, dt)

print("Ad:")
print(Ad)
print("\nBd:")
print(Bd)


# MPC parameters
N = 20

# State cost
Q = np.diag([
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0
])

# Input cost
R = np.diag([
    0.1,
    0.1
])

# Terminal cost
P = 10.0 * Q


# ------------------------------------------------------------
# Constraints
# ------------------------------------------------------------

# Input constraints:
# u = [tau_theta, tau_beta]
u_min = np.array([-10.0, -10.0])
u_max = np.array([ 10.0,  10.0])

# State constraint on theta
# State order is:
# [phi, delta, theta, gamma, beta, phi_dot, delta_dot, theta_dot, gamma_dot, beta_dot]
theta_index = 2
theta_min = -50.0
theta_max = 50.0


# Initial condition
x0 = np.zeros(dim_x)
x0[3] = 0.1


# Closed-loop simulation
N_sim = 200
x_hist = np.zeros((N_sim + 1, dim_x))
u_hist = np.zeros((N_sim, dim_u))
x_hist[0, :] = x0

for t in range(N_sim):
    u, u_bar = solve_mpc(
        Ad, Bd, Q, R, P, x_hist[t, :], N,
        u_min, u_max,
        theta_index, theta_min, theta_max
    )

    u_hist[t, :] = u
    x_hist[t + 1, :] = Ad @ x_hist[t, :] + Bd @ u


print("\nFirst control input:")
print(u_hist[0, :])

print("\nFinal state:")
print(x_hist[-1, :])

print("\nMaximum theta during simulation:")
print(np.max(x_hist[:, theta_index]))

print("\nMinimum theta during simulation:")
print(np.min(x_hist[:, theta_index]))