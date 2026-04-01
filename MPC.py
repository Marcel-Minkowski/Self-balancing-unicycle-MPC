import numpy as np
import scipy as sp
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import linprog
from qpsolvers import solve_qp

import Test_numerical_solution as model
import utils

np.set_printoptions(precision=5, suppress=True)

A_c = np.array(model.A, dtype=float)
B_c = np.array(model.B, dtype=float)

dim_x = A_c.shape[0]
dim_u = B_c.shape[1]

dt = 0.02


def discretize_system(Ac, Bc, dt):
    nx = Ac.shape[0]
    nu = Bc.shape[1]

    ABc = np.zeros((nx + nu, nx + nu))
    ABc[:nx, :nx] = Ac
    ABc[:nx, nx:] = Bc

    expm_ABc = expm(ABc * dt)

    Ad = expm_ABc[:nx, :nx]
    Bd = expm_ABc[:nx, nx:]

    return Ad, Bd


def box_constraints(lb, ub):
    num_con = 2 * len(lb)
    A = np.kron(np.eye(len(lb)), [[1], [-1]])

    b = np.zeros(num_con)
    for i in range(num_con):
        b[i] = ub[i // 2] if i % 2 == 0 else -lb[i // 2]

    goodrows = np.logical_and(~np.isinf(b), ~np.isnan(b))
    A = A[goodrows]
    b = b[goodrows]

    return A, b


def compute_maximal_admissible_set(F, A, b, max_iter=100):
    dim_con = A.shape[0]
    A_inf_hist = []
    b_inf_hist = []

    Ft = F
    A_inf = A.copy()
    b_inf = b.copy()
    A_inf_hist.append(A_inf.copy())
    b_inf_hist.append(b_inf.copy())

    for _ in range(max_iter):
        f_obj = A @ Ft
        stop_flag = True

        for i in range(dim_con):
            res = linprog(
                c=-f_obj[i],
                A_ub=A_inf,
                b_ub=b_inf,
                bounds=[(None, None)] * A.shape[1],
                method="highs"
            )

            if not res.success:
                stop_flag = False
                break

            x = res.x
            if f_obj[i] @ x > b[i]:
                stop_flag = False
                break

        if stop_flag:
            break

        A_inf = np.vstack((A_inf, A @ Ft))
        b_inf = np.hstack((b_inf, b))
        Ft = F @ Ft

        A_inf_hist.append(A_inf.copy())
        b_inf_hist.append(b_inf.copy())

    return A_inf_hist, b_inf_hist


def find_lqr_invariant_set(A, B, K, lb_x, ub_x, lb_u, ub_u):
    A_x, b_x = box_constraints(lb_x, ub_x)
    A_u, b_u = box_constraints(lb_u, ub_u)

    A_lqr = A_u @ K
    b_lqr = b_u

    A_con = np.vstack((A_lqr, A_x))
    b_con = np.hstack((b_lqr, b_x))

    F = A + B @ K

    A_inf_hist, b_inf_hist = compute_maximal_admissible_set(F, A_con, b_con)

    return A_inf_hist, b_inf_hist


def computeX1(G, H, psi, Ad, Bd, P, gamma):
    dim_u_local = Bd.shape[1]

    G_ = np.vstack((G, P @ Ad))
    H_ = np.vstack((H, P @ Bd))
    psi_ = np.hstack((psi, -gamma))
    psi_ = np.expand_dims(psi_, axis=1)

    A, b = utils.proj_input(G_, H_, psi_, 1, dim_u_local)
    b = np.atleast_1d(-b.squeeze())

    return A, b


def computeXn(A, B, K, N, lb_x, ub_x, lb_u, ub_u):
    A_x, b_x = box_constraints(lb_x, ub_x)
    A_u, b_u = box_constraints(lb_u, ub_u)

    A_lqr = A_u @ K
    b_lqr = b_u

    A_con = np.vstack((A_lqr, A_x))
    b_con = np.hstack((b_lqr, b_x))

    F = A + B @ K

    A_inf_hist, b_inf_hist = compute_maximal_admissible_set(F, A_con, b_con)
    A_inf = A_inf_hist[-1]
    b_inf = b_inf_hist[-1]

    GH = sp.linalg.block_diag(A_x, A_u)
    G = GH[:, :dim_x]
    H = GH[:, dim_x:]
    psi = -np.hstack((b_x, b_u))

    Xns = [(A_inf, b_inf)]

    for _ in range(N):
        P, gamma = Xns[-1]
        P, gamma = computeX1(G, H, psi, A, B, P, gamma)
        Xns.append((P, gamma))

    return Xns


def build_terminal_constraint(T, S, x0, A_f, b_f):
    T_N = T[-dim_x:, :]
    S_N = S[-dim_x:, :]

    G_term = A_f @ S_N
    g_term = b_f - A_f @ (T_N @ x0)

    return G_term, g_term


def solve_mpc_with_terminal_set(
    Ad, Bd, Q, R, P, x0, N,
    u_lb, u_ub,
    theta_index, theta_min, theta_max,
    A_f, b_f
):
    T, S = utils.gen_prediction_matrices(Ad, Bd, N)
    H, h = utils.gen_cost_matrices(Q, R, P, T, S, x0, N)

    D = np.zeros((1, dim_x))
    D[0, theta_index] = 1.0

    c_lb = np.array([theta_min])
    c_ub = np.array([theta_max])

    G, g = utils.gen_constraint_matrices(
        x0=x0,
        A=Ad,
        B=Bd,
        T=T,
        S=S,
        N=N,
        u_lb=u_lb,
        u_ub=u_ub,
        D=D,
        c_lb=c_lb,
        c_ub=c_ub
    )

    G_term, g_term = build_terminal_constraint(T, S, x0, A_f, b_f)

    G_all = np.vstack((G, G_term))
    g_all = np.hstack((g, g_term))

    u_bar = solve_qp(H, h, G=G_all, h=g_all, solver="quadprog")
    if u_bar is None:
        raise ValueError("QP infeasible")

    x_bar = T @ x0 + S @ u_bar
    x_bar = x_bar.reshape((N + 1, dim_x))
    u_bar = u_bar.reshape((N, dim_u))

    return x_bar, u_bar


Ad, Bd = discretize_system(A_c, B_c, dt)

print("Ad:")
print(Ad)
print("\nBd:")
print(Bd)

N = 50

Q = np.diag([
    10.0, 10.0, 10.0, 10.0, 10.0,
    10.0, 10.0, 10.0, 10.0, 10.0
])

R = np.diag([
    0.1,
    0.1
])

P_inf = solve_discrete_are(Ad, Bd, Q, R)
K_inf = -np.linalg.solve(R + Bd.T @ P_inf @ Bd, Bd.T @ P_inf @ Ad)

print("\nP_inf:")
print(P_inf)
print("\nK_inf:")
print(K_inf)

u_min = np.array([-10.0, -10.0])
u_max = np.array([10.0, 10.0])

theta_index = 2
theta_min = -50.0
theta_max = 50.0

lb_x = -np.inf * np.ones(dim_x)
ub_x = np.inf * np.ones(dim_x)
lb_x[theta_index] = theta_min
ub_x[theta_index] = theta_max

lb_u = u_min.copy()
ub_u = u_max.copy()

A_inf_hist, b_inf_hist = find_lqr_invariant_set(
    Ad, Bd, K_inf, lb_x, ub_x, lb_u, ub_u
)
A_f = A_inf_hist[-1]
b_f = b_inf_hist[-1]

print("\nNumber of terminal-set inequalities:")
print(A_f.shape[0])

Xns = computeXn(Ad, Bd, K_inf, N, lb_x, ub_x, lb_u, ub_u)
print("\nComputed X_n sets:")
print(len(Xns))

x0 = np.zeros(dim_x)
x0[3] = 0.1

N_sim = 200
x_hist = np.zeros((N_sim + 1, dim_x))
u_hist = np.zeros((N_sim, dim_u))
x_hist[0, :] = x0

for t in range(N_sim):
    x_bar, u_bar = solve_mpc_with_terminal_set(
        Ad, Bd, Q, R, P_inf, x_hist[t, :], N,
        u_min, u_max,
        theta_index, theta_min, theta_max,
        A_f, b_f
    )

    u = u_bar[0]
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