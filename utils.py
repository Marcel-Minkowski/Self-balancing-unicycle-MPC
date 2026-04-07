import numpy as np
from qpsolvers import solve_qp
import cvxpy as cp
from scipy.linalg import expm
from lqr_set import remove_redundant_constraints
import scipy as sp
import geopandas as gpd
from scipy.optimize import linprog
import scipy


def dimensions(A, B):
    return A.shape[0], B.shape[1]

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

def gen_prediction_matrices(Ad, Bd, N):
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]
    
    T = np.zeros(((dim_x * (N + 1), dim_x)))
    S = np.zeros(((dim_x * (N + 1), dim_u * N)))

    # Condensing
    power_matricies = []    # power_matricies = [I, A, A^2, ..., A^N]
    power_matricies.append(np.eye(dim_x))
    for k in range(N):
        power_matricies.append(power_matricies[k] @ Ad)

    for k in range(N + 1):
        T[k * dim_x: (k + 1) * dim_x, :] = power_matricies[k]
        for j in range(N):
            if k > j:
                S[k * dim_x:(k + 1) * dim_x, j * dim_u:(j + 1) * dim_u] = power_matricies[k - j - 1] @ Bd

    return T, S

def gen_cost_matrices(Q, R, P, T, S, x0, N):
    dim_x = Q.shape[0]
    
    Q_bar = np.zeros(((dim_x * (N + 1), dim_x * (N + 1))))
    Q_bar[-dim_x:, -dim_x:] = P
    Q_bar[:dim_x * N, :dim_x * N] = np.kron(np.eye(N), Q) 
    R_bar = np.kron(np.eye(N), R)
    
    H = S.T @ Q_bar @ S + R_bar
    h = S.T @ Q_bar @ T @ x0
    
    H = 0.5 * (H + H.T) # Ensure symmetry!
    H += np.eye(H.shape[0]) * 1e-8 #ensure positive definitness
    # print("eigenvalues = np.linalg.eigvals(h)", np.linalg.eigvals(H))

    return H, h

def gen_constraint_matrices(x0, A, B, T, S, N, u_lb, u_ub, D = None, c_lb = None, c_ub = None,
):
    nx = A.shape[0] #dimensions of states
    nu = B.shape[1] #dimensions of inputs

    # Input constraints
    Iu = np.eye(nu * N)
    Gu = np.vstack([Iu, -Iu])
    gu = np.hstack([
         np.kron(np.ones(N), u_ub),
        -np.kron(np.ones(N), u_lb)
    ])
    
    # State constraints
    if D is not None:
        T_tilde = T[:-nx, :]
        S_tilde = S[:-nx, :]
        D_tilde = np.kron(np.eye(N), D)

        DS_tilde = D_tilde @ S_tilde
        Gx = np.vstack([DS_tilde, -DS_tilde])
        DT_tilde = D_tilde @ T_tilde

        gx = np.hstack([
             np.kron(np.ones(N), c_ub) - DT_tilde @ x0,
            -np.kron(np.ones(N), c_lb) + DT_tilde @ x0
        ])
        G = np.vstack([Gu, Gx])
        g = np.hstack([gu, gx])
    else:
        G = Gu
        g = gu
    
    return G, g

def solve_condensed_mpc(x0, A, B, Q, R, P, N, u_lb, u_ub, D = None, c_lb = None, c_ub = None, with_terminal_constraint=False, A_inf = None, b_inf = None):
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    T, S = gen_prediction_matrices(A, B, N)
    H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)
    G, g = gen_constraint_matrices(x0, A, B, T, S, N, u_lb, u_ub, D, c_lb, c_ub)

    Gf = None
    gf = None
    if with_terminal_constraint:
        #from chat
        SN = S[-dim_x:, :] #S at step N
        TN = T[-dim_x:, :] #T at step N

        # Define as inequalities
        G_term = A_inf @ SN #projecting the state constraints onto inputs
        g_term = b_inf.flatten() - (A_inf @ TN @ x0).flatten()

        # Append to the path constraints G and g
        G = np.vstack([G, G_term])
        g = np.hstack([g, g_term])

    u_bar = solve_qp(H, h, G=G, h=g, solver='quadprog')

        # SN = S[-dim_x:, :]  # rows corresponding to x_N
        # TN = T[-dim_x:, :]  # rows corresponding to x_N
        #
        # Gf = A_inf @ SN
        # gf = b_inf - A_inf @ TN @ x0

        # print("SN.shape:", SN.shape)
        # print("A_inf.shape:", A_inf.shape)

        # Gf = S[-dim_x:, :]
        # gf = -T[-dim_x:, :] @ x0

    # print("Gf.shape:", Gf.shape)
    # print("gf.shape:", gf.shape)
    # print("G.shape:", G.shape)
    # print("g.shape:", g.shape)

    # u_bar = solve_qp(H, h,
    #                  A=Gf, b=gf,
    #                  G=G, h=g,
    #                  solver = 'quadprog')

    # print("Just calculated u bar and it is ", u_bar)

    if u_bar is None:
        print("CRITICAL: Solver failed to find a solution. The problem is infeasible.")
        # Check if constraints are the culprit by trying to solve without them:
        u_bar_test = solve_qp(H, h, solver='quadprog')
        print("Unconstrained solution exists:", u_bar_test is not None)
        return None, None

    x_bar = T @ x0 + S @ u_bar
    x_bar = x_bar.reshape((N + 1, dim_x))
    u_bar = u_bar.reshape((N, dim_u))

    return x_bar, u_bar

def remove_zero_rows(A, b):
    """
    Removes rows of A that are all zeros and the corresponding elements in b.
    """
    keeprows = np.any(A, axis=1)
    A = A[keeprows, :]
    b = b[keeprows]
    
    return A, b 

def proj_single_input(G, H, psi):
    # Define the sets by basing on the i-th column of H
   
    I_0 = np.where(H == 0)[0]
    I_p = np.where(H > 0)[0]
    I_m = np.where(H < 0)[0]

    # Set the row of matrix [P gamma]

    # Define C
    C = np.hstack((G, psi))

    # Define row by row [P gamma]
    aux = []
    for i in I_0:
        aux.append(C[i])

    for i in I_p:
        for j in I_m:
            aux.append(H[i]*C[j] - H[j]*C[i])

    # Return the desired matrix/vector
    aux = np.array(aux)
    P = aux[:,:-1]
    gamma = aux[:,[-1]]
    
    P, gamma = remove_zero_rows(P, gamma)

    return P, gamma

def fm_elim(G, H, psi):
    '''
    Performs one step of Fourier-Motzkin elimination for inequality constraints.
    '''
    I0 = np.where(H == 0)[0]
    Ip = np.where(H > 0)[0]
    Im = np.where(H < 0)[0]
    
    E = np.hstack([G, psi])    
    Ee = np.vstack([E[I0, :], np.kron(H[Ip], E[Im, :]) - np.kron(E[Ip, :], H[Im])])
    P = Ee[:, :-1]
    gamma = np.expand_dims(Ee[:, -1], axis=1)
    P, gamma = remove_zero_rows(P, gamma)
    
    return P, gamma

def proj_input(G, H, psi, N, dim_u):
    G_i = np.hstack((G, H[:,:-1]))
    H_i = np.expand_dims(H[:,-1], axis=1)
    psi_i = psi

    for i in range(N * dim_u, 0, -1):
        P_i, gamma_i = proj_single_input(G_i, H_i, psi_i)
        # P_i, gamma_i = fm_elim(G_i, H_i, psi_i)

        G_i = P_i[:,:-1]
        H_i = np.expand_dims(P_i[:,-1], axis=1)
        psi_i = gamma_i

    return P_i, gamma_i


# def solve_mpc_condensed(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub):
#     dim_u = Bd.shape[1]
#     dim_x = Ad.shape[0]
#     T, S = gen_prediction_matrices(Ad, Bd, N)
#     H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)
#
#     # print("np.linalg.cond(H)", np.linalg.cond(H))
#     # print("H shape", H.shape)
#     # print("h shape", h.shape)
#
#     Gu_bar, gu_bar = gen_constraint_matrices_diff(u_lb, u_ub, N)
#
#     # print("G shape", Gu_bar.shape)
#     # print("g shape", gu_bar.shape)
#
#     u_bar = solve_qp(H, h, G=Gu_bar, h=gu_bar, solver='quadprog')
#     x_bar = T @ x0 + S @ u_bar
#     x_bar = x_bar.reshape((N + 1, dim_x))
#     u_bar = u_bar.reshape((N, dim_u))
#
#     return x_bar, u_bar


# def gen_constraint_matrices_diff(u_lb, u_ub, N):
#     dim_u = u_lb.shape[0]
#
#     Gu = np.zeros((2 * dim_u, dim_u))
#     Gu[0: dim_u, :] = np.eye(dim_u)
#     Gu[dim_u:, :] = -np.eye(dim_u)
#
#     gu = np.zeros(2 * dim_u)
#     gu[0: dim_u] = u_ub
#     gu[dim_u:] = -u_lb
#
#     Gu_bar = np.kron(np.eye(N), Gu)
#     gu_bar = np.kron(np.ones(N), gu)
#
#     return Gu_bar, gu_bar

# def solve_mpc(
#         A, B, Q, R, P, N,
#         D, c_lb, c_ub,
#         x0,
#         with_terminal_constraint=False
# ):
#     dim_x = A.shape[0]
#     dim_u = B.shape[1]
#
#     x_bar = cp.Variable((N + 1, dim_x))
#     u_bar = cp.Variable((N, dim_u))
#
#     cost = 0.
#     constraints = [x_bar[0, :] == x0]
#     for k in range(N):
#         cost += 0.5 * cp.quad_form(x_bar[k, :], Q) + 0.5 * cp.quad_form(u_bar[k, :], R)
#         constraints += [x_bar[k + 1, :] == A @ x_bar[k, :] + B @ u_bar[k, :]]
#         constraints += [D @ x_bar[k + 1, :] <= c_ub]
#         constraints += [D @ x_bar[k + 1, :] >= c_lb]
#
#     cost += 0.5 * cp.quad_form(x_bar[N, :], P)
#
#     if with_terminal_constraint:
#         constraints += [x_bar[N, :] == 0]
#
#     prob = cp.Problem(cp.Minimize(cost), constraints)
#     prob.solve()
#
#     assert prob.status == cp.OPTIMAL, "Solver failed to find optimal solution"
#
#     return x_bar.value, u_bar.value

# def stability_check(A, B, K):
#     #open loop stability
#     open_loop_eigenvalues = np.linalg.eig(A)[0]
#     #closed loop stability
#     A_cl = A - B @ K
#     closed_loop_eigenvalues = np.linalg.eig(A_cl)[0]
#
#     print("Open Loop Eigenvalues:")
#     print(open_loop_eigenvalues)
#     print("Closed Loop Eigenvalues:")
#     print(closed_loop_eigenvalues)

"""
TERMINAL SET FUNCTIONS
"""
def terminal_set(A, B, K, lb_x, lb_u, ub_x, ub_u):
    A_x, b_x = box_constraints(lb_x, ub_x)
    A_u, b_u = box_constraints(lb_u, ub_u)

    A_lqr = A_u @ K
    b_lqr = b_u

    A_con = np.vstack((A_lqr, A_x))
    b_con = np.hstack((b_lqr, b_x))

    A_inf_hist, b_inf_hist = find_lqr_invariant_set(A, B, K, lb_x, ub_x, lb_u, ub_u)
    A_inf, b_inf = A_inf_hist[-1], b_inf_hist[-1] #take only the constraints when the set has converged

    # _, A_inf, b_inf, _, _ = remove_redundant_constraints(A_inf_hist[-1], b_inf_hist[-1])

    #convert to arrays
    A_inf = np.atleast_2d(A_inf_hist[-1])
    b_inf = np.atleast_1d(b_inf_hist[-1])

    return A_inf, b_inf


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


def find_lqr_invariant_set(A, B, K, lb_x, ub_x, lb_u, ub_u):
    A_x, b_x = box_constraints(lb_x, ub_x)
    A_u, b_u = box_constraints(lb_u, ub_u)

    A_lqr = A_u @ K
    b_lqr = b_u

    A_con = np.vstack((A_lqr, A_x))
    b_con = np.hstack((b_lqr, b_x))

    F = A - B @ K
    # print("eigenvalues of F", np.linalg.eigvals(F))

    A_inf_hist, b_inf_hist = compute_maximal_admissible_set(F, A_con, b_con)

    return A_inf_hist, b_inf_hist


def compute_maximal_admissible_set(F, A, b, max_iter=100):
    '''
    Compute the maximal admissible set for the system x_{t+1} = F x_t subject to A x_t <= b.
    '''
    dim_con = A.shape[0]
    A_inf_hist = []
    b_inf_hist = []

    Ft = F
    A_inf = A
    b_inf = b
    A_inf_hist.append(A_inf)
    b_inf_hist.append(b_inf)

    for t in range(max_iter):
        f_obj = A @ Ft
        stop_flag = True
        for i in range(dim_con):
            n = A_inf.shape[1]
            bounds = [(-1e6, 1e6)] * n

            res = linprog(-f_obj[i], A_ub=A_inf, b_ub=b_inf, method="highs")

            if not res.success:
                print("LP failed:", res.message)
                print("Status:", res.status)
                print("f_obj[i]:", f_obj[i])
                raise ValueError("Linear program failed")

            x = res.x

            if f_obj[i] @ x > b[i]:
            # x = linprog(-f_obj[i], A_ub=A_inf, b_ub=b_inf, method="highs")["x"]
            # print("x", x)
            # # x = solve_qp(np.zeros((2, 2)), -f_obj[i], A_inf, b_inf, solver="") # Actually, this is not a QP, but a LP. It is better to use a LP solver.
            # if f_obj[i] @ x > b[i]:
                stop_flag = False
                break

        if stop_flag:
            break

        A_inf = np.vstack((A_inf, A @ Ft))
        b_inf = np.hstack((b_inf, b))
        Ft = F @ Ft
        A_inf_hist.append(A_inf)
        b_inf_hist.append(b_inf)

    return A_inf_hist, b_inf_hist

#REGION OF ATTRACTION FUNCTIONS
def computeX1(G, H, psi, Ad, Bd, P, gamma):  # TODO: Add support for the point constraint
    '''
    Computes the feasible set X_1 for the system x^+ = Ax + Bu subject to constraints Gx + Hu <= psi and x^+ \in Xf.
    '''
    dim_u = Bd.shape[1]

    # print("G shape", G.shape)
    # print("P length", len(P))
    # print("P[0] shape", P[0].shape)
    # print("Ad shape", Ad.shape)
    # print("P@Ad shape", (P @ Ad).shape)
    G_ = np.vstack((G, P @ Ad))
    H_ = np.vstack((H, P @ Bd))
    psi_ = np.hstack((psi, -gamma))

    psi_ = np.expand_dims(psi_, axis=1)
    print("stuck here")
    A, b = proj_input(G_, H_, psi_, 1, dim_u)
    b = -b.squeeze()
    print("unstuck")
    return A, b


def computeXn(A, B, K, N, lb_x, ub_x, lb_u, ub_u):
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    A_x, b_x = box_constraints(lb_x, ub_x)
    A_u, b_u = box_constraints(lb_u, ub_u)

    A_lqr = A_u @ K
    b_lqr = b_u

    A_con = np.vstack((A_lqr, A_x))
    b_con = np.hstack((b_lqr, b_x))

    F = A - B @ K

    A_inf_hist, b_inf_hist = compute_maximal_admissible_set(F, A_con, b_con)
    A_inf, b_inf = A_inf_hist[-1], b_inf_hist[-1] #take only the converged constraints
    # _, A_inf, b_inf, _, _ = remove_redundant_constraints(A_inf_hist[-1], b_inf_hist[-1])

    GH = sp.linalg.block_diag(A_x, A_u)
    G = GH[:, :dim_x]
    H = GH[:, dim_x:]
    psi = -np.hstack((b_x, b_u))

    # Xns = [(A_inf_hist[-1], b_inf_hist[-1])]
    Xns = [(A_inf, b_inf)]

    for i in range(N):
        print("i =", i)
        P, gamma = Xns[-1]
        print("P", P.shape)
        print("gamma", gamma.shape)
        P, gamma = computeX1(G, H, psi, A, B, P, gamma)
        # _, P, gamma, _, _ = remove_redundant_constraints(P, gamma)
        Xns.append((P, gamma))
        i += 1

    return Xns
