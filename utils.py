import numpy as np
from qpsolvers import solve_qp
import cvxpy as cp


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
    
    return H, h

def gen_constraint_matrices(x0, A, B, T, S, N, u_lb, u_ub, D = None, c_lb = None, c_ub = None,
):
    nx = A.shape[0]
    nu = B.shape[1]
    
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

def solve_condensed_mpc(x0, A, B, Q, R, P, N, u_lb, u_ub, D = None, c_lb = None, c_ub = None, with_terminal_constraint=False):
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    T, S = gen_prediction_matrices(A, B, N)
    H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)
    G, g = gen_constraint_matrices(x0, A, B, T, S, N, u_lb, u_ub, D, c_lb, c_ub)

    Gf = None
    gf = None
    if with_terminal_constraint:
        Gf = S[-dim_x:, :]
        gf = -T[-dim_x:, :] @ x0

    u_bar = solve_qp(H, h,
                     A=Gf, b=gf,
                     G=G, h=g,
                     solver = 'quadprog')

    if u_bar is None:
        print("CRITICAL: Solver failed to find a solution. The problem is infeasible.")
        # Check if constraints are the culprit by trying to solve without them:
        u_bar_test = solve_qp(H, h, solver='quadprog')
        print("Unconstrained solution exists:", u_bar_test is not None)
        return None, None  # Or handle the error gracefully


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


def solve_mpc_condensed(Ad, Bd, Q, R, P, x0, N, u_lb, u_ub):
    dim_u = Bd.shape[1]
    dim_x = Ad.shape[0]
    T, S = gen_prediction_matrices(Ad, Bd, N)
    H, h = gen_cost_matrices(Q, R, P, T, S, x0, N)

    # print("np.linalg.cond(H)", np.linalg.cond(H))
    # print("H shape", H.shape)
    # print("h shape", h.shape)

    Gu_bar, gu_bar = gen_constraint_matrices_diff(u_lb, u_ub, N)

    # print("G shape", Gu_bar.shape)
    # print("g shape", gu_bar.shape)

    u_bar = solve_qp(H, h, G=Gu_bar, h=gu_bar, solver='quadprog')
    x_bar = T @ x0 + S @ u_bar
    x_bar = x_bar.reshape((N + 1, dim_x))
    u_bar = u_bar.reshape((N, dim_u))

    return x_bar, u_bar


def gen_constraint_matrices_diff(u_lb, u_ub, N):
    dim_u = u_lb.shape[0]

    Gu = np.zeros((2 * dim_u, dim_u))
    Gu[0: dim_u, :] = np.eye(dim_u)
    Gu[dim_u:, :] = -np.eye(dim_u)

    gu = np.zeros(2 * dim_u)
    gu[0: dim_u] = u_ub
    gu[dim_u:] = -u_lb

    Gu_bar = np.kron(np.eye(N), Gu)
    gu_bar = np.kron(np.ones(N), gu)

    return Gu_bar, gu_bar

def solve_mpc(
        A, B, Q, R, P, N,
        D, c_lb, c_ub,
        x0,
        with_terminal_constraint=False
):
    dim_x = A.shape[0]
    dim_u = B.shape[1]

    x_bar = cp.Variable((N + 1, dim_x))
    u_bar = cp.Variable((N, dim_u))

    cost = 0.
    constraints = [x_bar[0, :] == x0]
    for k in range(N):
        cost += 0.5 * cp.quad_form(x_bar[k, :], Q) + 0.5 * cp.quad_form(u_bar[k, :], R)
        constraints += [x_bar[k + 1, :] == A @ x_bar[k, :] + B @ u_bar[k, :]]
        constraints += [D @ x_bar[k + 1, :] <= c_ub]
        constraints += [D @ x_bar[k + 1, :] >= c_lb]

    cost += 0.5 * cp.quad_form(x_bar[N, :], P)

    if with_terminal_constraint:
        constraints += [x_bar[N, :] == 0]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    assert prob.status == cp.OPTIMAL, "Solver failed to find optimal solution"

    return x_bar.value, u_bar.value