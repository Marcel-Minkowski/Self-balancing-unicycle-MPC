import numpy as np
import Test_numerical_solution as model
from utils import solve_condensed_mpc
from my_utils import discretize_system
#from my_utils import solve_condensed_mpc
import matplotlib.pyplot as plt
from control import dare



# Import the continuous-time linearized system from your other file
A_c = np.array(model.A, dtype=float)
B_c = np.array(model.B, dtype=float)

# State and input dimensions
dim_x = A_c.shape[0]
dim_u = B_c.shape[1]

dt = 0.02
Ad, Bd = discretize_system(A_c, B_c, dt)


x0 = np.array([0.5, 0.5])
N = 3


alpha = 0.2
Q = alpha * np.eye(2)
R = np.eye(2)

P_inf, _, K_inf = dare(Ad, Bd, Q, R)
K_inf = -K_inf


u_lb = np.array([-1., -1.])
u_ub = np.array([ 1.,  1.])
D = np.array([[1., 0.], [0., 0.]])
c_lb = np.array([-np.inf, -np.inf])
c_ub = np.array([5.,       np.inf])

N_sim = 30
x_hist = np.zeros((N_sim + 1, dim_x))
u_hist = np.zeros((N_sim, dim_u))
x_hist[0, :] = x0

for t in range(N_sim):
    x_bar, u_bar = solve_condensed_mpc(x_hist[t, :], A, B, Q, R, P, N, u_lb, u_ub, D, c_lb, c_ub)
    x_hist[t + 1, :] = A @ x_hist[t, :] + B @ u_bar[0, :]
    u_hist[t, :] = u_bar[0, :]
    

### Plotting ###

plt.figure()
plt.plot(x_hist[:, 0], x_hist[:, 1], '-o', label='Closed-loop trajectory', markersize=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid()
plt.show()