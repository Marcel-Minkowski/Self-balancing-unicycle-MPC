import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
#import scipy
from scipy.linalg import null_space

np.set_printoptions(precision=5, suppress=True)

"""
VARIABLE NAMES
-------------------------------------------------------------------------------
STATES:
phi: Rope angle
theta: Rolling Wheel Rolling Angle
delta: Rolling Wheel Tilting Angle
gamma: Unicycle Body Tilting Angle
beta: Balancing Wheel Rolling Angle

INPUTS:
tau_theta, tau_beta = Torques applied at theta, beta

CONSTANTS:
R: Radius of Rope Movement
r1 m1: Radius, Mass of Rolling Wheel
r2, m2: Radius, Mass of Balancing Wheel
r_rope, m_rope: Radius, Mass of Rope
L_rope,: Length of Rope
l1: Length from center of Rolling Wheel to Center of Unicycle Body
l2: Length from Center of Unicycle Body to Center of Balancing Wheel
m_rod: Mass of Unicycle Body
-------------------------------------------------------------------------------

"""

phi, delta, theta, gamma, beta = me.dynamicsymbols('phi theta delta gamma beta')
R, l1, l2, r1, r2, r3, r_rope, L_rope = sm.symbols("R l1 l2 r1 r2 r3 r_rope L_rope", real=True)
m1, m2, m3, m_rod, m_rope, g = sm.symbols("m1 m2 m3 m_rod m_rope g", real=True)
tau_theta, tau_beta = sm.symbols("tau_theta tau_beta", real=True)

t = me.dynamicsymbols._t

q = sm.Matrix([phi, delta, theta, gamma, beta])
qd = q.diff(t)


N = me.ReferenceFrame('N')
A = me.ReferenceFrame('A')
#B1 = me.ReferenceFrame('B1')
B2 = me.ReferenceFrame('B2')
B3 = me.ReferenceFrame('B3')
C = me.ReferenceFrame('C')
D = me.ReferenceFrame('D')
#E = me.ReferenceFrame('E')

A.orient_axis(N, phi, N.x)
#B1.orient_axis(A, ksi, A.z)
B2.orient_axis(A, delta, A.x)
B3.orient_axis(B2, theta, B2.y)
C.orient_axis(B2, gamma, B2.y) 
D.orient_axis(C, beta, C.x)
#E.orient_axis(C, psi, C.z)

O = me.Point('O')
PA = me.Point('PA')
PB12 = me.Point('PB12')
PB3 = me.Point('PB3')
PC = me.Point('PC')
PD = me.Point('PD')
#PE = me.Point('PE')

O.set_vel(N, 0)
PA.set_pos(O, 1/2 * L_rope * N.x + R*sm.sin(phi)*N.y + R*(1 - sm.cos(phi))*N.z)
PB12.set_pos(PA, 0)
PB3.set_pos(PB12, r1*sm.sin(theta)*B2.x + r1*B2.z)
PC.set_pos(PB3, l1*C.z)
PD.set_pos(PB3, (l1 + l2)*C.z)
#PE.set_pos(PB12, (l1 + l2)*B2.z)

#Vectors from Each Point to Origin
O_r_PA = PA.pos_from(O).express(N)
O_r_PB12 = PB12.pos_from(O).express(N)
O_r_PB3 = PB3.pos_from(O).express(N)
O_r_PC = PC.pos_from(O).express(N)
O_r_PD = PD.pos_from(O).express(N)
#O_r_PE = PE.pos_from(O).express(N)

#Velocities of each Point wrt RF N
O_r_PA_dot = O_r_PA.dt(N)
O_r_PB12_dot = O_r_PB12.dt(N)
O_r_PB3_dot = O_r_PB3.dt(N)
O_r_PC_dot = O_r_PC.dt(N)
O_r_PD_dot = O_r_PD.dt(N)
#O_r_PE_dot = O_r_PE.dt(N)

#Velocities squared
PA_squared = O_r_PA_dot.dot(O_r_PA_dot)
PB12_squared = O_r_PB12_dot.dot(O_r_PB12_dot)
PB3_squared = O_r_PB3_dot.dot(O_r_PB3_dot)
PC_squared = O_r_PC_dot.dot(O_r_PC_dot)
PD_squared = O_r_PD_dot.dot(O_r_PD_dot)
#PE_squared = O_r_PE_dot.dot(O_r_PE_dot)

#Translational Kinetic Energies
TA = sm.Rational(1, 2) * m_rope * PA_squared
TB3 = sm.Rational(1, 2) * m1 * PB3_squared
TC = sm.Rational(1, 2) * m_rod * PC_squared
TD = sm.Rational(1, 2) * m2 * PD_squared
#TE = sm.Rational(1, 2) * m3 * PE_squared

#Rotational Velocities
A_ang_vel = A.ang_vel_in(N)
#B1_ang_vel = B1.ang_vel_in(N)
B2_ang_vel = B2.ang_vel_in(N)
B3_ang_vel = B3.ang_vel_in(N)
C_ang_vel = C.ang_vel_in(N)
D_ang_vel = D.ang_vel_in(N)
#E_ang_vel = E.ang_vel_in(N)

#Inertias
I_A = sm.Rational(1, 2) * m_rope * r_rope**2
#I_B1 = sm.Rational(1, 4) * m1 * r1**2
I_B2 = sm.Rational(1, 4) * m1 * r1**2
I_B3 = sm.Rational(1, 2) * m1 * r1**2
I_C = (sm.Rational(1, 12) * m_rod * (l1 + l2)**2) * 2
I_D = m2 * r2**2
#I_E = m3 * r3**2

#Rotational Kinetic Energies
T_rot_A = I_A * A_ang_vel.dot(A_ang_vel)
T_rot_B = sm.Rational(1, 2) * I_B3 * B3_ang_vel.dot(B3_ang_vel) 
T_rot_C = sm.Rational(1, 2) * I_C * C_ang_vel.dot(C_ang_vel)
T_rot_D = sm.Rational(1, 2) * I_D * D_ang_vel.dot(D_ang_vel)
#T_rot_E = sm.Rational(1, 2) * I_E * E_ang_vel.dot(E_ang_vel)

#Potential Energies
PE_PA = m_rope * g * O_r_PA.dot(N.z)
PE_PB3 = m1 * g * O_r_PB3.dot(N.z)
PE_PC = m_rod * g * O_r_PC.dot(N.z)
PE_PD = m2 * g * O_r_PD.dot(N.z)
#PE_PE = m3 * g * O_r_PE.dot(N.z)

#Total KE and PE of system
KE_system = TA + TB3 + TC + TD  + T_rot_A + T_rot_B + T_rot_C + T_rot_D 
PE_system = PE_PA + PE_PB3 + PE_PC + PE_PD

#Lagrange for Equations of motion
L = KE_system - PE_system

dL_dq = sm.Matrix([sm.diff(L, qi) for qi in q])
dL_dqd = sm.Matrix([sm.diff(L, qdi) for qdi in qd])

M = dL_dqd.jacobian(qd) #mass matrix
Aq = dL_dqd.jacobian(q) #

Q = sm.Matrix([0, 0, tau_theta, 0, tau_beta]) #generalised forces
forcing = Q - (Aq * qd - dL_dq)

params = [R, l1, l2, r1, r2, r3, r_rope, m1, m2, m3, m_rod, m_rope, g, L_rope]
inputs = [tau_theta, tau_beta]
state_syms = list(q) + list(qd)

#Converting the system to numeric
M_num = sm.lambdify(state_syms + params, M, modules="numpy")
forcing_num = sm.lambdify(state_syms + inputs + params, forcing, modules="numpy")


def make_state_space_function(constants):
    p = [constants[s] for s in params]

    def f(x, u):
        x = np.asarray(x, dtype=float).reshape(10)
        u = np.asarray(u, dtype=float).reshape(2)

        args_M = list(x) + p
        args_F = list(x) + list(u) + p

        M_eval = np.array(M_num(*args_M), dtype=float)
        F_eval = np.array(forcing_num(*args_F), dtype=float).reshape(5)

        qd_eval = x[5:]
        qdd_eval = np.linalg.solve(M_eval, F_eval)

        return np.concatenate([qd_eval, qdd_eval])

    return f

constants = {
    R: 1.0,
    l1: 0.5,
    l2: 0.4,
    r1: 0.1,
    r2: 0.08,
    r3: 0.08,
    r_rope: 0.02,
    m1: 2.0,
    m2: 1.0,
    m3: 1.0,
    m_rod: 0.8,
    m_rope: 0.2,
    g: 9.81,
    L_rope: 10
}

f = make_state_space_function(constants)


x0 = np.zeros(10)
x0[3] = 0.1
u0 = np.array([0, 0])

xdot0 = f(x0, u0)
# print("xdot(x0,u0) =")
# print(xdot0)

###################################################################

def linearize_numerically(f, x_e, u_e, eps=1e-6):
    x_e = np.asarray(x_e, dtype=float)
    u_e = np.asarray(u_e, dtype=float)

    n = len(x_e)
    m = len(u_e)

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    f0 = np.asarray(f(x_e, u_e), dtype=float)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        f_plus = np.asarray(f(x_e + dx, u_e), dtype=float)
        f_minus = np.asarray(f(x_e - dx, u_e), dtype=float)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        f_plus = np.asarray(f(x_e, u_e + du), dtype=float)
        f_minus = np.asarray(f(x_e, u_e - du), dtype=float)
        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B


def controllability_matrix(A, B):
    n = A.shape[0]
    C = B
    AB = B
    for _ in range(1, n):
        AB = A @ AB
        C = np.hstack((C, AB))
    return C

x_equilibrium = np.zeros(10)
u_equilibrium = np.zeros(2)
A, B = linearize_numerically(f, x_equilibrium, u_equilibrium)

C = controllability_matrix(A, B)
rank_C = np.linalg.matrix_rank(C)

# print("Controllability rank:", rank_C)
# print("State dimension:", A.shape[0])

N = null_space(C.T)
# print("Uncontrollable subspace basis:\n", N)


#State Constraints

#Total rotations to stay on rope
max_rad_rolling_wheel = (L_rope/r1).evalf(subs={L_rope:10, r1:0.1})
min_rolling_angle = -(max_rad_rolling_wheel/2)
max_rolling_angle = (max_rad_rolling_wheel/2)




