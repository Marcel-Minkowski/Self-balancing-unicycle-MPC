week2 formulation

get Ad and Bd in discrete form

Define horizon length N

Get T and S, using T, S = gen_prediction_matrices(Ad, Bd, N)

Define Q and R

Define Q_bar and R_bar (uses np.kron)

Calculate H

_________________________________________ week3 projection

Define constraints. There are two constrains - on states and inputs. They take the form:
A*x_k <= b  (But naming here is very misleding, so I prefer to use different names than A and b)
C*u_k <= d  (But naming here is very misleding, so I prefer to use different names than C and d)
So define constraints, by defining vector (array) A, b, C, d.

Define A_bar, b_bar, C_bar, d_bar (matrices composed of matrices) so that it can be used for many states and not only for single one.SyntaxWarning

Define G, H and psi, using A_bar, b_bar, C_bar, d_bar. This is just to have a different representation of the constraints. Just another form:
G*x_0 + H*U *psi <= 0

Next, you try to sort out the inequlities from above. To do that you only use proj_input which projects the inequalities in terms of pure input. 
You only use proj_input function, but this function uses proj_single_input, which uses remove_zero_rows. But you just do: P, gamma = proj_input(G, H, psi, N, dim_u)


_______________________________ week3 projection

Define Ad, Bd, inital conditions, horizon length

Define Q, R, P matrices (for P here they used 0)

Define input and state constraints. u_lb, u_ub for inputs. D, c_lb, c_up for states.
    
Define simulation length N_sim

Create storage for states and inputs: u_hist, x_hist
    
for t in N_sim: compute MPC (solve_condensed_mpc function)

solve_condensed_mpc solves mpc problem for the current state. 

If you want to impose terminal constraint too in this approach, you can set with_termianl_set = True, in the solve_condensed_mpc
What this does is, it ensures that by the end of the prediction horizon the system will be in equlibrium position (x_N = 0)



