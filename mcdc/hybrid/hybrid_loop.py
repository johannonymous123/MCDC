import numpy as np

from numpy import ascontiguousarray as cga
from numba import njit, objmode

import mcdc.adapt as adapt
import mcdc.hybrid.hybrid_kernel as hybrid_kernel
import mcdc.kernel as kernel
import mcdc.src.geometry as geometry
import mcdc.type_ as type_

from mcdc.constant import *
from mcdc.print_ import (
    print_error,
    print_hybrid_eigenvalue_exit_code,
    print_hybrid_eigenvalue_progress,
    print_msg,
    print_progress,
    print_progress_hybrid,
)
from mcdc.type_ import hybrid_score_list


# =========================================================================
# Validate inputs
# =========================================================================


def hybrid_validate_inputs(input_deck):
    hybrid = input_deck.technique["hybrid"]
    eigenmode = input_deck.setting["mode_eigenvalue"]

    # Batched mode has only been built for eigenvalue problems (so far)
    if hybrid["mode"] == "batched" and not eigenmode:
        print_error(
            "Invalid run mode. hybridMC batched mode has not been built for fixed source problems."
        )

    # Check fixed source solver
    if hybrid["fixed_source_solver"] not in ["source iteration", "gmres"]:
        print_error(
            f"Invalid fixed source solver, '{hybrid['fixed_source_solver']}'. Available iteration solvers inlcude ['source iteration', 'gmres']"
        )

    # Check sample method
    if hybrid["sample_method"] not in ["random", "halton"]:
        print_error(
            f"Unsupported sample method, '{hybrid['sample_method']}'. Available sample methods: ['halton', 'random']."
        )

    # Check run mode
    if hybrid["mode"] not in ["fixed", "batched"]:
        with objmode():
            print_error(
                f"Unsupported run mode, '{hybrid['mode']}'. Available hybridMC modes are ['fixed', 'batched']"
            )

    # Check scores
    for score in list(hybrid["score_list"].keys()):
        if score not in hybrid_score_list:
            print_error(
                f"Unsupported score, '{score}'. Available hybridMC scores are {hybrid_score_list}"
            )

    # Check N_inactive & N_active batches for batched mode
    if eigenmode and hybrid["mode"] == "batched":
        if (
            input_deck.setting["N_inactive"] == 0
            and input_deck.setting["N_active"] == 0
        ):
            print_error(
                "Specify N_inactive and N_active batches for hybridMC batched mode."
            )


# =============================================================================
# hybridMC Simulation
# =============================================================================


@njit
def hybrid_simulation(mcdc_arr):
    # Ensure `mcdc` exists for the lifetime of the program
    # by intentionally leaking their memory
    adapt.leak(mcdc_arr)
    mcdc = mcdc_arr[0]

    # Preprocessing
    hybrid = mcdc["technique"]["hybrid"]
    hybrid_kernel.hybrid_preprocess(mcdc)
    hybrid_kernel.samples_init(mcdc)
    hybrid_kernel.sn_init(mcdc)
    if hybrid["mode"] == "batched":
        hybrid["iterations_max"] = (
            mcdc["setting"]["N_active"] + mcdc["setting"]["N_inactive"] - 1
        )

    # Iterative Solve
    if mcdc["setting"]["mode_eigenvalue"]:
        power_iteration(mcdc)
    else:
        if hybrid["fixed_source_solver"] == "source iteration":
            source_iteration(mcdc)
        if hybrid["fixed_source_solver"] == "gmres":
            gmres(mcdc)

    # Post processing
    hybrid_kernel.hybrid_tally_closeout(mcdc)


# =============================================================================
# Iterative Solvers
# =============================================================================


@njit
def source_iteration(mcdc):
    simulation_end = False
    hybrid = mcdc["technique"]["hybrid"]
    total_source_old = hybrid["total_source"].copy()
    time_steps = hybrid["mesh"]["t"]
    Nt = len(time_steps)
    # reset particle bank size
    kernel.set_bank_size(mcdc["bank_source"], 0)
    # initialize particles with LDS
    hybrid_kernel.hybrid_prepare_particles(mcdc)
    
  #  for times in time_steps:
   #     hybrid_particle_sweep(0, times)    #ToDo: Iterate
    #    discrete_ordinates_sweep()
     #   hybrid_particle_sweep(tn_start,times)
      #  tn_start = tn_end
        
        
        
    while not simulation_end:
        mcdc["technique"]["hybrid"]["time_step_idx"]+=1
        hybrid_time_step(mcdc)

        
        #hybrid_sweep(mcdc)
        
        hybrid["iteration_count"] += 1
        # calculate norm of sources
        hybrid["residual"] = hybrid_kernel.hybrid_res(hybrid["total_source"], total_source_old)
        # hybridMC convergence criteria
        
        
        if (hybrid["iteration_count"] == hybrid["iterations_max"]) or (
            hybrid["residual"] <= hybrid["tol"] or hybrid["time_steps_idx"]== Nt-1
        ):
            simulation_end = True

        # Print progress
        if not mcdc["setting"]["mode_eigenvalue"]:
            with objmode():
                print_progress_hybrid(mcdc)

        # set  source_old = current source
        total_source_old = hybrid["total_source"].copy()
        
     # sum resultant flux on all processors
    hybrid_kernel.hybrid_reduce_tallies(hybrid)

@njit
def power_iteration(mcdc):
    simulation_end = False
    hybrid = mcdc["technique"]["hybrid"]
    tol = hybrid["tol"]
    maxit = hybrid["iterations_max"]
    score_bin = hybrid["score"]
    k_old = mcdc["k_eff"]
    fission_source_old = score_bin["fission-source"]["bin"].copy()

    while not simulation_end:
        # Scramble samples if in batched mode
        if hybrid["mode"] == "batched":
            hybrid_kernel.scramble_samples(mcdc)
        # Run sweep
        hybrid_sweep(mcdc)
        # Reset counter for inner iteration
        hybrid["iteration_count"] += 1
        # Update k_eff
        mcdc["k_eff"] *= score_bin["fission-source"]["bin"][0] / fission_source_old[0]
        # Calculate diff in keff
        hybrid["residual"] = abs(mcdc["k_eff"] - k_old) / k_old
        k_old = mcdc["k_eff"]
        # Store outter iteration values
        score_bin["effective-fission-outter"] = score_bin["effective-fission"][
            "bin"
        ].copy()
        fission_source_old = score_bin["fission-source"]["bin"].copy()

        # Batch mode
        if hybrid["mode"] == "batched":
            mcdc["idx_cycle"] += 1
            hybrid_kernel.hybrid_eigenvalue_tally_closeout_history(mcdc)
            if mcdc["cycle_active"]:
                # Only accumulate statistics
                hybrid_kernel.hybrid_tally_closeout_history(mcdc)
            # Entering active cycle ?
            if mcdc["idx_cycle"] >= mcdc["setting"]["N_inactive"]:
                mcdc["cycle_active"] = True

        # Print progress
        with objmode():
            if hybrid["mode"] == "fixed":
                print_hybrid_eigenvalue_progress(mcdc)
            else:
                print_hybrid_eigenvalue_progress(mcdc)

        # hybridMC convergence criteria
        if (hybrid["iteration_count"] == maxit) or (hybrid["residual"] <= tol):
            simulation_end = True
            if hybrid["mode"] == "fixed":
                with objmode():
                    print_hybrid_eigenvalue_exit_code(mcdc)


@njit
def gmres(mcdc):
    """
    GMRES solver.
    ----------
    Linear Krylov solver. Solves problem of the form Ax = b.
    This function is almost entirely linear algebra operations and does not
    directly use any functions in mcdc/kernel.py or mcdc/loop.py

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
        Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
        http://www-users.cs.umn.edu/~saad/books.html
    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html

    code adapted from: https://github.com/pygbe/pygbe/blob/master/pygbe/gmres.py

    """
    hybrid = mcdc["technique"]["hybrid"]
    max_iter = hybrid["iterations_max"]
    R = hybrid["krylov_restart"]
    tol = hybrid["tol"]

    fixed_source = hybrid["fixed_source"]
    single_vector = hybrid["fixed_source"].size
    b = np.zeros_like(hybrid["total_source"])
    b[:single_vector] = np.reshape(fixed_source, fixed_source.size)
    X = hybrid["total_source"].copy()
    # initial residual
    r = b - AxV(X, b, mcdc)
    normr = np.linalg.norm(r)

    # Defining dimension
    dimen = X.size
    # Set number of outer and inner iterations
    if R > dimen:
        # set number of outter iterations to max allowable (A.shape[0])
        R = dimen
    max_inner = R
    xtype = np.float64

    # max_outer should be max_iter/max_inner but this might not be an integer
    # so we get the ceil of the division.
    # In the inner loop there is a if statement to break in case max_iter is
    # reached.
    max_outer = int(np.ceil(max_iter / max_inner))

    # Check initial guess ( scaling by b, if b != 0, must account for
    # case when norm(b) is very small)
    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol * normb:
        return X, 0

    iteration = 0

    # GMRES starts here
    for outer in range(max_outer):
        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        Q = []
        H = np.zeros((max_inner + 1, max_inner + 1), dtype=xtype)
        V = np.zeros((max_inner + 1, dimen), dtype=xtype)

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []
        V[0, :] = (1.0 / normr) * r
        vs.append(V[0, :])

        # Saving initial residual to be used to calculate the rel_resid
        if iteration == 0:
            res_0 = normb

        # RHS vector in the Krylov space
        g = np.zeros((dimen,), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            # New search direction
            v = V[inner + 1, :]
            v[:] = AxV(vs[-1], b, mcdc)
            vs.append(v)

            # Modified Gram Schmidt
            for k in range(inner + 1):
                vk = vs[k]
                alpha = np.dot(vk, v)
                H[inner, k] = alpha
                v[:] = vk * (-alpha) + v[:]

            normv = np.linalg.norm(v)
            H[inner, inner + 1] = normv

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
                v[:] = (1.0 / H[inner, inner + 1]) * v

            # Apply for Givens rotations to H
            if inner > 0:
                for j in range(inner):
                    Qloc = Q[j]
                    H[inner, :][j : j + 2] = np.dot(Qloc, H[inner, :][j : j + 2])

            # Calculate and apply next complex-valued Givens rotations

            # If max_inner = dimen, we don't need to calculate, this
            # is unnecessary for the last inner iteration when inner = dimen -1
            if inner != dimen - 1:
                if H[inner, inner + 1] != 0:
                    # Caclulate matrix rotations
                    c, s, _ = kernel.lartg(H[inner, inner], H[inner, inner + 1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]], dtype=xtype)
                    Q.append(Qblock)

                    # Apply Givens Rotations to RHS for the linear system in
                    # the krylov space.
                    g[inner : inner + 2] = np.dot(Qblock, g[inner : inner + 2])

                    # Apply Givens rotations to H
                    H[inner, inner] = np.dot(Qblock[0, :], H[inner, inner : inner + 2])
                    H[inner, inner + 1] = 0.0

            iteration += 1

            if inner < max_inner - 1:
                normr = abs(g[inner + 1])
                rel_resid = normr / res_0
                hybrid["residual"] = rel_resid

            hybrid["iteration_count"] += 1
            if not mcdc["setting"]["mode_eigenvalue"]:
                with objmode():
                    print_progress_hybrid(mcdc)

            if rel_resid < tol:
                break
            if hybrid["iteration_count"] >= max_iter:
                break

        # end inner loop, back to outer loop
        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y = np.linalg.solve(H[0 : inner + 1, 0 : inner + 1].T, g[0 : inner + 1])
        update = np.ravel(np.dot(cga(V[: inner + 1, :].T), y.reshape(-1, 1)))
        X = X + update
        aux = AxV(X, b, mcdc)
        r = b - aux
        normr = np.linalg.norm(r)
        rel_resid = normr / res_0
        hybrid["residual"] = rel_resid
        if rel_resid < tol:
            break
        if hybrid["iteration_count"] >= max_iter:
            return


# =============================================================================
# Lower Level loops
# =============================================================================


@njit
def hybrid_loop_particle(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]
    current_t_idx = mcdc["technique"]["hybrid"]["time_step_idx"]
    current_t = mcdc["technique"]["hybrid"]["mesh"]["t"][current_t_idx]
    prev_t = mcdc["technique"]["hybrid"]["mesh"]["t"][current_t_idx-1]
    while P["alive"] and P["t"]<current_t:  #Move Particles to end of current step
        hybrid_step_particle(P_arr, prog)
    if P["hybrid"]["birth_time"] < prev_t and P["alive"]:    #Particles that were around current step won't be relabeled and move on next step 
        adapt.add_source(P_arr, mcdc)
        

@njit
def hybrid_step_particle(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]

    # Determine and move to event
    hybrid_kernel.hybrid_move_to_event(P_arr, mcdc)
    event = P["event"]

    # The & operator here is a bitwise and.
    # It is used to determine if an event type is part of the particle event.

    # Surface crossing
    if event & EVENT_SURFACE_CROSSING:
        hybrid_kernel.hybrid_surface_crossing(P_arr, prog)
        if event & EVENT_DOMAIN_CROSSING:
            if not (
                mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_REFLECTIVE
                or mcdc["surfaces"][P["surface_ID"]]["BC"] == BC_VACUUM
            ):
                kernel.domain_crossing(P_arr, mcdc)

    # Lattice or mesh crossing (skipped if surface crossing)
    elif event & EVENT_LATTICE_CROSSING or event & EVENT_IQMC_MESH:
        if event & EVENT_DOMAIN_CROSSING:
            kernel.domain_crossing(P_arr, mcdc)

    # Apply weight roulette
    if P["alive"] and mcdc["technique"]["weight_roulette"]:
        # check if weight has fallen below threshold
        if abs(P["w"]) <= mcdc["technique"]["wr_threshold"]:
            kernel.weight_roulette(P_arr, mcdc)


@njit
def hybrid_loop_source(mcdc):
    work_size = mcdc["bank_source"]["size"][0]
    N_prog = 0
    # loop over particles
    for idx_work in range(work_size):
        P_arr = mcdc["bank_source"]["particles"][idx_work : (idx_work + 1)]
        P = P_arr[0]
        mcdc["bank_source"]["size"] -= 1
        kernel.add_particle(P_arr, mcdc["bank_active"])

        # Loop until active bank is exhausted
        while mcdc["bank_active"]["size"] > 0:
            P_arr = adapt.local_array(1, type_.particle)
            P = P_arr[0]
            # Get particle from active bank
            kernel.get_particle(P_arr, mcdc["bank_active"], mcdc)
            # Particle loop
            hybrid_loop_particle(P_arr, mcdc)

        # Progress printout
        percent = (idx_work + 1.0) / work_size
        if mcdc["setting"]["progress_bar"] and int(percent * 100.0) > N_prog:
            N_prog += 1
            with objmode():
                print_progress(percent, mcdc)


@njit
def hybrid_sweep(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    # tally sweep count
    hybrid["sweep_count"] += 1
    # reset particle bank size
    kernel.set_bank_size(mcdc["bank_source"], 0)
    # initialize particles with LDS
    hybrid_kernel.hybrid_prepare_particles(mcdc)
    # reset tallies for next loop
    hybrid_kernel.hybrid_reset_tallies(hybrid)
    # sweep particles
    hybrid_loop_source(mcdc)
    # sum resultant flux on all processors
    hybrid_kernel.hybrid_reduce_tallies(hybrid)
    # update source = scattering + fission/keff + fixed
    hybrid_kernel.hybrid_update_source(mcdc)
    # combine source tallies into one vector
    hybrid_kernel.hybrid_consolidate_sources(mcdc)


@njit
def hybrid_time_step(mcdc):
    n_particles = mcdc["setting"]["N_particle"]
    n_directions = mcdc["technique"]["hybrid"]["SN"]["ordinates"].shape[0]
    hybrid_particle_sweep(mcdc)
    kernel.distribute_work(n_directions,mcdc)
    hybrid_SN_sweep(mcdc)
    kernel.distribute_work(n_particles,mcdc)
    hybrid_relabel(mcdc)
    

@njit
def hybrid_particle_sweep(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    # sweep particles
    hybrid_loop_source(mcdc)
    kernel.allreduce_array(hybrid["uncollided_flux"])

    
          
    
@njit    
def hybrid_SN_sweep(mcdc):
    hybrid = mcdc["technique"]["hybrid"]    
    iterate = True
    iterations = 0
    while iterate:
        iterations += 1
        err = single_SN_sweep(mcdc)
        kernel.allreduce_array(hybrid["SN"]["collided_flux"])
        
        
        iterate = err > hybrid["tol"] or iterations > hybrid["iterations_max"]
    

@njit
def single_SN_sweep(mcdc):
    sn = mcdc["technique"]["hybrid"]["SN"]  
    mesh = mcdc["technique"]["hybrid"]["mesh"] 
    x_deg = sn["x_degree"]
    y_deg = sn["y_degree"]
    z_deg = sn["z_degree"]
    ordinates = sn["ordinates"]
    n_directions = ordinates.shape[1]
    n_ord_tot = ordinates.shape[0]
    omega =np.zeros([4])
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    max_err = INF
    for i_ordinates in range(n_ord_tot):
        index = 0 
        omega[3] = ordinates[i_ordinates,-1]
        if x_deg != -1:
            omega[0] = ordinates[i_ordinates,index]
            index += 1
        if y_deg != -1:
            omega[1] = ordinates[i_ordinates,index]
            index += 1
        if z_deg != -1:
            omega[2] = ordinates[i,index]
        x_range = range(Nx) if omega[0] >= 0 else range(Nx-1, -1, -1)
        y_range = range(Ny) if omega[1] >= 0 else range(Ny-1, -1, -1)
        z_range = range(Nz) if omega[2] >= 0 else range(Nz-1, -1, -1)
        
    # Traverse the mesh in the determined order
        for x in x_range:
            for y in y_range:
                for z in z_range:
                
                    err=solve_SN(omega,i_ordinates,x,y,z,mcdc)
                    max_err = min(max_err,err)

    update_collided_flux(mcdc)        
    return max_err

@njit
def update_collided_flux(mcdc):
    k_eff = mcdc["k_eff"]

    hybrid = mcdc["technique"]["hybrid"]    
    t_idx = hybrid["time_step_idx"] 
    mesh = hybrid["mesh"]
    ordinates = hybrid["SN"]["ordinates"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    flux = hybrid["SN"]["collided_flux"]    
    flux =  np.tensordot(hybrid["SN"]["coef"], ordinates[:,-1], axes=([-1], [0]))
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):        
                mat_idx = mcdc["technique"]["hybrid"]["material_idx"][t_idx-1, x, y, z] 
                material = mcdc["materials"][mat_idx]
                chi_s = material["chi_s"]
                SigmaS = material["scatter"]
                eff_scatter = np.tensordot(chi_s,SigmaS[:,None,None,None,None,None,None]*flux[:,:,:,:,x,y,z], axes=([0],[0]))
                material = mcdc["nuclides"][mat_idx]

                chi_p = material["chi_p"]
                chi_d = material["chi_d"]
                nu_p = material["nu_p"]
                nu_d = material["nu_d"]
                SigmaF = material["fission"]
                F_p = np.tensordot(chi_p, (nu_p * SigmaF)[:,None,None,None,None,None,None] * flux[:,:,:,:,x,y,z], axes=([0],[0]))
        
                F_d = nu_d.T[:, :, None, None, None, None, None] * SigmaF[:, None, None, None, None, None, None] * flux[:,:,:,:,x,y,z]
                F_d = F_d.sum(axis=1)  # shape: (J, I, J, L, X, Y, Z)
                F_d = np.tensordot(chi_d, F_d, axes=([0], [0]))  # shape: (G, I, J, L, X, Y, Z)
                eff_fission = F_p+F_d
    flux = eff_fission/k_eff+eff_scatter    
     
@njit
def solve_SN(Omega,i_ordinates,x,y,z,mcdc):

    
   
    hybrid = mcdc["technique"]["hybrid"]    
    err = 0

    
    tensor = ["tensor_x", "tensor_y", "tensor_z"] 
    
    signs = np.sign(Omega[0:-1]).astype(int)
    effective_uncollided_flux = hybrid["SN"]["uncollided_flux"][:,x,y,z]
    effective_collided_flux = hybrid["SN"]["collided_flux"][:,:,:,:,x,y,z]
    coef_loc = hybrid["SN"]["coef"][:,:,:,:,x,y,z,i_ordinates]
    coef_down_x, coef_down_y, coef_down_z = downstream_coef(x,y,z,i_ordinates,signs,mcdc)
    
    #updown = int(Omega[idx]<0)
    #tensor = hybrid["SN"][tensors[idx]][:,:,updown,:]
    tensor_x = hybrid["SN"]["tensor_x"][:,:,int(Omega[0]<0),:]
    tensor_y = hybrid["SN"]["tensor_y"][:,:,int(Omega[1]<0),:]
    tensor_z = hybrid["SN"]["tensor_z"][:,:,int(Omega[2]<0),:]    
        
    ##Set up RHS
    RHS = np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,-1],tensor_y[:,:,-1],tensor_z[:,:,-1],effective_collided_flux)    
    RHS[:,0,0,0] += effective_uncollided_flux
    RHS -= Omega[0]*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,-2],tensor_y[:,:,-1],tensor_z[:,:,-1],coef_down_x)
    RHS -= Omega[1]*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,-1],tensor_y[:,:,-2],tensor_z[:,:,-1],coef_down_y)
    RHS -= Omega[2]*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,-1],tensor_y[:,:,-1],tensor_z[:,:,-2],coef_down_z)
    
    
    t_idx = hybrid["time_step_idx"] 
    t = hybrid["mesh"]["t"]
    mat_idx = mcdc["technique"]["hybrid"]["material_idx"][t_idx-1, x, y, z] 
    material = mcdc["materials"][mat_idx]
    
    dt_inv = 1/(t[t_idx]-t[t_idx-1])    
    sigmaT =  material["total"]+1/material["speed"]*dt_inv
    n_dim = np.count_nonzero(Omega[0:-1])
    idz = np.concatenate([np.nonzero(Omega[0:-1])[0], np.where(Omega[0:-1]==0)[0]])
    
    dxyz_inverse  = np.ones(3)
    if Omega[0] !=0:
        dxyz_inverse[0] = 1/(hybrid["mesh"]["x"][x+1]-hybrid["mesh"]["x"][x])
    if Omega[1] !=0:
        dxyz_inverse[1]  = 1/(hybrid["mesh"]["y"][y+1]-hybrid["mesh"]["y"][y])    
    if Omega[2] !=0:
        dxyz_inverse[2]  = 1/(hybrid["mesh"]["z"][z+1]-hybrid["mesh"]["z"][z])
    
    idx1 = idz[0]
    d2_in = dxyz_inverse[1]
    d3_in = dxyz_inverse[2]
    new_coef = (d2_in*d3_in)/Omega[idx1]*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,0],tensor_y[:,:,0],tensor_z[:,:,0],RHS)
    if sigmaT != 0 and n_dim ==1:
        d1_in = dxyz_inverse[1]
        idx2 =idz[1]
        new_coef += (d2_in*d3_in)**3/Omega[idx1]**3*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,1],tensor_y[:,:,1],tensor_z[:,:,1],RHS) 
        new_coef += d1_in*d2_in**2*d3_in**3/sigmaT/(Omega[idx1]**2)* np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,2],tensor_y[:,:,2],tensor_z[:,:,2],RHS) 
    if n_dim > 1:
        d1_in = dxyz[1]
        idx2 = idz[1]
        new_coef += (d2_in*d3_in)**3/Omega[idx1]**3*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,1],tensor_y[:,:,1],tensor_z[:,:,1],RHS) 
        new_coef += d1_in*d2_in**2*d3_in**3/(Omega[idx1]**2)/Omega[idx2]* np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,2],tensor_y[:,:,2],tensor_z[:,:,2],RHS) 
        if sigmaT != 0 and n_dim ==2:
            new_coef += d1_in**3*d2_in**2*d3_in**5/Omega[idx1]**2/Omega[idx2]**3*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,3],tensor_y[:,:,3],tensor_z[:,:,3],RHS) 
            new_coef += d1_in**3*d2_in**3*d3_in**4/Omega[idx1]**2/Omega[idx2]**2/sigmaT * np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,4],tensor_y[:,:,4],tensor_z[:,:,4],RHS) 
    if n_dim == 3:
        idx3 = idz[2]
        new_coef += d1_in**3*d2_in**2*d3_in**5/Omega[idx1]**2/Omega[idx2]**3*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,3],tensor_y[:,:,3],tensor_z[:,:,3],RHS) 
        new_coef += d1_in**3*d2_in**3*d3_in**4/Omega[idx1]**2/Omega[idx2]**2/Omega[idx3] *np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,4],tensor_y[:,:,4],tensor_z[:,:,4],RHS) 
    if sigmaT != 0 and n_dim ==3:
        new_coef += d1_in**5*d2_in**3*d3_in**4/Omega[idx1]**2/Omega[idx2]**2/Omega[idx3]**3*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,5],tensor_y[:,:,5],tensor_z[:,:,5],RHS) 
        new_coef += (d1_in*d2_in*d3_in)**5/Omega[idx1]**2/Omega[idx2]**2/Omega[idx3]**2/sigmaT*np.einsum("ij,kl,mn,gjln->gikm", tensor_x[:,:,6],tensor_y[:,:,6],tensor_z[:,:,6],RHS) 

        
        
    err+= np.linalg.norm(new_coef - coef_loc) 
    hybrid["SN"]["coef"][:,:,:,:,x,y,z,i_ordinates] = new_coef    
        
    return err  
     
@njit 
def downstream_coef(x,y,z,i_ordinates,signs, mcdc):    
     
    mesh = mcdc["technique"]["hybrid"]["mesh"] 
    if x == 0 or x == mesh["Nx"]-1:
        #Vacuum boundaries, reflective not done yet
        down_x = np.zeros_like(mcdc["technique"]["hybrid"]["SN"]["coef"][:,:,:,:,x,y,z,i_ordinates])
    else:
        down_x = np.zeros_like(mcdc["technique"]["hybrid"]["SN"]["coef"][:,:,:,:,x-signs[0],y,z,i_ordinates])

    if y == 0 or y == mesh["Ny"]-1:
        #Vacuum boundaries, reflective not done yet
        down_y = np.zeros_like(mcdc["technique"]["hybrid"]["SN"]["coef"][:,:,:,:,x,y,z,i_ordinates])
    else:
        down_y = np.zeros_like(mcdc["technique"]["hybrid"]["SN"]["coef"][:,:,:,:,x,y-signs[1],z,i_ordinates])

    if z == 0 or z == mesh["Nz"]-1:
        #Vacuum boundaries, reflective not done yet
        down_z = np.zeros_like(mcdc["technique"]["hybrid"]["SN"]["coef"][:,:,:,:,x,y,z,i_ordinates])
    else:
        down_z = np.zeros_like(mcdc["technique"]["hybrid"]["SN"]["coef"][:,:,:,:,x,y,z-signs[0],i_ordinates])

    return down_x, down_y,down_z


@njit 
def find_bcs(coords,signs,idx,mcdc):
    """only vacuum conditions so far
    for reflective return original coords, alternate, where 
    alternate =(1,-1,1,-1,...) with size of degree in idx direction"""
    
    mesh = mcdc["technique"]["hybrid"]["mesh"] 
    bounds =[mesh["Nx"],  mesh["Ny"],mesh["Nz"]]
    if ((coords[idx]-signs[idx]) < 0) or ((coords[idx]-signs[idx]) >= bounds[idx]):
        return tuple(coords), 0           
    coords[idx]-=signs[idx]
    return tuple(coords), 1
    
    
    
   
    
@njit    
def hybrid_relabel(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    # Particles born this time_step are reborn, scine source changed
    hybrid_kernel.hybrid_reset_particles(mcdc)
    
    # sweep particles
    hybrid_loop_source(mcdc)
    hybrid["SN"]["uncollided_flux"].fill(0)
# ===========================================================================
# GMRES Linear operator
# =============================================================================


@njit
def AxV(V, b, mcdc):
    """
    Linear operator to be used with GMRES.
    Calculate action of A on input vector V, where A is a transport sweep
    and V is the total source (constant and tilted).
    """
    hybrid = mcdc["technique"]["hybrid"]
    hybrid["total_source"] = V.copy()
    # distribute segments of V to appropriate sources
    hybrid_kernel.hybrid_distribute_sources(mcdc)
    hybrid_sweep(mcdc)
    v_out = hybrid["total_source"].copy()
    axv = V - (v_out - b)

    return axv
