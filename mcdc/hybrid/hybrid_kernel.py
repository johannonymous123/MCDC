import math
import numpy as np

from mpi4py import MPI
from numba import objmode, literal_unroll

import mcdc.type_ as type_
import mcdc.adapt as adapt
import mcdc.src.geometry as geometry
import mcdc.src.mesh as mesh_
import mcdc.src.physics as physics
import mcdc.src.surface as surface_

from mcdc.adapt import toggle
from mcdc.constant import *
from mcdc.kernel import (
    allreduce_array,
    move_particle,
)
from mcdc.type_ import hybrid_score_list


# =========================================================================
# Sampling Operations
# =========================================================================


@toggle("hybridMC")
def samples_init(mcdc):
    N, dim = mcdc["technique"]["hybrid"]["samples"].shape
    N_start = mcdc["mpi_work_start"]
    if mcdc["technique"]["hybrid"]["sample_method"] == "halton":
        mcdc["technique"]["hybrid"]["samples"] = halton(N, dim, skip=N_start)
        mcdc["technique"]["hybrid"]["samples"][0,:].sort()
    if mcdc["technique"]["hybrid"]["sample_method"] == "random":
        mcdc["technique"]["hybrid"]["samples"] = random(N, dim)
        mcdc["technique"]["hybrid"]["samples"][0,:].sort()


@toggle("hybridMC")
def scramble_samples(mcdc):
    # TODO: use MCDC seed system
    seed_batch = np.int64(mcdc["setting"]["N_particle"] * mcdc["idx_cycle"] + 1)
    hybrid = mcdc["technique"]["hybrid"]
    N, dim = hybrid["samples"].shape
    N_start = mcdc["mpi_work_start"]

    if hybrid["sample_method"] == "halton":
        hybrid["samples"] = rhalton(N, dim, seed=seed_batch, skip=N_start)
    if hybrid["sample_method"] == "random":
        hybrid["samples"] = random(N, dim, seed=seed_batch)


@toggle("hybridMC")
def rhalton(N, dim, seed=12345, skip=0):
    np.random.seed(seed)
    primes = np.array((2, 3, 5, 7, 11, 13, 17, 19, 23, 29), dtype=np.int64)
    halton = np.zeros((N, dim), dtype=np.float64)

    for D in range(dim):
        b = primes[D]
        # b = np.int64(2)
        ind = np.arange(skip, skip + N, dtype=np.int64)
        b2r = 1 / b
        ans = np.zeros(ind.shape, dtype=np.float64)
        res = ind.copy()
        while (1.0 - b2r) < 1.0:
            dig = np.mod(res, b)
            perm = np.random.permutation(b)
            pdig = perm[dig]
            ans = ans + pdig.astype(np.float64) * b2r
            b2r = b2r / np.float64(b)
            res = ((res - dig) / b).astype(np.int64)
        halton[:, D] = ans

    return halton


@toggle("hybridMC")
def halton(N, dim, skip=0):
    # TODO: find more efficient implementation of Halton Sequence
    primes = np.array((2, 3, 5, 7, 11, 13, 17, 19, 23, 29), dtype=np.int64)
    halton = np.zeros((N, dim), dtype=np.float64)

    for D in range(dim):
        b = primes[D]
        n, d = 0, 1
        for i in range(skip + N):
            x = d - n
            if x == 1:
                n = 1
                d *= b
            else:
                y = d // b
                while x <= y:
                    y //= b
                n = (b + 1) * y - x
            if i >= skip:
                halton[i - skip, D] = n / d

    return halton


@toggle("hybridMC")
def random(N, dim, seed=123456):
    np.random.seed(seed)
    return np.random.rand(N, dim)


# =============================================================================
# Preprocess functions
# =============================================================================


@toggle("hybridMC")
def hybrid_preprocess(mcdc):
    # set bank source
    hybrid = mcdc["technique"]["hybrid"]
    eigenmode = mcdc["setting"]["mode_eigenvalue"]
    # generate material index
    hybrid_generate_material_idx(mcdc)
    if hybrid["source"].all() == 0.0:
        # use material index to generate a first guess for the source
        hybrid_prepare_source(mcdc)
        hybrid_update_source(mcdc)
    if eigenmode:
        hybrid_prepare_nusigmaf(mcdc)

    hybrid_consolidate_sources(mcdc)


@toggle("hybridMC")
def hybrid_generate_material_idx(mcdc):
    """
    This algorithm is meant to loop through every spatial cell of the
    hybridMC mesh and assign a material index according to the material_ID at
    the center of the cell.

    Therefore, the whole cell is treated as the material located at the
    center of the cell, regardless of whethere there are more materials
    present.

    A crude but quick approximation.
    """
    mesh = mcdc["technique"]["hybrid"]["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # create particle to utilize cell finding functions
    P_temp_arr = adapt.local_array(1, type_.particle)
    P_temp = P_temp_arr[0]
    # set default attributes
    P_temp["alive"] = True

    x_mid = 0.5 * (mesh["x"][1:] + mesh["x"][:-1])
    y_mid = 0.5 * (mesh["y"][1:] + mesh["y"][:-1])
    z_mid = 0.5 * (mesh["z"][1:] + mesh["z"][:-1])

    # loop through every cell
    for t in range(Nt):
        for i in range(Nx):
            x = x_mid[i]
            for j in range(Ny):
                y = y_mid[j]
                for k in range(Nz):
                    z = z_mid[k]

                    # assign cell center position
                    P_temp["t"] = t
                    P_temp["x"] = x
                    P_temp["y"] = y
                    P_temp["z"] = z
                    P_temp["material_ID"] = -1
                    P_temp["cell_ID"] = -1
                    P_temp["g"] = 0

                    # set material_ID
                    geometry.locate_particle(P_temp_arr, mcdc)

                    # assign material index
                    mcdc["technique"]["hybrid"]["material_idx"][t, i, j, k] = P_temp[
                        "material_ID"
                    ]



@toggle("hybridMC")
def hybrid_prepare_nusigmaf(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    mesh = hybrid["mesh"]
    flux = hybrid["score"]["flux"]["bin"]
    fission_source = hybrid["score"]["fission-source"]["bin"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # calculate nu*SigmaF for every cell
    for t in range(Nt):
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    t = 0
                    mat_idx = hybrid["material_idx"][t, i, j, k]
                    material = mcdc["materials"][mat_idx]
                    fission_source += hybrid_fission_source(flux[:, t, i, j, k], material)


@toggle("hybridMC")
def hybrid_prepare_source(mcdc):
    """
    Iterates trhough all spatial cells to calculate the hybridMC source. The source
    is a combination of the user input Fixed-Source plus the calculated
    Scattering-Source and Fission-Sources. Resutls are stored in
    mcdc['technique']['hybrid_source'], a matrix of size [G,Nt,Nx,Ny,Nz].

    """
    hybrid = mcdc["technique"]["hybrid"]
    mesh = hybrid["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1

    fission = np.zeros_like(hybrid["source"])
    scatter = np.zeros_like(hybrid["source"])

    # calculate source for every cell and group in the hybrid_mesh
    for t in range(Nt):
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    mat_idx = hybrid["material_idx"][t, i, j, k]
                    # we can vectorize the multigroup calculation here
                    flux = hybrid["score"]["flux"]["bin"][:, t, i, j, k]
                    fission[:, t, i, j, k] = hybrid_effective_fission(flux, mat_idx, mcdc)
                    scatter[:, t, i, j, k] = hybrid_effective_scattering(
                        flux, mat_idx, mcdc
                    )
    hybrid["score"]["effective-scattering"]["bin"] = scatter
    hybrid["score"]["effective-fission"]["bin"] = fission
    hybrid["score"]["effective-fission-outter"] = fission


# =============================================================================
# Particle Operations
# =============================================================================


@toggle("hybridMC")
def hybrid_prepare_particles(mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    # total number of particles
    N_particle = mcdc["setting"]["N_particle"]
    # number of particles this processor will handle
    N_work = mcdc["mpi_work_size"]

    # low discrepency sequence
    samples = hybrid["samples"]
    # source
    Q = hybrid["source"]
    mesh = hybrid["mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # total number of spatial cells
    N_total = Nx * Ny * Nz
    # outter mesh boundaries for sampling position
    xa = mesh["x"][0]
    xb = mesh["x"][-1]
    ya = mesh["y"][0]
    yb = mesh["y"][-1]
    za = mesh["z"][0]
    zb = mesh["z"][-1]
    ta = mesh["t"][0]
    tb = mesh["t"][-1]

    for n in range(N_work):
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)
        P_new["g"] = 0
        P_new["t"] = hybrid_sample_position(xa, xb, samples[n, 0])
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, samples[n, 1])
        P_new["y"] = hybrid_sample_position(ya, yb, samples[n, 2])
        P_new["z"] = hybrid_sample_position(za, zb, samples[n, 3])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            samples[n, 4], samples[n, 5]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)
        q = Q[:, t, x, y, z].copy()
        dV = hybrid_cell_volume(x, y, z, mesh)
        # Source tilt
        hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["hybrid"]["w"] = q * dV * N_total / N_particle
        P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = P_new["t"]
        # add to source bank
        adapt.add_source(P_new_arr, mcdc)

@toggle("hybridMC")
def hybrid_reset_particles(mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    # total number of particles
    N_particle = mcdc["setting"]["N_particle"]
    # number of particles this processor will handle
    
    #N_work = mcdc["mpi_work_size"]

    # low discrepency sequence
    samples = hybrid["samples"]
    # source
    Q = hybrid["source"]
    eff_collided_flux = hybrid["SN"]["collided_flux"]
    eff_uncollided_flux = hybrid["SN"]["uncollided_flux"]
    
    mesh = hybrid["mesh"]
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # total number of spatial cells
    N_total = Nx * Ny * Nz
    # outter mesh boundaries for sampling position
    xa = mesh["x"][0]
    xb = mesh["x"][-1]
    ya = mesh["y"][0]
    yb = mesh["y"][-1]
    za = mesh["z"][0]
    zb = mesh["z"][-1]
    ta = mesh["t"][0]
    tb = mesh["t"][-1]
    t_prev = mesh["t"][hybrid["time_step_idx"]-1]
    t_curr = mesh["t"][hybrid["time_step_idx"]]
    t_idx = hybrid["time_step_idx"] 
    
    
    ###############3
    ###############3
    resample_start = np.searchsorted(hybrid["samples"][:,0],t_prev , side='left')
    resample_end = np.searchsorted(hybrid["samples"][:,0],t_curr , side='right')                               
                                     #####
    for n in range(resample_start,resample_end):
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)
        P_new["g"] = 0
        P_new["t"] = hybrid_sample_position(xa, xb, samples[n, 0])
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, samples[n, 1])
        P_new["y"] = hybrid_sample_position(ya, yb, samples[n, 2])
        P_new["z"] = hybrid_sample_position(za, zb, samples[n, 3])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            samples[n, 4], samples[n, 5]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)
        
        g, x_deg, y_deg, z_deg = eff_collided_flux[...,x,y,z].shape


        phi_x = np.polynomial.legendre.legval((P_new["x"]-mesh["x"][x])/(mesh["x"][x]-mesh["x"][x+1]), np.eye(x_deg))  # shape (ni,)
        phi_y = np.polynomial.legendre.legval((P_new["y"]-mesh["y"][y])/(mesh["y"][y]-mesh["y"][y+1]), np.eye(y_deg))  # shape (nj,)
        phi_z = np.polynomial.legendre.legval((P_new["z"]-mesh["z"][z])/(mesh["z"][z]-mesh["z"][z+1]), np.eye(z_deg))  # shape (nk,)


        f = np.einsum('gijk,i,j,k->g', eff_collided_flux[...,x,y,z], phi_x, phi_y, phi_z)

        q = Q[:, t, x, y, z].copy()+max([f,0])+eff_uncollided_flux[:,x,y,z].copy()
        dV = hybrid_cell_volume(x, y, z, mesh)
        # Source tilt
        hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["hybrid"]["w"] = q * dV * N_total / N_particle
        P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = ta-1
        # add to source bank
        adapt.add_source(P_new_arr, mcdc)


@toggle("hybridMC")
def sn_init(mcdc):
    ordinates_init(mcdc)
    tensor_init(mcdc)


@toggle("hybridMC")    
def ordinates_init(mcdc):
    sn = mcdc["technique"]["hybrid"]["SN"]
    work_start = mcdc["mpi_work_start"]
    work_size = mcdc["mpi_work_size"]
    shape = sn["ordinates"].shape
    nodes, weights = np.polynomial.legendre.leggauss(sn["n_ordinates"])
    weights /= sum(weights)
    if shape[1]==2:
        sn["ordinates"][:,0] = nodes[work_start:work_start+work_size]
        sn["ordinates"][:,1] = weights[work_start:work_start+work_size]
    else:
        n_polar = 2*sn["n_ordinates"]
        angles = np.linspace(0, 2*np.pi, n_polar, endpoint=False) + np.pi/n_polar 
        unit_circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
        if shape[1] == 3:
            nodes = nodes[len(nodes)//2:]
            weights = weights[len(weights)//2:]    
            ordinates = np.array([
                (np.sqrt(1-node**2)*x, np.sqrt(1-node**2)*y, 1/n_polar*weight) 
                for (x, y) in unit_circle_points for node, weight in zip(nodes, weights)
                ])
        if shape[1] == 4:    
            ordinates = np.array([
                (np.sqrt(1-node**2)*x, np.sqrt(1-node**2)*y, node,1/n_polar * weight) 
                for (x, y) in unit_circle_points for node, weight in zip(nodes, weights)
                ])
        sn["ordinates"] = ordinates[work_start:work_start+work_size,:]

@toggle("hybridMC")
def tensor_init(mcdc):
    sn = mcdc["technique"]["hybrid"]["SN"]    
    x_deg = sn["x_degree"]
    y_deg = sn["y_degree"]
    z_deg = sn["z_degree"]
    if x_deg > -1:
        build_tensor(mcdc, 1,0)
    else:
        sn["tensor_x"]+=1

    if y_deg > -1:
        build_tensor(mcdc, 2 if x_deg > -1 else 1,1)
    else:
        sn["tensor_y"]+=1
    if z_deg > -1:
        degree = 3 if x_deg > -1 and y_deg > -1 else 2 if x_deg > -1 or y_deg > -1 else 1
        build_tensor(mcdc, degree,2)
    else:
        sn["tensor_z"]+=1
    
def build_tensor(mcdc, flag, axis):
    """eq for ceof is given as 
       (Omega*B+sigma_t N)coef = P neighbor_coef + sigma_s N old_coef in every dimmension
       This sets up tensors to invert this equation in a simple functional form
    """
    sn = mcdc["technique"]["hybrid"]["SN"]
    tensor = ["tensor_x","tensor_y","tensor_z"][axis]
    deg = ["x_degree","y_degree","z_degree"][axis]
    deg = sn[deg]
    directions=sn["ordinates"].shape[1]-1
    if deg == 0:
        Ip = In = np.array([[1]])
    else:     
    # Initialize an (n+1) x (n+1) matrix with zeros
        Ip = np.zeros((deg+1, deg+1))
    
    # Set the required entries
        Ip[0, 0] = 0.5  # Top-left entry
        Ip[deg, deg] = 0.5  # Bottom-right entry
    
    # Set the first lower and upper diagonals
        np.fill_diagonal(Ip[1:], 0.5)  # Lower diagonal
        np.fill_diagonal(Ip[:, 1:], -0.5)  # Upper diagonal
    
        In = Ip.copy()
        In[0,0] -= 1
        In[-1,-1] -= 1
        #Bn[-1,0] = Bn[-1,0] - 2 * (-1)**deg 
    
    
    N_inv = np.diag([2*i + 1 for i in range(deg+1)])
    N = np.diag([1/(2*i + 1) for i in range(deg+1)])
    Pp = np.fromfunction(lambda i, j: (-1)**i, (deg+1, deg+1), dtype=int)
    Pn = -Pp.T
    
    alternating = np.array([(-1)**i for i in range(deg + 1)])
    D = np.diag(alternating)
    
    sn[tensor][:,:,0,-1] = Ip@N
    sn[tensor][:,:,1,-1] = In@N
    sn[tensor][:,:,0,-2] = Ip@Pp
    sn[tensor][:,:,1,-2] = In@Pn
    sn[tensor][:,:,0,0] = np.eye(deg+1)#np.linalg.inv(Bp.T)
    sn[tensor][:,:,1,0] = np.eye(deg+1)#np.linalg.inv(Bn.T)    
    sn[tensor][:,:,0,1] = D
    sn[tensor][:,:,1,1] = D
    
@toggle("hybridMC")
def hybrid_cell_volume(x, y, z, mesh):
    """
    Calculate the volume of the cartesian spatial cell.

    """
    dx = dy = dz = 1
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF):
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF):
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF):
        dz = mesh["z"][z + 1] - mesh["z"][z]
    dV = dx * dy * dz
    return dV


@toggle("hybridMC")
def hybrid_sample_position(a, b, sample):
    return a + (b - a) * sample


@toggle("hybridMC")
def hybrid_sample_isotropic_direction(sample1, sample2):
    """
    Sample the an isotropic direction using samples between [0,1].

    """
    # Sample polar cosine and azimuthal angle uniformly
    mu = 2.0 * sample1 - 1.0
    azi = 2.0 * PI * sample2

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    uy = math.cos(azi) * c
    uz = math.sin(azi) * c
    ux = mu
    return ux, uy, uz


@toggle("hybridMC")
def hybrid_sample_group(sample, G):
    """
    Uniformly sample energy group using a random sample between [0,1].

    """
    return int(np.floor(sample * G))


# =========================================================================
# Move to Event
# =========================================================================


@toggle("hybridMC")
def hybrid_move_to_event(P_arr, mcdc):
    # ==================================================================================
    # Preparation (as needed)
    # ==================================================================================

    P = P_arr[0]

    # Multigroup preparation
    #   In MG mode, particle speed is material-dependent.
    if mcdc["setting"]["mode_MG"]:
        # If material is not identified yet, locate the particle
        if P["material_ID"] == -1:
            if not geometry.locate_particle(P_arr, mcdc):
                # Particle is lost
                P["event"] = EVENT_LOST
                return

    # ==================================================================================
    # Geometry inspection
    # ==================================================================================
    #   - Set particle top cell and material IDs (if not lost)
    #   - Set surface ID (if surface hit)
    #   - Return distance to boundary (surface or lattice)
    #   - Return geometry event type (surface or lattice crossing or particle lost)

    d_boundary = geometry.inspect_geometry(P_arr, mcdc)

    # Particle is lost?
    if P["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Get distances to other events
    # ==================================================================================

    # Distance to domain decomposition mesh
    d_domain = INF
    speed = physics.get_speed(P_arr, mcdc)
    if mcdc["technique"]["domain_decomposition"]:
        d_domain = mesh_.structured.get_crossing_distance(
            P_arr, speed, mcdc["technique"]["dd_mesh"]
        )

    # Distance to hybrid mesh
    d_mesh = mesh_.structured.get_crossing_distance(
        P_arr, speed, mcdc["technique"]["hybrid"]["mesh"]
    )

    # =========================================================================
    # Determine event(s)
    # =========================================================================
    # TODO: Make a function to better maintain the repeating operation

    distance = d_boundary

    # Check distance to domain
    if d_domain < distance - COINCIDENCE_TOLERANCE:
        distance = d_domain
        P["event"] = EVENT_DOMAIN_CROSSING
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_domain, distance):
        P["event"] += EVENT_DOMAIN_CROSSING

    # Check distance to mesh
    if d_mesh < distance - COINCIDENCE_TOLERANCE:
        distance = d_mesh
        P["event"] = EVENT_HYBRIDMC_MESH
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_mesh, distance):
        P["event"] += EVENT_HYBRIDMC_MESH

    # =========================================================================
    # Move particle
    # =========================================================================

    # score hybridMC tallies
    hybrid_score_tallies(P_arr, distance, mcdc)
    # attenuate particle weight
    hybrid_continuous_weight_reduction(P_arr, distance, mcdc)
    # kill particle if it falls below weight threshold
    if abs(P["w"]) <= mcdc["technique"]["hybrid"]["w_min"]:
        P["alive"] = False

    # Move particle
    move_particle(P_arr, distance, mcdc)


@toggle("hybridMC")
def hybrid_continuous_weight_reduction(P_arr, distance, mcdc):
    """
    Continuous weight reduction technique based on particle track-length.
    """
    P = P_arr[0]
    material = mcdc["materials"][P["material_ID"]]
    SigmaT = material["total"][:]
    w = P["hybrid"]["w"]
    P["hybrid"]["w"] = w * np.exp(-distance * SigmaT)
    P["w"] = P["hybrid"]["w"].sum()


# =============================================================================
# Surface crossing
# =============================================================================


@toggle("hybridMC")
def hybrid_surface_crossing(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]
    surface = mcdc["surfaces"][P["surface_ID"]]
    if surface["BC"] == BC_VACUUM:
        P["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_.reflect(P_arr, surface)

    # Need to check new cell later?
    if P["alive"] and not surface["BC"] == BC_REFLECTIVE:
        P["cell_ID"] = -1


# =============================================================================
# hybridMC Source Operations
# =============================================================================


@toggle("hybridMC")
def hybrid_update_source(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    keff = mcdc["k_eff"]
    scatter = hybrid["score"]["effective-scattering"]["bin"]
    fixed = hybrid["fixed_source"]
    if mcdc["setting"]["mode_eigenvalue"]:
        fission = hybrid["score"]["effective-fission-outter"]
    else:
        fission = hybrid["score"]["effective-fission"]["bin"]
    hybrid["source"] = scatter + (fission / keff) + fixed


@toggle("hybridMC")
def hybrid_tilt_source(t, x, y, z, P_arr, Q, mcdc):
    P = P_arr[0]
    hybrid = mcdc["technique"]["hybrid"]
    score_list = hybrid["score_list"]
    score_bin = hybrid["score"]
    mesh = hybrid["mesh"]
    dx = mesh["x"][x + 1] - mesh["x"][x]
    dy = mesh["y"][y + 1] - mesh["y"][y]
    dz = mesh["z"][z + 1] - mesh["z"][z]
    x_mid = mesh["x"][x] + (0.5 * dx)
    y_mid = mesh["y"][y] + (0.5 * dy)
    z_mid = mesh["z"][z] + (0.5 * dz)
    # linear x-component
    if score_list["source-x"]:
        Q += score_bin["source-x"]["bin"][:, t, x, y, z] * (P["x"] - x_mid)
    # linear y-component
    if score_list["source-y"]:
        Q += score_bin["source-y"]["bin"][:, t, x, y, z] * (P["y"] - y_mid)
    # linear z-component
    if score_list["source-z"]:
        Q += score_bin["source-z"]["bin"][:, t, x, y, z] * (P["z"] - z_mid)


@toggle("hybridMC")
def hybrid_distribute_sources(mcdc):
    """
    This function is meant to distribute hybrid_total_source to the relevant
    invidual source contributions, e.x. source_total -> source, source-x,
    source-y, source-z, etc.

    """
    hybrid = mcdc["technique"]["hybrid"]
    total_source = hybrid["total_source"].copy()
    shape = hybrid["source"].shape
    size = hybrid["source"].size
    score_list = hybrid["score_list"]
    score_bin = hybrid["score"]
    Vsize = 0

    # effective source
    hybrid["source"] = np.reshape(total_source[Vsize : (Vsize + size)].copy(), shape)
    Vsize += size

    # source tilting arrays
    tilt_list = [
        "source-x",
        "source-y",
        "source-z",
    ]
    for name in literal_unroll(tilt_list):
        if score_list[name]:
            score_bin[name]["bin"] = np.reshape(
                total_source[Vsize : (Vsize + size)], shape
            )
            Vsize += size


@toggle("hybridMC")
def hybrid_consolidate_sources(mcdc):
    """
    This function is meant to collect the relevant invidual source
    contributions, e.x. source, source-x, source-y, source-z, source-xy, etc.
    and combine them into one vector (source_total)

    """
    hybrid = mcdc["technique"]["hybrid"]
    total_source = hybrid["total_source"]
    size = hybrid["source"].size
    score_list = hybrid["score_list"]
    score_bin = hybrid["score"]
    Vsize = 0

    # effective source
    total_source[Vsize : (Vsize + size)] = np.reshape(hybrid["source"].copy(), size)
    Vsize += size

    # source tilting arrays
    tilt_list = [
        "source-x",
        "source-y",
        "source-z",
    ]
    for name in literal_unroll(tilt_list):
        if score_list[name]:
            total_source[Vsize : (Vsize + size)] = np.reshape(
                score_bin[name]["bin"], size
            )
            Vsize += size


# =============================================================================
# Tally Operations
# =============================================================================
# TODO: Not all ST tallies have been built for case where SigmaT = 0.0


@toggle("hybridMC")
def hybrid_score_tallies(P_arr, distance, mcdc):
    """
    Tally the scalar flux and linear source tilt.

    """
    P = P_arr[0]
    hybrid = mcdc["technique"]["hybrid"]
    score_list = hybrid["score_list"]
    score_bin = hybrid["score"]
    # Get indices
    mesh = hybrid["mesh"]
    material = mcdc["materials"][P["material_ID"]]
    w = P["hybrid"]["w"]
    SigmaT = material["total"]
    mat_id = P["material_ID"]
    k_eff = mcdc["k_eff"]
    x, y, z, t, outside = mesh_.structured.get_indices(P_arr, mesh)
    if outside:
        return

    dt = dx = dy = dz = 1.0
    if (mesh["t"][t] != -INF) and (mesh["t"][t] != INF):
        dt = mesh["t"][t + 1] - mesh["t"][t]
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF):
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF):
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF):
        dz = mesh["z"][z + 1] - mesh["z"][z]

    dV = dx * dy * dz * dt

    flux = hybrid_flux(SigmaT, w, distance, dV)
    effective_scatter = hybrid_effective_scattering(flux, mat_id, mcdc)
    effective_fission =  hybrid_effective_fission(
        flux, mat_id, mcdc
    ) 
    hybrid["SN"]["uncollided_flux"][:,x,y,z]+= effective_scatter+effective_fission/k_eff
    
    
    current_t_idx = mcdc["technique"]["hybrid"]["time_step_idx"]
    
    prev_t = mcdc["technique"]["hybrid"]["mesh"]["t"][current_t_idx-1]
    
    if P["hybrid"]["birth_time"] < prev_t:
        score_bin["flux"]["bin"][:, t, x, y, z] += flux
    
        # Score effective source tallies
        score_bin["effective-scattering"]["bin"][
            :, t, x, y, z
        ] += hybrid_effective_scattering(flux, mat_id, mcdc)
        score_bin["effective-fission"]["bin"][:, t, x, y, z] += hybrid_effective_fission(
            flux, mat_id, mcdc
        )
    
        if score_list["fission-source"]:
            score_bin["fission-source"]["bin"] += hybrid_fission_source(flux, material)
    
        if score_list["fission-power"]:
            score_bin["fission-power"]["bin"][:, t, x, y, z] += hybrid_fission_power(
                flux, material
            )
    
        if score_list["source-x"]:
            x_mid = mesh["x"][x] + (dx * 0.5)
            tilt = hybrid_linear_tilt(P["ux"], P["x"], dx, x_mid, dy, dz, w, distance, SigmaT)
            score_bin["source-x"]["bin"][:, t, x, y, z] += hybrid_effective_source(
                tilt, mat_id, mcdc
            )
    
        if score_list["source-y"]:
            y_mid = mesh["y"][y] + (dy * 0.5)
            tilt = hybrid_linear_tilt(P["uy"], P["y"], dy, y_mid, dx, dz, w, distance, SigmaT)
            score_bin["source-y"]["bin"][:, t, x, y, z] += hybrid_effective_source(
                tilt, mat_id, mcdc
            )
    
        if score_list["source-z"]:
            z_mid = mesh["z"][z] + (dz * 0.5)
            tilt = hybrid_linear_tilt(P["uz"], P["z"], dz, z_mid, dx, dy, w, distance, SigmaT)
            score_bin["source-z"]["bin"][:, t, x, y, z] += hybrid_effective_source(
                tilt, mat_id, mcdc
            )


@toggle("hybridMC")
def hybrid_flux(SigmaT, w, distance, dV):
    # Score Flux
    if SigmaT.all() > 0.0:
        return w * (1 - np.exp(-(distance * SigmaT))) / (SigmaT * dV)
    else:
        return distance * w / dV


@toggle("hybridMC")
def hybrid_fission_source(phi, material):
    SigmaF = material["fission"]
    nu_f = material["nu_f"]
    return np.sum(nu_f * SigmaF * phi)


@toggle("hybridMC")
def hybrid_fission_power(phi, material):
    SigmaF = material["fission"]
    return SigmaF * phi


@toggle("hybridMC")
def hybrid_effective_fission(phi, mat_id, mcdc):
    """
    Calculate the fission source for use with hybridMC.

    """
    # TODO: Now, only single-nuclide material is allowed
    material = mcdc["nuclides"][mat_id]
    chi_p = material["chi_p"]
    chi_d = material["chi_d"]
    nu_p = material["nu_p"]
    nu_d = material["nu_d"]
    SigmaF = material["fission"]
    F_p = np.dot(chi_p.T, nu_p * SigmaF * phi)
    F_d = np.dot(chi_d.T, (nu_d.T * SigmaF * phi).sum(axis=1))
    F = F_p + F_d

    return F


@toggle("hybridMC")
def hybrid_effective_scattering(phi, mat_id, mcdc):
    """
    Calculate the scattering source for use with hybridMC.

    """
    material = mcdc["materials"][mat_id]
    chi_s = material["chi_s"]
    SigmaS = material["scatter"]
    return np.dot(chi_s.T, SigmaS * phi)


@toggle("hybridMC")
def hybrid_effective_source(phi, mat_id, mcdc):
    S = hybrid_effective_scattering(phi, mat_id, mcdc)
    F = hybrid_effective_fission(phi, mat_id, mcdc)
    return S + F


@toggle("hybridMC")
def hybrid_linear_tilt(mu, x, dx, x_mid, dy, dz, w, distance, SigmaT):
    if SigmaT.all() > 1e-12:
        a = mu * (
            w * (1 - (1 + distance * SigmaT) * np.exp(-SigmaT * distance)) / SigmaT**2
        )
        b = (x - x_mid) * (w * (1 - np.exp(-SigmaT * distance)) / SigmaT)
        Q = 12 * (a + b) / (dx**3 * dy * dz)
    else:
        Q = mu * w * distance ** (2) / 2 + w * (x - x_mid) * distance
    return Q


@toggle("hybridMC")
def hybrid_reset_tallies(hybrid):
    score_bin = hybrid["score"]
    score_list = hybrid["score_list"]

    hybrid["source"].fill(0.0)
    for name in literal_unroll(hybrid_score_list):
        if score_list[name]:
            score_bin[name]["bin"].fill(0.0)


@toggle("hybridMC")
def hybrid_reduce_tallies(hybrid):
    score_bin = hybrid["score"]
    score_list = hybrid["score_list"]

    for name in literal_unroll(hybrid_score_list):
        if score_list[name]:
            allreduce_array(score_bin[name]["bin"])


# =============================================================================
# Tally History Operations
# =============================================================================


@toggle("hybridMC")
def hybrid_tally_closeout_history(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    score_bin = hybrid["score"]
    score_list = hybrid["score_list"]

    for name in literal_unroll(hybrid_score_list):
        if score_list[name]:
            score_bin[name]["mean"] += score_bin[name]["bin"]
            score_bin[name]["sdev"] += np.square(score_bin[name]["bin"])


@toggle("hybridMC")
def hybrid_tally_closeout(mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    score_bin = hybrid["score"]
    score_list = hybrid["score_list"]

    if hybrid["mode"] == "fixed":
        for name in literal_unroll(hybrid_score_list):
            if score_list[name]:
                score_bin[name]["mean"] = score_bin[name]["bin"]

    if hybrid["mode"] == "batched":
        N_history = mcdc["setting"]["N_active"]
        for name in literal_unroll(hybrid_score_list):
            if score_list[name]:
                score_bin[name]["mean"] /= N_history
                allreduce_array(score_bin[name]["sdev"])
                score_bin[name]["sdev"] = np.sqrt(
                    (
                        score_bin[name]["sdev"] / N_history
                        - np.square(score_bin[name]["mean"])
                    )
                    / (N_history - 1)
                )


@toggle("hybridMC")
def hybrid_eigenvalue_tally_closeout_history(mcdc):
    idx_cycle = mcdc["idx_cycle"]

    # store outter iteration values
    mcdc["k_cycle"][idx_cycle] = mcdc["k_eff"]

    # Accumulate running average
    if mcdc["cycle_active"]:
        mcdc["k_avg"] += mcdc["k_eff"]
        mcdc["k_sdv"] += mcdc["k_eff"] * mcdc["k_eff"]
        N = mcdc["idx_cycle"] - mcdc["setting"]["N_inactive"]
        mcdc["k_avg_running"] = mcdc["k_avg"] / N
        if N == 1:
            mcdc["k_sdv_running"] = 0.0
        else:
            mcdc["k_sdv_running"] = math.sqrt(
                (mcdc["k_sdv"] / N - mcdc["k_avg_running"] ** 2) / (N - 1)
            )


# =============================================================================
# Misc
# =============================================================================


@toggle("hybridMC")
def hybrid_res(source_new, source_old):
    """
    Calculate residual between iterations.

    """
    size = source_new.size
    source_new = np.linalg.norm(source_new.reshape((size,)), ord=2)
    source_old = np.linalg.norm(source_old.reshape((size,)), ord=2)
    return (source_new - source_old) / source_old
