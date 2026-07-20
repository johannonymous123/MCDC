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
    distribute_work,
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
        samples = halton(N, dim, skip=N_start)
        # samples = samples[samples[:, 0].argsort()]
        mcdc["technique"]["hybrid"]["samples"] = samples
    if mcdc["technique"]["hybrid"]["sample_method"] == "random":
        samples = random(N, dim)
        # samples = samples[samples[:, 0].argsort()]
        mcdc["technique"]["hybrid"]["samples"] = samples


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
    # detect boundary conditions at mesh faces
    hybrid_generate_boundary_bc(mcdc)
    if np.all(hybrid["source"] == 0.0):
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
def hybrid_generate_boundary_bc(mcdc):
    """
    Detect boundary conditions at the 6 faces of the hybrid mesh domain.
    Iterates over all surfaces and checks which ones are at the mesh boundaries.
    For plane-x surfaces: x_position = -J
    For plane-y surfaces: y_position = -J
    For plane-z surfaces: z_position = -J
    """
    mesh = mcdc["technique"]["hybrid"]["mesh"]
    sn = mcdc["technique"]["hybrid"]["SN"]

    # Default all BCs to vacuum
    sn["bc_x_low"] = BC_VACUUM
    sn["bc_x_high"] = BC_VACUUM
    sn["bc_y_low"] = BC_VACUUM
    sn["bc_y_high"] = BC_VACUUM
    sn["bc_z_low"] = BC_VACUUM
    sn["bc_z_high"] = BC_VACUUM

    # Mesh boundary coordinates
    x_low = mesh["x"][0]
    x_high = mesh["x"][-1]
    y_low = mesh["y"][0]
    y_high = mesh["y"][-1]
    z_low = mesh["z"][0]
    z_high = mesh["z"][-1]

    # Tolerance for matching surface position to mesh boundary
    tol = COINCIDENCE_TOLERANCE

    # Iterate over all surfaces
    N_surface = len(mcdc["surfaces"])
    for i in range(N_surface):
        surface = mcdc["surfaces"][i]
        surface_type = surface["type"]
        bc = surface["BC"]

        # Surface types are bitflags - use bitwise AND to check
        # For plane-x surfaces: position = -J
        if surface_type & SURFACE_PLANE_X:
            pos = -surface["J"]
            if abs(pos - x_low) < tol:
                sn["bc_x_low"] = bc
            elif abs(pos - x_high) < tol:
                sn["bc_x_high"] = bc

        # For plane-y surfaces: position = -J
        elif surface_type & SURFACE_PLANE_Y:
            pos = -surface["J"]
            if abs(pos - y_low) < tol:
                sn["bc_y_low"] = bc
            elif abs(pos - y_high) < tol:
                sn["bc_y_high"] = bc

        # For plane-z surfaces: position = -J
        elif surface_type & SURFACE_PLANE_Z:
            pos = -surface["J"]
            if abs(pos - z_low) < tol:
                sn["bc_z_low"] = bc
            elif abs(pos - z_high) < tol:
                sn["bc_z_high"] = bc


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
                    fission_source += hybrid_fission_source(
                        flux[:, t, i, j, k], material
                    )


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
                    fission[:, t, i, j, k] = hybrid_effective_fission(
                        flux, mat_idx, mcdc
                    )
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
def hybrid_initialize_particle_state(P_arr, hybridized, scatter_count):
    """Initialize the per-history state used to choose MC or S_N transport."""
    P = P_arr[0]
    P["hybrid"]["birth_weight"] = P["w"]
    P["hybrid"]["p_scatter"] = scatter_count
    P["hybrid"]["mat_scatter"] = 0
    P["hybrid"]["last_material_ID"] = -1
    P["hybrid"]["hybridized"] = hybridized


@toggle("hybridMC")
def hybrid_prepare_particles(mcdc):
    # number of particles this processor will handle
    N_work = mcdc["mpi_work_size"]
    N_Q, N_I, N_B, N_P = distribute_particles(N_work, mcdc)
    if N_Q > 0:
        hybrid_prepare_domain_particles(N_Q, mcdc)
    if N_I > 0:
        hybrid_prepare_init_particles(N_Q, N_Q + N_I, mcdc)
    if N_P > 0:
        hybrid_prepare_point_particles(N_Q + N_I, N_Q + N_I + N_P, mcdc)
    N_start = N_Q + N_I + N_P
    for i in range(6):
        if N_B[i] > 0:
            N_end = N_start + N_B[i]
            hybrid_prepare_boundary_particles(N_start, N_end, i, mcdc)
            N_start = N_end


@toggle("hybridMC")
def hybrid_prepare_domain_particles(N_Q, mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    # total number of particles
    N_particle = N_Q  # mcdc["setting"]["N_particle"]

    samples = hybrid["samples"]
    # source
    Q = hybrid["source"]
    mesh = hybrid["mesh"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    Ng = mesh["Ng"]
    # total number of spatial cells
    # Find nonzero indices in Q to determine bounds
    nonzero = np.argwhere(Q != 0)
    if nonzero.size == 0:
        # Fallback to full mesh if Q is all zero
        xa = mesh["x"][0]
        xb = mesh["x"][-1]
        ya = mesh["y"][0]
        yb = mesh["y"][-1]
        za = mesh["z"][0]
        zb = mesh["z"][-1]
        x_min_idx, x_max_idx = 0, Nx - 1
        y_min_idx, y_max_idx = 0, Ny - 1
        z_min_idx, z_max_idx = 0, Nz - 1
    else:
        # nonzero indices: [g, t, x, y, z]
        x_min_idx = np.min(nonzero[:, 2])
        x_max_idx = np.max(nonzero[:, 2])
        y_min_idx = np.min(nonzero[:, 3])
        y_max_idx = np.max(nonzero[:, 3])
        z_min_idx = np.min(nonzero[:, 4])
        z_max_idx = np.max(nonzero[:, 4])
        xa = mesh["x"][x_min_idx]
        xb = mesh["x"][x_max_idx + 1]
        ya = mesh["y"][y_min_idx]
        yb = mesh["y"][y_max_idx + 1]
        za = mesh["z"][z_min_idx]
        zb = mesh["z"][z_max_idx + 1]

    ta = mesh["t"][0]
    tb = mesh["t"][-1]

    # Number of cells in each dimension between bounds
    Nx_eff = x_max_idx - x_min_idx + 1
    Ny_eff = y_max_idx - y_min_idx + 1
    Nz_eff = z_max_idx - z_min_idx + 1

    N_total = Nt * Ng * Nx_eff * Ny_eff * Nz_eff
    g = mesh["g"]
    g_coarse = mesh["g_coarse"]

    for n in range(N_Q):
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)
        g_idx = int(samples[n, 0] * Ng)
        P_new["g"] = g_idx
        # Find the coarse group index for the current fine group
        for i in range(len(g_coarse) - 1):
            if g_coarse[i] <= g[g_idx] < g_coarse[i + 1]:
                P_new["hybrid"]["g_coarse"] = i
                break

        P_new["t"] = hybrid_sample_position(ta, tb, samples[n, 1])
        # P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, samples[n, 2])
        P_new["y"] = hybrid_sample_position(ya, yb, samples[n, 3])
        P_new["z"] = hybrid_sample_position(za, zb, samples[n, 4])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            samples[n, 5], samples[n, 6]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)
        q = Q[g_idx, t, x, y, z].copy()
        dV = hybrid_cell_volume(x, y, z, t, mesh)
        # Source tilt
        # hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["w"] = q * dV * N_total / N_particle
        # P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = P_new["t"]
        hybrid_initialize_particle_state(P_new_arr, False, 0)

        # add to source bank
        if P_new["w"] > 0:
            adapt.add_future(P_new_arr, mcdc)


@toggle("hybridMC")
def hybrid_prepare_domain_particles_backup(N_Q, mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    # total number of particles
    N_particle = N_Q  # mcdc["setting"]["N_particle"]

    samples = hybrid["samples"]
    # source
    Q = hybrid["source"]
    mesh = hybrid["mesh"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    Ng = mesh["Ng"]
    # total number of spatial cells
    # Find nonzero indices in Q to determine bounds
    nonzero = np.argwhere(Q != 0)
    if nonzero.size == 0:
        # Fallback to full mesh if Q is all zero
        xa = mesh["x"][0]
        xb = mesh["x"][-1]
        ya = mesh["y"][0]
        yb = mesh["y"][-1]
        za = mesh["z"][0]
        zb = mesh["z"][-1]
        x_min_idx, x_max_idx = 0, Nx - 1
        y_min_idx, y_max_idx = 0, Ny - 1
        z_min_idx, z_max_idx = 0, Nz - 1
    else:
        # nonzero indices: [g, t, x, y, z]
        x_min_idx = np.min(nonzero[:, 2])
        x_max_idx = np.max(nonzero[:, 2])
        y_min_idx = np.min(nonzero[:, 3])
        y_max_idx = np.max(nonzero[:, 3])
        z_min_idx = np.min(nonzero[:, 4])
        z_max_idx = np.max(nonzero[:, 4])
        xa = mesh["x"][x_min_idx]
        xb = mesh["x"][x_max_idx + 1]
        ya = mesh["y"][y_min_idx]
        yb = mesh["y"][y_max_idx + 1]
        za = mesh["z"][z_min_idx]
        zb = mesh["z"][z_max_idx + 1]

    ta = mesh["t"][0]
    tb = mesh["t"][-1]

    # Number of cells in each dimension between bounds
    Nx_eff = x_max_idx - x_min_idx + 1
    Ny_eff = y_max_idx - y_min_idx + 1
    Nz_eff = z_max_idx - z_min_idx + 1

    N_total = Nt * Ng * Nx_eff * Ny_eff * Nz_eff
    g = mesh["g"]
    g_coarse = mesh["g_coarse"]

    for n in range(N_Q):
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)
        g_idx = int(samples[n, 0] * Ng)
        P_new["g"] = g_idx
        # Find the coarse group index for the current fine group
        for i in range(len(g_coarse) - 1):
            if g_coarse[i] <= g[g_idx] < g_coarse[i + 1]:
                P_new["hybrid"]["g_coarse"] = i
                break

        P_new["t"] = hybrid_sample_position(ta, tb, samples[n, 1])
        # P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, samples[n, 2])
        P_new["y"] = hybrid_sample_position(ya, yb, samples[n, 3])
        P_new["z"] = hybrid_sample_position(za, zb, samples[n, 4])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            samples[n, 5], samples[n, 6]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)
        q = Q[g_idx, t, x, y, z].copy()
        dV = hybrid_cell_volume(x, y, z, t, mesh)
        # Source tilt
        # hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["w"] = q * dV * N_total / N_particle
        # P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = P_new["t"]
        hybrid_initialize_particle_state(P_new_arr, False, 0)

        # add to source bank
        if P_new["w"] > 0:
            adapt.add_future(P_new_arr, mcdc)


@toggle("hybridMC")
def hybrid_prepare_point_particles(N_start, N_end, mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    sources = mcdc["sources"]
    # total number of particles
    N_particle = N_end - N_start
    mesh = hybrid["mesh"]
    g = mesh["g"]
    g_coarse = mesh["g_coarse"]
    samples = hybrid["samples"]

    # Find which source entry to use for each particle
    # Compute cumulative probabilities
    probs = np.array([src["prob"] for src in sources])
    cum_probs = np.cumsum(probs)
    cum_probs /= cum_probs[-1]  # Ensure normalization
    src_idx = 0
    source = sources[src_idx]
    # For each n, determine which source entry to use
    for n in range(N_start, N_end):
        frac = (n - N_start) / (N_end - N_start)
        if frac > cum_probs[src_idx]:
            src_idx += 1
            source = sources[src_idx]

        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)
        # Sample group index according to source["group"] probabilities
        g_idx = np.random.choice(len(source["group"]), p=source["group"])
        P_new["g"] = g_idx
        # Find the coarse group index for the current fine group
        for i in range(len(g_coarse) - 1):
            if g_coarse[i] <= g[g_idx] < g_coarse[i + 1]:
                P_new["hybrid"]["g_coarse"] = i
                break

        P_new["t"] = hybrid_sample_position(
            source["time"][0], source["time"][1], samples[n, 1]
        )
        # P_new["rng_seed"] = 0
        # assign direction
        if source["box"]:
            P_new["x"] = hybrid_sample_position(
                source["box_x"][0], source["box_x"][1], samples[n, 2]
            )
            P_new["y"] = hybrid_sample_position(
                source["box_y"][0], source["box_y"][1], samples[n, 3]
            )
            P_new["z"] = hybrid_sample_position(
                source["box_z"][0], source["box_z"][1], samples[n, 4]
            )

        else:
            P_new["x"] = source["x"]
            P_new["y"] = source["y"]
            P_new["z"] = source["z"]

            # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            samples[n, 5], samples[n, 6]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)

        P_new["w"] = source["prob"] * hybrid["pt_source_total"] / N_particle
        # P_new["w"] = P_new["hybrid"]["w"].sum()
        ta = mesh["t"][0]
        P_new["hybrid"]["birth_time"] = ta - 1
        hybrid_initialize_particle_state(P_new_arr, False, 0)

        # add to source bank
        if P_new["w"] > 0:
            adapt.add_future(P_new_arr, mcdc)
        hybrid["samples"][n, 0] = -1  # Avoids resampling


@toggle("hybridMC")
def hybrid_prepare_init_particles(N_start, N_end, mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    # total number of particles
    N_particle = N_end - N_start

    samples = hybrid["samples"]
    # source
    Q = hybrid["phi0"]
    mesh = hybrid["mesh"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    Ng = mesh["Ng"]
    g = mesh["g"]
    g_coarse = mesh["g_coarse"]

    # total number of spatial cells
    N_total = Nx * Ny * Nz * Nt * Ng
    # outter mesh boundaries for sampling position
    xa = mesh["x"][0]
    xb = mesh["x"][-1]
    ya = mesh["y"][0]
    yb = mesh["y"][-1]
    za = mesh["z"][0]
    zb = mesh["z"][-1]
    ta = mesh["t"][0]
    tb = mesh["t"][-1]

    for n in range(N_start, N_end):
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)

        g_idx = int(samples[n, 0] * Ng)
        P_new["g"] = g_idx
        # Find the coarse group index for the current fine group
        for i in range(len(g_coarse) - 1):
            if g_coarse[i] <= g[g_idx] < g_coarse[i + 1]:
                P_new["hybrid"]["g_coarse"] = i
                break
        P_new["t"] = ta
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, samples[n, 2])
        P_new["y"] = hybrid_sample_position(ya, yb, samples[n, 3])
        P_new["z"] = hybrid_sample_position(za, zb, samples[n, 4])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            samples[n, 5], samples[n, 6]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)
        q = Q[g_idx, x, y, z].copy()
        dV = hybrid_space_volume(x, y, z, mesh)
        # Source tilt
        # hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["w"] = q * dV * N_total / N_particle
        # P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = ta - 1
        hybrid_initialize_particle_state(P_new_arr, False, 0)

        # add to source bank
        if P_new["w"] > 0:
            adapt.add_future(P_new_arr, mcdc)
        hybrid["samples"][n, 0] = -1  # Avoids resampling


@toggle("hybridMC")
def hybrid_prepare_boundary_particles(N_start, N_end, idx, mcdc):
    """
    Create N_particles assigning the position, direction, and group from the
    QMC Low-Discrepency Sequence. Particles are added to the bank_source.

    Particles are prepared as a batch in hybridMC so that we only have to call the
    low-discprenecy sequence function once for fixed-seed mode or once per sweep
    for batched mode.

    """
    hybrid = mcdc["technique"]["hybrid"]
    # total number of particles
    N_particle = N_end - N_start

    samples = hybrid["samples"]

    bdry = [
        "boundary_x_pos",
        "boundary_x_neg",
        "boundary_y_pos",
        "boundary_y_neg",
        "boundary_z_pos",
        "boundary_z_neg",
    ]
    # source
    Q = hybrid[bdry[idx]]
    mesh = hybrid["mesh"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    Ng = mesh["Ng"]
    g = mesh["g"]
    g_coarse = mesh["g_coarse"]

    # total number of spatial cells
    N_total = Nx * Ny * Nz * Nt * Ng
    # outter mesh boundaries for sampling position
    xa = mesh["x"][0]
    xb = mesh["x"][-1]
    ya = mesh["y"][0]
    yb = mesh["y"][-1]
    za = mesh["z"][0]
    zb = mesh["z"][-1]
    ta = mesh["t"][0]
    tb = mesh["t"][-1]

    if idx == 0:
        xb = xa
    if idx == 1:
        xa = xb
    if idx == 2:
        yb = ya
    if idx == 3:
        ya = yb
    if idx == 4:
        zb = za
    if idx == 5:
        za = zb

    for n in range(N_start, N_end):
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group, time, and rng_seed (not used)
        g_idx = int(samples[n, 0] * Ng)
        P_new["g"] = g_idx

        # Find the coarse group index for the current fine group
        for coarse_idx in range(len(g_coarse) - 1):
            if g_coarse[coarse_idx] <= g[g_idx] < g_coarse[coarse_idx + 1]:
                P_new["hybrid"]["g_coarse"] = coarse_idx
                break

        P_new["t"] = hybrid_sample_position(ta, tb, samples[n, 1])
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, samples[n, 2])
        P_new["y"] = hybrid_sample_position(ya, yb, samples[n, 3])
        P_new["z"] = hybrid_sample_position(za, zb, samples[n, 4])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_boundary_direction(
            samples[n, 5], samples[n, 6], idx
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)

        if idx in [4, 5]:
            q = Q[g_idx, t, x, y].copy()
            # total number of spatial cells
            N_total = Nx * Ny * Nt
        elif idx in [2, 3]:
            q = Q[g_idx, t, x, z].copy()
            # total number of spatial cells
            N_total = Nx * Nz * Nt
        elif idx in [0, 1]:
            q = Q[g_idx, t, y, z].copy()
            # total number of spatial cells
            N_total = Ny * Nz * Nt
        dV = hybrid_boundary_volume(x, y, z, t, idx, mesh)
        # Source tilt
        # hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["w"] = q * dV * N_total / N_particle
        # P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = ta - 1
        hybrid_initialize_particle_state(P_new_arr, False, 0)

        # add to source bank
        if P_new["w"] > 0:
            adapt.add_future(P_new_arr, mcdc)
        hybrid["samples"][n, 0] = -1  # Avoids resampling


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

    # N_work = mcdc["mpi_work_size"]

    # low discrepency sequence
    samples = hybrid["samples"]
    # source
    Q = hybrid["source"]
    eff_collided_flux = hybrid["SN"]["collided_flux"]
    eff_uncollided_flux = hybrid["SN"]["uncollided_flux"]
    eff_flux_n_collision = hybrid["SN"]["flux_n_collisions"]

    mesh = hybrid["mesh"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    Ng = mesh["Ng"]
    g = mesh["g"]
    g_coarse = mesh["g_coarse"]

    # total number of spatial cells
    # Compute the support of Q + eff_collided_flux + eff_uncollided_flux + eff_flux_n_collision
    # Create boolean support array of shape (Nx, Ny, Nz)
    # total_support[x,y,z] = 1 if any term is nonzero at [..., x, y, z]

    # Q: (Ng, Nt, Nx, Ny, Nz) -> check if any value is nonzero along (Ng, Nt) axes
    Q_support = np.any(Q != 0, axis=(0, 1))  # shape (Nx, Ny, Nz)

    # eff_collided_flux: (Ng_coarse, x_deg, y_deg, z_deg, Nx, Ny, Nz)
    eff_collided_support = np.any(
        eff_collided_flux != 0, axis=(0, 1, 2, 3)
    )  # shape (Nx, Ny, Nz)

    # eff_uncollided_flux: (Ng, Nx, Ny, Nz)
    eff_uncollided_support = np.any(
        eff_uncollided_flux != 0, axis=0
    )  # shape (Nx, Ny, Nz)

    # eff_flux_n_collision: (Ng, Nx, Ny, Nz)
    eff_flux_n_collision_support = np.any(
        eff_flux_n_collision != 0, axis=0
    )  # shape (Nx, Ny, Nz)

    total_support = (
        Q_support
        | eff_collided_support
        | eff_uncollided_support
        | eff_flux_n_collision_support
    )

    nonzero = np.argwhere(total_support)
    if nonzero.size == 0:
        # Fallback to full mesh if all zero
        xa = mesh["x"][0]
        xb = mesh["x"][-1]
        ya = mesh["y"][0]
        yb = mesh["y"][-1]
        za = mesh["z"][0]
        zb = mesh["z"][-1]
        x_min_idx, x_max_idx = 0, Nx - 1
        y_min_idx, y_max_idx = 0, Ny - 1
        z_min_idx, z_max_idx = 0, Nz - 1
    else:
        # nonzero indices: [x, y, z]
        x_min_idx = np.min(nonzero[:, 0])
        x_max_idx = np.max(nonzero[:, 0])
        y_min_idx = np.min(nonzero[:, 1])
        y_max_idx = np.max(nonzero[:, 1])
        z_min_idx = np.min(nonzero[:, 2])
        z_max_idx = np.max(nonzero[:, 2])
        xa = mesh["x"][x_min_idx]
        xb = mesh["x"][x_max_idx + 1]
        ya = mesh["y"][y_min_idx]
        yb = mesh["y"][y_max_idx + 1]
        za = mesh["z"][z_min_idx]
        zb = mesh["z"][z_max_idx + 1]

    # Number of cells in each dimension between bounds
    Nx_eff = x_max_idx - x_min_idx + 1
    Ny_eff = y_max_idx - y_min_idx + 1
    Nz_eff = z_max_idx - z_min_idx + 1

    N_total = Ng * Nx_eff * Ny_eff * Nz_eff
    ta = mesh["t"][0]
    tb = mesh["t"][-1]
    t_prev = (mesh["t"][hybrid["time_step_idx"] - 1] - ta) / (tb - ta)
    t_curr = (mesh["t"][hybrid["time_step_idx"]] - ta) / (tb - ta)
    # t_idx = hybrid["time_step_idx"]

    ###############
    ###############

    # LDS column layout (must match initial sampling in hybrid_prepare_domain_particles):
    # col 0 = group, col 1 = time, col 2 = x, col 3 = y, col 4 = z, col 5/6 = direction
    LDS_COL_GROUP = 0
    LDS_COL_TIME = 1
    LDS_COL_X = 2
    LDS_COL_Y = 3
    LDS_COL_Z = 4
    LDS_COL_MU = 5
    LDS_COL_AZI = 6

    t_prev = (mesh["t"][hybrid["time_step_idx"] - 1] - ta) / (tb - ta)
    t_curr = (mesh["t"][hybrid["time_step_idx"]] - ta) / (tb - ta)

    mask = (hybrid["samples"][:, LDS_COL_TIME] >= t_prev) & (
        hybrid["samples"][:, LDS_COL_TIME] < t_curr
    )
    # Number of particles to reset in this time step
    N_particle = np.count_nonzero(mask)
    indices = np.where(mask)[0]
    for n in indices:
        # Create new particle
        P_new_arr = adapt.local_array(1, type_.particle_record)
        P_new = P_new_arr[0]
        # assign initial group using same column as initial sampling
        g_idx = int(hybrid["samples"][n, LDS_COL_GROUP] * Ng)
        P_new["g"] = g_idx

        # Find the coarse group index for the current fine group
        for coarse_idx in range(len(g_coarse) - 1):
            if g_coarse[coarse_idx] <= g[g_idx] < g_coarse[coarse_idx + 1]:
                P_new["hybrid"]["g_coarse"] = coarse_idx
                break

        P_new["t"] = hybrid_sample_position(ta, tb, hybrid["samples"][n, LDS_COL_TIME])
        P_new["rng_seed"] = 0
        # assign direction
        P_new["x"] = hybrid_sample_position(xa, xb, hybrid["samples"][n, LDS_COL_X])
        P_new["y"] = hybrid_sample_position(ya, yb, hybrid["samples"][n, LDS_COL_Y])
        P_new["z"] = hybrid_sample_position(za, zb, hybrid["samples"][n, LDS_COL_Z])
        # Sample isotropic direction
        P_new["ux"], P_new["uy"], P_new["uz"] = hybrid_sample_isotropic_direction(
            hybrid["samples"][n, LDS_COL_MU], hybrid["samples"][n, LDS_COL_AZI]
        )
        x, y, z, t, outside = mesh_.structured.get_indices(P_new_arr, mesh)

        x_deg, y_deg, z_deg = eff_collided_flux[coarse_idx, ..., x, y, z].shape

        phi_x = np.polynomial.legendre.legval(
            (P_new["x"] - mesh["x"][x]) / (mesh["x"][x] - mesh["x"][x + 1]),
            np.eye(x_deg),
        )  # shape (ni,)
        phi_y = np.polynomial.legendre.legval(
            (P_new["y"] - mesh["y"][y]) / (mesh["y"][y] - mesh["y"][y + 1]),
            np.eye(y_deg),
        )  # shape (nj,)
        phi_z = np.polynomial.legendre.legval(
            (P_new["z"] - mesh["z"][z]) / (mesh["z"][z] - mesh["z"][z + 1]),
            np.eye(z_deg),
        )  # shape (nk,)

        group_ratio = (g[g_idx] - g[g_idx + 1]) / (
            g_coarse[coarse_idx] - g_coarse[coarse_idx + 1]
        )
        f = (
            group_ratio
            * np.einsum(
                "gijk,i,j,k->g", eff_collided_flux[..., x, y, z], phi_x, phi_y, phi_z
            )[coarse_idx]
        )

        q = (
            Q[g_idx, t, x, y, z].copy()
            + max([f, 0])
            + eff_flux_n_collision[g_idx, x, y, z].copy()
            + eff_uncollided_flux[g_idx, x, y, z].copy()
        )
        dV = hybrid_cell_volume(x, y, z, t, mesh)
        # Source tilt
        hybrid_tilt_source(t, x, y, z, P_new_arr, q, mcdc)
        # set particle weight
        P_new["w"] = q * dV * N_total / N_particle
        # P_new["w"] = P_new["hybrid"]["w"].sum()
        P_new["hybrid"]["birth_time"] = ta - 1
        hybrid_initialize_particle_state(P_new_arr, True, hybrid["n_scatter"])

        # add to source bank
        if P_new["w"] > 0:
            adapt.add_source(P_new_arr, mcdc)


@toggle("hybridMC")
def distribute_particles(N_work, mcdc):
    hybrid = mcdc["technique"]["hybrid"]
    mesh = hybrid["mesh"]

    # Compute mesh spacings, handling INF/-INF boundaries
    def safe_diff(arr):
        diff = np.diff(arr)
        for i in range(len(diff)):
            if (
                arr[i] == INF
                or arr[i] == -INF
                or arr[i + 1] == INF
                or arr[i + 1] == -INF
            ):
                diff[i] = 1
        return diff

    dt = safe_diff(mesh["t"])
    dx = safe_diff(mesh["x"])
    dy = safe_diff(mesh["y"])
    dz = safe_diff(mesh["z"])
    Q = hybrid["source"]
    Phi_init = hybrid["phi0"]
    Bdry_x_p = hybrid["boundary_x_pos"]
    Bdry_x_n = hybrid["boundary_x_neg"]
    Bdry_y_p = hybrid["boundary_y_pos"]
    Bdry_y_n = hybrid["boundary_y_neg"]
    Bdry_z_p = hybrid["boundary_z_pos"]
    Bdry_z_n = hybrid["boundary_z_neg"]
    PT_source = hybrid["pt_source_total"]

    vol_space = np.outer(dx, np.outer(dy, dz)).reshape(
        len(dx), len(dy), len(dz)
    )  # (nx, ny, nz)
    vol_spacetime = np.outer(dt, vol_space.ravel()).reshape(
        len(dt), len(dx), len(dy), len(dz)
    )  # (nt, nx, ny, nz)

    # Phi_init: (ng, nx, ny, nz)
    integral_phi_init = np.sum(Phi_init * vol_space, axis=(1, 2, 3))  # shape (ng,)
    total_phi_init = np.sum(integral_phi_init)  # scalar

    integral_Q = np.sum(Q * vol_spacetime, axis=(1, 2, 3, 4))  # shape (ng,)
    total_Q = np.sum(integral_Q)

    # x boundaries: (nt, ny, nz)
    area_yz = np.outer(dy, dz).reshape(len(dy), len(dz))  # (ny, nz)
    bdry_vol_x = np.outer(dt, area_yz.ravel()).reshape(len(dt), len(dy), len(dz))
    integral_bdry_x_pos = np.sum(Bdry_x_p * bdry_vol_x, axis=(1, 2, 3))  # (ng,)
    total_bdry_x_pos = np.sum(integral_bdry_x_pos)
    integral_bdry_x_neg = np.sum(Bdry_x_n * bdry_vol_x, axis=(1, 2, 3))  # (ng,)
    total_bdry_x_neg = np.sum(integral_bdry_x_neg)

    # y boundaries: (nt, nx, nz)
    area_xz = np.outer(dx, dz).reshape(len(dx), len(dz))  # (nx, nz)
    bdry_vol_y = np.outer(dt, area_xz.ravel()).reshape(len(dt), len(dx), len(dz))
    integral_bdry_y_pos = np.sum(Bdry_y_p * bdry_vol_y, axis=(1, 2, 3))  # (ng,)
    total_bdry_y_pos = np.sum(integral_bdry_y_pos)
    integral_bdry_y_neg = np.sum(Bdry_y_n * bdry_vol_y, axis=(1, 2, 3))  # (ng,)
    total_bdry_y_neg = np.sum(integral_bdry_y_neg)

    # z boundaries: (nt, nx, ny)
    area_xy = np.outer(dx, dy).reshape(len(dx), len(dy))  # (nx, ny)
    bdry_vol_z = np.outer(dt, area_xy.ravel()).reshape(len(dt), len(dx), len(dy))
    integral_bdry_z_pos = np.sum(Bdry_z_p * bdry_vol_z, axis=(1, 2, 3))  # (ng,)
    total_bdry_z_pos = np.sum(integral_bdry_z_pos)
    integral_bdry_z_neg = np.sum(Bdry_z_n * bdry_vol_z, axis=(1, 2, 3))  # (ng,)
    total_bdry_z_neg = np.sum(integral_bdry_z_neg)

    T_B = np.array(
        [
            total_bdry_x_pos,
            total_bdry_x_neg,
            total_bdry_y_pos,
            total_bdry_y_neg,
            total_bdry_z_pos,
            total_bdry_z_neg,
        ]
    )
    T_total = total_phi_init + total_Q + np.sum(T_B) + PT_source

    # Handle T_total == 0 to avoid division by zero
    if T_total == 0:
        w_Q = 1.0
        w_I = 0.0
        w_B = np.zeros(6)
        w_P = 0.0
    else:
        # Weight for Q (source)
        w_Q = 0.5 + 0.5 * (total_Q / T_total)

        # Normalize weights for I and B, handle denominator == 0
        denom = total_phi_init + np.sum(T_B) + PT_source
        if denom == 0:
            w_I = 0.0
            w_B = np.zeros(6)
            w_P = 0.0
        else:
            w_I = total_phi_init / denom
            w_B = T_B / denom  # vector of 6
            w_P = PT_source / denom
    N_Q = int(round(N_work * w_Q))

    # Remaining work
    N_rest = N_work - N_Q

    # Allocate
    N_I = int(round(N_rest * w_I))
    N_B = np.round(N_rest * w_B).astype(int)  # shape (6,)
    N_P = int(round(N_rest * w_P))

    # Fix rounding drift
    drift = N_work - (N_Q + N_I + N_P + np.sum(N_B))
    N_Q += drift  # or fix most contributing part
    return N_Q, N_I, N_B, N_P


@toggle("hybridMC")
def sn_init(mcdc):
    # Distribute work based on n_directions for ordinate initialization
    n_directions = mcdc["technique"]["hybrid"]["SN"]["n_directions"]
    distribute_work(n_directions, mcdc)
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
    if shape[1] == 2:
        sn["ordinates"][:, 0] = nodes[work_start : work_start + work_size]
        sn["ordinates"][:, 1] = weights[work_start : work_start + work_size]
    else:
        n_polar = 2 * sn["n_ordinates"]
        angles = np.linspace(0, 2 * np.pi, n_polar, endpoint=False) + np.pi / n_polar
        unit_circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
        if shape[1] == 3:
            nodes = nodes[len(nodes) // 2 :]
            weights = weights[len(weights) // 2 :]
            ordinates = np.array(
                [
                    (
                        np.sqrt(1 - node**2) * x,
                        np.sqrt(1 - node**2) * y,
                        1 / n_polar * weight,
                    )
                    for (x, y) in unit_circle_points
                    for node, weight in zip(nodes, weights)
                ]
            )
        if shape[1] == 4:
            ordinates = np.array(
                [
                    (
                        np.sqrt(1 - node**2) * x,
                        np.sqrt(1 - node**2) * y,
                        node,
                        1 / n_polar * weight,
                    )
                    for (x, y) in unit_circle_points
                    for node, weight in zip(nodes, weights)
                ]
            )
        sn["ordinates"] = ordinates[work_start : work_start + work_size, :]

    # Build ordinate reflection mappings for reflective BCs
    ordinates_init_reflection_maps(mcdc)


@toggle("hybridMC")
def ordinates_init_reflection_maps(mcdc):
    """
    Build reflection mappings for each direction.
    For x-reflection: find ordinate j such that Omega_x(j) = -Omega_x(i)
    and Omega_y(j) = Omega_y(i), Omega_z(j) = Omega_z(i)

    For 1D (shape[1]==2): ordinates[:,0] is mu (x-direction), symmetric around 0
    For 2D (shape[1]==3): ordinates[:,0] is Omega_x, ordinates[:,1] is Omega_y
    For 3D (shape[1]==4): ordinates[:,0] is Omega_x, [:,1] is Omega_y, [:,2] is Omega_z
    """
    sn = mcdc["technique"]["hybrid"]["SN"]
    ordinates = sn["ordinates"]
    n_ord = sn["n_directions"]
    shape = ordinates.shape

    # Initialize to identity (self-reflection) as fallback
    for i in range(n_ord):
        sn["reflect_x"][i] = i
        sn["reflect_y"][i] = i
        sn["reflect_z"][i] = i

    tol = 1e-10

    if shape[1] == 2:
        # 1D: only x-direction, ordinates[:,0] = mu
        # For Gauss-Legendre, nodes are symmetric: if mu[i] exists, -mu[i] also exists
        for i in range(n_ord):
            mu_i = ordinates[i, 0]
            for j in range(n_ord):
                mu_j = ordinates[j, 0]
                if abs(mu_j + mu_i) < tol:  # mu_j = -mu_i
                    sn["reflect_x"][i] = j
                    break

    elif shape[1] == 3:
        # 2D: ordinates[:,0] = Omega_x, ordinates[:,1] = Omega_y
        for i in range(n_ord):
            ox_i, oy_i = ordinates[i, 0], ordinates[i, 1]
            for j in range(n_ord):
                ox_j, oy_j = ordinates[j, 0], ordinates[j, 1]
                # X-reflection: Omega_x -> -Omega_x, Omega_y unchanged
                if abs(ox_j + ox_i) < tol and abs(oy_j - oy_i) < tol:
                    sn["reflect_x"][i] = j
                # Y-reflection: Omega_y -> -Omega_y, Omega_x unchanged
                if abs(ox_j - ox_i) < tol and abs(oy_j + oy_i) < tol:
                    sn["reflect_y"][i] = j

    elif shape[1] == 4:
        # 3D: ordinates[:,0] = Omega_x, [:,1] = Omega_y, [:,2] = Omega_z
        for i in range(n_ord):
            ox_i, oy_i, oz_i = ordinates[i, 0], ordinates[i, 1], ordinates[i, 2]
            for j in range(n_ord):
                ox_j, oy_j, oz_j = ordinates[j, 0], ordinates[j, 1], ordinates[j, 2]
                # X-reflection
                if (
                    abs(ox_j + ox_i) < tol
                    and abs(oy_j - oy_i) < tol
                    and abs(oz_j - oz_i) < tol
                ):
                    sn["reflect_x"][i] = j
                # Y-reflection
                if (
                    abs(ox_j - ox_i) < tol
                    and abs(oy_j + oy_i) < tol
                    and abs(oz_j - oz_i) < tol
                ):
                    sn["reflect_y"][i] = j
                # Z-reflection
                if (
                    abs(ox_j - ox_i) < tol
                    and abs(oy_j - oy_i) < tol
                    and abs(oz_j + oz_i) < tol
                ):
                    sn["reflect_z"][i] = j


@toggle("hybridMC")
def tensor_init(mcdc):
    sn = mcdc["technique"]["hybrid"]["SN"]
    x_deg = sn["x_degree"]
    y_deg = sn["y_degree"]
    z_deg = sn["z_degree"]
    if x_deg > -1:
        build_tensor(mcdc, 1, 0)
    else:
        sn["tensor_x"] += 1

    if y_deg > -1:
        build_tensor(mcdc, 2 if x_deg > -1 else 1, 1)
    else:
        sn["tensor_y"] += 1
    if z_deg > -1:
        degree = (
            3 if x_deg > -1 and y_deg > -1 else 2 if x_deg > -1 or y_deg > -1 else 1
        )
        build_tensor(mcdc, degree, 2)
    else:
        sn["tensor_z"] += 1


def build_tensor(mcdc, flag, axis):
    """eq for ceof is given as
    (Omega*B+sigma_t N)coef = P neighbor_coef + sigma_s N old_coef in every dimmension
    This sets up tensors to invert this equation in a simple functional form
    """
    sn = mcdc["technique"]["hybrid"]["SN"]
    tensor = ["tensor_x", "tensor_y", "tensor_z"][axis]
    deg = ["x_degree", "y_degree", "z_degree"][axis]
    deg = sn[deg]
    directions = sn["ordinates"].shape[1] - 1
    if deg == 0:
        Ip = In = np.array([[1]])
    else:
        # Initialize an (n+1) x (n+1) matrix with zeros
        Ip = np.zeros((deg + 1, deg + 1))

        # Set the required entries
        Ip[0, 0] = 0.5  # Top-left entry
        Ip[deg, deg] = 0.5  # Bottom-right entry

        # Set the first lower and upper diagonals
        np.fill_diagonal(Ip[1:], 0.5)  # Lower diagonal
        np.fill_diagonal(Ip[:, 1:], -0.5)  # Upper diagonal

        In = Ip.copy()
        In[0, 0] -= 1
        In[-1, -1] -= 1
        # Bn[-1,0] = Bn[-1,0] - 2 * (-1)**deg

    N_inv = np.diag([2 * i + 1 for i in range(deg + 1)])
    N = np.diag([1 / (2 * i + 1) for i in range(deg + 1)])
    Pp = np.fromfunction(lambda i, j: (-1) ** i, (deg + 1, deg + 1), dtype=int)
    Pn = -Pp.T

    alternating = np.array([(-1) ** i for i in range(deg + 1)])
    D = np.diag(alternating)

    sn[tensor][:, :, 0, -1] = Ip @ N
    sn[tensor][:, :, 1, -1] = In @ N
    sn[tensor][:, :, 0, -2] = Ip @ Pp
    sn[tensor][:, :, 1, -2] = In @ Pn
    sn[tensor][:, :, 0, 0] = np.eye(deg + 1)  # np.linalg.inv(Bp.T)
    sn[tensor][:, :, 1, 0] = np.eye(deg + 1)  # np.linalg.inv(Bn.T)
    sn[tensor][:, :, 0, 1] = D
    sn[tensor][:, :, 1, 1] = D


@toggle("hybridMC")
def hybrid_cell_volume(x, y, z, t, mesh):
    """
    Calculate the volume of the cartesian spatial cell.

    """
    dx = dy = dz = dt = 1
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF):
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF):
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF):
        dz = mesh["z"][z + 1] - mesh["z"][z]
    if (mesh["t"][t] != -INF) and (mesh["t"][t] != INF):
        dt = mesh["t"][t + 1] - mesh["t"][t]

    dV = dx * dy * dz * dt
    return dV


@toggle("hybridMC")
def hybrid_space_volume(x, y, z, mesh):
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
def hybrid_boundary_volume(x, y, z, t, idx, mesh):
    """
    Calculate the volume of the cartesian spatial cell.

    """
    dx = dy = dz = dt = 1
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF) and idx > 1:
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF) and not 2 <= idx < 4:
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF) and idx < 4:
        dz = mesh["z"][z + 1] - mesh["z"][z]
    if (mesh["t"][t] != -INF) and (mesh["t"][t] != INF):
        dt = mesh["t"][t + 1] - mesh["t"][t]

    dV = dx * dy * dz * dt
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


def hybrid_sample_boundary_direction(sample1, sample2, idx):
    azi = 2.0 * PI * sample2
    mu = (-1) ** idx * np.sqrt(sample1)
    c = (1.0 - mu**2) ** 0.5

    if idx < 2:
        uy = math.cos(azi) * c
        uz = math.sin(azi) * c
        ux = mu
    if 2 <= idx < 4:
        ux = math.cos(azi) * c
        uz = math.sin(azi) * c
        uy = mu
    if 4 <= idx:
        uy = math.cos(azi) * c
        ux = math.sin(azi) * c
        uz = mu
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

# Select one experimental label rule here. Keep exactly one assignment
# uncommented; these are intentionally source-level switches, not input options.
HYBRIDIZATION_STANDARD = 0
HYBRIDIZATION_MATERIAL = 1
HYBRIDIZATION_WEIGHT = 2
HYBRIDIZATION_DIRECTION = 3

HYBRIDIZATION_MODE = HYBRIDIZATION_STANDARD
# HYBRIDIZATION_MODE = HYBRIDIZATION_MATERIAL
# HYBRIDIZATION_MODE = HYBRIDIZATION_WEIGHT
# HYBRIDIZATION_MODE = HYBRIDIZATION_DIRECTION


@toggle("hybridMC")
def hybrid_material_scatter_limit(material_ID, n_scatter_max):
    """Return the hard-coded local scatter limit for the current material."""
    # Reed baseline: the central [-2, 2] material is created first and therefore
    # has material_ID 0. All other materials use the global per-step maximum.
    if material_ID == 0:
        return 1
    return n_scatter_max


@toggle("hybridMC")
def hybrid_update_hybridization_status(P_arr, mcdc):
    """Update the particle's MC/S_N label at an existing transport event."""
    P = P_arr[0]
    hybrid = mcdc["technique"]["hybrid"]
    n_scatter_max = hybrid["n_scatter"]

    # A newly located material starts a new material-local scatter count.
    # The total p_scatter count is intentionally not reset here.
    material_ID = P["material_ID"]
    if P["hybrid"]["last_material_ID"] == -1:
        P["hybrid"]["last_material_ID"] = material_ID
    elif P["hybrid"]["last_material_ID"] != material_ID:
        P["hybrid"]["last_material_ID"] = material_ID
        P["hybrid"]["mat_scatter"] = 0

    max_scatter_reached = P["hybrid"]["p_scatter"] >= n_scatter_max

    if HYBRIDIZATION_MODE == HYBRIDIZATION_MATERIAL:
        # Entering a new material resets mat_scatter, but p_scatter still
        # enforces the overall per-time-step maximum. This rule may relabel a
        # particle as MC again on material entry until that maximum is reached.
        material_limit = hybrid_material_scatter_limit(material_ID, n_scatter_max)
        hybridized = max_scatter_reached or P["hybrid"]["mat_scatter"] >= material_limit
    elif HYBRIDIZATION_MODE == HYBRIDIZATION_WEIGHT:
        # Change label at the first existing event after the particle reaches
        # 10% of its birth weight.
        hybridized = (
            P["hybrid"]["hybridized"]
            or max_scatter_reached
            or abs(P["w"]) <= 0.1 * abs(P["hybrid"]["birth_weight"])
        )
    elif HYBRIDIZATION_MODE == HYBRIDIZATION_DIRECTION:
        # Particles moving left are irreversibly hybridized.
        hybridized = P["hybrid"]["hybridized"] or max_scatter_reached or P["ux"] < 0.0
    else:
        # Preserve the current input-driven n_scatter behavior.
        hybridized = max_scatter_reached

    P["hybrid"]["hybridized"] = hybridized


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

    # Update the MC/S_N label after geometry has identified the material and
    # before this track segment is scored or a collision distance is sampled.
    hybrid_update_hybridization_status(P_arr, mcdc)

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
    # Distance to next collision
    if not P["hybrid"]["hybridized"]:
        d_collision = distance_to_scatter(P_arr, mcdc)
    else:
        d_collision = INF

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

    # CHeck distance to Collision
    if d_collision < distance - COINCIDENCE_TOLERANCE:
        distance = d_collision
        P["event"] = EVENT_COLLISION
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_collision, distance):
        P["event"] += EVENT_COLLISION

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
def distance_to_scatter(P_arr, mcdc):
    P = P_arr[0]
    # Get total cross-section
    material = mcdc["materials"][P["material_ID"]]
    SigmaS = get_MacroXS(XS_SCATTER, material, P_arr, mcdc)

    # Vacuum material?
    if SigmaS == 0.0:
        return INF

    # Sample collision distance
    xi = np.random.rand()
    distance = -math.log(xi) / SigmaS
    return distance


@toggle("hybridMC")
def get_MacroXS(type_, material, P_arr, mcdc):
    P = P_arr[0]
    # Multigroup XS
    g = P["g"]
    if mcdc["setting"]["mode_MG"]:
        # Cross sections
        if type_ == XS_TOTAL:
            return material["total"][g]
        elif type_ == XS_SCATTER:
            return material["scatter"][g]
        elif type_ == XS_CAPTURE:
            return material["capture"][g]
        elif type_ == XS_FISSION:
            return material["fission"][g]

        # Productions
        elif type_ == XS_NU_SCATTER:
            nu = material["nu_s"][g]
            scatter = material["scatter"][g]
            return nu * scatter
        elif type_ == XS_NU_FISSION:
            nu = material["nu_f"][g]
            fission = material["fission"][g]
            return nu * fission
        elif type_ == XS_NU_FISSION_PROMPT:
            nu_p = material["nu_p"][g]
            fission = material["fission"][g]
            return nu_p * fission
        elif type_ == XS_NU_FISSION_DELAYED:
            nu_d = 0.0
            for j in range(material["J"]):
                nu_d += material["nu_d"][g, j]
            fission = material["fission"][g]
            return nu_d * fission

    # Continuous-energy XS
    MacroXS = 0.0
    E = P["E"]

    # Sum over all nuclides
    for i in range(material["N_nuclide"]):
        ID_nuclide = material["nuclide_IDs"][i]
        nuclide = mcdc["nuclides"][ID_nuclide]

        # Get nuclide density
        N = material["nuclide_densities"][i]

        # Get microscopic cross-section
        microXS = get_microXS(type_, nuclide, E)

        # Accumulate
        MacroXS += N * microXS

    return MacroXS


def hybrid_continuous_weight_reduction(P_arr, distance, mcdc):
    """
    Continuous weight reduction technique based on particle track-length.
    """
    P = P_arr[0]
    material = mcdc["materials"][P["material_ID"]]
    if not P["hybrid"]["hybridized"]:
        Sigma = material["capture"][P["g"]]
    else:
        Sigma = material["total"][P["g"]]

    w = P["w"]
    P["w"] = w * np.exp(-distance * Sigma)
    # P["w"] = P["hybrid"]["w"].sum()


# =============================================================================
# Scattering
# =============================================================================
@toggle("hybridMC")
def scattering(P_arr, prog):
    P = P_arr[0]
    mu0 = 2.0 * np.random.rand() - 1.0

    # Scatter direction
    azi = 2.0 * PI * np.random.rand()
    P["ux"], P["uy"], P["uz"] = scatter_direction(P["ux"], P["uy"], P["uz"], mu0, azi)

    P["hybrid"]["p_scatter"] += 1
    P["hybrid"]["mat_scatter"] += 1


@toggle("hybridMC")
def scatter_direction(ux, uy, uz, mu0, azi):
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu0**2) ** 0.5

    if uz != 1.0:
        B = (1.0 - uz**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uz * cos_azi - uy * sin_azi) * C
        uy_new = uy * mu0 + (uy * uz * cos_azi + ux * sin_azi) * C
        uz_new = uz * mu0 - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the scattering formula
    else:
        B = (1.0 - uy**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uy * cos_azi - uz * sin_azi) * C
        uz_new = uz * mu0 + (uz * uy * cos_azi + ux * sin_azi) * C
        uy_new = uy * mu0 - cos_azi * Ac * B

    return ux_new, uy_new, uz_new


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
    g = P["g"]
    g_coarse = P["hybrid"]["g_coarse"]
    mesh = hybrid["mesh"]
    material = mcdc["materials"][P["material_ID"]]
    w = P["w"]
    mat_id = P["material_ID"]
    k_eff = mcdc["k_eff"]
    x, y, z, t, outside = mesh_.structured.get_indices(P_arr, mesh)
    if outside:
        return

    dV = hybrid_cell_volume(x, y, z, t, mesh)
    SigmaT = material["total"][g]
    SigmaA = material["capture"][g]

    # Choose which cross section to use for flux
    Sigma = SigmaT if P["hybrid"]["hybridized"] else SigmaA
    flux = hybrid_flux(Sigma, w, distance, dV)

    # Effective sources
    eff_scatter = hybrid_effective_scattering(flux, mat_id, mcdc, g)
    eff_fission = hybrid_effective_fission(flux, mat_id, mcdc, g)

    current_t_idx = hybrid["time_step_idx"]
    prev_t = mesh["t"][current_t_idx - 1]

    # Score SN fluxes
    if not P["hybrid"]["hybridized"]:
        if P["hybrid"]["birth_time"] > prev_t:
            hybrid["SN"]["uncollided_flux"][:, x, y, z] += (
                eff_scatter + eff_fission / k_eff
            )
    else:
        hybrid["SN"]["flux_n_collisions"][:, x, y, z] += (
            eff_scatter + eff_fission / k_eff
        )

    # Score tallies only for particles born before prev_t
    pure_mc = (
        hybrid["n_scatter"] >= INF and HYBRIDIZATION_MODE == HYBRIDIZATION_STANDARD
    )
    if P["hybrid"]["birth_time"] < prev_t or pure_mc:
        score_bin["flux"]["bin"][g, t, x, y, z] += flux
        score_bin["effective-scattering"]["bin"][:, t, x, y, z] += eff_scatter
        score_bin["effective-fission"]["bin"][:, t, x, y, z] += eff_fission

        if score_list["fission-source"]:
            score_bin["fission-source"]["bin"][:, t, x, y, z] += hybrid_fission_source(
                flux, material, g
            )

        if score_list["fission-power"]:
            score_bin["fission-power"]["bin"][:, t, x, y, z] += hybrid_fission_power(
                flux, material, g
            )


@toggle("hybridMC")
def condense_groups(source, mcdc):
    """
    Condense the source vector to the coarse group structure.
    """
    hybrid = mcdc["technique"]["hybrid"]
    mesh = hybrid["mesh"]
    if mesh["Ng"] == mesh["Ng_coarse"]:
        return source
    g = mesh["g"][:-1]
    g_coarse = mesh["g_coarse"]
    out = np.zeros(len(g_coarse) - 1)
    for j in range(len(g_coarse) - 1):
        mask = (g >= g_coarse[j]) & (g < g_coarse[j + 1])
        out[j] = np.sum(source[mask])
    return out


@toggle("hybridMC")
def hybrid_flux(SigmaT, w, distance, dV):
    # Score Flux
    if SigmaT.all() > 0.0:
        return w * (1 - np.exp(-(distance * SigmaT))) / (SigmaT * dV)
    else:
        return distance * w / dV


@toggle("hybridMC")
def hybrid_fission_source(phi, material, g=0):
    SigmaF = material["fission"]
    nu_f = material["nu_f"]
    return np.sum(nu_f * SigmaF * phi)


@toggle("hybridMC")
def hybrid_fission_power(phi, material, g=0):
    SigmaF = material["fission"]
    return SigmaF * phi


@toggle("hybridMC")
def hybrid_effective_fission(phi, mat_id, mcdc, g=0):
    """
    Calculate the fission source for use with hybridMC, including prompt and delayed.

    If phi is a scalar, g must be provided and phi is the flux for group g.
    If phi is a vector, phi is the flux for all groups and g is ignored.
    """
    # Prompt part from material
    material = mcdc["materials"][mat_id]
    chi_p = material["chi_p"]
    nu_p = material["nu_p"]
    SigmaF = material["fission"]

    # Delayed part from nuclide
    nuclide = mcdc["nuclides"][mat_id]
    chi_d = nuclide["chi_d"]
    nu_d = nuclide["nu_d"]

    if np.isscalar(phi):
        # Scalar phi: use group g
        F_p = chi_p[:, g] * nu_p[g] * SigmaF[g] * phi
        # Delayed: sum over all delayed groups
        # chi_d: (G, J), nu_d: (G, J), SigmaF: (G,), phi: scalar
        # F_d[g] = sum_j chi_d[g, j] * sum_g' nu_d[g', j] * SigmaF[g'] * phi
        # delayed_sum = np.sum(nu_d[:, :] * SigmaF[:, None], axis=0) * phi
        # F_d = np.dot(chi_d, delayed_sum)
        F = F_p  # + F_d
    else:
        # Vector phi: sum over all groups
        # Prompt part
        F_p = chi_p @ (nu_p * SigmaF * phi)  # shape (G,)

        # Delayed part
        # chi_d: (G, J), nu_d: (G, J), SigmaF: (G,), phi: (G,)
        # F_d[g] = sum_j chi_d[g, j] * sum_g' nu_d[g', j] * SigmaF[g'] * phi[g']
        # delayed_sum = np.sum(nu_d * SigmaF[:, None] * phi[:, None], axis=0)
        # F_d = chi_d @ delayed_sum
        # F = F_p + F_d

        F = F_p  # + F_d

    return F


@toggle("hybridMC")
def hybrid_effective_scattering(phi, mat_id, mcdc, g=0):
    """
    Calculate the scattering source for use with hybridMC.

    If phi is a scalar, g must be provided and phi is the flux for group g.
    If phi is a vector, phi is the flux for all groups and g is ignored.
    """
    material = mcdc["materials"][mat_id]
    chi_s = material["chi_s"]
    SigmaS = material["scatter"]

    if np.isscalar(phi):
        # Scalar phi: use group g
        # chi_s[g, :] gives prob of scattering FROM group g TO each output group
        S_s = chi_s[g, :] * SigmaS[g] * phi
    else:
        # Vector phi: sum over incoming groups (axis=0), consistent with iqmc convention
        # S_s[g'] = sum_g chi_s[g, g'] * SigmaS[g] * phi[g]  = (chi_s.T @ (SigmaS * phi))
        S_s = np.dot(chi_s.T, SigmaS * phi)
    return S_s


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
