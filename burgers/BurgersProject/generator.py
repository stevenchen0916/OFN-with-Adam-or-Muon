"""Functions extracted unchanged from the stored Burgers generator. No work on import."""
import numpy as np

def build_inputs_from_grf(
    space,
    x_in_1d,
    n_samples,
    center_u0=True,
    center_f=True,
    normalize_u0=True,
    normalize_f=True,
    u0_amp_range=(0.3, 0.5),
    f_amp_range=(0.1, 0.5),
    seed=None,
):
    """
    Build periodic initial condition u0(x) and spatial forcing f(x)
    directly from a periodic GRF evaluated on x.

    Output:
        u0_vals: shape (n_samples, m_branch)
        f_vals:  shape (n_samples, m_branch)
    """
    rng = np.random.default_rng(seed)

    x_sensors = np.asarray(x_in_1d, dtype=np.float32)[:, None]

    feat1 = space.random(n_samples)
    raw_u0 = space.eval_batch(feat1, x_sensors).astype(np.float32)

    feat2 = space.random(n_samples)
    raw_f = space.eval_batch(feat2, x_sensors).astype(np.float32)

    if raw_u0.ndim == 3 and raw_u0.shape[-1] == 1:
        raw_u0 = raw_u0[:, :, 0]
    if raw_f.ndim == 3 and raw_f.shape[-1] == 1:
        raw_f = raw_f[:, :, 0]

    u0_vals = raw_u0.copy()
    f_vals = raw_f.copy()

    if center_u0:
        u0_vals = u0_vals - np.mean(u0_vals, axis=1, keepdims=True)

    if center_f:
        f_vals = f_vals - np.mean(f_vals, axis=1, keepdims=True)

    if normalize_u0:
        rms_u0 = np.sqrt(np.mean(u0_vals ** 2, axis=1, keepdims=True) + 1e-12)
        u0_vals = u0_vals / rms_u0

    if normalize_f:
        rms_f = np.sqrt(np.mean(f_vals ** 2, axis=1, keepdims=True) + 1e-12)
        f_vals = f_vals / rms_f

    u0_amp = rng.uniform(u0_amp_range[0], u0_amp_range[1], size=(n_samples, 1)).astype(np.float32)
    f_amp = rng.uniform(f_amp_range[0], f_amp_range[1], size=(n_samples, 1)).astype(np.float32)

    u0_vals = u0_vals * u0_amp
    f_vals = f_vals * f_amp

    return u0_vals.astype(np.float32), f_vals.astype(np.float32)

def trunk_periodic_features(x_vals, t_vals, L=1.0):
    """
    Input:
        x_vals, t_vals: meshgrid flattened arrays, shape (P,)
    Output:
        shape (P, 5)
        [cos(2pi x/L), sin(2pi x/L), cos(4pi x/L), sin(4pi x/L), t]
    """
    x_vals = np.asarray(x_vals, dtype=np.float32)
    t_vals = np.asarray(t_vals, dtype=np.float32)

    feat = np.stack([
        np.cos(2.0 * np.pi * x_vals / L),
        np.sin(2.0 * np.pi * x_vals / L),
        np.cos(4.0 * np.pi * x_vals / L),
        np.sin(4.0 * np.pi * x_vals / L),
        t_vals,
    ], axis=1).astype(np.float32)

    return feat

def interp_periodic(u_on_grid, x_grid, x_query, L=1.0):
    """
    Periodic 1D interpolation on [0, L).
    """
    x_grid = np.asarray(x_grid, dtype=np.float64)
    u_on_grid = np.asarray(u_on_grid, dtype=np.float64)
    x_query = np.asarray(x_query, dtype=np.float64)

    x_query_mod = np.mod(x_query, L)

    x_ext = np.concatenate([x_grid, np.array([L], dtype=x_grid.dtype)])
    u_ext = np.concatenate([u_on_grid, np.array([u_on_grid[0]], dtype=u_on_grid.dtype)])

    return np.interp(x_query_mod, x_ext, u_ext)

def solve_burgers_forced_periodic(
    nu,
    u0_in,
    f_in,
    x_in,
    x_out=None,
    T=1.0,
    nx_solve=1024,
    nt_solve=3000,
    cfl=0.40,
    Nt=150,
):
    """
    Periodic 1D forced viscous Burgers:
        u_t + (1/2 u^2)_x = nu * u_xx + f(x)

    Spatial discretization:
        - Rusanov flux for nonlinear advection
        - central difference for diffusion
    Time stepping:
        - explicit Euler with adaptive dt based on advection + diffusion CFL
    """
    L = 1.0

    if x_out is None:
        x_out = x_in

    x = np.linspace(0.0, L, nx_solve, endpoint=False, dtype=np.float64)
    dx = L / nx_solve

    u = interp_periodic(u0_in, x_in, x, L=L).astype(np.float64)
    f = interp_periodic(f_in, x_in, x, L=L).astype(np.float64)

    t_eval = np.linspace(0.0, T, Nt, dtype=np.float64)
    u_snap = np.zeros((Nt, len(x_out)), dtype=np.float64)
    u_snap[0, :] = interp_periodic(u, x, x_out, L=L)

    next_k = 1
    next_t = t_eval[next_k] if next_k < Nt else None
    t = 0.0

    dt_nominal = T / nt_solve

    step = 0
    max_steps = 2000000

    while t < T - 1e-12 and step < max_steps:
        step += 1

        umax = np.max(np.abs(u))

        dt_adv = cfl * dx / (umax + 1e-12)
        dt_diff = 0.5 * cfl * dx * dx / (nu + 1e-12)
        dt_step = min(dt_nominal, dt_adv, dt_diff, T - t)

        uR = np.roll(u, -1)
        flux_left = 0.5 * u ** 2
        flux_right = 0.5 * uR ** 2
        a = np.maximum(np.abs(u), np.abs(uR))

        flux_num = 0.5 * (flux_left + flux_right) - 0.5 * a * (uR - u)
        adv = -(flux_num - np.roll(flux_num, 1)) / dx

        diff = nu * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

        u = u + dt_step * (adv + diff + f)
        t += dt_step

        while next_k < Nt and t >= next_t - 1e-12:
            u_snap[next_k, :] = interp_periodic(u, x, x_out, L=L)
            next_k += 1
            next_t = t_eval[next_k] if next_k < Nt else None

        if next_k >= Nt:
            break

    if step >= max_steps:
        raise RuntimeError("Time stepping hit safety cap; dt_step may be extremely small.")

    return u_snap.astype(np.float32)
