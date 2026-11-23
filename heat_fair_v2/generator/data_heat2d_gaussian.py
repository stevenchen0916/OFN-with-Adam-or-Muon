import os
import sys

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from .gaussian_generator import generate_gaussian_family

def trunk_features_xyz(x_vals, y_vals, t_vals):
    x_vals = np.asarray(x_vals, dtype=np.float32)
    y_vals = np.asarray(y_vals, dtype=np.float32)
    t_vals = np.asarray(t_vals, dtype=np.float32)
    return np.stack([x_vals, y_vals, t_vals], axis=1).astype(np.float32)


def make_time_window(T_full, Nt_trunk, T_start=None, T_end=None):
    if T_start is None:
        T_start = 0.0
    if T_end is None:
        T_end = T_full

    if not (0.0 <= T_start < T_end <= T_full):
        raise ValueError(f"Require 0 <= T_start < T_end <= T_full, got {T_start}, {T_end}, {T_full}")

    t_used = np.linspace(T_start, T_end, Nt_trunk).astype(np.float32)
    return t_used, float(T_start), float(T_end)


def sample_kappa(n_samples, kappa_min=0.02, kappa_max=0.20, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(kappa_min, kappa_max, size=(n_samples, 1)).astype(np.float32)


def build_gaussian_initial_inputs(
    x_branch_1d,
    y_branch_1d,
    n_samples,
    shape_mode="circular",
    amp_range=(0.8, 1.2),
    x0_range=(0.45, 0.55),
    y0_range=(0.45, 0.55),
    sigma_range=(0.08, 0.16),
    sigma_x_range=(0.08, 0.16),
    sigma_y_range=(0.08, 0.16),
    seed=None,
):
    U, params = generate_gaussian_family(
        x=x_branch_1d,
        y=y_branch_1d,
        n_samples=n_samples,
        shape_mode=shape_mode,
        amp_range=amp_range,
        x0_range=x0_range,
        y0_range=y0_range,
        sigma_range=sigma_range,
        sigma_x_range=sigma_x_range,
        sigma_y_range=sigma_y_range,
        seed=seed,
    )
    return U.reshape(n_samples, -1).astype(np.float32), params


def interpolate_field_2d(field_branch_flat, x_branch, y_branch, x_out, y_out):
    m = len(x_branch)
    field_branch = field_branch_flat.reshape(m, m)

    interp = RegularGridInterpolator(
        (y_branch, x_branch),
        field_branch,
        bounds_error=False,
        fill_value=None,
    )

    xv, yv = np.meshgrid(x_out, y_out, indexing="xy")
    pts = np.column_stack([yv.ravel(), xv.ravel()])
    return interp(pts).reshape(len(y_out), len(x_out)).astype(np.float64)


def solve_heat_2d_kappa(
    kappa,
    u0_in,
    x_branch,
    y_branch,
    x_out=None,
    y_out=None,
    T=1.0,
    nx_solve=64,
    ny_solve=64,
    dt=None,
    Nt=20,
    t_eval=None,
):
    """
    2D heat equation with zero Dirichlet BC:

        u_t = kappa * (u_xx + u_yy)
    """
    if x_out is None:
        x_out = np.linspace(0.0, 1.0, nx_solve, dtype=np.float32)
    if y_out is None:
        y_out = np.linspace(0.0, 1.0, ny_solve, dtype=np.float32)

    x = np.linspace(0.0, 1.0, nx_solve, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny_solve, dtype=np.float64)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if dt is None:
        dt_stable = 0.5 / (kappa * (2.0 / dx**2 + 2.0 / dy**2))
        dt = 0.8 * dt_stable

    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps

    if t_eval is None:
        t_eval = np.linspace(0.0, T, Nt, dtype=np.float64)
    else:
        t_eval = np.asarray(t_eval, dtype=np.float64)
        Nt = len(t_eval)
        if Nt == 0:
            raise ValueError("t_eval must contain at least one time point.")
        if np.any(np.diff(t_eval) < 0):
            raise ValueError("t_eval must be nondecreasing.")
        if t_eval[0] < -1e-12 or t_eval[-1] > T + 1e-12:
            raise ValueError("t_eval must lie within [0, T].")

    u = interpolate_field_2d(u0_in, x_branch, y_branch, x, y)

    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    snapshots = np.zeros((Nt, len(y_out), len(x_out)), dtype=np.float32)

    def interp_solution_to_out(u_grid):
        if np.array_equal(x, x_out) and np.array_equal(y, y_out):
            return u_grid.astype(np.float32)

        interp = RegularGridInterpolator(
            (y, x),
            u_grid,
            bounds_error=False,
            fill_value=None,
        )
        xv, yv = np.meshgrid(x_out, y_out, indexing="xy")
        pts = np.column_stack([yv.ravel(), xv.ravel()])
        return interp(pts).reshape(len(y_out), len(x_out)).astype(np.float32)

    next_k = 0
    if np.isclose(t_eval[0], 0.0):
        snapshots[0] = interp_solution_to_out(u)
        next_k = 1

    next_t = t_eval[next_k] if next_k < Nt else None

    for step in range(1, nsteps + 1):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
            + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        )

        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (kappa * lap[1:-1, 1:-1])

        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0

        t = step * dt

        while next_k < Nt and t >= next_t - 1e-12:
            snapshots[next_k] = interp_solution_to_out(u)
            next_k += 1
            next_t = t_eval[next_k] if next_k < Nt else None

    if next_k < Nt:
        raise RuntimeError("Not all requested t_eval points were filled. Try reducing dt.")

    return snapshots.astype(np.float32)

def preview_one_sample(
    m_branch=10,
    T=1.0,
    Nt_plot=20,
    T_start=None,
    T_end=None,
    nx_plot=64,
    ny_plot=64,
    kappa=0.1,
    shape_mode="circular",
    amp_range=(0.8, 1.2),
    x0_range=(0.45, 0.55),
    y0_range=(0.45, 0.55),
    sigma_range=(0.08, 0.16),
    sigma_x_range=(0.08, 0.16),
    sigma_y_range=(0.08, 0.16),
    out_png="heat2d_kappa_gaussian_preview.png",
    seed=42,
):
    x_branch = np.linspace(0.0, 1.0, num=m_branch).astype(np.float32)
    y_branch = np.linspace(0.0, 1.0, num=m_branch).astype(np.float32)

    x_dense = np.linspace(0.0, 1.0, num=nx_plot).astype(np.float32)
    y_dense = np.linspace(0.0, 1.0, num=ny_plot).astype(np.float32)

    xv_b, yv_b = np.meshgrid(x_branch, y_branch, indexing="xy")

    u0_vals, params = build_gaussian_initial_inputs(
        x_branch_1d=x_branch,
        y_branch_1d=y_branch,
        n_samples=1,
        shape_mode=shape_mode,
        amp_range=amp_range,
        x0_range=x0_range,
        y0_range=y0_range,
        sigma_range=sigma_range,
        sigma_x_range=sigma_x_range,
        sigma_y_range=sigma_y_range,
        seed=seed,
    )
    u0_branch = u0_vals[0]
    param = params[0]

    u0_dense = interpolate_field_2d(u0_branch, x_branch, y_branch, x_dense, y_dense)

    t_used, T_start_actual, T_end_actual = make_time_window(
        T_full=T,
        Nt_trunk=Nt_plot,
        T_start=T_start,
        T_end=T_end,
    )

    U = solve_heat_2d_kappa(
        kappa=float(kappa),
        u0_in=u0_branch,
        x_branch=x_branch,
        y_branch=y_branch,
        x_out=x_dense,
        y_out=y_dense,
        T=T,
        nx_solve=nx_plot,
        ny_solve=ny_plot,
        t_eval=t_used,
    )

    snap_idx = np.linspace(0, Nt_plot - 1, 4).astype(int)
    vmin = min(U[k].min() for k in snap_idx)
    vmax = max(U[k].max() for k in snap_idx)

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11.5))

    im0 = axes[0, 0].imshow(
        u0_dense,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="rainbow",
    )
    axes[0, 0].scatter(xv_b.ravel(), yv_b.ravel(), s=5, c="black")
    axes[0, 0].set_title(r"Input 2: $u_0(x,y)$")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[0, 1].axis("off")
    info_text = f"shape_mode = {shape_mode}\nkappa = {kappa:.4f}\namp = {param['amp']:.3f}\nx0 = {param['x0']:.3f}\ny0 = {param['y0']:.3f}"
    if shape_mode == "circular":
        info_text += f"\nsigma = {param['sigma']:.3f}"
    else:
        info_text += f"\nsigma_x = {param['sigma_x']:.3f}\nsigma_y = {param['sigma_y']:.3f}"

    axes[0, 1].text(
        0.05,
        0.95,
        info_text,
        va="top",
        ha="left",
        fontsize=12,
    )

    snapshot_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
    for ax, k in zip(snapshot_axes, snap_idx):
        im = ax.imshow(
            U[k],
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=vmin,
            vmax=vmax,
            cmap="rainbow",
        )
        ax.set_title(f"$u(x,y,t)$ at $t={t_used[k]:.3f}$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved preview to: {out_png}")
    print(f"shape_mode = {shape_mode}")
    print(f"kappa = {kappa:.6f}")
    print(f"u0 range: [{u0_branch.min():.4f}, {u0_branch.max():.4f}]")
    print(f"Time window used: [{T_start_actual:.4f}, {T_end_actual:.4f}] with {len(t_used)} time points")


def run_heat2d_kappa_gaussian(
    m_branch,
    nx_trunk,
    ny_trunk,
    T,
    Nt_trunk,
    num_train,
    num_test,
    kappa_min=0.02,
    kappa_max=0.20,
    shape_mode="circular",
    amp_range=(0.8, 1.2),
    x0_range=(0.45, 0.55),
    y0_range=(0.45, 0.55),
    sigma_range=(0.08, 0.16),
    sigma_x_range=(0.08, 0.16),
    sigma_y_range=(0.08, 0.16),
    problem="HEAT2D_kappa_gaussian",
    T_start=None,
    T_end=None,
    nx_solve=64,
    ny_solve=64,
):
    print("Generating 2D heat data with scalar kappa + Gaussian initial...", flush=True)
    os.makedirs("data", exist_ok=True)

    x_branch = np.linspace(0.0, 1.0, num=m_branch).astype(np.float32)
    y_branch = np.linspace(0.0, 1.0, num=m_branch).astype(np.float32)

    x_trunk = np.linspace(0.0, 1.0, num=nx_trunk).astype(np.float32)
    y_trunk = np.linspace(0.0, 1.0, num=ny_trunk).astype(np.float32)

    t_used, T_start_actual, T_end_actual = make_time_window(
        T_full=T,
        Nt_trunk=Nt_trunk,
        T_start=T_start,
        T_end=T_end,
    )

    Xg, Yg, Tg = np.meshgrid(x_trunk, y_trunk, t_used, indexing="xy")
    x_flat = Xg.reshape(-1)
    y_flat = Yg.reshape(-1)
    t_flat = Tg.reshape(-1)

    xt = trunk_features_xyz(x_flat, y_flat, t_flat)
    npoints_output = len(t_used) * nx_trunk * ny_trunk

    def make_raw_dataset(n_samples, seed_offset=0):
        kappa_vals = sample_kappa(
            n_samples=n_samples,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            seed=1234 + seed_offset,
        )

        u0_vals, params = build_gaussian_initial_inputs(
            x_branch_1d=x_branch,
            y_branch_1d=y_branch,
            n_samples=n_samples,
            shape_mode=shape_mode,
            amp_range=amp_range,
            x0_range=x0_range,
            y0_range=y0_range,
            sigma_range=sigma_range,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_y_range,
            seed=5678 + seed_offset,
        )

        U = np.zeros((n_samples, npoints_output), dtype=np.float32)

        for i in tqdm(range(n_samples)):
            u_txy = solve_heat_2d_kappa(
                kappa=float(kappa_vals[i, 0]),
                u0_in=u0_vals[i],
                x_branch=x_branch,
                y_branch=y_branch,
                x_out=x_trunk,
                y_out=y_trunk,
                T=T,
                nx_solve=nx_solve,
                ny_solve=ny_solve,
                t_eval=t_used,
            )
            U[i, :] = np.transpose(u_txy, (1, 2, 0)).reshape(-1)

        return kappa_vals.astype(np.float32), u0_vals.astype(np.float32), U.astype(np.float32), params

    kappa_train, u0_train, y_train, params_train = make_raw_dataset(num_train, seed_offset=0)
    kappa_test, u0_test, y_test, params_test = make_raw_dataset(num_test, seed_offset=100000)

    X_train = np.array([
        kappa_train,
        u0_train,
        xt.astype(np.float32),
    ], dtype=object)

    X_test = np.array([
        kappa_test,
        u0_test,
        xt.astype(np.float32),
    ], dtype=object)

    np.savez_compressed(
        f"data/{problem}_train.npz",
        X_train=X_train,
        y_train=y_train,
    )
    np.savez_compressed(
        f"data/{problem}_test.npz",
        X_test=X_test,
        y_test=y_test,
    )
    np.savez_compressed(
        f"data/{problem}_different_width_train.npz",
        X_train=X_train,
        y_train=y_train,
    )
    np.savez_compressed(
        f"data/{problem}_different_width_test.npz",
        X_test=X_test,
        y_test=y_test,
    )

    print(f"Saved dataset: {problem}")
    print(f"shape_mode: {shape_mode}")
    print(f"Branch input 1 dimension (kappa): 1")
    print(f"Branch input 2 dimension (u0): {m_branch} x {m_branch} = {m_branch * m_branch}")
    print(f"Trunk points used: {nx_trunk} x {ny_trunk} x {len(t_used)} = {npoints_output}")
    print(f"Time window used in dataset: [{T_start_actual:.4f}, {T_end_actual:.4f}]")
    print(f"Original full time horizon: [0.0000, {T:.4f}]")
    print(f"Retained/re-sampled time points: {len(t_used)}")
    print(f"kappa range in dataset: [{kappa_min:.4f}, {kappa_max:.4f}]")


def main():
    m_branch = 10
    nx_trunk = 20
    ny_trunk = 20
    Nt_trunk = 20

    T = 1.0
    num_train = 2000
    num_test = 200

    kappa_min = 0.0005
    kappa_max = 0.05


    shape_mode = "elliptic"   # "circular" or "elliptic"
    amp_range=[0.8, 1.2]
    x0_range=[0.3, 0.7]
    y0_range=[0.3, 0.7]

    sigma_range=[0.22, 0.32]#宽度
    sigma_x_range=[0.15, 0.3]#宽度
    sigma_y_range=[0.15, 0.3]#宽度

    T_start = None
    T_end = None

    preview_one_sample(
        m_branch=m_branch,
        T=T,
        Nt_plot=20,
        T_start=T_start,
        T_end=T_end,
        nx_plot=64,
        ny_plot=64,
        kappa=0.01,
        shape_mode=shape_mode,
        amp_range=amp_range,
        x0_range=x0_range,
        y0_range=y0_range,
        sigma_range=sigma_range,
        sigma_x_range=sigma_x_range,
        sigma_y_range=sigma_y_range,
        out_png="heat2d_kappa_gaussian_preview.png",
        seed=42,
    )

    problem_name = f"HEAT2D_kappa_gaussian_{shape_mode}_v2"

    run_heat2d_kappa_gaussian(
        m_branch=m_branch,
        nx_trunk=nx_trunk,
        ny_trunk=ny_trunk,
        T=T,
        Nt_trunk=Nt_trunk,
        num_train=num_train,
        num_test=num_test,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        shape_mode=shape_mode,
        amp_range=amp_range,
        x0_range=x0_range,
        y0_range=y0_range,
        sigma_range=sigma_range,
        sigma_x_range=sigma_x_range,
        sigma_y_range=sigma_y_range,
        problem=problem_name,
        T_start=T_start,
        T_end=T_end,
        nx_solve=64,
        ny_solve=64,
    )


if __name__ == "__main__":
    main()
