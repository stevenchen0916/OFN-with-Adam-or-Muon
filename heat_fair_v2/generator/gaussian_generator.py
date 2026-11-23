import numpy as np
import matplotlib.pyplot as plt

def gaussian_2d_circular(
    x,
    y,
    amp=1.0,
    x0=0.5,
    y0=0.5,
    sigma=0.12,
):
    """
    Circular 2D Gaussian:
        u(x,y) = amp * exp(-((x-x0)^2 + (y-y0)^2) / (2 sigma^2))
    """
    xv, yv = np.meshgrid(x, y, indexing="xy")
    u = amp * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / (2.0 * sigma ** 2))
    return u.astype(np.float32)


def gaussian_2d_elliptic(
    x,
    y,
    amp=1.0,
    x0=0.5,
    y0=0.5,
    sigma_x=0.10,
    sigma_y=0.16,
):
    """
    Axis-aligned elliptic 2D Gaussian:
        u(x,y) = amp * exp(-0.5 * [((x-x0)/sigma_x)^2 + ((y-y0)/sigma_y)^2])
    """
    xv, yv = np.meshgrid(x, y, indexing="xy")
    u = amp * np.exp(
        -0.5 * (
            ((xv - x0) / sigma_x) ** 2
            + ((yv - y0) / sigma_y) ** 2
        )
    )
    return u.astype(np.float32)


def sample_gaussian_params(
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
    rng = np.random.default_rng(seed)

    amps = rng.uniform(amp_range[0], amp_range[1], size=n_samples).astype(np.float32)
    x0s = rng.uniform(x0_range[0], x0_range[1], size=n_samples).astype(np.float32)
    y0s = rng.uniform(y0_range[0], y0_range[1], size=n_samples).astype(np.float32)

    if shape_mode == "circular":
        sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=n_samples).astype(np.float32)
        return {
            "amp": amps,
            "x0": x0s,
            "y0": y0s,
            "sigma": sigmas,
        }

    elif shape_mode == "elliptic":
        sigma_xs = rng.uniform(sigma_x_range[0], sigma_x_range[1], size=n_samples).astype(np.float32)
        sigma_ys = rng.uniform(sigma_y_range[0], sigma_y_range[1], size=n_samples).astype(np.float32)
        return {
            "amp": amps,
            "x0": x0s,
            "y0": y0s,
            "sigma_x": sigma_xs,
            "sigma_y": sigma_ys,
        }

    else:
        raise ValueError(f"Unsupported shape_mode: {shape_mode}")


def generate_gaussian_family(
    x,
    y,
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
    params_raw = sample_gaussian_params(
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

    U = []
    params = []

    for i in range(n_samples):
        amp = float(params_raw["amp"][i])
        x0 = float(params_raw["x0"][i])
        y0 = float(params_raw["y0"][i])

        if shape_mode == "circular":
            sigma = float(params_raw["sigma"][i])
            u = gaussian_2d_circular(
                x=x,
                y=y,
                amp=amp,
                x0=x0,
                y0=y0,
                sigma=sigma,
            )
            params.append(
                {
                    "shape_mode": "circular",
                    "amp": amp,
                    "x0": x0,
                    "y0": y0,
                    "sigma": sigma,
                }
            )

        elif shape_mode == "elliptic":
            sigma_x = float(params_raw["sigma_x"][i])
            sigma_y = float(params_raw["sigma_y"][i])
            u = gaussian_2d_elliptic(
                x=x,
                y=y,
                amp=amp,
                x0=x0,
                y0=y0,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
            )
            params.append(
                {
                    "shape_mode": "elliptic",
                    "amp": amp,
                    "x0": x0,
                    "y0": y0,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                }
            )

        else:
            raise ValueError(f"Unsupported shape_mode: {shape_mode}")

        U.append(u)

    return np.stack(U, axis=0).astype(np.float32), params


def preview_gaussian_family(
    n_show=6,
    nx=100,
    ny=100,
    shape_mode="circular",
    amp_range=(0.8, 1.2),
    x0_range=(0.45, 0.55),
    y0_range=(0.45, 0.55),
    sigma_range=(0.08, 0.16),
    sigma_x_range=(0.08, 0.16),
    sigma_y_range=(0.08, 0.16),
    seed=42,
    out_png="gaussian_family_preview.png",
):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)

    U, params = generate_gaussian_family(
        x=x,
        y=y,
        n_samples=n_show,
        shape_mode=shape_mode,
        amp_range=amp_range,
        x0_range=x0_range,
        y0_range=y0_range,
        sigma_range=sigma_range,
        sigma_x_range=sigma_x_range,
        sigma_y_range=sigma_y_range,
        seed=seed,
    )

    ncols = 3
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    vmin = U.min()
    vmax = U.max()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_show:
            im = ax.imshow(
                U[i],
                origin="lower",
                extent=[0, 1, 0, 1],
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
                cmap='rainbow'
            )
            p = params[i]

            if p["shape_mode"] == "circular":
                title = (
                    f"amp={p['amp']:.2f}, x0={p['x0']:.2f}\n"
                    f"y0={p['y0']:.2f}, sigma={p['sigma']:.3f}"
                )
            else:
                title = (
                    f"amp={p['amp']:.2f}, x0={p['x0']:.2f}\n"
                    f"y0={p['y0']:.2f}, sx={p['sigma_x']:.3f}, sy={p['sigma_y']:.3f}"
                )

            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.axis("off")

    #fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved preview to: {out_png}")


if __name__ == "__main__":
    preview_gaussian_family(
        n_show=6,
        nx=120,
        ny=120,
        shape_mode="elliptic",   # or "circular"
        amp_range=(0.6, 1.0),
        x0_range=(0.4, 0.6),
        y0_range=(0.4, 0.6),
        sigma_range=(0.22, 0.32),#宽度
        sigma_x_range=(0.15, 0.3),#宽度
        sigma_y_range=(0.15, 0.3),#宽度
        seed=42,
        out_png="gaussian_family_preview.png",
    )
