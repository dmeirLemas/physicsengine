import numpy as np

from physengine.modules.galaxy_gravity import bh_acc_grouped


def ics(n, M, a, r_max, gen: np.random.Generator):
    u_max = (r_max / (r_max + a)) ** 2
    u = gen.uniform(0, u_max, n)

    r = a * np.sqrt(u) / (1 - np.sqrt(u))

    cos_theta = gen.uniform(-1, 1, n)
    phi = gen.uniform(0, 2 * np.pi, n)

    sin_theta = np.sqrt(1 - cos_theta**2)

    pos = r[:, None] * np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]).T
    mass = M / n * np.ones(n)

    return pos, mass


def sample_disc(n, M, R_d, z_0, R_max, gen):
    R_grid = np.linspace(0, R_max, 4000)

    d = R_grid / R_d
    FR = 1 - (1 + d) * np.exp(-d)
    FR /= FR[-1]

    u = gen.uniform(0, 1, n)
    R = np.interp(u, FR, R_grid)

    phi = gen.uniform(0, 2 * np.pi, n)
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    w = gen.uniform((1 - np.tanh(10)) / 2, (1 + np.tanh(10)) / 2, n)
    z = z_0 * np.arctanh(2 * w - 1)
    pos = np.stack([x, y, z], axis=-1)
    mass = M / n * np.ones(n)

    return pos, mass


def circular_velocity(r_query, pos, mass, G=1.0):
    r_all = np.linalg.norm(pos, axis=1)
    order = np.argsort(r_all)
    r_sorted = r_all[order]
    m_enclosed = np.cumsum(mass[order])

    r_query = np.asarray(r_query, dtype=float)
    m_query = np.interp(r_query, r_sorted, m_enclosed, left=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(G * m_query / r_query)
    return np.where(r_query > 0.0, v, 0.0)


def disc_rotation(disc_pos, pos, mass, G=1.0):
    x, y = disc_pos[:, 0], disc_pos[:, 1]
    R = np.hypot(x, y)
    speed = circular_velocity(R, pos, mass, G)

    vel = np.zeros_like(disc_pos)
    safe_R = np.where(R > 0.0, R, 1.0)
    vel[:, 0] = -speed * y / safe_R
    vel[:, 1] = speed * x / safe_R
    return vel


def enclosed_mass_and_potential(r_query, pos, mass, G=1.0):
    r_all = np.linalg.norm(pos, axis=1)
    order = np.argsort(r_all)
    r_sorted = r_all[order]
    m_sorted = mass[order]
    m_enclosed = np.cumsum(m_sorted)
    # Phi(r) = -G [M(<r) / r + sum over shells outside r of m_k / r_k]
    outer = np.cumsum((m_sorted / np.maximum(r_sorted, 1e-12))[::-1])[::-1]

    r_query = np.asarray(r_query, dtype=float)
    m_q = np.interp(r_query, r_sorted, m_enclosed, left=0.0)
    idx = np.searchsorted(r_sorted, r_query, side="right")
    outer_q = np.append(outer, 0.0)[idx]
    phi = -G * (m_q / np.maximum(r_query, 1e-12) + outer_q)
    return m_q, phi


def hernquist_dispersion(r_query, a, r_max, pos, mass, G=1.0, n_grid=2000):
    r_grid = np.geomspace(1e-3 * a, r_max, n_grid)
    rho = 1.0 / (r_grid * (r_grid + a) ** 3)  # normalisation cancels in sigma^2
    m_enc, _ = enclosed_mass_and_potential(r_grid, pos, mass, G)
    f = rho * G * m_enc / r_grid**2

    # integral from r to r_max of f dr', by the trapezoid rule summed from the outer edge inwards
    segments = 0.5 * (f[1:] + f[:-1]) * np.diff(r_grid)
    integral = np.append(np.cumsum(segments[::-1])[::-1], 0.0)
    sigma2 = integral / rho

    r_query = np.clip(np.asarray(r_query, dtype=float), r_grid[0], r_grid[-1])
    return np.sqrt(np.interp(np.log(r_query), np.log(r_grid), sigma2))


def isotropic_velocities(comp_pos, a, r_max, pos, mass, gen, G=1.0, escape_fraction=0.95):
    r = np.linalg.norm(comp_pos, axis=1)
    sigma = hernquist_dispersion(r, a, r_max, pos, mass, G)
    _, phi = enclosed_mass_and_potential(r, pos, mass, G)
    v_max = escape_fraction * np.sqrt(-2.0 * phi)

    vel = np.zeros_like(comp_pos)
    todo = np.arange(len(r))
    for _ in range(100):
        if todo.size == 0:
            break
        vel[todo] = gen.normal(0.0, 1.0, (todo.size, 3)) * sigma[todo, None]
        todo = todo[np.linalg.norm(vel[todo], axis=1) > v_max[todo]]
    return vel


def midplane_rotation_curve(pos, mass, R_max, eps, theta=0.5, G=1.0, n_R=96, n_phi=32, smooth=5):
    """Circular speed in the disc plane from the actual (softened) gravity of all particles.

    Massless test points on rings in the plane z = 0 get their acceleration from
    the tree; v_c^2 = -R <a_R>, averaged around each ring. Unlike
    circular_velocity this includes the flattening of the disc, which pulls
    harder in its own plane than the same mass spread over a sphere.
    Returns (R_grid, v_grid).
    """
    R_grid = np.geomspace(0.02 * R_max, 1.2 * R_max, n_R)
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    Rr, pp = np.meshgrid(R_grid, phi, indexing="ij")
    probes = np.stack(
        [(Rr * np.cos(pp)).ravel(), (Rr * np.sin(pp)).ravel(), np.zeros(Rr.size)], axis=1
    )

    a = bh_acc_grouped(
        np.vstack([pos, probes]), np.concatenate([mass, np.zeros(len(probes))]), G, eps, theta
    )
    a = a[len(pos) :]
    a_R = (a[:, 0] * probes[:, 0] + a[:, 1] * probes[:, 1]) / Rr.ravel()
    v2 = np.maximum(-R_grid * a_R.reshape(n_R, n_phi).mean(axis=1), 0.0)

    pad = smooth // 2  # running mean in log R: kappa needs a smooth slope
    v2 = np.convolve(np.pad(v2, pad, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    return R_grid, np.sqrt(v2)


def epicyclic_frequencies(R, pos, mass, G=1.0, n_grid=512, smooth=9, rotation_curve=None):
    R = np.asarray(R, dtype=float)
    if rotation_curve is not None:
        grid, v = rotation_curve
    else:
        # fixed grid over the whole galaxy, so the result does not depend on which radii are asked for
        r_all = np.linalg.norm(pos, axis=1)
        grid = np.geomspace(max(np.min(r_all), 1e-3), np.max(r_all), n_grid)
        v = circular_velocity(grid, pos, mass, G)
        # running mean in log R: the particle rotation curve is noisy, kappa needs its slope
        pad = smooth // 2
        v = np.convolve(np.pad(v, pad, mode="edge"), np.ones(smooth) / smooth, mode="valid")

    omega2 = (v / grid) ** 2
    kappa2 = np.maximum(
        grid * np.gradient(omega2, grid) + 4.0 * omega2, 0.0
    )  # kappa^2 = R dOmega^2/dR + 4 Omega^2
    omega = np.sqrt(np.interp(R, grid, omega2))
    kappa = np.sqrt(np.interp(R, grid, kappa2))
    return omega, kappa


def disc_kinematics(
    R, M, R_d, z_0, R_max, pos, mass, Q_ref=1.5, R_ref=None, G=1.0, rotation_curve=None
):
    R = np.asarray(R, dtype=float)
    if R_ref is None:
        R_ref = 2.5 * R_d
    x_max = R_max / R_d
    sigma0 = (
        M / (2.0 * np.pi * R_d**2) / (1.0 - (1.0 + x_max) * np.exp(-x_max))
    )  # central Sigma of the truncated disc

    def surface_density(r):
        return sigma0 * np.exp(-r / R_d)

    omega, kappa = epicyclic_frequencies(
        np.append(R, R_ref), pos, mass, G, rotation_curve=rotation_curve
    )
    omega_ref, kappa_ref = omega[-1], kappa[-1]
    omega, kappa = omega[:-1], kappa[:-1]

    sigma_R_ref = Q_ref * 3.36 * G * surface_density(R_ref) / kappa_ref
    sigma_R = sigma_R_ref * np.exp(-(R - R_ref) / (2.0 * R_d))
    sigma_z = np.sqrt(np.pi * G * surface_density(R) * z_0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio2 = np.where(omega > 0.0, kappa**2 / (4.0 * omega**2), 1.0)
    sigma_phi = sigma_R * np.sqrt(ratio2)

    if rotation_curve is not None:
        v_c = np.interp(R, *rotation_curve)
    else:
        v_c = circular_velocity(R, pos, mass, G)
    v_phi2 = v_c**2 + sigma_R**2 * (1.0 - ratio2 - 2.0 * R / R_d)
    v_phi = np.sqrt(np.maximum(v_phi2, 0.0))
    return v_phi, sigma_R, sigma_phi, sigma_z


def disc_velocities(
    disc_pos, M, R_d, z_0, R_max, pos, mass, gen, Q_ref=1.5, G=1.0, rotation_curve=None
):
    x, y = disc_pos[:, 0], disc_pos[:, 1]
    R = np.hypot(x, y)
    v_phi, sigma_R, sigma_phi, sigma_z = disc_kinematics(
        R, M, R_d, z_0, R_max, pos, mass, Q_ref, G=G, rotation_curve=rotation_curve
    )

    n = len(R)
    v_R = gen.normal(0.0, 1.0, n) * sigma_R
    v_t = v_phi + gen.normal(0.0, 1.0, n) * sigma_phi
    v_z = gen.normal(0.0, 1.0, n) * sigma_z

    safe_R = np.where(R > 0.0, R, 1.0)
    cos_p, sin_p = x / safe_R, y / safe_R
    return np.stack([v_R * cos_p - v_t * sin_p, v_R * sin_p + v_t * cos_p, v_z], axis=1)
