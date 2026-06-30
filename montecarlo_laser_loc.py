import math
import time
import numpy as np

import HAL
import WebGUI
import Frequency

PX_PER_M = 101.1
MAP_SIZE = 1012
OCC_THRESHOLD = 128

MAP_URL_CANDIDATES = [
    "/RoboticsAcademy/exercises/montecarlo_laser_loc/frontend/resources/mapgrannyannie.png",
    "/RoboticsAcademy/exercises/static/exercises/montecarlo_laser_loc/frontend/resources/mapgrannyannie.png",
    "resources/mapgrannyannie.png",
    "mapgrannyannie.png",
]

GLOBAL_INIT     = False
INIT_SPREAD_XY  = 0.5
INIT_SPREAD_YAW = math.radians(25)

DEMO_BAD_INIT    = True
DEMO_OFFSET_XY   = (1.3, -0.8)
DEMO_SPREAD_XY   = 0.8
DEMO_SPREAD_YAW  = math.radians(12)
DEMO_HOLD_SECONDS = 4.0

NUM_PARTICLES   = 3000
NUM_BEAMS       = 50

ALPHA1 = 0.01
ALPHA2 = 0.005
ALPHA3 = 0.08
ALPHA4 = 0.005

SIGMA_HIT = 0.2
Z_HIT     = 0.85
Z_RAND    = 0.15
MAX_FIELD = 2.0

RESAMPLE_RATIO = 0.5
ROUGHEN_XY  = 0.02
ROUGHEN_YAW = math.radians(0.5)

ALPHA_SLOW = 0.002
ALPHA_FAST = 0.10
INJECT_CAP = 0.10

ENABLE_WANDER = True
CRUISE_V = 0.4
TURN_W   = 0.9
FRONT_STOP_DIST = 0.6

RATE_HZ = 10


def build_likelihood_field(occ):
    try:
        from scipy import ndimage
        dist_px = ndimage.distance_transform_edt(~occ)
    except Exception:
        dist_px = _edt_fallback(occ)
    dist_m = dist_px / PX_PER_M
    return np.minimum(dist_m, MAX_FIELD)


def _edt1d(f):
    n = len(f)
    d = np.empty(n)
    v = np.zeros(n, dtype=np.intp)
    z = np.empty(n + 1)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def _edt_fallback(occ):
    small = occ[::2, ::2]
    INF = 1e12
    g = np.where(small, 0.0, INF)
    for i in range(g.shape[0]):
        g[i, :] = _edt1d(g[i, :])
    for j in range(g.shape[1]):
        g[:, j] = _edt1d(g[:, j])
    dist_px = np.sqrt(g) * 2.0
    return np.repeat(np.repeat(dist_px, 2, axis=0), 2, axis=1)[:MAP_SIZE, :MAP_SIZE]


def load_map():
    img = None
    for url in MAP_URL_CANDIDATES:
        try:
            candidate = WebGUI.getMap(url)
            if candidate is not None and getattr(candidate, "size", 0) > 0:
                img = candidate
                print("[MCL] mapa carregado de:", url)
                break
        except Exception:
            continue
    if img is None:
        raise RuntimeError(
            "Não consegui carregar o mapa. Ajusta MAP_URL_CANDIDATES para o "
            "caminho correto de mapgrannyannie.png no teu ambiente."
        )
    gray = img[..., 0] if img.ndim == 3 else img
    if gray.max() <= 1.0:
        gray = gray * 255.0
    occ = gray < OCC_THRESHOLD
    return occ


def world_to_map_idx(x, y):
    row = np.clip(np.rint(PX_PER_M * (4.2 + y)).astype(np.intp), 0, MAP_SIZE - 1)
    col = np.clip(np.rint(PX_PER_M * (5.7 - x)).astype(np.intp), 0, MAP_SIZE - 1)
    return row, col


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def sample_free_poses(field, n):
    parts = np.empty((n, 3))
    filled = 0
    x_lo, x_hi = -4.3, 5.7
    y_lo, y_hi = -4.2, 5.8
    while filled < n:
        m = (n - filled) * 3
        xs = np.random.uniform(x_lo, x_hi, m)
        ys = np.random.uniform(y_lo, y_hi, m)
        row, col = world_to_map_idx(xs, ys)
        clear = field[row, col] > 0.15
        xs, ys = xs[clear], ys[clear]
        take = min(len(xs), n - filled)
        parts[filled:filled + take, 0] = xs[:take]
        parts[filled:filled + take, 1] = ys[:take]
        parts[filled:filled + take, 2] = np.random.uniform(-math.pi, math.pi, take)
        filled += take
    return parts


def init_particles_global(field, n):
    return sample_free_poses(field, n), np.full(n, 1.0 / n)


def init_particles_local(field, n, x0, y0, yaw0, spread_xy=None, spread_yaw=None):
    sxy = INIT_SPREAD_XY if spread_xy is None else spread_xy
    syaw = INIT_SPREAD_YAW if spread_yaw is None else spread_yaw
    parts = np.empty((n, 3))
    parts[:, 0] = np.random.normal(x0, sxy, n)
    parts[:, 1] = np.random.normal(y0, sxy, n)
    parts[:, 2] = np.random.normal(yaw0, syaw, n)
    parts[:, 2] = (parts[:, 2] + math.pi) % (2 * math.pi) - math.pi
    return parts, np.full(n, 1.0 / n)


def motion_update(parts, dtrans, drot1, drot2):
    n = len(parts)
    sd_rot1  = math.sqrt(ALPHA1 * drot1 ** 2 + ALPHA2 * dtrans ** 2)
    sd_trans = math.sqrt(ALPHA3 * dtrans ** 2 + ALPHA4 * (drot1 ** 2 + drot2 ** 2))
    sd_rot2  = math.sqrt(ALPHA1 * drot2 ** 2 + ALPHA2 * dtrans ** 2)

    r1 = drot1  - np.random.normal(0.0, sd_rot1,  n)
    tr = dtrans - np.random.normal(0.0, sd_trans, n)
    r2 = drot2  - np.random.normal(0.0, sd_rot2,  n)

    th = parts[:, 2]
    parts[:, 0] += tr * np.cos(th + r1)
    parts[:, 1] += tr * np.sin(th + r1)
    parts[:, 2] = (th + r1 + r2 + math.pi) % (2 * math.pi) - math.pi
    return parts


def measurement_update(parts, weights, field, beam_angles, beam_ranges):
    rel = beam_angles - math.pi / 2.0
    th = parts[:, 2][:, None]
    glob = th + rel[None, :]
    ex = parts[:, 0][:, None] + beam_ranges[None, :] * np.cos(glob)
    ey = parts[:, 1][:, None] + beam_ranges[None, :] * np.sin(glob)

    row, col = world_to_map_idx(ex, ey)
    dist = field[row, col]

    gauss = np.exp(-(dist ** 2) / (2.0 * SIGMA_HIT ** 2))
    p = Z_HIT * gauss + Z_RAND
    logw = np.sum(np.log(p), axis=1)
    nbeams = max(1, p.shape[1])
    w_avg = float(np.exp(np.mean(logw) / nbeams))

    logw -= logw.max()
    new_w = weights * np.exp(logw)

    s = new_w.sum()
    if s <= 0 or not np.isfinite(s):
        new_w = np.full(len(weights), 1.0 / len(weights))
    else:
        new_w /= s
    return new_w, w_avg


def systematic_resample(parts, weights, field, p_inject=0.0):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform()) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    new_parts = parts[idx].copy()
    new_parts[:, 0] += np.random.normal(0.0, ROUGHEN_XY, n)
    new_parts[:, 1] += np.random.normal(0.0, ROUGHEN_XY, n)
    new_parts[:, 2] += np.random.normal(0.0, ROUGHEN_YAW, n)
    new_parts[:, 2] = (new_parts[:, 2] + math.pi) % (2 * math.pi) - math.pi

    n_inject = int(round(min(max(p_inject, 0.0), INJECT_CAP) * n))
    if n_inject > 0:
        sel = np.random.choice(n, n_inject, replace=False)
        new_parts[sel] = sample_free_poses(field, n_inject)

    return new_parts, np.full(n, 1.0 / n)


def neff(weights):
    return 1.0 / np.sum(weights ** 2)


def estimate_pose(parts, weights):
    x = np.sum(weights * parts[:, 0])
    y = np.sum(weights * parts[:, 1])
    s = np.sum(weights * np.sin(parts[:, 2]))
    c = np.sum(weights * np.cos(parts[:, 2]))
    yaw = math.atan2(s, c)
    return x, y, yaw


def wander(laser):
    vals = np.asarray(laser.values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        HAL.setV(0.0); HAL.setW(0.0); return
    n = len(laser.values)
    front = np.asarray(laser.values[int(0.4 * n):int(0.6 * n)], dtype=float)
    front = front[np.isfinite(front) & (front > 0)]
    if front.size and front.min() < FRONT_STOP_DIST:
        HAL.setV(0.0); HAL.setW(TURN_W)
    else:
        HAL.setV(CRUISE_V); HAL.setW(0.0)


print("[MCL] a inicializar...")
occ = load_map()
field = build_likelihood_field(occ)


def _valid(p):
    return abs(p.x) + abs(p.y) + abs(p.yaw) > 1e-9

for _ in range(100):
    if _valid(HAL.getPose3d()):
        break
    time.sleep(0.05)

if not _valid(HAL.getOdom()) and _valid(HAL.getPose3d()):
    print("[MCL] /odom_noisy indisponível -> a usar /odom (getPose3d) no motion model.")
    read_odom = HAL.getPose3d
else:
    read_odom = HAL.getOdom

prev = read_odom()
prev_x, prev_y, prev_yaw = prev.x, prev.y, prev.yaw

if GLOBAL_INIT:
    particles, weights = init_particles_global(field, NUM_PARTICLES)
    print("[MCL] %d partículas (init GLOBAL). A localizar..." % NUM_PARTICLES)
elif DEMO_BAD_INIT:
    cx = prev_x + DEMO_OFFSET_XY[0]
    cy = prev_y + DEMO_OFFSET_XY[1]
    particles, weights = init_particles_local(
        field, NUM_PARTICLES, cx, cy, prev_yaw,
        spread_xy=DEMO_SPREAD_XY, spread_yaw=DEMO_SPREAD_YAW)
    print("[MCL] DEMO: a partir de POSE INICIAL RUIM. A convergir...")
    _hold_end = time.time() + DEMO_HOLD_SECONDS
    while time.time() < _hold_end:
        ex0, ey0, eyaw0 = estimate_pose(particles, weights)
        WebGUI.showPosition(ex0, ey0, eyaw0)
        WebGUI.showParticles([[float(p[0]), float(p[1]), float(p[2])] for p in particles])
        Frequency.tick(RATE_HZ)
else:
    particles, weights = init_particles_local(
        field, NUM_PARTICLES, prev_x, prev_y, prev_yaw)
    print("[MCL] %d partículas (init LOCAL @ odom). A localizar..." % NUM_PARTICLES)

w_slow = 0.0
w_fast = 0.0

while True:
    odom = read_odom()
    laser = HAL.getLaserData()

    dx = odom.x - prev_x
    dy = odom.y - prev_y
    dtrans = math.hypot(dx, dy)
    if dtrans < 1e-3:
        drot1 = 0.0
    else:
        drot1 = angle_diff(math.atan2(dy, dx), prev_yaw)
    drot2 = angle_diff(odom.yaw - prev_yaw, drot1)
    prev_x, prev_y, prev_yaw = odom.x, odom.y, odom.yaw

    if dtrans > 1e-3 or abs(drot1) > 1e-3 or abs(drot2) > 1e-3:
        particles = motion_update(particles, dtrans, drot1, drot2)

    vals = np.asarray(laser.values, dtype=float)
    nrays = len(vals)
    p_inject = 0.0
    if nrays > 0:
        angles_all = np.linspace(laser.minAngle, laser.maxAngle, nrays)
        idx = np.linspace(0, nrays - 1, min(NUM_BEAMS, nrays)).astype(int)
        b_ang = angles_all[idx]
        b_rng = vals[idx]
        maxr = laser.maxRange if laser.maxRange and laser.maxRange > 0 else 10.0
        good = np.isfinite(b_rng) & (b_rng > 0.0) & (b_rng < maxr * 0.99)
        b_ang, b_rng = b_ang[good], b_rng[good]

        if b_rng.size >= 3:
            weights, w_avg = measurement_update(
                particles, weights, field, b_ang, b_rng)
            w_slow += ALPHA_SLOW * (w_avg - w_slow) if w_slow > 0 else w_avg
            w_fast += ALPHA_FAST * (w_avg - w_fast) if w_fast > 0 else w_avg
            if w_slow > 0:
                p_inject = 1.0 - (w_fast / w_slow)

    est_x, est_y, est_yaw = estimate_pose(particles, weights)
    WebGUI.showPosition(est_x, est_y, est_yaw)
    WebGUI.showParticles([[float(p[0]), float(p[1]), float(p[2])] for p in particles])

    if neff(weights) < RESAMPLE_RATIO * NUM_PARTICLES or p_inject > 0.0:
        particles, weights = systematic_resample(
            particles, weights, field, p_inject)

    if ENABLE_WANDER:
        wander(laser)

    Frequency.tick(RATE_HZ)
