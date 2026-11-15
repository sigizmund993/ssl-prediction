"""Датасет для прогноза движения роботов: npz от extract.py → признаки и ответы.

Пример = (робот, кадр). Всё считается в системе координат робота: начало в его положении, ось x вдоль скорости
(если он медленнее MOVING мм/с — на мяч), ось y отражается так, чтобы мяч был слева (y >= 0). Длины в метрах,
скорости в м/с. Ответ — на сколько робот отклонился от прогноза «постоянная скорость» через HORIZONS секунд.
Признаки каждого робота считаются одинаково для обеих команд: «свои» и «чужие» — относительно него самого.
"""
import numpy as np

DT = 1 / 60                         # сетка времени, с
HORIZONS = (0.25, 0.5, 0.75, 1.0)   # на сколько вперёд прогноз, с
LAGS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)  # история скорости, с назад
N_MATES, N_OPP = 3, 3               # сколько ближайших своих и чужих (для этого робота) идёт в признаки
MOVING = 100.0                      # мм/с: быстрее — ось x вдоль скорости
RUNNING = (2, 3, 8, 9)              # NORMAL_START, FORCE_START, DIRECT_FREE_YELLOW/BLUE: игра идёт
STOP = 1
MAX_SPEED = 6000.0                  # мм/с: быстрее робот не ездит — это сбой трекера или перенос робота
DEFAULT_FIELD = (9000.0, 6000.0)    # если в логе нет геометрии (дивизион B)


def feature_names():
    names = ['v_x', 'v_y', 'moving']
    for lag in LAGS:
        names += [f'dv{lag}_x', f'dv{lag}_y']
    names += ['cos_theta', 'sin_theta', 'omega', 'has_theta']
    names += ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'has_ball']
    for kind, n in (('mate', N_MATES), ('opp', N_OPP)):
        for j in range(n):
            names += [f'{kind}{j}_x', f'{kind}{j}_y', f'{kind}{j}_vx', f'{kind}{j}_vy', f'{kind}{j}_present']
    names += ['wall_fwd', 'wall_back', 'wall_left', 'wall_right']
    names += ['own_goal_x', 'own_goal_y', 'opp_goal_x', 'opp_goal_y', 'side_known']
    names += ['running', 'stop', 'keeper', 'nearest_to_ball']
    return names


def load_grid(path):
    """Кадры трекера (~73 Гц, неровно) → ровная сетка 60 Гц по ближайшему кадру."""
    d = np.load(path)
    t = d['t']
    grid = np.arange(t[0], t[-1], DT)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    idx = np.where(np.abs(t[prev] - grid) < np.abs(t[idx] - grid), prev, idx)
    field = d['field'] if 'field' in d and np.isfinite(d['field'][0]) else np.array(DEFAULT_FIELD)
    return dict(grid=grid - grid[0], ok=np.abs(t[idx] - grid) < 0.03, robots=d['robots'][idx].astype(np.float64),
                ball=d['ball'][idx], cmd=d['cmd'][idx], blue_pos=d['blue_pos'][idx] if 'blue_pos' in d else np.full(len(idx), -1),
                keeper=d['keeper'][idx] if 'keeper' in d else np.zeros((len(idx), 2), int), field=field[:2])


def _rot(v, c, s, m):
    """Вектор(ы) мира → система робота: поворот на угол (c, s) и отражение y знаком m. v: (..., 2), c/s/m: (n,)."""
    shape = (-1,) + (1,) * (v.ndim - 2)
    c, s, m = c.reshape(shape), s.reshape(shape), m.reshape(shape)
    return np.stack([c * v[..., 0] + s * v[..., 1], m * (c * v[..., 1] - s * v[..., 0])], axis=-1)


def build(path, step=3, log_id=0):
    """Признаки X (M, F), ответы Y (M, H, 2) и сведения о примерах meta по одному npz. step — брать каждый step-й кадр."""
    g = load_grid(path)
    R, B, cmd, ok = g['robots'], g['ball'], g['cmd'], g['ok']
    G = len(ok)
    lag_n = [int(round(l / DT)) for l in LAGS]
    hor_n = [int(round(h / DT)) for h in HORIZONS]
    # окно [i - max лаг, i + max горизонт] целиком из хороших кадров
    bad = np.r_[0, np.cumsum(~ok)]
    frames = np.arange(max(lag_n), G - max(hor_n), step)
    frames = frames[(bad[frames + max(hor_n) + 1] - bad[frames - max(lag_n)] == 0) & np.isin(cmd[frames], RUNNING + (STOP,))]
    half = g['field'] / 2
    has_ball_all = np.isfinite(B[:, 0])
    # ближайший к мячу в каждой команде
    dball = np.linalg.norm(R[:, :, :2] - B[:, None, :2], axis=2)
    dball = np.where(np.isfinite(dball), dball, np.inf)
    nearest = np.stack([np.argmin(dball[:, :16], axis=1), 16 + np.argmin(dball[:, 16:], axis=1)], axis=1)

    Xs, Ys, metas = [], [], []
    for k in range(32):
        team = k // 16
        P, Vel = R[:, k, :2], R[:, k, 2:4]
        i = frames
        valid = np.isfinite(P[i, 0])
        for n in lag_n:
            valid &= np.isfinite(P[i - n, 0])
        for n in hor_n:
            valid &= np.isfinite(P[i + n, 0]) & (np.linalg.norm(P[i + n] - P[i], axis=1) < MAX_SPEED * n * DT)
        valid &= np.linalg.norm(Vel[i], axis=1) < MAX_SPEED
        i = i[valid]
        if len(i) == 0:
            continue
        p, v = P[i], Vel[i]
        speed = np.linalg.norm(v, axis=1)
        has_ball = has_ball_all[i]
        to_ball = np.where(has_ball[:, None], B[i, :2] - p, 0.0)
        moving = speed > MOVING
        ang = np.where(moving, np.arctan2(v[:, 1], v[:, 0]), np.arctan2(to_ball[:, 1], to_ball[:, 0]))
        c, s = np.cos(ang), np.sin(ang)
        one = np.ones(len(i))
        m = np.where(_rot(to_ball, c, s, one)[:, 1] < 0, -1.0, 1.0)

        f = [_rot(v, c, s, m) / 1000, moving[:, None].astype(float)]
        for n in lag_n:
            f.append(_rot(Vel[i - n] - v, c, s, m) / 1000)
        theta, omega = R[i, k, 4], R[i, k, 5]
        has_theta = np.isfinite(theta)
        rel = np.where(has_theta, theta - ang, 0.0)
        f.append(np.stack([np.cos(rel) * has_theta, m * np.sin(rel) * has_theta, m * np.where(np.isfinite(omega), omega, 0.0), has_theta], axis=1))
        f.append(np.concatenate([_rot(to_ball, c, s, m) / 1000, _rot(np.where(has_ball[:, None], B[i, 2:4], 0.0), c, s, m) / 1000,
                                 has_ball[:, None]], axis=1))
        # ближайшие свои и чужие
        others = np.array([o for o in range(32) if o != k])
        rel_pos = R[i][:, others, :2] - p[:, None, :]
        dist = np.linalg.norm(rel_pos, axis=2)
        for same, count in ((True, N_MATES), (False, N_OPP)):
            mask = (others // 16 == team) == same
            d = np.where(mask[None, :] & np.isfinite(dist), dist, np.inf)
            order = np.argsort(d, axis=1)[:, :count]
            rows = np.arange(len(i))[:, None]
            present = np.isfinite(d[rows, order])
            pos = np.where(present[..., None], rel_pos[rows, order], 0.0)
            vel = np.where(present[..., None], R[i][:, others, 2:4][rows, order], 0.0)
            block = np.concatenate([_rot(pos, c, s, m) / 1000, _rot(vel, c, s, m) / 1000, present[..., None]], axis=2)
            f.append(block.reshape(len(i), -1))
        # сколько до границы поля вперёд, назад, влево и вправо (в системе робота), не больше 6 м
        ex = np.stack([c, s], axis=1)
        ey = np.stack([-s, c], axis=1) * m[:, None]
        walls = []
        for d in (ex, -ex, ey, -ey):
            with np.errstate(divide='ignore', invalid='ignore'):
                tt = (np.sign(d) * half - p) / d
            tt = np.where(np.abs(d) < 1e-9, np.inf, tt)
            walls.append(np.clip(np.min(tt, axis=1), 0, 6000) / 1000)
        f.append(np.stack(walls, axis=1))
        # свои и чужие ворота
        side_known = g['blue_pos'][i] >= 0
        blue_sign = np.where(g['blue_pos'][i] == 1, 1.0, -1.0)
        own_sign = blue_sign if team == 1 else -blue_sign
        own = np.stack([own_sign * half[0], np.zeros(len(i))], axis=1) - p
        opp = np.stack([-own_sign * half[0], np.zeros(len(i))], axis=1) - p
        f.append(np.concatenate([_rot(own, c, s, m) / 1000 * side_known[:, None], _rot(opp, c, s, m) / 1000 * side_known[:, None],
                                 side_known[:, None]], axis=1))
        running = np.isin(cmd[i], RUNNING)
        f.append(np.stack([running, cmd[i] == STOP, g['keeper'][i, team] == k % 16, nearest[i, team] == k], axis=1).astype(float))
        Xs.append(np.concatenate(f, axis=1).astype(np.float32))

        y = np.stack([_rot(P[i + n] - p - v * (n * DT), c, s, m) / 1000 for n in hor_n], axis=1)
        Ys.append(y.astype(np.float32))
        acc = _rot((v - Vel[i - lag_n[1]]) / LAGS[1], c, s, m) / 1000
        metas.append(dict(log=np.full(len(i), log_id), frame=i, time=g['grid'][i], robot=np.full(len(i), k), running=running,
                          speed=speed / 1000, vel=_rot(v, c, s, m) / 1000, acc=acc))
    meta = {key: np.concatenate([mm[key] for mm in metas]) for key in metas[0]}
    return np.concatenate(Xs), np.concatenate(Ys), meta


def build_all(paths, step=3):
    parts = [build(p, step, j) for j, p in enumerate(paths)]
    meta = {key: np.concatenate([p[2][key] for p in parts]) for key in parts[0][2]}
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), meta


def split(meta, paths, val_logs=(), block=60.0, purge=1.5):
    """Маски обучающей и проверочной частей. val_logs — имена npz, целиком уходящих в проверку (так честнее всего:
    другая игра, другие команды). Если их нет, проверка — каждый пятый кусок времени по block секунд; purge секунд
    у краёв каждого куска выбрасываются, чтобы соседние (почти одинаковые) кадры не попали в обе части."""
    if val_logs:
        import os
        ids = [j for j, p in enumerate(paths) if os.path.basename(p) in [os.path.basename(v) for v in val_logs]]
        if not ids:
            raise SystemExit(f'проверочных логов {val_logs} нет среди {paths}')
        val = np.isin(meta['log'], ids)
        return ~val, val
    n = np.floor(meta['time'] / block)
    inside = meta['time'] - n * block
    keep = (inside > purge) & (inside < block - purge)
    val = (n % 5 == 4)
    return ~val & keep, val & keep
