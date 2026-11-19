"""Прогнозы без нейросети. Все возвращают ответ в том же виде, что у модели: отклонение от прогноза
«постоянная скорость» в системе робота, (M, H, 2), метры. meta — из features.build."""
import numpy as np
from features import HORIZONS

H = np.array(HORIZONS)


def const_velocity(meta):
    return np.zeros((len(meta['speed']), len(H), 2), dtype=np.float32)


def accel_decay(meta, tau):
    """Текущее ускорение (по скорости 0.1 с назад) затухает со временем tau: a(t) = a·exp(−t/tau)."""
    k = tau * tau * (H / tau - 1 + np.exp(-H / tau))
    return (meta['acc'][:, None, :] * k[None, :, None]).astype(np.float32)


def brake(meta, decel):
    """Робот тормозит по прямой с замедлением decel (м/с²) до остановки."""
    speed = np.linalg.norm(meta['vel'], axis=1)
    unit = meta['vel'] / np.maximum(speed, 1e-9)[:, None]
    t_stop = speed / decel
    dist = np.where(H[None, :] < t_stop[:, None], speed[:, None] * H[None, :] - decel * H[None, :] ** 2 / 2, (speed ** 2 / (2 * decel))[:, None])
    return ((dist - speed[:, None] * H[None, :])[:, :, None] * unit[:, None, :]).astype(np.float32)


def mean_error(pred, Y):
    return float(np.linalg.norm(pred - Y, axis=2).mean())


def fit_param(fn, grid, meta, Y):
    """Параметр прогноза подбирается по средней ошибке на обучающей части."""
    errs = [mean_error(fn(meta, p), Y) for p in grid]
    return grid[int(np.argmin(errs))]


class Linear:
    """Линейная регрессия по тем же признакам, что у модели (с небольшой регуляризацией)."""

    def fit(self, X, Y, ridge=1e-3, chunk=1 << 18):
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        n = X.shape[1] + 1
        AtA, AtY = np.zeros((n, n)), np.zeros((n, Y[0].size))
        # по частям: float64-копия всей матрицы признаков на десятках игр не помещается в память
        for j in range(0, len(X), chunk):
            A = np.c_[(X[j:j + chunk] - self.mean) / self.std, np.ones(len(X[j:j + chunk]))].astype(np.float64)
            AtA += A.T @ A
            AtY += A.T @ Y[j:j + chunk].reshape(len(A), -1).astype(np.float64)
        reg = ridge * len(X) * np.eye(n)
        reg[-1, -1] = 0
        self.W = np.linalg.solve(AtA + reg, AtY)
        return self

    def predict(self, X):
        A = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        return (A @ self.W).reshape(len(X), len(H), 2).astype(np.float32)


def fit_all(Xtr, Ytr, mtr):
    """Подобрать параметры на обучающей части. Возвращает {название: функция (X, meta) → прогноз}."""
    tau = fit_param(accel_decay, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0], mtr, Ytr)
    dec = fit_param(brake, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], mtr, Ytr)
    lin = Linear().fit(Xtr, Ytr)
    return {
        'постоянная скорость': lambda X, m: const_velocity(m),
        f'ускорение, затухание {tau} с': lambda X, m: accel_decay(m, tau),
        f'торможение {dec} м/с²': lambda X, m: brake(m, dec),
        'линейная регрессия': lambda X, m: lin.predict(X),
    }
