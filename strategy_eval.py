"""Опыт: прогноз по нашей стратегии (strategy_replay.py → runs/strategy/<игра>.npz) и бустинг, который его поправляет.

1. Без обучения: робот едет к точке, которую выбрала бы наша стратегия (model.rollout — «к цели с торможением»);
   ускорение и предельная скорость — константы стратегии и лучшие из сетки по обучающим играм.
2. XGBoost (как boost.py --engine xgb) на 67 признаках, к которым добавлены признаки стратегии и/или команда
   робота (one-hot по имени): видно отдельно, что даёт стратегия и что даёт знание команды.

    venv/bin/python strategy_eval.py data/*.npz --out runs/strategy-eval --step 6 --val-logs a.npz ...
"""
import argparse, gc, os, time
import numpy as np
import torch
import baselines, evaluate, features
import model as M
from features import HORIZONS

STRATEGY_ACC, STRATEGY_VMAX = 6.0, 2.0  # const.MAX_ACCELERATION, MAX_SPEED стратегии (IS_SIMULATOR_USED), м/с², м/с
STRAT_NAMES = ['strat_target_x', 'strat_target_y', 'strat_has_target', 'strat_vel_x', 'strat_vel_y', 'strat_has_vel'] + \
    [f'strat_roll_{h}_{a}' for h in HORIZONS for a in 'xy']


def strategy_inputs(paths, meta, src):
    """Точка назначения и скорость стратегии в системе робота (м, м/с; NaN → 0 и флаг) и имя команды робота."""
    n = len(meta['log'])
    target, vel = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
    has_t, has_v = np.zeros(n, bool), np.zeros(n, bool)
    team = np.empty(n, object)
    for j, p in enumerate(paths):
        rows = np.where(meta['log'] == j)[0]
        if not len(rows):
            continue
        g = features.load_grid(p)
        names = np.load(p)['names']
        s = np.load(os.path.join(src, os.path.basename(p)))
        i, k = meta['frame'][rows], meta['robot'][rows]
        r = (i - s['frames'][0]) // (s['frames'][1] - s['frames'][0])
        assert np.all(s['frames'][r] == i), 'кадры стратегии не совпадают с примерами: другой --step?'
        # та же система робота, что в features.build
        P, V, B = g['robots'][i, k, :2], g['robots'][i, k, 2:4], g['ball'][i]
        has_ball = np.isfinite(B[:, 0])
        to_ball = np.where(has_ball[:, None], B[:, :2] - P, 0.0)
        moving = np.linalg.norm(V, axis=1) > features.MOVING
        ang = np.where(moving, np.arctan2(V[:, 1], V[:, 0]), np.arctan2(to_ball[:, 1], to_ball[:, 0]))
        c, sn = np.cos(ang), np.sin(ang)
        m = np.where(features._rot(to_ball, c, sn, np.ones(len(rows)))[:, 1] < 0, -1.0, 1.0)
        tg, cv = s['target'][r, k].astype(np.float64), s['cmd_vel'][r, k].astype(np.float64)
        has_t[rows], has_v[rows] = np.isfinite(tg[:, 0]), np.isfinite(cv[:, 0])
        target[rows] = np.nan_to_num(features._rot(tg - P, c, sn, m) / 1000)
        vel[rows] = np.nan_to_num(features._rot(cv, c, sn, m) / 1000)
        team[rows] = names[k // 16]
    return target, has_t, vel, has_v, team


def roll(v0, target, has_t, acc, vmax, batch=262144):
    """Отклонение от «постоянной скорости» (M, H, 2), если робот едет к точке стратегии; без точки — 0."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = np.zeros((len(v0), len(HORIZONS), 2), np.float32)
    for j in range(0, len(v0), batch):
        sl = slice(j, j + batch)
        n = len(v0[sl])
        with torch.no_grad():
            y = M.rollout(torch.from_numpy(v0[sl]).to(dev), torch.from_numpy(target[sl]).to(dev),
                          torch.full((n,), acc, device=dev), torch.full((n,), vmax, device=dev))
        out[sl] = y.cpu().numpy()
    out[~has_t] = 0
    torch.cuda.empty_cache()  # память нужна XGBoost
    return out


def fit_roll(v0, target, has_t, Y, running):
    """Ускорение и предельная скорость с наименьшей средней ошибкой через 1 с (игра идёт, есть точка)."""
    use = np.where(running & has_t)[0]
    use = np.random.default_rng(0).choice(use, min(len(use), 200000), replace=False)
    best = None
    for acc in (1.0, 2.0, 3.0, 4.0, 6.0):
        for vmax in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            err = np.linalg.norm(roll(v0[use], target[use], has_t[use], acc, vmax)[:, -1] - Y[use, -1], axis=1).mean()
            if best is None or err < best[0]:
                best = (err, acc, vmax)
    return best[1], best[2]


def xgb_fit(Xfit, Yfit, Xstop, Ystop, Xva, rounds, info, label):
    import xgboost as xgb
    params = dict(objective='reg:squarederror', learning_rate=0.1, max_depth=10, min_child_weight=200, colsample_bytree=0.9,
                  subsample=0.5, max_bin=63, tree_method='hist', device='cuda')
    dfit = xgb.QuantileDMatrix(Xfit, max_bin=63)
    dstop = xgb.QuantileDMatrix(Xstop, ref=dfit, max_bin=63)
    dva = xgb.DMatrix(Xva)
    pred = np.zeros((len(Xva), len(HORIZONS), 2), np.float32)
    trees = []
    for h in range(len(HORIZONS)):
        for ax in range(2):
            dfit.set_label(Yfit[:, h, ax])
            dstop.set_label(Ystop[:, h, ax])
            b = xgb.train(params, dfit, rounds, evals=[(dstop, 'stop')], early_stopping_rounds=50, verbose_eval=False)
            pred[:, h, ax] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
            trees.append(b.best_iteration)
            del b
            gc.collect()  # иначе бустеры держат память видеокарты до конца и последний вариант не помещается
    info.append(f'{label}: признаков {Xfit.shape[1]}, деревьев {min(trees)}–{max(trees)}')
    print(info[-1], flush=True)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-logs', nargs='*', default=[])
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--src', default='runs/strategy', help='папка с выходом strategy_replay.py')
    ap.add_argument('--rounds', type=int, default=1000)
    ap.add_argument('--no-boost', action='store_true', help='только прогноз стратегии без обучения')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    X, Y, meta = features.build_all(a.logs, a.step)
    target, has_t, svel, has_v, team = strategy_inputs(a.logs, meta, a.src)
    tr, va = features.split(meta, a.logs, a.val_logs)
    v0 = X[:, :2].copy()
    print(f'примеров: обучение {tr.sum()}, проверка {va.sum()}; точка стратегии есть у {has_t.mean() * 100:.0f} % '
          f'(игра идёт: {has_t[meta["running"]].mean() * 100:.0f} %), датасет за {time.time() - t0:.0f} с', flush=True)

    acc, vmax = fit_roll(v0[tr], target[tr], has_t[tr], Y[tr], meta['running'][tr])
    mva = {k: v[va] for k, v in meta.items()}
    preds = {name: fn(X[va], mva) for name, fn in baselines.fit_all(X[tr], Y[tr], {k: v[tr] for k, v in meta.items()}).items()}
    preds[f'стратегия ({STRATEGY_ACC:g} м/с², {STRATEGY_VMAX:g} м/с)'] = roll(v0[va], target[va], has_t[va], STRATEGY_ACC, STRATEGY_VMAX)
    roll_all = roll(v0, target, has_t, acc, vmax)
    preds[f'стратегия ({acc:g} м/с², {vmax:g} м/с, подобрано)'] = roll_all[va]
    info = [f'точка стратегии есть у {has_t[va].mean() * 100:.0f} % проверочных примеров '
            f'(игра идёт: {has_t[va & meta["running"]].mean() * 100:.0f} %)']

    if not a.no_boost:
        S = np.concatenate([target, has_t[:, None], svel, has_v[:, None], roll_all.reshape(len(X), -1)], axis=1).astype(np.float32)
        teams = sorted(set(team))
        T = (team[:, None] == np.array(teams)[None, :]).astype(np.float32)
        # остановка — по 3 последним обучающим играм, как в boost.py
        train_ids = np.unique(meta['log'][tr])
        stop = tr & np.isin(meta['log'], train_ids[-3:])
        fit = tr & ~stop
        variants = (('XGBoost', [X]), ('XGBoost + команда', [X, T]), ('XGBoost + стратегия', [X, S]),
                    ('XGBoost + стратегия + команда', [X, S, T]))
        for v, (label, parts) in enumerate(variants):
            # прогноз каждого варианта сразу на диск: если следующий упадёт, готовые не пересчитываются
            path = f'{a.out}/pred_{v}.npy'
            if os.path.exists(path):
                preds[label] = np.load(path)
                info.append(f'{label}: из {path}')
                continue
            Z = np.concatenate(parts, axis=1)
            preds[label] = xgb_fit(Z[fit], Y[fit], Z[stop], Y[stop], Z[va], a.rounds, info, label)
            np.save(path, preds[label])
            del Z
            gc.collect()

    lines = [f'логи: {len(a.logs)}, проверка: {a.val_logs}', *info]

    def out(s):
        print(s)
        lines.append(s)
    evaluate.report(preds, Y[va], mva, None, out)
    # только примеры, где у стратегии есть точка: там её прогноз и сравнивать
    sub = {k: v[has_t[va]] for k, v in mva.items()}
    out('\n--- только примеры с точкой стратегии ---')
    evaluate.report({k: v[has_t[va]] for k, v in preds.items()}, Y[va][has_t[va]], sub, None, out)
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'\nвсего {time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()
