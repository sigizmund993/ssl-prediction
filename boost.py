"""Опыт: градиентный бустинг (LightGBM) на тех же признаках, что у сети. Отдельная модель на каждую координату
каждого горизонта (4×2), без sigma. Остановка по ошибке на трёх последних обучающих играх (не на проверочных).
Нужен, чтобы понять, упирается ли точность в архитектуру сети или во входные данные.

    python boost.py data/*.npz --out runs/имя --step 6 --val-logs a.npz ...
"""
import argparse, os, time
import numpy as np
import lightgbm as lgb
import baselines, evaluate, features
from features import HORIZONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-logs', nargs='*', default=[])
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--rounds', type=int, default=1000)
    ap.add_argument('--leaves', type=int, default=255)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--train-frac', type=float, default=1.0, help='доля обучающих примеров (случайно): на всех 66 играх долго')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    X, Y, meta = features.build_all(a.logs, a.step)
    tr, va = features.split(meta, a.logs, a.val_logs)
    train_ids = np.unique(meta['log'][tr])
    stop = tr & np.isin(meta['log'], train_ids[-max(1, min(3, len(train_ids) // 4)):])  # для ранней остановки
    fit = tr & ~stop & (np.random.default_rng(0).random(len(tr)) < a.train_frac)
    Xfit, Yfit, Xstop, Ystop = X[fit], Y[fit], X[stop], Y[stop]
    Xva, Yva = X[va], Y[va]
    mva = {k: v[va] for k, v in meta.items()}
    print(f'примеров: обучение {len(Xfit)}, остановка {len(Xstop)}, проверка {len(Xva)} (датасет за {time.time() - t0:.0f} с)', flush=True)

    params = dict(objective='l2', learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.5, bagging_freq=1, max_bin=63, num_threads=os.cpu_count(), verbose=-1)
    dfit = lgb.Dataset(Xfit, free_raw_data=False, params={'max_bin': 63})
    dstop = lgb.Dataset(Xstop, reference=dfit, free_raw_data=False)
    pred = np.zeros_like(Yva)
    info = []
    for h in range(len(HORIZONS)):
        for ax in range(2):
            t1 = time.time()
            dfit.set_label(Yfit[:, h, ax])
            dstop.set_label(Ystop[:, h, ax])
            b = lgb.train(params, dfit, a.rounds, valid_sets=[dstop],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            b.save_model(f'{a.out}/h{HORIZONS[h]}_{"xy"[ax]}.txt')
            pred[:, h, ax] = b.predict(Xva, num_iteration=b.best_iteration)
            info.append(f'{HORIZONS[h]} с, {"xy"[ax]}: деревьев {b.best_iteration}')
            print(info[-1], f'({time.time() - t1:.0f} с)', flush=True)
    np.save(f'{a.out}/pred_val.npy', pred)

    preds = {name: fn(Xva, mva) for name, fn in baselines.fit_all(X[tr], Y[tr], {k: v[tr] for k, v in meta.items()}).items()}
    preds['бустинг'] = pred
    lines = [f'логи: {len(a.logs)}, проверка: {a.val_logs}', f'примеров: обучение {len(Xfit)}, остановка {len(Xstop)}, '
             f'проверка {len(Xva)}; листьев {a.leaves}, шаг {a.lr}', '; '.join(info)]

    def out(s):
        print(s)
        lines.append(s)
    evaluate.report(preds, Yva, mva, None, out)
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'\nвсего {time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()
