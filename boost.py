"""Опыт: градиентный бустинг (LightGBM; XGBoost или CatBoost на видеокарте — --engine xgb, cat) на тех же признаках, что у сети. Отдельная модель на каждую координату
каждого горизонта (4×2), без sigma. Остановка по ошибке на трёх последних обучающих играх (не на проверочных).
Нужен, чтобы понять, упирается ли точность в архитектуру сети или во входные данные.

    python boost.py data/*.npz --out runs/имя --step 6 --val-logs a.npz ...
"""
import argparse, os, time
import numpy as np
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
    ap.add_argument('--engine', choices=['lgb', 'xgb', 'cat'], default='lgb', help='xgb — XGBoost, cat — CatBoost, оба на cuda')
    ap.add_argument('--depth', type=int, default=10, help='глубина деревьев у xgb и cat')
    ap.add_argument('--train-frac', type=float, default=1.0, help='доля обучающих примеров (случайно): на всех 66 играх долго')
    ap.add_argument('--no-keepers', action='store_true', help='не учить на вратарях (проверка та же)')
    ap.add_argument('--no-stop', action='store_true', help='учить только на кадрах «игра идёт» (проверка та же)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    X, Y, meta = features.build_all(a.logs, a.step)
    tr, va = features.split(meta, a.logs, a.val_logs)
    tr = features.restrict(meta, tr, keepers=not a.no_keepers, stop=not a.no_stop)
    train_ids = np.unique(meta['log'][tr])
    stop = tr & np.isin(meta['log'], train_ids[-max(1, min(3, len(train_ids) // 4)):])  # для ранней остановки
    fit = tr & ~stop & (np.random.default_rng(0).random(len(tr)) < a.train_frac)
    Xfit, Yfit, Xstop, Ystop = X[fit], Y[fit], X[stop], Y[stop]
    Xva, Yva = X[va], Y[va]
    mva = {k: v[va] for k, v in meta.items()}
    print(f'примеров: обучение {len(Xfit)}, остановка {len(Xstop)}, проверка {len(Xva)} (датасет за {time.time() - t0:.0f} с)', flush=True)

    pred = np.zeros_like(Yva)
    info = []
    if a.engine == 'lgb':
        import lightgbm as lgb
        params = dict(objective='l2', learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200, feature_fraction=0.9,
                      bagging_fraction=0.5, bagging_freq=1, max_bin=63, num_threads=os.cpu_count(), verbose=-1)
        dfit = lgb.Dataset(Xfit, free_raw_data=False, params={'max_bin': 63})
        dstop = lgb.Dataset(Xstop, reference=dfit, free_raw_data=False)
    elif a.engine == 'cat':
        import catboost as cb
        params = dict(loss_function='RMSE', learning_rate=a.lr, depth=a.depth, iterations=a.rounds, border_count=63,
                      task_type='GPU', od_type='Iter', od_wait=50, use_best_model=True, verbose=False, train_dir=a.out)
    else:
        import xgboost as xgb
        params = dict(objective='reg:squarederror', learning_rate=a.lr, max_depth=a.depth, min_child_weight=200,
                      colsample_bytree=0.9, subsample=0.5, max_bin=63, tree_method='hist', device='cuda')
        dfit = xgb.QuantileDMatrix(Xfit, max_bin=63)
        dstop = xgb.QuantileDMatrix(Xstop, ref=dfit, max_bin=63)
        dva = xgb.DMatrix(Xva)
    for h in range(len(HORIZONS)):
        for ax in range(2):
            t1 = time.time()
            name = f'{a.out}/h{HORIZONS[h]}_{"xy"[ax]}'
            if a.engine == 'cat':
                b = cb.CatBoostRegressor(**params)
                b.fit(Xfit, Yfit[:, h, ax], eval_set=(Xstop, Ystop[:, h, ax]))
                b.save_model(name + '.cbm')
                pred[:, h, ax] = b.predict(Xva)
                info.append(f'{HORIZONS[h]} с, {"xy"[ax]}: деревьев {b.get_best_iteration()}')
                print(info[-1], f'({time.time() - t1:.0f} с)', flush=True)
                continue
            dfit.set_label(Yfit[:, h, ax])
            dstop.set_label(Ystop[:, h, ax])
            if a.engine == 'lgb':
                b = lgb.train(params, dfit, a.rounds, valid_sets=[dstop],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
                b.save_model(name + '.txt')
                pred[:, h, ax] = b.predict(Xva, num_iteration=b.best_iteration)
            else:
                b = xgb.train(params, dfit, a.rounds, evals=[(dstop, 'stop')], early_stopping_rounds=50, verbose_eval=False)
                b.save_model(name + '.ubj')  # двоичный формат: в 1.7 раза меньше json
                pred[:, h, ax] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
            info.append(f'{HORIZONS[h]} с, {"xy"[ax]}: деревьев {b.best_iteration}')
            print(info[-1], f'({time.time() - t1:.0f} с)', flush=True)
    np.save(f'{a.out}/pred_val.npy', pred)

    preds = {name: fn(Xva, mva) for name, fn in baselines.fit_all(X[tr], Y[tr], {k: v[tr] for k, v in meta.items()}).items()}
    preds['бустинг'] = pred
    lines = [f'логи: {len(a.logs)}, проверка: {a.val_logs}', f'примеров: обучение {len(Xfit)}, остановка {len(Xstop)}, '
             f'проверка {len(Xva)}; {a.engine}, ' + (f'листьев {a.leaves}' if a.engine == 'lgb' else f'глубина {a.depth}') + f', шаг {a.lr}', '; '.join(info)]

    def out(s):
        print(s)
        lines.append(s)
    evaluate.report(preds, Yva, mva, None, out)
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'\nвсего {time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()
