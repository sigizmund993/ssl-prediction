"""Ошибки прогнозов на проверочной части: прогнозы без обучения (baselines.py) и модель, если есть.

    python evaluate.py [--run runs/default] [--val-logs игра.npz ...] data/*.npz
"""
import argparse, os
import numpy as np
import baselines, features
from features import HORIZONS

# радиус круга, в который круговое гауссово распределение попадает с вероятностью 50 % и 90 %, в единицах sigma
R50, R90 = np.sqrt(2 * np.log(2)), np.sqrt(2 * np.log(10))


def subsets(meta):
    return {'все примеры': np.ones(len(meta['speed']), bool), 'игра идёт': meta['running'],
            'игра идёт, быстрее 0.5 м/с': meta['running'] & (meta['speed'] > 0.5)}


def report(preds, Y, meta, sigma=None, out=print):
    """preds: {название: прогноз (M, H, 2)}. Ошибка — расстояние между прогнозом и настоящим положением, мм."""
    for sub, mask in subsets(meta).items():
        out(f'\n{sub}: примеров {int(mask.sum())}. Ошибка, мм: среднее / медиана / p90')
        out(f'{"":34s}' + ''.join(f'{f"через {h} с":>22s}' for h in HORIZONS))
        for name, pred in preds.items():
            err = np.linalg.norm(pred[mask] - Y[mask], axis=2) * 1000
            cells = [f'{err[:, j].mean():.0f} / {np.median(err[:, j]):.0f} / {np.percentile(err[:, j], 90):.0f}' for j in range(len(HORIZONS))]
            out(f'{name:34s}' + ''.join(f'{c:>22s}' for c in cells))
    if sigma is not None:
        err = np.linalg.norm(preds['модель'] - Y, axis=2)
        out('\nразброс модели (sigma): доля примеров, где ошибка меньше радиуса, в который модель обещает 50 % и 90 %')
        out(''.join(f'  через {h} с: {np.mean(err[:, j] < R50 * sigma[:, j]) * 100:.0f} % и {np.mean(err[:, j] < R90 * sigma[:, j]) * 100:.0f} %;'
                    for j, h in enumerate(HORIZONS)))
        out('медиана sigma, мм:' + ''.join(f'  {np.median(sigma[:, j]) * 1000:.0f}' for j in range(len(HORIZONS))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--run', default=None, help='папка обучения с model.pt')
    ap.add_argument('--val-logs', nargs='*', default=[])
    ap.add_argument('--step', type=int, default=3)
    a = ap.parse_args()
    X, Y, meta = features.build_all(a.logs, a.step)
    tr, va = features.split(meta, a.logs, a.val_logs)
    mtr = {k: v[tr] for k, v in meta.items()}
    mva = {k: v[va] for k, v in meta.items()}
    print(f'примеров: обучение {int(tr.sum())}, проверка {int(va.sum())}, признаков {X.shape[1]}')
    preds = {name: fn(X[va], mva) for name, fn in baselines.fit_all(X[tr], Y[tr], mtr).items()}
    sigma = None
    if a.run and os.path.exists(f'{a.run}/model.pt'):
        import model as M
        net, _ = M.load(f'{a.run}/model.pt')
        preds['модель'], sigma = net.predict(X[va])
    report(preds, Y[va], mva, sigma)


if __name__ == '__main__':
    main()
