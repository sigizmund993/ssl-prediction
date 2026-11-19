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


def model_preds(net, X, Y):
    """Прогнозы сети для отчёта: ({название: (M, H, 2)}, sigma (M, H) для «модель», варианты или None).
    У сети с вариантами «модель» — самый вероятный вариант; добавляются среднее по вариантам и лучший из них — тот,
    что ближе всего к настоящей траектории (подсказка: в игре ответа не знаем, это потолок для планировщика,
    проверяющего все варианты). X — признаки или кортеж входов (признаки, объекты) у SetPredictor."""
    import model as M
    inputs = X if isinstance(X, tuple) else (X,)
    if not isinstance(net, M.MixturePredictor):
        pred, sigma = net.predict(*inputs)
        return {'модель': pred}, sigma, None
    means, sigmas, w = net.predict(X)
    rows = np.arange(len(X))
    top = w.argmax(1)
    best = np.linalg.norm(means - Y[:, None], axis=3).mean(2).argmin(1)
    preds = {'модель': means[rows, top], 'модель, среднее по вариантам': (means * w[:, :, None, None]).sum(1),
             'модель, лучший вариант (знаем ответ)': means[rows, best]}
    return preds, sigmas[rows, top], (means, sigmas, w)


def report(preds, Y, meta, sigma=None, out=print, modes=None, min_weight=0.1):
    """preds: {название: прогноз (M, H, 2)}. Ошибка — расстояние между прогнозом и настоящим положением, мм.
    sigma — разброс для preds['модель']; modes — (средние, sigma, веса) вариантов у сети с вариантами."""
    for sub, mask in subsets(meta).items():
        out(f'\n{sub}: примеров {int(mask.sum())}. Ошибка, мм: среднее / медиана / p90')
        out(f'{"":38s}' + ''.join(f'{f"через {h} с":>22s}' for h in HORIZONS))
        for name, pred in preds.items():
            err = np.linalg.norm(pred[mask] - Y[mask], axis=2) * 1000
            cells = [f'{err[:, j].mean():.0f} / {np.median(err[:, j]):.0f} / {np.percentile(err[:, j], 90):.0f}' for j in range(len(HORIZONS))]
            out(f'{name:38s}' + ''.join(f'{c:>22s}' for c in cells))
    if sigma is not None:
        err = np.linalg.norm(preds['модель'] - Y, axis=2)
        out('\nразброс модели (sigma): доля примеров, где ошибка меньше радиуса, в который модель обещает 50 % и 90 %')
        out(''.join(f'  через {h} с: {np.mean(err[:, j] < R50 * sigma[:, j]) * 100:.0f} % и {np.mean(err[:, j] < R90 * sigma[:, j]) * 100:.0f} %;'
                    for j, h in enumerate(HORIZONS)))
        out('медиана sigma, мм:' + ''.join(f'  {np.median(sigma[:, j]) * 1000:.0f}' for j in range(len(HORIZONS))))
        out('круг «90 %»: средняя площадь, м²:' + ''.join(f'  {np.mean(np.pi * (R90 * sigma[:, j]) ** 2):.3f}' for j in range(len(HORIZONS))))
    if modes is not None:
        means, sigmas, w = modes
        ws = -np.sort(-w, axis=1)
        out(f'\nварианты: {w.shape[1]}, средние веса по убыванию: ' + ', '.join(f'{x:.2f}' for x in ws.mean(0))
            + f'; главный вариант весит меньше 0.6 у {np.mean(ws[:, 0] < 0.6) * 100:.0f} % примеров')
        # зона «90 %» для планировщика: круги R90·sigma всех вариантов с весом не меньше min_weight
        use = (w >= min_weight)[:, :, None]
        err = np.linalg.norm(means - Y[:, None], axis=3)
        inside = ((err < R90 * sigmas) & use).any(1)
        area = (np.pi * (R90 * sigmas) ** 2 * use).sum(1)
        out(f'круги «90 %» вариантов с весом от {min_weight}: доля примеров внутри хотя бы одного и средняя суммарная площадь, м²:')
        out(''.join(f'  через {h} с: {inside[:, j].mean() * 100:.0f} %, {area[:, j].mean():.3f};' for j, h in enumerate(HORIZONS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--run', default=None, help='папка обучения с model.pt')
    ap.add_argument('--val-logs', nargs='*', default=[])
    ap.add_argument('--step', type=int, default=3)
    a = ap.parse_args()
    cfg = None
    if a.run and os.path.exists(f'{a.run}/model.pt'):
        import model as M
        net, cfg = M.load(f'{a.run}/model.pt')
    sets = cfg is not None and cfg.get('arch') == 'sets'
    X, Y, meta, *rest = features.build_all(a.logs, a.step, agents=sets)
    tr, va = features.split(meta, a.logs, a.val_logs)
    mtr = {k: v[tr] for k, v in meta.items()}
    mva = {k: v[va] for k, v in meta.items()}
    print(f'примеров: обучение {int(tr.sum())}, проверка {int(va.sum())}, признаков {X.shape[1]}')
    preds = {name: fn(X[va], mva) for name, fn in baselines.fit_all(X[tr], Y[tr], mtr).items()}
    sigma = modes = None
    if cfg is not None:
        if cfg.get('features') != features.feature_names() or (sets and cfg.get('agents') != features.AGENT_NAMES):
            raise SystemExit(f'{a.run}: модель обучена на других признаках ({cfg["n_features"]}, сейчас '
                             f'{len(features.feature_names())}) — переобучи')
        mp, sigma, modes = model_preds(net, (X[va], rest[0][va]) if sets else X[va], Y[va])
        preds.update(mp)
    report(preds, Y[va], mva, sigma, modes=modes)


if __name__ == '__main__':
    main()
