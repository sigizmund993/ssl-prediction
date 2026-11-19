"""Насколько прогноз зависит от команды: обучение на роботах одной команды против обучения на всех.

    python by_team.py data/*.npz [--teams TRAPS "The Bots"] [--test-games 2] [--step 6] [--out runs/teams]

Для каждой команды T тест — её роботы в последних по дате --test-games играх T. Обучение (тестовые игры всех
выбранных команд исключены везде):
  только T                  роботы T в остальных играх T;
  все команды               все роботы всех остальных игр;
  все, столько же примеров  то же, случайно урезанное до размера «только T» (отделяет пользу от объёма данных);
  без T                     все роботы, кроме роботов T (T как незнакомый соперник).
Модель «только T» проверяется и на роботах других выбранных команд. Эпоха выбирается по средней ошибке на части
обучающих примеров (каждый пятый кусок по 60 с), тест в выборе не участвует.
"""
import argparse, copy, os, time
import numpy as np
import torch
import baselines, features
import model as M
from features import HORIZONS


def train(X, Y, meta, mask, epochs, seed, hidden=128, layers=3, lr=1e-3, batch=2048):
    """Сеть на примерах mask; лучшая эпоха — по средней ошибке (по всем горизонтам) на внутренней проверке."""
    torch.manual_seed(seed)
    idx = np.flatnonzero(mask)
    sub = {k: v[idx] for k, v in meta.items()}
    tr_m, va_m = features.split(sub, [], ())
    tr, va = idx[tr_m], idx[va_m]
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    net = M.Predictor(X.shape[1], hidden, layers)
    net.x_mean.copy_(torch.from_numpy(Xtr.mean(0)))
    net.x_std.copy_(torch.from_numpy(Xtr.std(0) + 1e-6))
    net.y_scale.copy_(torch.from_numpy(np.sqrt((Ytr ** 2).sum(-1).mean(0)).astype(np.float32)))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n_batches = max(1, len(Xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * n_batches)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr) / net.y_scale[None, :, None]
    best, best_state, best_epoch = float('inf'), None, 0
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for j in range(n_batches):
            b = perm[j * batch:(j + 1) * batch]
            loss = M.nll(*net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        pred, _ = net.predict(Xva)
        err = np.linalg.norm(pred - Yva, axis=2).mean() * 1000
        if err < best:
            best, best_state, best_epoch = err, copy.deepcopy(net.state_dict()), epoch + 1
    net.load_state_dict(best_state)
    return net, dict(n=len(tr), inner_val=len(va), best_epoch=best_epoch)


def cells(pred, Y, mask):
    err = np.linalg.norm(pred[mask] - Y[mask], axis=2) * 1000
    return [f'{err[:, j].mean():.0f} / {np.median(err[:, j]):.0f} / {np.percentile(err[:, j], 90):.0f}' for j in range(len(HORIZONS))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--teams', nargs='+', default=['TRAPS', 'The Bots'])
    ap.add_argument('--test-games', type=int, default=2)
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/teams')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    paths = sorted(a.logs)  # имена начинаются с даты — порядок по времени
    t0 = time.time()
    X, Y, meta = features.build_all(paths, a.step)
    names = [tuple(str(s) for s in np.load(p)['names']) for p in paths]
    yellow = np.array([n[0] for n in names])
    blue = np.array([n[1] for n in names])
    team = np.where(meta['robot'] < 16, yellow[meta['log']], blue[meta['log']])
    print(f'примеров {len(X)}, датасет за {time.time() - t0:.0f} с')

    games = {T: [j for j, n in enumerate(names) if T in n] for T in a.teams}
    test_games = {T: g[-a.test_games:] for T, g in games.items()}
    all_test = np.isin(meta['log'], sorted({j for g in test_games.values() for j in g}))
    lines = []

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    for T in a.teams:
        if not games[T]:
            raise SystemExit(f'команды {T} нет в логах; есть: {sorted(set(yellow) | set(blue))}')
        out(f'{T}: игр {len(games[T])}, тест: {[os.path.basename(paths[j]) for j in test_games[T]]}')

    # обучающие наборы
    sets = {'все команды': ~all_test}
    for T in a.teams:
        sets[f'только {T}'] = ~all_test & (team == T) & np.isin(meta['log'], games[T])
        sets[f'без {T}'] = ~all_test & (team != T)
        k = int(sets[f'только {T}'].sum())
        keep = np.zeros(len(X), bool)
        keep[rng.choice(np.flatnonzero(~all_test), k, replace=False)] = True
        sets[f'все, столько же примеров, как у {T}'] = keep

    nets, info, lins = {}, {}, {}
    for name, mask in sets.items():
        t1 = time.time()
        nets[name], info[name] = train(X, Y, meta, mask, a.epochs, a.seed)
        print(f'  {name}: обучение {info[name]["n"]} примеров, лучшая эпоха {info[name]["best_epoch"]}, {time.time() - t1:.0f} с', flush=True)
        if name == 'все команды' or name.startswith('только'):
            lins[name] = baselines.Linear().fit(X[mask], Y[mask])

    for T in a.teams:
        test = np.isin(meta['log'], test_games[T]) & (team == T)
        Xt, Yt = X[test], Y[test]
        mt = {k: v[test] for k, v in meta.items()}
        rows = [('постоянная скорость', None, baselines.const_velocity(mt))]
        rows += [(f'линейная регрессия, {n}', None, lins[n].predict(Xt)) for n in ['все команды', f'только {T}']]
        order = [f'только {T}', 'все команды', f'все, столько же примеров, как у {T}', f'без {T}']
        order += [f'только {U}' for U in a.teams if U != T]
        rows += [(f'модель, {n}', info[n]['n'], nets[n].predict(Xt)[0]) for n in order]
        for sub, mask in [('игра идёт', mt['running']), ('игра идёт, быстрее 0.5 м/с', mt['running'] & (mt['speed'] > 0.5))]:
            out(f'\nтест {T}, {sub}: примеров {int(mask.sum())}. Ошибка, мм: среднее / медиана / p90')
            out(f'{"":48s}{"обучение":>10s}' + ''.join(f'{f"через {h} с":>18s}' for h in HORIZONS))
            for name, n, pred in rows:
                out(f'{name:48s}{(str(n) if n else ""):>10s}' + ''.join(f'{c:>18s}' for c in cells(pred, Yt, mask)))
        out()
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'всего {time.time() - t0:.0f} с; отчёт в {a.out}/report.txt')


if __name__ == '__main__':
    main()
