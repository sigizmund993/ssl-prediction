"""Обучение модели прогноза на логах.

    python train.py [--out runs/default] [--val-logs игра.npz ...] [--epochs 30] data/*.npz

В папке --out остаются model.pt (веса и настройки), weights.npz (веса без torch, см. model.export_npz) и
report.txt (ошибки на проверочной части рядом с прогнозами без обучения).
"""
import argparse, os, time
import numpy as np
import torch
import baselines, evaluate, features
import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out', default='runs/default')
    ap.add_argument('--val-logs', nargs='*', default=[], help='игры, целиком уходящие в проверку')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--step', type=int, default=3, help='брать каждый step-й кадр сетки 60 Гц')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--modes', type=int, default=1, help='сколько вариантов будущего выдаёт сеть (1 — один прогноз)')
    ap.add_argument('--arch', choices=['mlp', 'goal', 'sets'], default='mlp',
                    help='mlp — Predictor/MixturePredictor, goal — точка назначения и физика, sets — сеть по всем объектам')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    sets = a.arch == 'sets'
    X, Y, meta, *rest = features.build_all(a.logs, a.step, agents=sets)
    tr, va = features.split(meta, a.logs, a.val_logs)
    if sets:
        Atr, Ava = rest[0][tr], rest[0][va]
        del rest  # объекты — самый большой массив, освобождаем до копирования признаков
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    mtr = {k: v[tr] for k, v in meta.items()}
    mva = {k: v[va] for k, v in meta.items()}
    n_features = X.shape[1]
    del X, Y, meta  # полный датасет больше не нужен: на десятках игр это гигабайты
    print(f'примеров: обучение {len(Xtr)}, проверка {len(Xva)}, признаков {n_features} (датасет за {time.time() - t0:.0f} с)')

    config = dict(n_features=n_features, hidden=a.hidden, layers=a.layers, horizons=features.HORIZONS, lags=features.LAGS,
                  features=features.feature_names(), logs=[os.path.basename(p) for p in a.logs], val_logs=a.val_logs, step=a.step,
                  modes=a.modes, arch=a.arch)
    if sets:
        config.update(agent_dim=Atr.shape[2], agents=features.AGENT_NAMES, n_agents=Atr.shape[1])
    net = M.build(config)
    if sets:
        sample = Atr[np.random.choice(len(Atr), min(len(Atr), 200000), replace=False)].astype(np.float32).reshape(-1, Atr.shape[2])
        sample = sample[sample[:, -1] > 0]  # нормировка только по присутствующим объектам
        net.a_mean.copy_(torch.from_numpy(sample.mean(0)))
        net.a_std.copy_(torch.from_numpy(sample.std(0) + 1e-3))
    net.x_mean.copy_(torch.from_numpy(Xtr.mean(0)))
    net.x_std.copy_(torch.from_numpy(Xtr.std(0) + 1e-6))
    # масштаб ответа: среднеквадратичное отклонение от «постоянной скорости» на каждом горизонте
    net.y_scale.copy_(torch.from_numpy(np.sqrt((Ytr ** 2).sum(-1).mean(0)).astype(np.float32)))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr) / net.y_scale[None, :, None]
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(Yva) / net.y_scale[None, :, None]
    inp_tr = (xt, torch.from_numpy(Atr)) if sets else (xt,)
    inp_va = (xv, torch.from_numpy(Ava)) if sets else (xv,)
    np_va = (Xva, Ava) if sets else Xva
    best = float('inf')
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for j in range(len(xt) // a.batch):
            idx = perm[j * a.batch:(j + 1) * a.batch]
            loss = M.loss(net, tuple(t[idx] for t in inp_tr), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        net.eval()
        with torch.no_grad():
            val = float(np.mean([M.loss(net, tuple(t[j:j + 65536] for t in inp_va), yv[j:j + 65536]).item()
                                 for j in range(0, len(xv), 65536)]))
        pred = evaluate.model_preds(net, np_va, Yva)[0]['модель']  # у сети с вариантами — самый вероятный
        err = np.linalg.norm(pred - Yva, axis=2).mean(0) * 1000
        mark = ''
        if val < best:
            # сохраняется лучшая по проверочной части эпоха
            best = val
            M.save(net, f'{a.out}/model.pt', config)
            mark = ' *'
        print(f'эпоха {epoch + 1:3d}: обучение {total / (len(xt) // a.batch):.4f}, проверка {val:.4f}, '
              f'средняя ошибка через 0.5 с {err[1]:.0f} мм, через 1 с {err[3]:.0f} мм{mark}', flush=True)

    net, _ = M.load(f'{a.out}/model.pt')
    M.export_npz(net, f'{a.out}/weights.npz')
    preds = {name: fn(Xva, mva) for name, fn in baselines.fit_all(Xtr, Ytr, mtr).items()}
    mp, sigma, modes = evaluate.model_preds(net, np_va, Yva)
    preds.update(mp)
    lines = [f'логи: {config["logs"]}, проверка: {a.val_logs or "каждый пятый кусок по 60 с"}',
             f'примеров: обучение {len(Xtr)}, проверка {len(Xva)}; {a.arch}, сеть {a.layers}×{a.hidden}, вариантов {a.modes}, эпох {a.epochs}, '
             f'параметров {sum(p.numel() for p in net.parameters())}']

    def out(s):
        print(s)
        lines.append(s)
    evaluate.report(preds, Yva, mva, sigma, out, modes=modes)
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'\nвсего {time.time() - t0:.0f} с; модель в {a.out}/')


if __name__ == '__main__':
    main()
