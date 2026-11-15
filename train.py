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
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    X, Y, meta = features.build_all(a.logs, a.step)
    tr, va = features.split(meta, a.logs, a.val_logs)
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    mtr = {k: v[tr] for k, v in meta.items()}
    mva = {k: v[va] for k, v in meta.items()}
    print(f'примеров: обучение {len(Xtr)}, проверка {len(Xva)}, признаков {X.shape[1]} (датасет за {time.time() - t0:.0f} с)')

    net = M.Predictor(X.shape[1], a.hidden, a.layers)
    net.x_mean.copy_(torch.from_numpy(Xtr.mean(0)))
    net.x_std.copy_(torch.from_numpy(Xtr.std(0) + 1e-6))
    # масштаб ответа: среднеквадратичное отклонение от «постоянной скорости» на каждом горизонте
    net.y_scale.copy_(torch.from_numpy(np.sqrt((Ytr ** 2).sum(-1).mean(0)).astype(np.float32)))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr) / net.y_scale[None, :, None]
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(Yva) / net.y_scale[None, :, None]

    config = dict(n_features=X.shape[1], hidden=a.hidden, layers=a.layers, horizons=features.HORIZONS, lags=features.LAGS,
                  features=features.feature_names(), logs=[os.path.basename(p) for p in a.logs], val_logs=a.val_logs, step=a.step)
    best = float('inf')
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for j in range(len(xt) // a.batch):
            idx = perm[j * a.batch:(j + 1) * a.batch]
            loss = M.nll(*net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        net.eval()
        with torch.no_grad():
            val = float(np.mean([M.nll(*net(xv[j:j + 65536]), yv[j:j + 65536]).item() for j in range(0, len(xv), 65536)]))
        pred, _ = net.predict(Xva)
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
    preds['модель'], sigma = net.predict(Xva)
    lines = [f'логи: {config["logs"]}, проверка: {a.val_logs or "каждый пятый кусок по 60 с"}',
             f'примеров: обучение {len(Xtr)}, проверка {len(Xva)}; сеть {a.layers}×{a.hidden}, эпох {a.epochs}, '
             f'параметров {sum(p.numel() for p in net.parameters())}']

    def out(s):
        print(s)
        lines.append(s)
    evaluate.report(preds, Yva, mva, sigma, out)
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'\nвсего {time.time() - t0:.0f} с; модель в {a.out}/')


if __name__ == '__main__':
    main()
