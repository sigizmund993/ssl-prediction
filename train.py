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
    ap.add_argument('--arch', choices=['mlp', 'goal', 'sets', 'tabm', 'ft'], default='mlp',
                    help='mlp — Predictor/MixturePredictor, goal — точка назначения и физика, sets — сеть по всем объектам, '
                         'tabm — ансамбль TabM-mini, ft — FT-Transformer по признакам')
    ap.add_argument('--emb', choices=['ple', 'periodic'], default=None, help='эмбеддинги чисел (model.NumEmbedding)')
    ap.add_argument('--emb-dim', type=int, default=16)
    ap.add_argument('--bins', type=int, default=16, help='отрезков у ple, частот у periodic')
    ap.add_argument('--k', type=int, default=16, help='членов ансамбля у tabm')
    ap.add_argument('--amp', action='store_true', help='bfloat16 на видеокарте (для ft)')
    ap.add_argument('--loss', choices=['nll', 'l2'], default='nll',
                    help='nll — правдоподобие; l2 — среднее по квадрату ошибки, sigma по nll отдельно (model.l2_nll)')
    ap.add_argument('--device', default='cpu', help='cpu или cuda: на cuda данные целиком лежат в памяти видеокарты')
    ap.add_argument('--no-keepers', action='store_true', help='не учить на вратарях (проверка та же)')
    ap.add_argument('--no-stop', action='store_true', help='учить только на кадрах «игра идёт» (проверка та же)')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    sets = a.arch == 'sets'
    X, Y, meta, *rest = features.build_all(a.logs, a.step, agents=sets)
    tr, va = features.split(meta, a.logs, a.val_logs)
    tr = features.restrict(meta, tr, keepers=not a.no_keepers, stop=not a.no_stop)
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
                  modes=a.modes, arch=a.arch, loss=a.loss, emb=a.emb, emb_dim=a.emb_dim, bins=a.bins, k=a.k)
    if sets:
        config.update(agent_dim=Atr.shape[2], agents=features.AGENT_NAMES, n_agents=Atr.shape[1])
    net = M.build(config)
    if sets:
        sample = Atr[np.random.choice(len(Atr), min(len(Atr), 200000), replace=False)].astype(np.float32).reshape(-1, Atr.shape[2])
        sample = sample[sample[:, -1] > 0]  # нормировка только по присутствующим объектам
        net.a_mean.copy_(torch.from_numpy(sample.mean(0)))
        net.a_std.copy_(torch.from_numpy(sample.std(0) + 1e-3))
    net.x_mean.copy_(torch.from_numpy(Xtr.mean(0)))
    # постоянный в обучении признак (флаг «вратарь» при --no-keepers) не растягиваем: иначе на проверке он ×1e6
    std = Xtr.std(0)
    net.x_std.copy_(torch.from_numpy(np.where(std > 1e-3, std, 1.0).astype(np.float32)))
    if a.emb == 'ple':
        # границы отрезков — квантили нормированных признаков по обучающей части
        sample = Xtr[np.random.choice(len(Xtr), min(len(Xtr), 500000), replace=False)]
        sample = (sample - net.x_mean.numpy()) / net.x_std.numpy()
        net.emb.edges.copy_(torch.from_numpy(np.quantile(sample, np.linspace(0, 1, a.bins + 1), axis=0).T.astype(np.float32)))
    # масштаб ответа: среднеквадратичное отклонение от «постоянной скорости» на каждом горизонте
    net.y_scale.copy_(torch.from_numpy(np.sqrt((Ytr ** 2).sum(-1).mean(0)).astype(np.float32)))
    dev = torch.device(a.device)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev) / net.y_scale[None, :, None]
    xv, yv = torch.from_numpy(Xva).to(dev), torch.from_numpy(Yva).to(dev) / net.y_scale[None, :, None]
    inp_tr = (xt, torch.from_numpy(Atr).to(dev)) if sets else (xt,)
    inp_va = (xv, torch.from_numpy(Ava).to(dev)) if sets else (xv,)
    np_va = (Xva, Ava) if sets else Xva
    best = float('inf')
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        total = torch.zeros((), device=dev)  # .item() на каждом шаге тормозит видеокарту
        for j in range(len(xt) // a.batch):
            idx = perm[j * a.batch:(j + 1) * a.batch]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.amp):
                loss = M.loss(net, tuple(t[idx] for t in inp_tr), yt[idx], a.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach()
        net.eval()
        chunk = getattr(net, 'PREDICT_BATCH', 65536)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.amp):
            val = float(np.mean([M.loss(net, tuple(t[j:j + chunk] for t in inp_va), yv[j:j + chunk], a.loss).item()
                                 for j in range(0, len(xv), chunk)]))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.amp):
            pred = evaluate.model_preds(net, np_va, Yva)[0]['модель']  # у сети с вариантами — самый вероятный
        err = np.linalg.norm(pred - Yva, axis=2).mean(0) * 1000
        mark = ''
        # сохраняется лучшая по проверочной части эпоха: при nll — по потерям, при l2 — по средней ошибке
        # (в потерях l2 есть ещё nll для sigma, она растёт от излишней уверенности и сбивала бы выбор)
        crit = val if a.loss == 'nll' else float(err.mean())
        if crit < best:
            best = crit
            M.save(net, f'{a.out}/model.pt', config)
            mark = ' *'
        print(f'эпоха {epoch + 1:3d}: обучение {total.item() / (len(xt) // a.batch):.4f}, проверка {val:.4f}, '
              f'средняя ошибка через 0.5 с {err[1]:.0f} мм, через 1 с {err[3]:.0f} мм{mark}', flush=True)

    net, _ = M.load(f'{a.out}/model.pt')
    M.export_npz(net, f'{a.out}/weights.npz')
    preds = {name: fn(Xva, mva) for name, fn in baselines.fit_all(Xtr, Ytr, mtr).items()}
    mp, sigma, modes = evaluate.model_preds(net, np_va, Yva)
    preds.update(mp)
    lines = [f'логи: {config["logs"]}, проверка: {a.val_logs or "каждый пятый кусок по 60 с"}',
             f'примеров: обучение {len(Xtr)}, проверка {len(Xva)}; {a.arch}, сеть {a.layers}×{a.hidden}, вариантов {a.modes}, эпох {a.epochs}, '
             f'потери {a.loss}, эмбеддинги {a.emb or "нет"}, '
             f'параметров {sum(p.numel() for p in net.parameters())}']

    def out(s):
        print(s)
        lines.append(s)
    evaluate.report(preds, Yva, mva, sigma, out, modes=modes)
    open(f'{a.out}/report.txt', 'w').write('\n'.join(lines) + '\n')
    print(f'\nвсего {time.time() - t0:.0f} с; модель в {a.out}/')


if __name__ == '__main__':
    main()
