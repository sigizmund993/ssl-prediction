"""Просмотр лога с прогнозами моделей: поле, роботы, мяч и куда каждая модель ведёт каждого робота на 0.25–1 с.

    venv/bin/python viewer.py [--port 8765]   →   http://127.0.0.1:8765

Игры — data/*.npz, модели — все папки runs/ с model.pt (сети) или 8 файлами деревьев h*_x/y (boost.py: xgb — .ubj или .json, cat, lgb),
плюс «постоянная скорость». Прогноз считается один раз на всю игру (features.build(all_frames=True), каждый кадр 60 Гц,
на CPU — десятки секунд) и хранится в памяти; страница (viewer.html) получает его кусками по ~10 с.
Модели с другими признаками, чем сейчас в features.py, в списке есть, но выбрать их нельзя.
"""
import argparse, ast, glob, json, os, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import torch
import features
import model as M
from features import HORIZONS, DT

R90 = np.sqrt(2 * np.log(10))  # радиус круга «90 %» в единицах sigma
HERE = os.path.dirname(os.path.abspath(__file__))
COMMANDS = ['HALT', 'STOP', 'NORMAL_START', 'FORCE_START', 'PREPARE_KICKOFF_YELLOW', 'PREPARE_KICKOFF_BLUE',
            'PREPARE_PENALTY_YELLOW', 'PREPARE_PENALTY_BLUE', 'DIRECT_FREE_YELLOW', 'DIRECT_FREE_BLUE',
            'INDIRECT_FREE_YELLOW', 'INDIRECT_FREE_BLUE', 'TIMEOUT_YELLOW', 'TIMEOUT_BLUE', 'GOAL_YELLOW', 'GOAL_BLUE',
            'BALL_PLACEMENT_YELLOW', 'BALL_PLACEMENT_BLUE']
TRAINED = set(features.RUNNING) | {features.STOP}  # команды, на которых учились модели


# ---------- модели ----------

def boost_files(d):
    """Деревья boost.py: 8 файлов h{гор}_{x|y}.{json|cbm|txt} → (движок, [[x, y] по горизонтам]) или None."""
    for ext, kind in (('ubj', 'xgb'), ('json', 'xgb'), ('cbm', 'cat'), ('txt', 'lgb')):
        files = [[f'{d}/h{h}_{ax}.{ext}' for ax in 'xy'] for h in HORIZONS]
        if all(os.path.exists(f) for pair in files for f in pair):
            return kind, files
    return None


def report_lines(d):
    try:
        return open(f'{d}/report.txt').read().splitlines()
    except OSError:
        return []


def feature_set(ref):
    """Имена признаков с подробным состоянием игры (features.REF_STATE) или без него."""
    old = features.REF_STATE
    features.REF_STATE = ref
    try:
        return features.feature_names()
    finally:
        features.REF_STATE = old


def model_ref(names):
    """С каким REF_STATE обучена модель (по именам признаков) или None, если признаки другие."""
    for ref in (False, True):
        if names == feature_set(ref):
            return ref
    return None


def boost_ref(d):
    """REF_STATE деревьев — по числу признаков первой модели."""
    kind, files = boost_files(d)
    n = None
    if kind == 'xgb':
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(files[0][0])
        n = b.num_features()
    elif kind == 'cat':
        import catboost as cb
        b = cb.CatBoostRegressor()
        b.load_model(files[0][0])
        n = len(b.feature_names_)
    else:
        import lightgbm as lgb
        n = lgb.Booster(model_file=files[0][0]).num_feature()
    for ref in (False, True):
        if n == len(feature_set(ref)):
            return ref
    return None


def list_models():
    out = [dict(id='cv', name='постоянная скорость', kind='cv', ok=True, val=None, about='без обучения')]
    for d in sorted(glob.glob(f'{HERE}/runs/*/')):
        d = d.rstrip('/')
        rid = os.path.basename(d)
        rep = report_lines(d)
        if os.path.exists(f'{d}/model.pt'):
            cfg = torch.load(f'{d}/model.pt', map_location='cpu', weights_only=False)['config']
            ref = model_ref(cfg.get('features'))
            ok = ref is not None and (cfg.get('arch') != 'sets' or cfg.get('agents') == features.AGENT_NAMES)
            about = f"{cfg.get('arch', 'mlp')}, {cfg.get('layers')}×{cfg.get('hidden')}, потери {cfg.get('loss', 'nll')}"
            if cfg.get('emb'):
                about += f", эмбеддинги {cfg['emb']}"
            if cfg.get('modes', 1) > 1:
                about += f", вариантов {cfg['modes']}"
            if ref:
                about += ', состояние игры'
            if not rep:
                about += ' — нет отчёта: ещё учится или остановлена (лучшая эпоха на сейчас)'

            out.append(dict(id=rid, name=rid, kind='net', ok=ok, val=[os.path.basename(v) for v in cfg.get('val_logs', [])],
                            about=about if ok else 'другие признаки — переобучить'))
        elif boost_files(d):
            kind, _ = boost_files(d)
            ref = boost_ref(d)
            val = None
            if rep and 'проверка: ' in rep[0]:
                try:
                    val = [os.path.basename(v) for v in ast.literal_eval(rep[0].split('проверка: ', 1)[1])]
                except (ValueError, SyntaxError):
                    pass
            about = {'xgb': 'XGBoost', 'cat': 'CatBoost', 'lgb': 'LightGBM'}[kind] + (', состояние игры' if ref else '')
            out.append(dict(id=rid, name=rid, kind=kind, ok=ref is not None, val=val,
                            about=about if ref is not None else 'другие признаки — переобучить'))
    return out


def predict_boost(d, X):
    """Отклонение от постоянной скорости (M, H, 2) в системе робота, метры."""
    kind, files = boost_files(d)
    pred = np.zeros((len(X), len(HORIZONS), 2), np.float32)
    for h, pair in enumerate(files):
        for ax, f in enumerate(pair):
            if kind == 'xgb':
                import xgboost as xgb
                b = xgb.Booster()
                b.load_model(f)
                b.set_param({'nthread': os.cpu_count(), 'device': 'cpu'})
                pred[:, h, ax] = b.inplace_predict(X)
            elif kind == 'cat':
                import catboost as cb
                b = cb.CatBoostRegressor()
                b.load_model(f)
                pred[:, h, ax] = b.predict(X)
            else:
                import lightgbm as lgb
                pred[:, h, ax] = lgb.Booster(model_file=f).predict(X)
    return pred


# ---------- игра и прогнозы ----------

def to_world(v, ang, mirror):
    """Вектор(ы) системы робота (..., 2) → мира: обратное к features._rot."""
    c, s = np.cos(ang), np.sin(ang)
    shape = (-1,) + (1,) * (v.ndim - 2)
    c, s, mirror = c.reshape(shape), s.reshape(shape), mirror.reshape(shape)
    b = v[..., 1] * mirror
    return np.stack([c * v[..., 0] - s * b, s * v[..., 0] + c * b], axis=-1)


class Game:
    """Одна игра: сетка 60 Гц для рисования, признаки всех кадров и прогнозы по моделям (кэш)."""

    def __init__(self, name):
        self.name = name
        self.path = f'{HERE}/data/{name}'
        d = np.load(self.path)
        self.g = features.load_grid(self.path)
        field = d['field'] if 'field' in d and np.isfinite(d['field'][0]) else np.array(features.DEFAULT_FIELD)
        self.field = [float(x) for x in field]
        self.names = [str(x) for x in d['names']] if 'names' in d else ['', '']
        self.G = len(self.g['grid'])
        self.data = {}  # agents → (X, Y, meta, A)
        self.preds = {}
        R = self.g['robots']
        hor_n = [int(round(h / DT)) for h in HORIZONS]
        P = np.concatenate([R[:, :, :2], np.full((max(hor_n), 32, 2), np.nan)])
        self.truth = np.stack([P[n:n + self.G] for n in hor_n], axis=2)  # (G, 32, H, 2) настоящее будущее
        self.cv = R[:, :, None, :2] + R[:, :, None, 2:4] * (np.array(HORIZONS)[None, None, :, None])

    def dataset(self, agents=False, ref=False):
        key = (agents, ref)
        if key not in self.data:
            if len(self.data) >= 2:  # сотни мегабайт на набор: держим не больше двух
                self.data.pop(next(iter(self.data)))
            old = features.REF_STATE
            features.REF_STATE = ref
            try:
                out = features.build(self.path, 1, 0, agents=agents, all_frames=True)
            finally:
                features.REF_STATE = old
            self.data[key] = out if agents else (*out, None)
        return self.data[key]

    def prediction(self, mid):
        """{'pts': (G, 32, H, 2) мм мира, 'sigma': (G, 32, H) мм или None, 'modes': (G, 32, K, H, 2), 'w': (G, 32, K)}."""
        if mid in self.preds:
            return self.preds[mid]
        if mid == 'cv':
            self.preds[mid] = dict(pts=self.cv, sigma=None, modes=None, w=None)
            return self.preds[mid]
        d = f'{HERE}/runs/{mid}'
        sigma = modes = w = None
        if os.path.exists(f'{d}/model.pt'):
            net, cfg = M.load(f'{d}/model.pt')
            X, Y, meta, A = self.dataset(cfg.get('arch') == 'sets', bool(model_ref(cfg['features'])))
            torch.set_num_threads(os.cpu_count())
            with torch.no_grad():
                if isinstance(net, M.MixturePredictor):
                    means, sig, w = net.predict(X)
                    top = w.argmax(1)
                    rows = np.arange(len(X))
                    dev = (means * w[:, :, None, None]).sum(1)  # среднее по вариантам: оно точнее самого вероятного
                    sigma, modes = sig[rows, top], means
                else:
                    dev, sigma = net.predict(X, A) if A is not None else net.predict(X)
        else:
            X, Y, meta, A = self.dataset(False, bool(boost_ref(d)))
            dev = predict_boost(d, X)
        g, k = meta['frame'], meta['robot']
        ang, mir = meta['ang'], meta['mirror']
        base = self.cv[g, k]  # (M, H, 2) постоянная скорость, мм
        pts = np.full((self.G, 32, len(HORIZONS), 2), np.nan, np.float32)
        pts[g, k] = base + to_world(dev, ang, mir) * 1000
        out = dict(pts=pts, sigma=None, modes=None, w=None)
        if sigma is not None:
            out['sigma'] = np.full((self.G, 32, len(HORIZONS)), np.nan, np.float32)
            out['sigma'][g, k] = sigma * 1000
        if modes is not None:
            K = modes.shape[1]
            out['modes'] = np.full((self.G, 32, K, len(HORIZONS), 2), np.nan, np.float32)
            out['modes'][g, k] = base[:, None] + to_world(modes, ang, mir) * 1000
            out['w'] = np.full((self.G, 32, K), np.nan, np.float32)
            out['w'][g, k] = w
        self.preds[mid] = out
        return out

    def chunk(self, models, start, n):
        R, B, cmd = self.g['robots'], self.g['ball'], self.g['cmd']
        frames = []
        for gi in range(max(0, start), min(self.G, start + n)):
            present = np.where(np.isfinite(R[gi, :, 0]))[0]
            fr = dict(t=round(float(self.g['grid'][gi]), 3), cmd=int(cmd[gi]),
                      ball=[round(float(B[gi, 0])), round(float(B[gi, 1]))] if np.isfinite(B[gi, 0]) else None,
                      robots=[[int(k), round(float(R[gi, k, 0])), round(float(R[gi, k, 1])),
                               round(float(R[gi, k, 4]), 2) if np.isfinite(R[gi, k, 4]) else None] for k in present],
                      truth={}, pred={})
            for k in present:
                tr = self.truth[gi, k]
                if np.isfinite(tr).all():
                    fr['truth'][int(k)] = np.round(tr.ravel()).astype(int).tolist()
            for mid in models:
                p = self.prediction(mid)
                rows = {}
                for k in present:
                    pts = p['pts'][gi, k]
                    if not np.isfinite(pts).all():
                        continue
                    r = dict(p=np.round(pts.ravel()).astype(int).tolist())
                    if p['sigma'] is not None:
                        r['s'] = np.round(p['sigma'][gi, k] * R90).astype(int).tolist()  # сразу радиус «90 %»
                    if p['modes'] is not None:
                        r['m'] = [[round(float(p['w'][gi, k, j]), 2)] + np.round(p['modes'][gi, k, j].ravel()).astype(int).tolist()
                                  for j in np.argsort(-p['w'][gi, k]) if p['w'][gi, k, j] >= 0.05]
                    rows[int(k)] = r
                fr['pred'][mid] = rows
            frames.append(fr)
        return frames


# ---------- сервер ----------

games, lock = {}, threading.Lock()


def get_game(name):
    if name not in games:
        if len(games) >= 2:  # признаки игры — сотни мегабайт: держим не больше двух
            games.pop(next(iter(games)))
        games[name] = Game(name)
    return games[name]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, ctype='application/json', code=200):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                return self.send(open(f'{HERE}/viewer.html', 'rb').read(), 'text/html')
            if u.path == '/api/list':
                logs = sorted(os.path.basename(p) for p in glob.glob(f'{HERE}/data/*.npz'))
                return self.send(dict(logs=logs, models=list_models(), commands=COMMANDS, trained=sorted(TRAINED),
                                      horizons=list(HORIZONS)))
            if u.path == '/api/open':
                models = [m for m in q.get('models', '').split(',') if m]
                t0 = time.time()
                with lock:  # одна тяжёлая загрузка за раз
                    game = get_game(q['log'])
                    for mid in models:
                        game.prediction(mid)
                return self.send(dict(frames=game.G, field=game.field, names=game.names, seconds=round(time.time() - t0, 1)))
            if u.path == '/api/chunk':
                models = [m for m in q.get('models', '').split(',') if m]
                with lock:
                    game = get_game(q['log'])
                    for mid in models:
                        game.prediction(mid)
                    return self.send(game.chunk(models, int(q['start']), int(q.get('n', 600))))
            return self.send(dict(error='нет такого адреса'), code=404)
        except Exception as e:  # ошибку — на страницу, сервер живёт дальше
            import traceback
            traceback.print_exc()
            return self.send(dict(error=f'{type(e).__name__}: {e}'), code=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    print(f'http://127.0.0.1:{a.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
