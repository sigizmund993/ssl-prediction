"""Модель прогноза: полносвязная сеть. Выход — среднее отклонение от «постоянной скорости» на каждом горизонте
(H × 2) и логарифм разброса sigma (H): ошибка считается круговым гауссовым распределением со своим sigma."""
import numpy as np
import torch
from torch import nn
from features import HORIZONS


class NumEmbedding(nn.Module):
    """Эмбеддинг чисел (Gorishniy и др., 2022): каждый нормированный признак → вектор длины d, своя маленькая сеть
    на признак. Так полносвязная сеть получает «пороги», как у деревьев.
    ple — кусочно-линейное кодирование по квантилям обучающей части (bins отрезков, границы — буфер edges, задаются
    в train.py), затем линейный слой на признак и ReLU. periodic — cos/sin от k частот на признак (частоты учатся,
    начальный разброс sigma), общий линейный слой и ReLU."""

    def __init__(self, n_features, kind, d=16, bins=16, sigma=1.0):
        super().__init__()
        self.kind = kind
        if kind == 'ple':
            self.register_buffer('edges', torch.linspace(-2, 2, bins + 1).repeat(n_features, 1))
            self.w = nn.Parameter(torch.randn(n_features, bins, d) / np.sqrt(bins))
            self.b = nn.Parameter(torch.zeros(n_features, d))
        elif kind == 'periodic':
            self.c = nn.Parameter(torch.randn(n_features, bins) * sigma)
            self.lin = nn.Linear(2 * bins, d)
        else:
            raise ValueError(kind)

    def forward(self, xn):
        """(B, F) нормированных признаков → (B, F, d)."""
        if self.kind == 'periodic':
            v = 2 * np.pi * self.c * xn[..., None]
            return torch.relu(self.lin(torch.cat([torch.cos(v), torch.sin(v)], -1)))
        lo, hi = self.edges[:, :-1], self.edges[:, 1:]
        width = hi - lo
        e = (xn[..., None] - lo) / width.clamp(min=1e-6)
        # крайние отрезки не обрезаются снаружи: значения за пределами обучающих продолжают линию
        e = torch.cat([e[..., :1].clamp(max=1), e[..., 1:-1].clamp(0, 1), e[..., -1:].clamp(min=0)], -1)
        e = e * (width > 1e-6)  # у дискретных признаков часть квантилей совпадает — такие отрезки пустые
        return torch.relu(torch.einsum('bft,ftd->bfd', e, self.w) + self.b)


class Predictor(nn.Module):
    PREDICT_BATCH = 65536

    def __init__(self, n_features, hidden=128, layers=3, emb=None, emb_dim=16, bins=16):
        super().__init__()
        self.n_h = len(HORIZONS)
        # emb — эмбеддинги чисел (NumEmbedding), идут в сеть вместе с самими нормированными признаками
        self.emb = NumEmbedding(n_features, emb, emb_dim, bins) if emb else None
        blocks, width = [], n_features * (1 + emb_dim if emb else 1)
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.n_h * 3)
        # нормировка входа и масштаб ответа по горизонтам; считаются по обучающей части и хранятся вместе с весами
        self.register_buffer('x_mean', torch.zeros(n_features))
        self.register_buffer('x_std', torch.ones(n_features))
        self.register_buffer('y_scale', torch.ones(self.n_h))

    def encode(self, x):
        """Вход сети: нормированные признаки и, если есть, их эмбеддинги."""
        xn = (x - self.x_mean) / self.x_std
        return xn if self.emb is None else torch.cat([xn, self.emb(xn).flatten(1)], 1)

    def forward(self, x):
        """Среднее и log sigma в нормированных единицах (ответ, делённый на y_scale)."""
        out = self.head(self.body(self.encode(x)))
        mean = out[:, :self.n_h * 2].reshape(-1, self.n_h, 2)
        log_sigma = out[:, self.n_h * 2:].clamp(-6, 4)
        return mean, log_sigma

    @torch.no_grad()
    def predict(self, X, batch=None):
        """numpy → (отклонение (M, H, 2), sigma (M, H)), метры."""
        self.eval()
        batch = batch or self.PREDICT_BATCH
        means, sigmas = [], []
        for j in range(0, len(X), batch):
            mean, log_sigma = self.combine(*self(torch.from_numpy(X[j:j + batch]).to(self.y_scale.device)))
            means.append((mean * self.y_scale[None, :, None]).float().cpu().numpy())
            sigmas.append((log_sigma.exp() * self.y_scale[None, :]).float().cpu().numpy())
        return np.concatenate(means), np.concatenate(sigmas)

    def combine(self, mean, log_sigma):
        """Выход forward → один прогноз; у ансамбля (TabMPredictor) — среднее по членам."""
        return mean, log_sigma


class MixturePredictor(Predictor):
    """Несколько вариантов будущего (modes): у каждого своя траектория (H × 2), свой sigma на каждом горизонте и вес.
    Вариант — траектория целиком (робот либо тормозит, либо едет дальше), поэтому вес один на все горизонты."""

    def __init__(self, n_features, hidden=128, layers=3, modes=3):
        super().__init__(n_features, hidden, layers)
        self.modes = modes
        self.head = nn.Linear(self.head.in_features, modes * (self.n_h * 3 + 1))

    def forward(self, x):
        """Средние (B, K, H, 2) и log sigma (B, K, H) в нормированных единицах, логиты весов (B, K)."""
        out = self.head(self.body(self.encode(x)))
        K, H = self.modes, self.n_h
        mean = out[:, :K * H * 2].reshape(-1, K, H, 2)
        log_sigma = out[:, K * H * 2:K * H * 3].reshape(-1, K, H).clamp(-6, 4)
        return mean, log_sigma, out[:, K * H * 3:]

    @torch.no_grad()
    def predict(self, X, batch=65536):
        """numpy → (отклонения (M, K, H, 2), sigma (M, K, H), метры; веса вариантов (M, K))."""
        self.eval()
        means, sigmas, weights = [], [], []
        for j in range(0, len(X), batch):
            mean, log_sigma, logits = self(torch.from_numpy(X[j:j + batch]).to(self.y_scale.device))
            means.append((mean * self.y_scale[None, None, :, None]).cpu().numpy())
            sigmas.append((log_sigma.exp() * self.y_scale[None, None, :]).cpu().numpy())
            weights.append(torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(means), np.concatenate(sigmas), np.concatenate(weights)


def rollout(v0, goal, acc, vmax, dt=1 / 60):
    """Робот едет к точке goal: желаемая скорость — к цели, не больше vmax и такая, чтобы успеть затормозить с
    ускорением acc (sqrt(2·acc·d)); изменение скорости за шаг не больше acc·dt. Всё в системе робота, м и м/с, старт
    из нуля со скоростью v0. Возвращает отклонение от «постоянной скорости» на горизонтах HORIZONS (B, H, 2)."""
    marks = {int(round(h / dt)): j for j, h in enumerate(HORIZONS)}
    p, v = torch.zeros_like(v0), v0
    out = [None] * len(HORIZONS)
    acc, vmax = acc[:, None], vmax[:, None]
    for step in range(1, max(marks) + 1):
        d = goal - p
        dist = torch.sqrt((d * d).sum(1, keepdim=True) + 1e-8)
        speed = torch.minimum(torch.minimum(vmax, torch.sqrt(2 * acc * dist + 1e-8)), dist * 20)
        dv = d / dist * speed - v
        norm = torch.sqrt((dv * dv).sum(1, keepdim=True) + 1e-8)
        v = v + dv * torch.clamp(acc * dt / norm, max=1.0)
        p = p + v * dt
        if step in marks:
            out[marks[step]] = p - v0 * (step * dt)
    return torch.stack(out, 1)


class GoalPredictor(Predictor):
    """Точка назначения и физика вместо свободной траектории: сеть выдаёт, куда едет робот (система робота, м), его
    ускорение (0.3–6 м/с²) и предельную скорость (0.3–4 м/с); траектория — rollout. Так прогноз всегда выполним
    роботом. sigma — как у Predictor. Признаки v_x, v_y (первые два) — текущая скорость, старт регулятора."""

    def __init__(self, n_features, hidden=128, layers=3):
        super().__init__(n_features, hidden, layers)
        self.head = nn.Linear(self.head.in_features, 4 + self.n_h)

    def physics(self, x):
        out = self.head(self.body(self.encode(x)))
        goal = out[:, :2] * 2.0
        acc = 0.3 + 5.7 * torch.sigmoid(out[:, 2])
        vmax = 0.3 + 3.7 * torch.sigmoid(out[:, 3])
        return goal, acc, vmax, out[:, 4:].clamp(-6, 4)

    def forward(self, x):
        goal, acc, vmax, log_sigma = self.physics(x)
        return rollout(x[:, :2], goal, acc, vmax) / self.y_scale[None, :, None], log_sigma


class TabMPredictor(Predictor):
    """TabM-mini (Gorishniy и др., 2024): k членов ансамбля в одной сети. Общее тело, у каждого члена свой множитель
    на входе (r, ±1 в начале) и своя голова; учатся все сразу на одних батчах, прогноз — среднее по членам."""

    def __init__(self, n_features, hidden=128, layers=3, emb=None, emb_dim=16, bins=16, k=16):
        super().__init__(n_features, hidden, layers, emb, emb_dim, bins)
        self.k = k
        width_in, width = self.body[0].in_features, self.head.in_features
        self.r = nn.Parameter(torch.randint(0, 2, (k, width_in)).float() * 2 - 1)
        self.head_w = nn.Parameter(torch.randn(k, width, self.n_h * 3) / np.sqrt(width))
        self.head_b = nn.Parameter(torch.zeros(k, self.n_h * 3))
        del self.head

    def forward(self, x):
        """Средние (B, k, H, 2) и log sigma (B, k, H) каждого члена."""
        h = self.body(self.encode(x)[:, None] * self.r)
        out = torch.einsum('bkw,kwo->bko', h, self.head_w) + self.head_b
        H = self.n_h
        return out[..., :H * 2].reshape(len(x), self.k, H, 2), out[..., H * 2:].clamp(-6, 4)

    def combine(self, mean, log_sigma):
        # разброс ансамбля: средняя дисперсия членов плюс разброс их средних (на одну координату)
        m = mean.mean(1)
        var = torch.exp(2 * log_sigma).mean(1) + ((mean - m[:, None]) ** 2).sum(-1).mean(1) / 2
        return m, 0.5 * torch.log(var)


class FTPredictor(Predictor):
    """FT-Transformer (Gorishniy и др., 2021): каждый признак — токен (x·w + b, свои w и b у признака), плюс токен
    CLS; трансформер внимания между признаками (не между роботами), выход — по CLS. hidden — ширина токена."""
    PREDICT_BATCH = 8192

    def __init__(self, n_features, hidden=64, layers=3, heads=8):
        super().__init__(n_features, hidden, 0)
        self.tok_w = nn.Parameter(torch.randn(n_features, hidden) / np.sqrt(hidden))
        self.tok_b = nn.Parameter(torch.zeros(n_features, hidden))
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) / np.sqrt(hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, self.n_h * 3))

    def forward(self, x):
        xn = (x - self.x_mean) / self.x_std
        t = xn[..., None] * self.tok_w + self.tok_b
        out = self.head(self.tr(torch.cat([self.cls.expand(len(x), -1, -1), t], 1))[:, 0])
        return out[:, :self.n_h * 2].reshape(-1, self.n_h, 2), out[:, self.n_h * 2:].clamp(-6, 4)


class SetPredictor(nn.Module):
    """Сеть по множеству объектов (Deep Sets): каждый объект (мяч и роботы, features.AGENT_NAMES) проходит одну и ту
    же сеть enc, результаты складываются и берётся максимум по присутствующим, это идёт вместе с обычными признаками
    робота в тело 3×hidden. Порядок объектов не важен, видны все роботы на поле. Выход — как у Predictor."""

    def __init__(self, n_features, agent_dim, hidden=128, layers=3, enc=64):
        super().__init__()
        self.n_h = len(HORIZONS)
        self.enc = nn.Sequential(nn.Linear(agent_dim, enc), nn.SiLU(), nn.Linear(enc, enc), nn.SiLU(), nn.Linear(enc, enc))
        blocks, width = [], n_features + 2 * enc
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.n_h * 3)
        self.register_buffer('x_mean', torch.zeros(n_features))
        self.register_buffer('x_std', torch.ones(n_features))
        self.register_buffer('a_mean', torch.zeros(agent_dim))
        self.register_buffer('a_std', torch.ones(agent_dim))
        self.register_buffer('y_scale', torch.ones(self.n_h))

    def forward(self, x, a):
        a = a.float()
        mask = a[..., -1:]  # «есть»
        h = self.enc((a - self.a_mean) / self.a_std) * mask
        hmax = torch.where(mask > 0, h, torch.full_like(h, -1e4)).max(1).values
        hmax = torch.where(hmax > -1e3, hmax, torch.zeros_like(hmax))
        z = torch.cat([(x - self.x_mean) / self.x_std, h.sum(1), hmax], 1)
        out = self.head(self.body(z))
        return out[:, :self.n_h * 2].reshape(-1, self.n_h, 2), out[:, self.n_h * 2:].clamp(-6, 4)

    @torch.no_grad()
    def predict(self, X, A, batch=65536):
        self.eval()
        means, sigmas = [], []
        for j in range(0, len(X), batch):
            dev = self.y_scale.device
            mean, log_sigma = self(torch.from_numpy(X[j:j + batch]).to(dev), torch.from_numpy(A[j:j + batch]).to(dev))
            means.append((mean * self.y_scale[None, :, None]).cpu().numpy())
            sigmas.append((log_sigma.exp() * self.y_scale[None, :]).cpu().numpy())
        return np.concatenate(means), np.concatenate(sigmas)


def nll(mean, log_sigma, target):
    """Минус логарифм правдоподобия кругового гауссова распределения на плоскости (без константы)."""
    err2 = ((mean - target) ** 2).sum(-1)
    return (2 * log_sigma + err2 / (2 * torch.exp(2 * log_sigma))).mean()


def mixture_nll(mean, log_sigma, logits, target):
    """То же для смеси траекторий: −log Σ_k w_k Π_h N(y_h | μ_kh, σ_kh), делённое на число горизонтов, чтобы при
    одном варианте совпадать с nll."""
    err2 = ((mean - target[:, None]) ** 2).sum(-1)
    logp = -(2 * log_sigma + err2 / (2 * torch.exp(2 * log_sigma))).sum(-1)
    return -torch.logsumexp(torch.log_softmax(logits, -1) + logp, -1).mean() / mean.shape[2]


def l2_nll(mean, log_sigma, target):
    """Среднее учится по квадрату ошибки (все примеры с равным весом, как у бустинга), sigma — по nll при уже
    выученном среднем: градиент nll в среднее не идёт, иначе nll делит ошибку на sigma² и трудные примеры почти
    не учатся. Первое слагаемое — средний квадрат ошибки на горизонт в нормированных единицах."""
    l2 = ((mean - target) ** 2).sum(-1).mean()
    return l2 + nll(mean.detach(), log_sigma, target)


def loss(net, inputs, y, kind='nll'):
    """inputs — кортеж входов сети: (x,) или (x, объекты) у SetPredictor. kind: nll или l2 (см. l2_nll)."""
    out = [o.float() for o in net(*inputs)]  # при --amp выход в bfloat16, потери считаем в float32
    if isinstance(net, MixturePredictor):
        if kind != 'nll':
            raise ValueError('у сети с вариантами только потери nll')
        return mixture_nll(*out, y)
    if isinstance(net, TabMPredictor):
        y = y[:, None]  # каждый член ансамбля учится на тот же ответ
    return (l2_nll if kind == 'l2' else nll)(*out, y)


def build(c):
    """Сеть по настройкам из model.pt."""
    arch = c.get('arch', 'mlp')
    if arch == 'sets':
        return SetPredictor(c['n_features'], c['agent_dim'], c['hidden'], c['layers'], c.get('enc', 64))
    emb = dict(emb=c.get('emb'), emb_dim=c.get('emb_dim', 16), bins=c.get('bins', 16))
    if arch == 'goal':
        return GoalPredictor(c['n_features'], c['hidden'], c['layers'])
    if arch == 'tabm':
        return TabMPredictor(c['n_features'], c['hidden'], c['layers'], k=c.get('k', 16), **emb)
    if arch == 'ft':
        return FTPredictor(c['n_features'], c['hidden'], c['layers'])
    if c.get('modes', 1) > 1:
        return MixturePredictor(c['n_features'], c['hidden'], c['layers'], c['modes'])
    return Predictor(c['n_features'], c['hidden'], c['layers'], **emb)


def save(model, path, config):
    torch.save({'state': model.state_dict(), 'config': config}, path)


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c = ck['config']
    model = build(c)
    model.load_state_dict(ck['state'])
    return model, c


def export_npz(model, path):
    """Веса простым набором массивов — чтобы посчитать сеть без torch (например, из C++): слой i — W{i}, b{i},
    между слоями SiLU, вход нормируется (x − x_mean) / x_std, выход: H×2 средних и H log sigma, умножить на y_scale.
    При modes = K > 1 выход: K×H×2 средних, K×H log sigma, K логитов весов (softmax).
    GoalPredictor, SetPredictor, TabM, FT и сети с эмбеддингами чисел так не выгружаются."""
    if type(model) not in (Predictor, MixturePredictor) or model.emb is not None:
        return
    arrays = {k: v.numpy() for k, v in model.state_dict().items() if k in ('x_mean', 'x_std', 'y_scale')}
    linears = [m for m in list(model.body) + [model.head] if isinstance(m, nn.Linear)]
    for j, lin in enumerate(linears):
        arrays[f'W{j}'] = lin.weight.detach().numpy()
        arrays[f'b{j}'] = lin.bias.detach().numpy()
    np.savez(path, horizons=np.array(HORIZONS), modes=np.array(getattr(model, 'modes', 1)), **arrays)
