"""Модель прогноза: полносвязная сеть. Выход — среднее отклонение от «постоянной скорости» на каждом горизонте
(H × 2) и логарифм разброса sigma (H): ошибка считается круговым гауссовым распределением со своим sigma."""
import numpy as np
import torch
from torch import nn
from features import HORIZONS


class Predictor(nn.Module):
    def __init__(self, n_features, hidden=128, layers=3):
        super().__init__()
        self.n_h = len(HORIZONS)
        blocks, width = [], n_features
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.n_h * 3)
        # нормировка входа и масштаб ответа по горизонтам; считаются по обучающей части и хранятся вместе с весами
        self.register_buffer('x_mean', torch.zeros(n_features))
        self.register_buffer('x_std', torch.ones(n_features))
        self.register_buffer('y_scale', torch.ones(self.n_h))

    def forward(self, x):
        """Среднее и log sigma в нормированных единицах (ответ, делённый на y_scale)."""
        out = self.head(self.body((x - self.x_mean) / self.x_std))
        mean = out[:, :self.n_h * 2].reshape(-1, self.n_h, 2)
        log_sigma = out[:, self.n_h * 2:].clamp(-6, 4)
        return mean, log_sigma

    @torch.no_grad()
    def predict(self, X, batch=65536):
        """numpy → (отклонение (M, H, 2), sigma (M, H)), метры."""
        self.eval()
        means, sigmas = [], []
        for j in range(0, len(X), batch):
            mean, log_sigma = self(torch.from_numpy(X[j:j + batch]))
            means.append((mean * self.y_scale[None, :, None]).numpy())
            sigmas.append((log_sigma.exp() * self.y_scale[None, :]).numpy())
        return np.concatenate(means), np.concatenate(sigmas)


class MixturePredictor(Predictor):
    """Несколько вариантов будущего (modes): у каждого своя траектория (H × 2), свой sigma на каждом горизонте и вес.
    Вариант — траектория целиком (робот либо тормозит, либо едет дальше), поэтому вес один на все горизонты."""

    def __init__(self, n_features, hidden=128, layers=3, modes=3):
        super().__init__(n_features, hidden, layers)
        self.modes = modes
        self.head = nn.Linear(self.head.in_features, modes * (self.n_h * 3 + 1))

    def forward(self, x):
        """Средние (B, K, H, 2) и log sigma (B, K, H) в нормированных единицах, логиты весов (B, K)."""
        out = self.head(self.body((x - self.x_mean) / self.x_std))
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
            mean, log_sigma, logits = self(torch.from_numpy(X[j:j + batch]))
            means.append((mean * self.y_scale[None, None, :, None]).numpy())
            sigmas.append((log_sigma.exp() * self.y_scale[None, None, :]).numpy())
            weights.append(torch.softmax(logits, -1).numpy())
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
        out = self.head(self.body((x - self.x_mean) / self.x_std))
        goal = out[:, :2] * 2.0
        acc = 0.3 + 5.7 * torch.sigmoid(out[:, 2])
        vmax = 0.3 + 3.7 * torch.sigmoid(out[:, 3])
        return goal, acc, vmax, out[:, 4:].clamp(-6, 4)

    def forward(self, x):
        goal, acc, vmax, log_sigma = self.physics(x)
        return rollout(x[:, :2], goal, acc, vmax) / self.y_scale[None, :, None], log_sigma


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
            mean, log_sigma = self(torch.from_numpy(X[j:j + batch]), torch.from_numpy(A[j:j + batch]))
            means.append((mean * self.y_scale[None, :, None]).numpy())
            sigmas.append((log_sigma.exp() * self.y_scale[None, :]).numpy())
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


def loss(net, inputs, y):
    """inputs — кортеж входов сети: (x,) или (x, объекты) у SetPredictor."""
    return mixture_nll(*net(*inputs), y) if isinstance(net, MixturePredictor) else nll(*net(*inputs), y)


def build(c):
    """Сеть по настройкам из model.pt."""
    arch = c.get('arch', 'mlp')
    if arch == 'sets':
        return SetPredictor(c['n_features'], c['agent_dim'], c['hidden'], c['layers'], c.get('enc', 64))
    if arch == 'goal':
        return GoalPredictor(c['n_features'], c['hidden'], c['layers'])
    if c.get('modes', 1) > 1:
        return MixturePredictor(c['n_features'], c['hidden'], c['layers'], c['modes'])
    return Predictor(c['n_features'], c['hidden'], c['layers'])


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
    GoalPredictor и SetPredictor так не выгружаются: им нужен rollout и сеть объектов."""
    if type(model) not in (Predictor, MixturePredictor):
        return
    arrays = {k: v.numpy() for k, v in model.state_dict().items() if k in ('x_mean', 'x_std', 'y_scale')}
    linears = [m for m in list(model.body) + [model.head] if isinstance(m, nn.Linear)]
    for j, lin in enumerate(linears):
        arrays[f'W{j}'] = lin.weight.detach().numpy()
        arrays[f'b{j}'] = lin.bias.detach().numpy()
    np.savez(path, horizons=np.array(HORIZONS), modes=np.array(getattr(model, 'modes', 1)), **arrays)
