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


def nll(mean, log_sigma, target):
    """Минус логарифм правдоподобия кругового гауссова распределения на плоскости (без константы)."""
    err2 = ((mean - target) ** 2).sum(-1)
    return (2 * log_sigma + err2 / (2 * torch.exp(2 * log_sigma))).mean()


def save(model, path, config):
    torch.save({'state': model.state_dict(), 'config': config}, path)


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = Predictor(ck['config']['n_features'], ck['config']['hidden'], ck['config']['layers'])
    model.load_state_dict(ck['state'])
    return model, ck['config']


def export_npz(model, path):
    """Веса простым набором массивов — чтобы посчитать сеть без torch (например, из C++): слой i — W{i}, b{i},
    между слоями SiLU, вход нормируется (x − x_mean) / x_std, выход: H×2 средних и H log sigma, умножить на y_scale."""
    arrays = {k: v.numpy() for k, v in model.state_dict().items() if k in ('x_mean', 'x_std', 'y_scale')}
    linears = [m for m in list(model.body) + [model.head] if isinstance(m, nn.Linear)]
    for j, lin in enumerate(linears):
        arrays[f'W{j}'] = lin.weight.detach().numpy()
        arrays[f'b{j}'] = lin.bias.detach().numpy()
    np.savez(path, horizons=np.array(HORIZONS), **arrays)
