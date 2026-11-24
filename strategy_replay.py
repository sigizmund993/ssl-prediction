"""Опыт: наша стратегия (../strategy) как прогноз чужих роботов. По кадрам лога стратегия играет за каждую из команд
по очереди и для каждого робота выдаёт действие; из действия берётся точка, куда она ведёт робота (target_pos у
GoToPointIgnore), и скорость, которую она бы ему дала. Без обучения: «робот делает то, что сделали бы мы».

    python3 strategy_replay.py data/игра.npz runs/strategy/игра.npz [--src runs/strategy-src] [--step 6]

Запускать системным python3 (зависимости стратегии: filterpy, scipy, osqp, pyzmq…), не venv/bin/python.
--src — копия репозитория стратегии с .env (DEBUG_MODE = False, DEBUG_HALF = 0): стратегия при импорте читает .env
из текущей папки и может переписать его по шаблону, поэтому рабочий ../strategy не трогаем.

В выходном npz: frames (K,) — номера кадров сетки 60 Гц (features.load_grid), target (K, 32, 2) — точка назначения,
cmd_vel (K, 32, 2) — скорость от стратегии, мм и мм/с, мировые координаты; NaN — робота нет или действие без точки;
state (K,) — состояние игры стратегии (имя const.State), errors — сколько кадров упало с исключением.
"""
import argparse, os, sys, time as _time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features  # noqa: E402

# команды судьи (Referee.Command)
HALT, STOP, NORMAL_START, FORCE_START = 0, 1, 2, 3
PREP_KICKOFF = {4: 'Y', 5: 'B'}
PREP_PENALTY = {6: 'Y', 7: 'B'}
DIRECT_FREE = {8: 'Y', 9: 'B'}
TIMEOUT = {12, 13}
BALL_PLACEMENT = {16: 'Y', 17: 'B'}


class Clock:
    """Время лога вместо time.time(): регуляторы и фильтры стратегии считают по нему dt."""
    now = 0.0

    def __call__(self):
        return self.now


def load_strategy(src):
    """Импорт стратегии из копии src; time() в её модулях подменяется на Clock."""
    os.chdir(src)
    sys.path[:0] = [src, os.path.join(src, 'strategy-bridge', 'src')]
    from bridge import const
    from bridge.auxiliary import aux, fld
    from bridge.strategy import strategy
    from bridge.router.actions.action import ActionDomain, ActionValues
    from bridge.router.actions.base_actions import Actions
    clock = Clock()
    for name, mod in list(sys.modules.items()):
        if name.startswith('bridge') and getattr(mod, 'time', None) is _time.time:
            mod.time = clock

    # точка назначения: последний GoToPointIgnore, через который прошло действие робота
    behavior = Actions.GoToPointIgnore.behavior

    def spy(self, domain, current_action):
        domain.robot.replay_target = self.target_pos
        # ограничение ускорения делит на время с прошлой команды; в повторе внутри кадра время стоит
        if domain.robot.prev_sended_time >= clock.now:
            domain.robot.prev_sended_time = clock.now - 0.1
        return behavior(self, domain, current_action)
    Actions.GoToPointIgnore.behavior = spy
    return dict(const=const, aux=aux, fld=fld, strategy=strategy, ActionDomain=ActionDomain, ActionValues=ActionValues,
                clock=clock)


class Referee:
    """Команды судьи → состояние игры стратегии и активная команда, как в её RefereeStateProcessor: после NORMAL_START
    и DIRECT_FREE игра идёт (RUN), когда мяч сдвинулся на 50 мм или прошло 10 с."""

    def __init__(self, S):
        self.S = S
        self.prev_cmd = None
        self.prepare = None  # ('kickoff' | 'penalty', команда)
        self.state, self.active = S['const'].State.HALT, S['const'].Color.ALL
        self.wait_start, self.wait_ball = None, None

    def color(self, c):
        return self.S['const'].Color.YELLOW if c == 'Y' else self.S['const'].Color.BLUE

    def update(self, cmd, t, ball):
        State, Color = self.S['const'].State, self.S['const'].Color
        if cmd != self.prev_cmd:
            self.prev_cmd = cmd
            self.wait_start = None
            if cmd == HALT:
                self.state, self.active = State.HALT, Color.ALL
            elif cmd == STOP or cmd in BALL_PLACEMENT:
                # расстановку мяча стратегии не отдаём (нет точки расстановки в логе): для остальных это STOP
                self.state, self.active = State.STOP, Color.ALL
            elif cmd in TIMEOUT:
                self.state, self.active = State.TIMEOUT, Color.ALL
            elif cmd == FORCE_START:
                self.state, self.active = State.RUN, Color.ALL
            elif cmd in PREP_KICKOFF:
                self.prepare = ('kickoff', PREP_KICKOFF[cmd])
                self.state, self.active = State.PREPARE_KICKOFF, self.color(PREP_KICKOFF[cmd])
            elif cmd in PREP_PENALTY:
                self.prepare = ('penalty', PREP_PENALTY[cmd])
                self.state, self.active = State.PREPARE_PENALTY, self.color(PREP_PENALTY[cmd])
            elif cmd == NORMAL_START:
                kind, team = self.prepare or ('kickoff', None)
                self.state = State.PENALTY if kind == 'penalty' else State.KICKOFF
                self.active = self.color(team) if team else Color.ALL
                self.wait_start, self.wait_ball = t, ball
            elif cmd in DIRECT_FREE:
                self.state, self.active = State.FREE_KICK, self.color(DIRECT_FREE[cmd])
                self.wait_start, self.wait_ball = t, ball
        if self.wait_start is not None and self.state in (State.KICKOFF, State.FREE_KICK):
            moved = ball is not None and self.wait_ball is not None and np.hypot(*(ball - self.wait_ball)) > 50
            if moved or t - self.wait_start > 10:
                self.state, self.active, self.wait_start = State.RUN, Color.ALL, None
        return self.state, self.active


class Team:
    """Стратегия, играющая за одну команду. Пересоздаётся, если сменились стороны поля или вратари."""

    def __init__(self, S, color, polarity, gk, enemy_gk):
        const = S['const']
        const.COLOR, const.POLARITY, const.GK, const.ENEMY_GK = color, polarity, gk, enemy_gk
        self.S, self.color, self.key = S, color, (polarity, gk, enemy_gk)
        self.field = S['fld'].Field(color)
        self.strategy = S['strategy'].Strategy()

    def step(self, t, R, ball, state, active):
        """R: (32, 6) робота кадра (x y vx vy theta omega, NaN — нет), ball: (4,) или None. → target, cmd_vel (32, 2)."""
        S, f, const, aux = self.S, self.field, self.S['const'], self.S['aux']
        const.COLOR, (const.POLARITY, const.GK, const.ENEMY_GK) = self.color, self.key
        t = float(t)
        S['clock'].now = t
        f.game_state, f.active_team = state, active
        if ball is not None:
            bx, by, bvx, bvy = (float(x) for x in ball)  # обычные float: np.float64 + poly1d в стратегии даёт массив
            f.ball._pos, f.ball._vel, f.ball.last_update_ = aux.Point(bx, by), aux.Point(bvx, bvy), t
            f.update_ball_history()
        for k, robot in enumerate(f.y_team + f.b_team):
            r = R[k]
            if not np.isfinite(r[0]):
                robot.used(0)
                continue
            robot._pos, robot._vel = aux.Point(float(r[0]), float(r[1])), aux.Point(float(r[2]), float(r[3]))
            robot._angle = float(r[4]) if np.isfinite(r[4]) else 0.0
            robot._anglevel = float(r[5]) if np.isfinite(r[5]) else 0.0
            robot.last_update_ = t
            if not robot.is_used():
                robot.used(1)
                robot.live_time_ = t
        f.update_active_allies([r for r in f.allies if r.is_used() and r.r_id != f.gk_id])
        f.update_active_enemies([r for r in f.enemies if r.is_used() and r.r_id != f.enemy_gk_id])
        holder = None
        for r in f.enemies + f.allies:  # как в FieldCreator: свои важнее
            if r.is_used() and f._is_ball_in(r):
                holder = r
        f.update_robot_with_ball(holder)
        f.last_update = t

        commands = self.strategy.process(f)
        we_active = active in (const.Color.ALL, self.color)
        target = np.full((32, 2), np.nan)
        vel = np.full((32, 2), np.nan)
        base = 0 if self.color == const.Color.YELLOW else 16
        for c in commands:
            robot = f.allies[c.r_id]
            if not (c.action.is_used and robot.is_used()):
                continue
            robot.replay_target = None
            values = S['ActionValues']()
            c.action.process(S['ActionDomain'](f, f.game_state, we_active, robot), values)
            vel[base + c.r_id] = values.vel.x, values.vel.y
            if robot.replay_target is not None:
                target[base + c.r_id] = robot.replay_target.x, robot.replay_target.y
        f.clear_images()
        return target, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('out')
    ap.add_argument('--src', default=os.path.join(HERE, 'runs', 'strategy-src'))
    ap.add_argument('--step', type=int, default=6, help='стратегия считается на каждом step-м кадре сетки 60 Гц')
    ap.add_argument('--limit', type=int, default=0, help='только первые N кадров (проверка скорости)')
    a = ap.parse_args()
    log, out = os.path.abspath(a.log), os.path.abspath(a.out)
    S = load_strategy(os.path.abspath(a.src))
    const = S['const']
    g = features.load_grid(log)
    R, B, cmd, ok, grid = g['robots'], g['ball'], g['cmd'], g['ok'], g['grid']
    lag = int(round(max(features.LAGS) / features.DT))
    frames = np.arange(lag, len(ok), a.step)  # те же кадры, что у примеров features.build с тем же step
    if a.limit:
        frames = frames[:a.limit]
    referee = Referee(S)
    teams = {}
    targets = np.full((len(frames), 32, 2), np.nan, np.float32)
    vels = np.full((len(frames), 32, 2), np.nan, np.float32)
    states, errors, t0 = [], 0, _time.time()
    for j, i in enumerate(frames):
        ball = B[i] if np.isfinite(B[i, 0]) else None
        state, active = referee.update(int(cmd[i]), grid[i], None if ball is None else ball[:2])
        states.append(state.name)
        if not ok[i] or g['blue_pos'][i] < 0 or state == const.State.HALT:
            continue
        # POLARITY стратегии: 1 — синие ворота на −x; blue_pos: 1 — синие защищают +x
        polarity = -1 if g['blue_pos'][i] == 1 else 1
        ky, kb = (int(x) for x in g['keeper'][i])
        for color, gk, enemy_gk in ((const.Color.YELLOW, ky, kb), (const.Color.BLUE, kb, ky)):
            key = (polarity, gk, enemy_gk)
            if color not in teams or teams[color].key != key:
                teams[color] = Team(S, color, *key)
            try:
                tg, vl = teams[color].step(grid[i], R[i], ball, state, active)
            except Exception:
                errors += 1
                if errors <= 3:
                    traceback.print_exc()
                teams.pop(color)  # после исключения состояние стратегии непредсказуемо
                continue
            sl = slice(0, 16) if color == const.Color.YELLOW else slice(16, 32)
            targets[j, sl], vels[j, sl] = tg[sl], vl[sl]
        if j % 2000 == 0:
            print(f'@@ кадр {j}/{len(frames)}, {_time.time() - t0:.0f} с, ошибок {errors}', flush=True)
    np.savez_compressed(out, frames=frames, target=targets, cmd_vel=vels, state=np.array(states), errors=errors)
    print(f'{os.path.basename(log)}: кадров {len(frames)}, ошибок {errors}, {_time.time() - t0:.0f} с, '
          f'с точкой назначения {np.isfinite(targets[..., 0]).mean() * 100:.0f} % (робот, кадр)')


if __name__ == '__main__':
    main()
