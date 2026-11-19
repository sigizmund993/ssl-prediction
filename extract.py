# Лог игры SSL (.log.gz, формат SSL_LOG_FILE) → numpy. Аргументы: лог, папка с python-модулями proto, выходной .npz.
# Берутся кадры трекера одной команды (TRACKER, по умолчанию TIGERs: в логах RoboCup пишут несколько трекеров),
# команды судьи и размеры поля из геометрии vision.
#
# Что в файле (N кадров трекера):
#   t         (N,)        время кадра, с
#   cmd       (N,)        команда судьи (Referee.Command), -1 до первого сообщения
#   ball      (N, 4)      мяч x y vx vy, мм и мм/с (NaN, если мяча нет)
#   robots    (N, 32, 6)  роботы x y vx vy theta omega: 0–15 жёлтые, 16–31 синие; мм, мм/с, рад, рад/с (NaN — нет)
#   blue_pos  (N,)        1 — синие защищают ворота на +x, 0 — на −x, -1 — неизвестно
#   keeper    (N, 2)      номера вратарей (жёлтый, синий)
#   field     (5,)        длина и ширина поля, ширина ворот, глубина и ширина штрафной, мм (NaN — геометрии в логе нет)
#   names     (2,)        названия команд (жёлтые, синие)
import gzip, os, struct, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from tracker import ssl_vision_wrapper_tracked_pb2 as T
from state import ssl_gc_referee_message_pb2 as R
from vision import ssl_vision_wrapper_pb2 as V

from google.protobuf.message import DecodeError

MESSAGES = {3: R.Referee, 4: V.SSL_WrapperPacket, 5: T.TrackerWrapperPacket}


def parse(typ, data):
    if typ not in MESSAGES or (typ == 4 and not np.isnan(field[0])):
        return None
    m = MESSAGES[typ]()
    m.ParseFromString(data)
    return m


TRACKER = os.environ.get('TRACKER', 'TIGERs')
f = gzip.open(sys.argv[1], 'rb')
f.read(16)  # заголовок файла
cmd, blue_pos, keeper, names = -1, -1, (0, 0), ('', '')
field = np.full(5, np.nan)
times, cmds, balls, robots, sides, keepers = [], [], [], [], [], []
sources = {}
broken = 0  # битые сообщения (в логах RoboCup изредка попадаются): пропускаются
while True:
    h = f.read(16)
    if len(h) < 16:
        break
    ts, typ, size = struct.unpack('>qii', h)
    data = f.read(size)
    try:
        parsed = parse(typ, data)
    except DecodeError:
        broken += 1
        continue
    if typ == 3:  # судья
        m = parsed
        cmd = m.command
        names = (m.yellow.name, m.blue.name)
        keeper = (m.yellow.goalkeeper, m.blue.goalkeeper)
        if m.HasField('blue_team_on_positive_half'):
            blue_pos = int(m.blue_team_on_positive_half)
    elif typ == 4 and np.isnan(field[0]):  # vision: нужна только геометрия, один раз
        m = parsed
        if m.HasField('geometry'):
            g = m.geometry.field
            field = np.array([g.field_length, g.field_width, g.goal_width,
                              g.penalty_area_depth if g.HasField('penalty_area_depth') else np.nan,
                              g.penalty_area_width if g.HasField('penalty_area_width') else np.nan], dtype=float)
    elif typ == 5:  # трекер
        m = parsed
        sources[m.source_name] = sources.get(m.source_name, 0) + 1
        if m.source_name != TRACKER:
            continue
        fr = m.tracked_frame
        if fr.timestamp <= 0:  # пустые кадры без времени (бывают в логах RoboCup) ломают сетку 60 Гц
            continue
        times.append(fr.timestamp)
        cmds.append(cmd)
        sides.append(blue_pos)
        keepers.append(keeper)
        b = fr.balls[0] if len(fr.balls) else None
        balls.append([b.pos.x * 1000, b.pos.y * 1000, b.vel.x * 1000, b.vel.y * 1000] if b else [np.nan] * 4)
        rr = np.full((32, 6), np.nan)
        for r in fr.robots:
            if r.robot_id.id < 16:
                k = r.robot_id.id + (16 if r.robot_id.team == 2 else 0)
                rr[k] = [r.pos.x * 1000, r.pos.y * 1000, r.vel.x * 1000, r.vel.y * 1000, r.orientation, r.vel_angular]
        robots.append(rr)
if not times:
    sys.exit(f'в логе нет кадров трекера {TRACKER}; есть: {sources}')
np.savez_compressed(sys.argv[3], t=np.array(times), cmd=np.array(cmds), ball=np.array(balls), robots=np.array(robots, dtype=np.float32),
                    blue_pos=np.array(sides), keeper=np.array(keepers), field=field, names=np.array(names))
print(f'кадров {len(times)}, команды (жёлтые, синие): {names}, поле {field}, трекеры в логе: {sources}'
      + (f', пропущено битых сообщений: {broken}' if broken else ''))
