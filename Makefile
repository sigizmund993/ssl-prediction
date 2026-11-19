PY    = venv/bin/python
PROTO = ../ssl_packet_package/proto/ssl
RUN  ?= runs/all-66

.PHONY: init proto extract train eval clean

# окружение: venv, numpy, protobuf, torch (CPU)
init:
	python3 -m venv venv
	venv/bin/pip install numpy protobuf
	venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# python-модули proto из соседнего ssl_packet_package
proto:
	mkdir -p pb
	cd $(PROTO) && protoc -I. --python_out=$(CURDIR)/pb tracker/*.proto vision/*.proto \
		state/ssl_gc_referee_message.proto state/ssl_gc_common.proto state/ssl_gc_game_event.proto geom/ssl_gc_geometry.proto

# make extract LOG=путь/к/игре.log.gz  →  data/<имя>.npz
extract: proto
	mkdir -p data
	$(PY) -I extract.py "$(LOG)" pb data/$(basename $(basename $(notdir $(LOG)))).npz

# обучение на всех data/*.npz; проверочная часть — каждый пятый кусок времени (или целые игры: VAL="a.npz b.npz");
# логи идут первыми: --val-logs забирает все аргументы после себя. Готовую модель не перезаписывает: RUN=runs/новое
train:
	@test ! -e $(RUN)/model.pt || { echo "$(RUN)/model.pt уже есть — укажи другую папку: make train RUN=runs/новое"; exit 1; }
	$(PY) train.py data/*.npz --out $(RUN) $(if $(STEP),--step $(STEP)) $(if $(MODES),--modes $(MODES)) $(if $(VAL),--val-logs $(VAL))

# ошибки прогнозов без обучения и модели на проверочной части
eval:
	$(PY) evaluate.py data/*.npz --run $(RUN) $(if $(STEP),--step $(STEP)) $(if $(VAL),--val-logs $(VAL))

clean:
	rm -rf pb runs
