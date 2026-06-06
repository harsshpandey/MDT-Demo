# MDT Wind Power Forecasting — Reproducible Pipeline
# Each step depends on the previous. Run sequentially or use `make all`.

PYTHON  ?= .venv/bin/python
PIP     ?= .venv/bin/pip
DATA    ?= data/raw/36565_23.03_72.56_2014_cc6ea7f2b4966ba2f914d889439754cc.csv
OUTDIR  ?= results

.PHONY: help venv install eda data data15 train train15 eval fuse compare serve test clean all

help:
	@echo "MDT Wind Power Forecasting — NREL India Site 36565 (Ahmedabad, 2014)"
	@echo "  make venv     — create .venv"
	@echo "  make install  — install pinned deps"
	@echo "  make eda      — write results/eda_report.md + eda_stats.json"
	@echo "  make data     — prepare HOURLY features (default)"
	@echo "  make data15   — prepare 15-MIN features (resampled from hourly)"
	@echo "  make train    — train 4 twins on hourly data → results/hourly/"
	@echo "  make train15  — train 4 twins on 15-min data → results/min15/"
	@echo "  make eval     — single-DT metrics matrix (uses current predictions/)"
	@echo "  make fuse     — run both MDT methods, write combination_results.csv"
	@echo "  make compare  — hourly vs 15-min side-by-side"
	@echo "  make serve    — Flask dashboard on :5001"
	@echo "  make test     — pytest suite (leakage + math + pipeline)"
	@echo "  make all      — eda → data → train → eval → fuse"

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -r requirements.txt

eda:
	$(PYTHON) eda.py --csv $(DATA)

data:
	$(PYTHON) data_pipeline.py --csv $(DATA) --out $(OUTDIR)/hourly

data15:
	$(PYTHON) data_pipeline.py --csv $(DATA) --out $(OUTDIR)/min15 --resample-15min

train:
	$(PYTHON) train.py --csv $(DATA) --out $(OUTDIR)/hourly

train15:
	$(PYTHON) train.py --csv $(DATA) --out $(OUTDIR)/min15 --resample-15min --predictions-dir predictions_min15

compare:
	$(PYTHON) compare.py

eval:
	$(PYTHON) evaluate.py

fuse:
	$(PYTHON) -c "from app import regenerate_combination_results; regenerate_combination_results()"

serve:
	$(PYTHON) app.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf $(OUTDIR) predictions __pycache__ */__pycache__ .pytest_cache

all: eda data train eval fuse
