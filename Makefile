PYTHON ?= python3

.PHONY: setup test smoke lint train run run-enhanced backtest supervise docs

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	PYTHONPATH=src $(PYTHON) -m compileall src

lint:
	$(PYTHON) -m py_compile src/auto_py_bot/*.py scripts/*.py

train:
	PYTHONPATH=src $(PYTHON) scripts/train_model.py

run:
	PYTHONPATH=src $(PYTHON) scripts/trading_bot.py

run-enhanced:
	PYTHONPATH=src $(PYTHON) scripts/enhanced_bot.py

backtest:
	PYTHONPATH=src $(PYTHON) scripts/backtest.py

supervise:
	PYTHONPATH=src $(PYTHON) scripts/run_24_7.py

docs:
	@echo 'Documentación disponible en docs/PROJECT_STRUCTURE.md y docs/ROADMAP_TASKS.md'
