POETRY ?= poetry

.PHONY: setup test smoke lint train run run-enhanced backtest supervise docs

setup:
	POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) install --no-interaction --sync

test:
	PYTHONPATH=src POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python -m compileall src

lint:
	POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python -m py_compile src/auto_py_bot/*.py scripts/*.py

train:
	PYTHONPATH=src POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python scripts/train_model.py

run:
	PYTHONPATH=src POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python scripts/trading_bot.py

run-enhanced:
	PYTHONPATH=src POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python scripts/enhanced_bot.py

backtest:
	PYTHONPATH=src POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python scripts/backtest.py

supervise:
	PYTHONPATH=src POETRY_VIRTUALENVS_IN_PROJECT=true $(POETRY) run python scripts/run_24_7.py

docs:
	@echo 'Documentación disponible en docs/PROJECT_STRUCTURE.md y docs/ROADMAP_TASKS.md'
