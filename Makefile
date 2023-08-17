VENV = .venv
VENV_ACTIVATE = PATH=$(shell pwd)/$(VENV)/bin
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

install: requirements-dev.txt
	rm -rf $(VENV)
	python3.8 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache .coverage**