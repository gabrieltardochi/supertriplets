VENV = .venv
PIP = $(VENV)/bin/pip3

install: pyproject.toml
	rm -rf $(VENV)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install .[dev] --no-cache-dir
clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache *.egg-info .coverage** .ruff_cache build dist