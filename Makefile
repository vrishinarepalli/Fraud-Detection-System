.PHONY: install train serve test lint format

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/
