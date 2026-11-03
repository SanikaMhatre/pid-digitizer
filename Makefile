.PHONY: install test lint eval ui api smoke
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests
eval:
	python -m pid.eval.run_eval --mode simulated --n 14
ui:
	streamlit run src/pid/app/ui.py
api:
	uvicorn pid.app.api:app --port 8000
smoke:
	python -m pid.smoke all
