.PHONY: setup run test app clean
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
run:
	PYTHONPATH=src .venv/bin/python -m var_engine.cli
test:
	PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
app:
	.venv/bin/streamlit run app/streamlit_app.py
clean:
	rm -rf .venv .pytest_cache **/__pycache__
