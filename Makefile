.PHONY: setup data train analyze figures run report test lint format notebooks clean

setup:
	uv sync --locked

data:
	uv run mnist-study data

train: data
	uv run mnist-study train

analyze:
	uv run mnist-study analyze

figures:
	uv run mnist-study figures

run: train analyze figures

report:
	uv run mnist-study report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# JUPYTER_PATH pins the project's own kernel, even if a user-level "python3" kernel exists.
notebooks:
	JUPYTER_PATH=.venv/share/jupyter uv run --group notebooks jupyter nbconvert \
		--to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/*.ipynb

clean:
	rm -rf data/interim .pytest_cache .ruff_cache
