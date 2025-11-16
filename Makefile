UV = uv

format:
	$(UV) run ruff format

lint:
	$(UV) run ruff check --select I --fix

test:
	$(UV) run pytest -v ./tests
