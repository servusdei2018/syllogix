UV = uv

format:
	$(UV) run ruff format

lint:
	$(UV) run ruff check --fix

test:
	$(UV) run pytest -v ./tests
