.PHONY: init
init:
	pip install pre-commit
	pre-commit install
	poetry install

.PHONY: test
test:
	poetry run pytest --cov=. --junitxml=pytest.xml --cov-report=term-missing services/
