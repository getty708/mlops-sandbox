.PHONY: init
init:
	pip install pre-commit
	pre-commit install
	poetry install --without=gpu

.PHONY: init-gpu
init-gpu: init
	poetry install --with=dev,gpu

.PHONY: test
test:
	poetry run pytest  -m "not test_requires_data" --cov=. --junitxml=pytest.xml --cov-report=term-missing services/
