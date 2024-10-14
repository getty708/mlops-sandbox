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

# =====================
#  Monitoring Services
# =====================
CMD_DOCKER_COMPOSE_MONITOR=docker-compose -f docker-compose-monitor.yaml

.PHONY: up-monitor
up-monitor:
	${CMD_DOCKER_COMPOSE_MONITOR} up -d

.PHONY: stop-monitor
stop-monitor:
	${CMD_DOCKER_COMPOSE_MONITOR} stop

.PHONY: down-monitor
down-monitor:
	${CMD_DOCKER_COMPOSE_MONITOR} down
