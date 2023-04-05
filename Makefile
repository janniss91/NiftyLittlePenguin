# TODO: create a build_local endpoint that runs setup.sh

PACKAGE_DIR=niftylittlepenguin
TEST_DIR = tests
SERVICE_DIR = inference_service

build_container:
	docker build -t ${PACKAGE_DIR} .

run_container:
	docker run -p 8000:80 -it --rm ${PACKAGE_DIR}

test:
	poetry run pytest -s ${TEST_DIR}/

format:
	poetry run black ${PACKAGE_DIR} ${TEST_DIR} ${SERVICE_DIR}
	poetry run isort ${PACKAGE_DIR} ${TEST_DIR} ${SERVICE_DIR}

checkformat:
	poetry run black --check --diff ${PACKAGE_DIR} ${TEST_DIR} ${SERVICE_DIR}
	poetry run isort --check-only --diff ${PACKAGE_DIR} ${TEST_DIR} ${SERVICE_DIR}

lint:
	poetry run pylint ${PACKAGE_DIR} ${TEST_DIR} ${SERVICE_DIR}

requirements:
	poetry export --format=requirements.txt > requirements.txt

tensorboard:
	tensorboard --logdir lightning_logs