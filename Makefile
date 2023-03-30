# TODO: create a build_local endpoint that runs setup.sh

{PACKAGE_DIR}=niftylittlepenguin

build_container:
	docker build -t ${PACKAGE_DIR} .

run_container:
	docker run -it --rm ${PACKAGE_DIR}

test:
	poetry run pytest -s tests/

format:
	poetry run black ${PACKAGE_DIR} tests
	poetry run isort ${PACKAGE_DIR} tests

checkformat:
	poetry run black --check --diff ${PACKAGE_DIR} tests
	poetry run isort --check-only --diff ${PACKAGE_DIR} tests

lint:
	poetry run pylint ${PACKAGE_DIR} tests

requirements:
	poetry export --format=requirements.txt > requirements.txt

tensorboard:
	tensorboard --logdir lightning_logs