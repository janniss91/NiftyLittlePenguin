# TODO: create a build_local endpoint that runs setup.sh

build_container:
	docker build -t niftylittlepenguin .

run_container:
	docker run -it --rm niftylittlepenguin

test:
	poetry run pytest -s tests/

format:
	poetry run black niftylittlepenguin tests

requirements:
	poetry export --format=requirements.txt > requirements.txt