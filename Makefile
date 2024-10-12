.DEFAULT_GOAL := default

.PHONY: test
test:
	poetry run pytest --cov=ccrestoration --cov-report=xml --cov-report=html

.PHONY: lint
lint:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

.PHONY: build
build:
	poetry build --format wheel

.PHONY: vs
vs:
	rm -f encoded.mkv
	vspipe -c y4m example/vapourSynth.py - | ffmpeg -i - -vcodec libx265 -crf 16 encoded.mkv
