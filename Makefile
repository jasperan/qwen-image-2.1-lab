SHELL := /bin/bash
PY := .venv/bin/python

.PHONY: setup download gallery edit serve test test-gpu test-server env-report

setup:
	bash scripts/setup.sh

download:
	bash scripts/download_model.sh

gallery:
	$(PY) examples/generate_gallery.py --config configs/prompts.json

edit:
	$(PY) examples/edit_image.py \
		--image outputs/gallery/00-product-teapot.png \
		--prompt "Change the glaze to a deep crimson red, keep the same photo style" \
		--out outputs/edits/teapot-crimson.png

serve:
	bash scripts/serve.sh

test:
	$(PY) -m pytest -q

test-gpu:
	$(PY) -m pytest -m slow -v

test-server:
	$(PY) -m pytest -m server -v

env-report:
	$(PY) -m pytest tests/test_env.py -v
