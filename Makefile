.PHONY: test smoke compile frontend-build up

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m cropfed.cli demo --rounds 5 --output artifacts/smoke-result.json

compile:
	python3 -m compileall -q src tests scripts

frontend-build:
	cd frontend && npm ci && npm run build

up:
	docker compose up --build
