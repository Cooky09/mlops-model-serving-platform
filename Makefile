install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m ml.train

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn app.main:app --reload
