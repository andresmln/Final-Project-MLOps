install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli 

format:	
	uv run black mylib/*.py cli/*.py api/*.py

lint:
	uv run pylint --rcfile=.pylintrc --ignore-patterns=test_.*\.py mylib/*.py cli/*.py api/*.py

refactor: format lint

all: install format lint test
