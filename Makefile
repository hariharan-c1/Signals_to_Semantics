.PHONY: check test repository-check

check: repository-check test

repository-check:
	python scripts/maintenance/check_repository.py

test:
	python -m unittest discover -s tests -v
