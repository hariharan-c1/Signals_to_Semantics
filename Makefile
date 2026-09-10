.PHONY: check demo demo-synthetic figures repository-check test

check: repository-check test demo demo-synthetic
	python scripts/maintenance/generate_readme_figures.py --check

demo:
	python -m signals_to_semantics demo

demo-synthetic:
	python -m signals_to_semantics demo-synthetic

figures:
	python scripts/maintenance/generate_readme_figures.py

repository-check:
	python scripts/maintenance/check_repository.py

test:
	python -m unittest discover -s tests -v
