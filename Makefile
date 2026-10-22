.PHONY: install test quick all

install:
	pip install -e ".[dev]"

test:
	pytest -q

# small subsample and 2 seeds, for a fast end-to-end check
quick:
	textaudit run --quick
	textaudit report

# full run on 20 Newsgroups (downloaded automatically by scikit-learn on first use)
all:
	textaudit run
	textaudit report
