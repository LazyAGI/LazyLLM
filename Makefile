.PHONY: lint lint-flake8 lint-print

lint-flake8:
	python -m flake8

lint-print:
	@matches=$$(grep -RIn --binary-files=without-match --include="*.py" --exclude-dir="__pycache__" --exclude="finetune.py" --exclude-dir="docs" 'print(' lazyllm \
		| grep -v '^\s*#' \
		| grep -v '# noqa print' || true); \
	if [ -n "$$matches" ]; then \
		count=$$(echo "$$matches" | wc -l); \
		echo "❌ Lint failed: found $$count print(...) statements in the codebase:"; \
		echo "$$matches"; \
		exit 1; \
	else \
		echo "✅ Lint passed: no print(...) statements found."; \
	fi

lint: lint-flake8 lint-print
