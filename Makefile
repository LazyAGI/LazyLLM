.PHONY: lint lint-flake8 lint-print

lint-flake8:
	python -m flake8

lint-flake8-only-diff:
	@echo "🔍 Collecting changed Python files..."
	@FILES=$$( \
		{ \
			git diff --name-status origin/main..HEAD -- 'lazyllm/**.py'  'docs/**.py' 'scripts/**.py' 'tests/**.py' 'examples/**.py'; \
			git diff --cached --name-status -- 'lazyllm/**.py' 'docs/**.py' 'scripts/**.py' 'tests/**.py' 'examples/**.py'; \
			git diff --name-status -- 'lazyllm/**.py' 'docs/**.py' 'scripts/**.py' 'tests/**.py' 'examples/**.py'; \
		} | awk '$$1 ~ /^(A|M)$$/ {print $$2}' \
	);  \
	if [ -n "$$FILES" ]; then \
		echo "➡️  Running flake8 on:"; \
		echo "$$FILES"; \
		echo "$$FILES" | xargs flake8; \
	else \
		echo "✅ No Python file changes to lint."; \
	fi

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
lint-only-diff: lint-flake8-only-diff lint-print
