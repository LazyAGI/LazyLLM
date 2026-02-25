.PHONY: lint lint-flake8 lint-print install-flake8

install-flake8:
	@for pkg in flake8-quotes flake8-bugbear; do \
		case $$pkg in \
			flake8-quotes) mod="flake8_quotes" ;; \
			flake8-bugbear) mod="bugbear" ;; \
		esac; \
		python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$$mod') else 1)" \
			|| pip install $$pkg; \
	done

lint-flake8:
	python -m flake8

.ONESHELL:
lint-flake8-only-diff:
	@set -e
	@if [ -n "${CHANGED_FILES}" ]; then \
		EXISTING=$$(for f in $(CHANGED_FILES); do [ -f "$$f" ] && echo "$$f"; done); \
		[ -n "$$EXISTING" ] && echo "$$EXISTING" | xargs flake8 || true; \
		exit 0; \
	fi

	@echo "🔍 Collecting changed Python files..."
	@FILES=$$( \
		{ \
			git diff --name-status origin/main..HEAD -- 'lazyllm/**.py'  'docs/**.py' 'scripts/**.py' 'tests/**.py' 'examples/**.py'; \
			git diff --cached --name-status -- 'lazyllm/**.py' 'docs/**.py' 'scripts/**.py' 'tests/**.py' 'examples/**.py'; \
			git diff --name-status -- 'lazyllm/**.py' 'docs/**.py' 'scripts/**.py' 'tests/**.py' 'examples/**.py'; \
		} | awk '$$1 ~ /^(A|M)$$/ {print $$2}' | sort -u \
	);  \
	EXISTING=$$(for f in $$FILES; do [ -f "$$f" ] && echo "$$f"; done); \
	if [ -n "$$EXISTING" ]; then \
		echo "➡️  Running flake8 on:"; \
		echo "$$EXISTING"; \
		echo "$$EXISTING" | xargs flake8; \
	else \
		echo "✅ No Python file changes to lint (or changed files no longer exist)."; \
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

lint: install-flake8 lint-flake8 lint-print
lint-only-diff: install-flake8 lint-flake8-only-diff lint-print
poetry-install:
	cp pyproject.toml pyproject.toml.backup; \
	python scripts/generate_toml_optional_deps.py; \
	poetry install; \
	mv pyproject.toml.backup pyproject.toml
poetry-lock:
	cp pyproject.toml pyproject.toml.backup; \
	python scripts/generate_toml_optional_deps.py; \
	poetry lock; \
	mv pyproject.toml.backup pyproject.toml
