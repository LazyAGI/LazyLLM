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

lint-flake8-only-diff:
	echo "$(CHANGED_FILES)" | xargs flake8; \

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
