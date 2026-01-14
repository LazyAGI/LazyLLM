# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LazyLLM is a low-code development framework for building multi-agent LLM applications. The framework follows a "prototype building -> data feedback -> iterative optimization" development cycle, allowing rapid prototyping with deployment to production supporting multiple users, fault tolerance, and high concurrency.

## Core Architecture

LazyLLM is organized around three main abstractions:

1. **Components** (`lazyllm/components/`): The smallest execution units. Components can wrap functions or bash commands and execute across platforms via launchers. Key base classes:
   - `LazyLLMDataprocBase`: Data processing components
   - `LazyLLMFinetuneBase`: Fine-tuning components
   - `LazyLLMDeployBase`: Deployment components
   - `LazyLLMValidateBase`: Validation/evaluation components

2. **Modules** (`lazyllm/module/`): Top-level components with training, deployment, inference, and evaluation capabilities:
   - `TrainableModule`: Local models with fine-tuning support
   - `OnlineChatModule` / `OnlineEmbeddingModule` / `OnlineMultiModalModule`: Online model services
   - `ServerModule`: Wraps functions/flows as API services
   - `ActionModule`: Wraps functions, modules, or flows
   - `WebModule`: Gradio-based web interface
   - `UrlModule`: Wraps external URLs

3. **Flows** (`lazyllm/flow/`): Define data streams between components:
   - `pipeline`: Sequential execution
   - `parallel`: Parallel execution (with `.sum`, `.product` modes)
   - `diverter`: Route to different branches
   - `loop`: Iterative execution
   - `switch` / `ifs`: Conditional branching
   - `warp`: Data transformation
   - `graph`: Complex DAG workflows

4. **Launchers** (`lazyllm/launcher/`): Platform abstraction for execution:
   - `EmptyLauncher`: Local execution (default)
   - `SlurmLauncher`: Slurm cluster scheduling
   - `ScoLauncher`: SenseCore platform
   - `K8sLauncher`: Kubernetes deployment

5. **Tools** (`lazyllm/tools/`): High-level tools:
   - RAG: `Document`, `Retriever`, `Reranker`, `SentenceSplitter`
   - Agents: `FunctionCall`, `ReactAgent`, `PlanAndSolveAgent`, `ReWOOAgent`
   - MCP integration (Model Context Protocol)
   - SQL integration

## Development Commands

```bash
# Installation
pip install -r requirements.txt              # Basic dependencies
lazyllm install standard                     # Standard features
lazyllm install full                         # All features

# Linting
make lint                                    # Full lint (flake8 + print check)
make lint-only-diff                          # Lint only changed files

# Testing
pytest tests/basic_tests/                    # Basic unit tests
pytest tests/advanced_tests/                 # Advanced integration tests
pytest tests/charge_tests/                   # Online service tests (requires API keys)
pytest -m "not skip_on_linux"                # Skip platform-specific tests

# CLI
lazyllm run chatbot                          # Quick chatbot
lazyllm run rag --documents=/path/to/data    # Quick RAG
lazyllm install <extra> <pkg>...             # Install optional dependencies
lazyllm deploy modelname                     # Deploy a model
```

## Configuration

Configuration is managed via `lazyllm.configs.Config`. Options can be set:
1. Environment variables: `LAZYLLM_<CONFIG_NAME>` (e.g., `LAZYLLM_HOME`)
2. Config file: `~/.lazyllm/config.json`
3. Directly: `lazyllm.config['key'] = value`

Key config paths:
- `LAZYLLM_HOME`: Default `~/.lazyllm` - stores config and models
- `LAZYLLM_DATA_PATH`: Data directory for tests/examples
- `LAZYLLM_DEFAULT_RECENT_K`: Default retrieval count

## Code Style

- Python 3.10-3.12 required
- Use single quotes for strings
- Max line length: 121
- No `print()` statements (use logging instead)
- CI checks flake8 with custom plugins (`flake8-no-direct-imports`)

## Key Module Patterns

When creating new modules or components:
1. Inherit from appropriate base class in `lazyllm/components/` or `lazyllm/module/`
2. Use the `register` decorator to make components discoverable
3. Use `@lazyllm.component_register('group_name')` for component organization
4. For flows, use context managers: `with pipeline() as ppl:`

## Testing Structure

- `tests/basic_tests/`: Fast unit tests, no external services
- `tests/advanced_tests/`: Integration tests, may require local models
- `tests/charge_tests/`: Online API tests (require API keys)
- `tests/engine_tests/`: Inference/finetune engine tests
- `tests/doc_check/`: Documentation consistency checks

Test markers: `run_on_change`, `ignore_cache_on_change`, `skip_on_win`, `skip_on_mac`, `skip_on_linux`

## Build System

- Uses `scikit-build-core` with C++ extensions in `csrc/`
- Python bindings via `pybind11`
- Build without C++: `python -m build --config-setting=skbuild.wheel.cmake=false`

## Important Notes

- Models are auto-downloaded to `~/.lazyllm/model/` (symlinked from ModelScope cache)
- Web modules use Gradio 5.49.1
- Online services support: OpenAI, SenseNova, Kimi, GLM, Qwen, Doubao, SiliconFlow, Minimax
- Local inference: vLLM, LMDeploy, LightLLM
- Local finetuning: LLaMA-Factory, Collie, PEFT
