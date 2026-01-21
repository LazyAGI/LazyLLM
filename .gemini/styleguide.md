# Code Review Guidance

## Overview  
This codebase frequently encounters risks related to configuration flexibility, inheritance complexity, and cross-system consistency. Tight coupling between components, ambiguous parameter handling, and platform-specific assumptions often lead to subtle bugs in distributed workflows. A structured review approach ensures robustness across deployment scenarios, prevents technical debt from implicit dependencies, and maintains API coherence as new modules integrate with existing systems.

## Review Categories  

### Configuration Management and Environment Interaction  
Configuration-related issues often stem from conflicting sources of truth, insufficient validation, or platform-specific assumptions. Hardcoded paths, unvalidated environment variables, and improper global state manipulation create unpredictable behavior across environments. Mismanagement here risks deployment failures, security vulnerabilities, and debugging challenges.  

Review points:  
- Prefer `lazyllm.config` over direct environment variable access for consistent configuration resolution  
- Validate TOML/payload inputs with fallback mechanisms for missing keys (e.g., default API key generation)  
- Avoid modifying global state in shared utilities to prevent cross-thread/session conflicts  
- Use OS-agnostic path normalization for cross-platform reliability (e.g., SQLite path validation)  

### Class Design and Inheritance Principles  
Inheritance hierarchies and class-level attributes require careful alignment between interface contracts and implementation details. Improper attribute scoping, signature mismatches, and redundant overrides erode maintainability. Poorly designed registries or base classes create cascading issues during extension.  

Review points:  
- Avoid tight coupling between base classes and implementation-specific configurations (e.g., `Register` depending on module-specific settings)  
- Ensure child class methods maintain behavioral consistency with parent signatures  
- Remove redundant methods that merely delegate to parent implementations without modification  
- Resolve registration conflicts arising from overlapping multiple inheritance conditions  

### API Design and Interface Consistency  
API surfaces must balance flexibility with explicitness to prevent misuse. Inconsistent naming conventions, hidden parameters in `**kwargs`, and misaligned documentation create confusion for consumers. Ambiguous parameter expectations undermine reliability in complex workflows.  

Review points:  
- Standardize parameter naming across implementations and documentation (e.g., `model`/`url` vs `model_name`/`base_url`)  
- Explicitly declare critical parameters in function signatures rather than obscuring them in variable arguments  
- Synchronize documented parameters (e.g., `criteria` vs `filters`) with implementation behavior  
- Clarify necessity of special-case logic in interfaces (e.g., `DEFAULT_INDEX_BODY` checks)  

### Concurrency and Resource Handling  
Multi-threaded execution and external resource management demand disciplined error handling and lifecycle control. Improper thread pool usage, unchecked global caching, and missing context managers lead to resource leaks or race conditions that manifest intermittently.  

Review points:  
- Use context managers (`with`)