'''Session-local tool discovery and atomic exposure changes.'''
import copy
import re
import threading
from typing import Literal, Optional

import docstring_parser
from lazyllm import locals
from .toolError import ToolExecutionError
from .tool_runtime import _set_tool_runtime_metadata


class ToolRetrieval:
    def __init__(self, manager, *, required, groups, estimate_tokens, threshold_tokens,
                 state_store=None, skill_dependencies=None):
        self.manager = manager
        self.required = tuple(required)
        self.groups = set(groups)
        self.estimate_tokens = estimate_tokens
        self.threshold_tokens = threshold_tokens
        self.state_store = state_store
        self.skill_dependencies = skill_dependencies
        self._lock = threading.RLock()
        self._index_key = None
        self._index = None
        self._vocab = None

    def catalog(self):
        return self.manager.atomic_tool_catalog()

    def _expand(self, names, catalog):
        result = []
        for name in names:
            matches = [name] if name in catalog else [
                key for key, entry in catalog.items() if name in self.groups and name in entry['groups']]
            if not matches:
                raise ToolExecutionError(f'Tool or group is unavailable: {name}')
            result.extend(matches)
        return list(dict.fromkeys(result))

    def _reconcile(self, state, catalog):
        loaded = [name for name in state.get('loaded', []) if name in catalog]
        required = [name for name in self.required if name in catalog]
        skills = {}
        for skill in state.get('skills', {}):
            declarations = self.skill_dependencies(skill) if self.skill_dependencies else None
            if declarations is not None:
                # Resolve each declaration separately; revoked dependencies cannot grant access.
                names, _ = self._resolve_dependencies(declarations, catalog)
                skills[skill] = names
                required.extend(names)
        required = list(dict.fromkeys([*required, 'search_tools', 'load_tools']))
        return {'loaded': list(dict.fromkeys([*loaded, *required])), 'skills': skills}, set(required)

    def _local_state(self):
        states = locals.get('_tool_retrieval_states')
        if states is None:
            states = {}
            locals['_tool_retrieval_states'] = states
        # Parallel tool execution shallow-copies locals. Share this per-agent holder
        # so committed updates remain visible to the next model round.
        return states.setdefault(self.manager._module_id, {})

    def _read(self):
        return self.state_store.read() if self.state_store else self._local_state()

    def _commit(self, update):
        if self.state_store:
            return self.state_store.update(update)
        holder = self._local_state()
        state = update(copy.deepcopy(holder))
        holder.clear()
        holder.update(state)
        return state

    def initialize(self):
        with self._lock:
            catalog = self.catalog()
            self._commit(lambda raw: self._reconcile(raw, catalog)[0])

    def descriptions(self):
        with self._lock:
            catalog = self.catalog()
            state, _ = self._reconcile(self._read(), catalog)
            return [catalog[name]['schema'] for name in state['loaded'] if name in catalog]

    def usage(self, state, catalog):
        used = self.estimate_tokens([catalog[name]['schema'] for name in state['loaded']])
        return {'threshold_tokens': self.threshold_tokens,
                'used_tokens_estimate': used, 'over_threshold': used > self.threshold_tokens}

    @staticmethod
    def _parameter_text(schema):
        parts = []
        if isinstance(schema, dict):
            if isinstance(schema.get('description'), str):
                parts.append(schema['description'])
            for key, value in schema.items():
                if key in ('properties', '$defs') and isinstance(value, dict):
                    for name, child in value.items():
                        parts.extend([name, ToolRetrieval._parameter_text(child)])
                elif isinstance(value, (dict, list)):
                    parts.append(ToolRetrieval._parameter_text(value))
        elif isinstance(schema, list):
            parts.extend(ToolRetrieval._parameter_text(child) for child in schema)
        return ' '.join(parts)

    def search(self, query, limit, detail):
        if not query.strip() or not 1 <= limit <= 5 or detail not in ('short', 'long'):
            raise ToolExecutionError('Use a non-empty query, limit 1..5, and detail short or long.')
        from lazyllm.thirdparty import bm25s
        with self._lock:
            catalog = self.catalog()
            state, _ = self._reconcile(self._read(), catalog)
            names = list(catalog)
            texts = []
            for name, entry in catalog.items():
                function = entry['schema']['function']
                words = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name).replace('_', ' ')
                texts.append(' '.join([name, words, entry['source'], *entry['groups'],
                                       function.get('description') or '',
                                       self._parameter_text(function.get('parameters', {}))]))
            key = tuple(zip(names, texts))
            if key != self._index_key:
                tokens = bm25s.tokenize(texts, stopwords='en', show_progress=False)
                self._index = bm25s.BM25()
                self._index.index(tokens, show_progress=False)
                self._vocab = tokens.vocab
                self._index_key = key
            query_tokens = bm25s.tokenize(query.replace('_', ' '), stopwords='en', show_progress=False)
            ids = [self._vocab[word] for word in query_tokens.vocab if word in self._vocab]
            if not ids:
                return []
            scores = self._index.get_scores_from_ids(ids)
            candidates = {}
            for name, score in zip(names, scores):
                if name in state['loaded'] or score <= 0:
                    continue
                entry = catalog[name]
                group = next((g for g in reversed(entry['groups']) if g in self.groups), None)
                candidate = group or name
                description = entry['group_description'] if group else entry['schema']['function']['description']
                if candidate not in candidates or score > candidates[candidate][0]:
                    candidates[candidate] = (float(score), {
                        'name': candidate, 'type': 'group' if group else 'tool',
                        'description': description if detail == 'long' else (
                            docstring_parser.parse(description or '').short_description or description),
                    })
            return [value[1] for _, value in sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))[:limit]]

    def load(self, tool_names, unload_tool_names):
        with self._lock:
            catalog = self.catalog()
            added = self._expand(tool_names, catalog)
            removed = set(self._expand(unload_tool_names, catalog))

            def update(raw):
                state, protected = self._reconcile(raw, catalog)
                if removed & protected:
                    raise ToolExecutionError(f'Cannot unload required tools: {sorted(removed & protected)}')
                state['loaded'] = [name for name in state['loaded'] if name not in removed]
                if set(added) - set(state['loaded']) and self.usage(state, catalog)['over_threshold']:
                    raise ToolExecutionError('Tool context exceeds the loading threshold; unload optional tools first.')
                state['loaded'] = list(dict.fromkeys([*state['loaded'], *added]))
                return state

            state = self._commit(update)
            return {'status': 'ok', 'loaded': added, 'unloaded': sorted(removed), **self.usage(state, catalog)}

    def _resolve_dependencies(self, declarations, catalog):
        if declarations is None:
            return [], []
        if isinstance(declarations, str):
            declarations = re.split(r'[\s,]+', declarations.strip())
        if not isinstance(declarations, list) or not all(isinstance(name, str) for name in declarations):
            return [], ['Invalid allowed-tools declaration']
        names, errors = [], []
        for name in filter(None, declarations):
            try:
                names.extend(self._expand([name], catalog))
            except ToolExecutionError as error:
                errors.append(str(error))
        return list(dict.fromkeys(names)), errors

    def load_skill(self, name, declarations):
        with self._lock:
            catalog = self.catalog()
            names, errors = self._resolve_dependencies(declarations, catalog)

            def update(raw):
                state, _ = self._reconcile(raw, catalog)
                state['skills'][name] = names
                state['loaded'] = list(dict.fromkeys([*state['loaded'], *names]))
                return state

            state = self._commit(update)
            return {'loaded': names, 'unavailable': errors, **self.usage(state, catalog)}

    def tools(self):
        def search_tools(query: str, limit: int = 5, detail: Literal['short', 'long'] = 'short') -> list:
            '''Find allowed tools without loading their schemas. Prefer English capability keywords.

            Args:
                query (str): English tool capability keywords, not a business data query.
                limit (int): Candidate count, between 1 and 5. Defaults to 5.
                detail (Literal['short', 'long']): Short summary or full description.
            '''
            return self.search(query, limit, detail)

        def load_tools(tool_names: Optional[list[str]] = None,
                       unload_tool_names: Optional[list[str]] = None) -> dict:
            '''Atomically load tools/groups and unload optional tools. Use new tools only next model round.

            Args:
                tool_names (Optional[list[str]]): Exact tool or group names to load.
                unload_tool_names (Optional[list[str]]): Tool or group names to unload before loading.
            '''
            return self.load(tool_names or [], unload_tool_names or [])
        resource = f'tool-retrieval:{self.manager._module_id}'
        _set_tool_runtime_metadata(search_tools, dict(
            execute_in_sandbox=False, host_file='NONE', read_keys=(resource,)))
        _set_tool_runtime_metadata(load_tools, dict(
            execute_in_sandbox=False, host_file='NONE', write_keys=(resource,)))
        return [search_tools, load_tools]
