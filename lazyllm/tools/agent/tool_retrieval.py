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
                 state_store=None, skill_dependencies=None, group_descriptions=None,
                 group_members=None, validate_load=None):
        self.manager = manager
        self.required = tuple(required)
        self.group_members = {name: tuple(members) for name, members in (group_members or {}).items()}
        self.groups = set(groups) | self.group_members.keys()
        self.validate_load = validate_load
        self.group_descriptions = dict(group_descriptions or {})
        self.estimate_tokens = estimate_tokens
        self.threshold_tokens = threshold_tokens
        self.state_store = state_store
        self.skill_dependencies = skill_dependencies
        self._lock = threading.RLock()
        self._index_key = None
        self._index = None
        self._vocab = None
        self._member_indexes = {}

    def catalog(self):
        catalog = self.manager.atomic_tool_catalog()
        for group, members in self.group_members.items():
            for name in members:
                if name in catalog:
                    entry = catalog[name]
                    catalog[name] = {**entry, 'groups': (*entry['groups'], group),
                                     'group_descriptions': {**entry['group_descriptions'],
                                                            group: self.group_descriptions.get(group, '')}}
        return catalog

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

    def _commit(self, update, catalog):
        def validated_update(raw):
            state = update(raw)
            if self.validate_load is not None:
                self.validate_load([catalog[name]['schema'] for name in state['loaded']])
            return state

        if self.state_store:
            return self.state_store.update(validated_update)
        holder = self._local_state()
        state = validated_update(copy.deepcopy(holder))
        holder.clear()
        holder.update(state)
        return state

    def initialize(self):
        with self._lock:
            catalog = self.catalog()
            self._commit(lambda raw: self._reconcile(raw, catalog)[0], catalog)

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

    @staticmethod
    def _words(name):
        return re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name).replace('_', ' ')

    @classmethod
    def _member_text(cls, name, entry):
        function = entry['schema']['function']
        return ' '.join([name, cls._words(name), function.get('description') or '',
                         cls._parameter_text(function.get('parameters', {}))])

    @staticmethod
    def _build_index(texts):
        from lazyllm.thirdparty import bm25s
        tokens = bm25s.tokenize(texts, stopwords='en', show_progress=False)
        index = bm25s.BM25()
        index.index(tokens, show_progress=False)
        return index, tokens.vocab

    @staticmethod
    def _scores(index, vocab, query):
        from lazyllm.thirdparty import bm25s
        tokens = bm25s.tokenize(query.replace('_', ' '), stopwords='en', show_progress=False)
        ids = [vocab[word] for word in tokens.vocab if word in vocab]
        return index.get_scores_from_ids(ids) if ids else []

    @staticmethod
    def _summary(description):
        return docstring_parser.parse(description or '').short_description or description or ''

    def _candidates(self, catalog, loaded):
        candidates = {}
        for name, entry in catalog.items():
            if name in loaded:
                continue
            group = next((g for g in reversed(entry['groups']) if g in self.groups), None)
            candidate = group or name
            if candidate not in candidates:
                description = (entry['group_descriptions'][group] if group else
                               entry['schema']['function'].get('description') or '')
                candidates[candidate] = {
                    'type': 'group' if group else 'tool', 'description': description,
                    'members': [], 'texts': [candidate, self._words(candidate), description] if group else [],
                }
            item = candidates[candidate]
            item['members'].append(name)
            item['texts'].append(' '.join([self._member_text(name, entry), entry['source'], *entry['groups']]))
        return candidates

    def _matched_members(self, name, members, catalog, query, loaded):
        rows = [(member, self._member_text(member, catalog[member])) for member in members if member not in loaded]
        key = tuple(rows)
        cached = self._member_indexes.get(name)
        if cached is None or cached[0] != key:
            index, vocab = self._build_index([text for _, text in rows])
            cached = self._member_indexes[name] = (key, index, vocab)
        scores = self._scores(cached[1], cached[2], query)
        ranked = sorted(((member, float(score)) for (member, _), score in zip(rows, scores) if score > 0),
                        key=lambda item: (-item[1], item[0]))[:3]
        return [{'name': member, 'description': self._summary(catalog[member]['schema']['function'].get('description'))}
                for member, _ in ranked]

    def search(self, query, limit, detail):
        if not query.strip() or not 1 <= limit <= 5 or detail not in ('short', 'long'):
            raise ToolExecutionError('Use a non-empty query, limit 1..5, and detail short or long.')
        with self._lock:
            catalog = self.catalog()
            state, _ = self._reconcile(self._read(), catalog)
            loaded = set(state['loaded'])
            candidates = self._candidates(catalog, loaded)
            if not candidates:
                return []
            names = list(candidates)
            texts = [' '.join(item['texts']) for item in candidates.values()]
            key = tuple(zip(names, texts))
            if key != self._index_key:
                self._index, self._vocab = self._build_index(texts)
                self._index_key = key
                self._member_indexes.clear()
            scores = self._scores(self._index, self._vocab, query)
            ranked = sorted(((name, float(score)) for name, score in zip(names, scores)
                             if score > 0),
                            key=lambda item: (-item[1], item[0]))[:limit]
            results = []
            for name, _ in ranked:
                item = candidates[name]
                description = (self.group_descriptions.get(name, item['description'])
                               if item['type'] == 'group' else item['description'])
                result = {'name': name, 'type': item['type'],
                          'description': description if detail == 'long' else self._summary(description)}
                if item['type'] == 'group':
                    result['matched_members'] = self._matched_members(name, item['members'], catalog, query, loaded)
                results.append(result)
            return results

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

            state = self._commit(update, catalog)
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

            state = self._commit(update, catalog)
            return {'loaded': names, 'unavailable': errors, **self.usage(state, catalog)}

    def tools(self):
        def search_tools(query: str, limit: int = 5, detail: Literal['short', 'long'] = 'short') -> list:
            '''Find allowed tools or groups without loading schemas. Prefer English capability keywords.

            Group member summaries explain matching capabilities, not every member. Load the group
            to expose complete member schemas next round; no get_*_methods activation is needed.

            Args:
                query (str): English tool capability keywords, not a business data query.
                limit (int): Candidate count, between 1 and 5. Defaults to 5.
                detail (Literal['short', 'long']): Short summary or full description.
            '''
            return self.search(query, limit, detail)

        def load_tools(tool_names: Optional[list[str]] = None,
                       unload_tool_names: Optional[list[str]] = None) -> dict:
            '''Atomically load tools/groups and unload optional tools. Use new tools only next model round.

            Grouped tools are usually complementary and intended to work together. Prefer loading the group;
            load an individual member only when the required capability is clearly limited to that tool.

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
