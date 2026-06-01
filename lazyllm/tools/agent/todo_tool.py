from lazyllm import locals as lazyllm_locals
from .toolsManager import register
from typing import List, Dict

_VALID_STATUSES = {'pending', 'in_progress', 'completed', 'cancelled'}

_TODO_HEADER = (
    'Successfully updated TODOs. Make sure to follow and update your TODO list as you make progress. '
    'Cancel and add new TODO tasks as needed when the user makes a correction or follow-up request.'
)


@register('builtin_tools', execute_in_sandbox=False)
def todo_write(todos: List[Dict], merge: bool = False) -> str:
    '''Write or update the todo list for the current agent session.

    Use this tool to create a plan at the start of a task, and update progress as you work.

    Args:
        todos (List[Dict]): Todo items to write or update; each dict needs id, content, status.
        merge (bool): If False, replace the list; if True, merge by id. Defaults to False.

    Returns:
        str: Confirmation message with the current state of the todo list.
'''
    for item in todos:
        if not all(k in item for k in ('id', 'content', 'status')):
            raise ValueError(f'Each todo item must contain "id", "content" and "status", got: {item}')
        if item['status'] not in _VALID_STATUSES:
            raise ValueError(
                f'Invalid status "{item["status"]}" for todo id="{item["id"]}". '
                f'Must be one of: {", ".join(sorted(_VALID_STATUSES))}.'
            )

    agent_ctx = lazyllm_locals['_lazyllm_agent']
    current: Dict[str, Dict] = agent_ctx.get('todo_list', {})

    if not merge:
        updated = {item['id']: item for item in todos}
    else:
        updated = dict(current)
        for item in todos:
            updated[item['id']] = item

    agent_ctx['todo_list'] = updated

    lines = [_TODO_HEADER, '', 'Here are the latest contents of your todo list:']
    for item in updated.values():
        lines.append(f'- **{item["status"].upper()}**: {item["content"]} (id: {item["id"]})')
    return '\n'.join(lines)
