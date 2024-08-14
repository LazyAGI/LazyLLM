from typing import Dict, Union, Any, List


def chat_history_to_str(history: List[Union[List[str], Dict[str, Any]]] = [], user_query: Union[str, None] = None):
    MAX_HISTORY_LEN = 20
    history_info = ""
    MAP_ROLE = {"user": "human", "assitant": "assitant"}
    if history:
        if isinstance(history[0], list):
            for chat_msg in history[-MAX_HISTORY_LEN:]:
                assert len(chat_msg) == 2
                history_info += f"human: {chat_msg[0]}\nassistant: {chat_msg[1]}\n"
        elif isinstance(history[0], dict):
            for chat_msg in history[-2 * MAX_HISTORY_LEN:]:
                cur_role = chat_msg.get("role", "")
                if cur_role not in MAP_ROLE:
                    continue
                history_info += f"{MAP_ROLE[cur_role]}\n"
        else:
            raise ValueError(f"Unexpected type for history: {type(history[0])}")
    if user_query:
        history_info += f"human: {user_query}\n"
    return history_info
