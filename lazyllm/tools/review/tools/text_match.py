import re
from difflib import SequenceMatcher
from typing import List, Dict

def normalize_text(s: str) -> str:
    s = re.sub(r"[ \t\n，。,.!?！；;：“”\"'（）()《》<>【】|]", '', s)
    s = re.sub(r'-{2,}', '', s)
    return s

def partial_match_ratio(a: str, b: str, isjunk=None) -> float:
    matcher = SequenceMatcher(isjunk, a, b)
    matches = matcher.get_matching_blocks()
    match_len = sum(block.size for block in matches if block.size > 1)
    return match_len / len(a) if len(a) > 0 else 0.0

def find_fuzzy_lines(lines: List[Dict], sentence: str, threshold: float = 0.08) -> List[int]:
    if not lines:
        return []

    sims = [
        partial_match_ratio(sentence, line.get('content', ''))
        for line in lines
    ]

    if not sims:
        return []

    start_idx = max(range(len(sims)), key=lambda i: sims[i])
    best_group = [start_idx]
    best_score = sims[start_idx]

    if best_score > 0.9:
        return best_group

    current_text = lines[start_idx].get('content', '')
    for i in range(start_idx - 1, -1, -1):
        new_text = lines[i].get('content', '') + current_text
        new_score = partial_match_ratio(sentence, new_text)
        if new_score > best_score + threshold:
            best_score = new_score
            best_group.insert(0, i)
            current_text = new_text
        else:
            break

    for j in range(start_idx + 1, len(lines)):
        new_text = current_text + lines[j].get('content', '')
        new_score = partial_match_ratio(sentence, new_text)
        if new_score > best_score + threshold:
            best_score = new_score
            best_group.append(j)
            current_text = new_text
        else:
            break

    return best_group

def coverage_ignore_order(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    b_set = set(b)
    count = sum(1 for ch in a if ch in b_set)
    return count / len(a)
